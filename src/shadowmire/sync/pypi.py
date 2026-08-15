import json
import logging
from os.path import normpath
from pathlib import Path
from urllib.parse import unquote

from ..constants import (
    IGNORE_THRESHOLD,
    PACKAGE_FILES_METADATA_ONLY,
    PACKAGE_NOT_FOUND_SERIAL,
)
from ..database import LocalVersionKV
from ..errors import PackageNotFoundError
from ..filesystem import overwrite
from ..filters import FileInclusionChecker
from ..http import create_requests_session, download
from ..pypi import PyPI
from ..simple import (
    file_url_to_local_path,
    file_url_to_local_url,
    get_existing_hrefs,
    get_release_files,
)
from .base import SyncBase

logger = logging.getLogger("shadowmire")


class SyncPyPI(SyncBase):
    def __init__(
        self, basedir: Path, local_db: LocalVersionKV, sync_packages: bool = False
    ) -> None:
        self.pypi = PyPI()
        self.session = create_requests_session()
        self.last_serial: int | None = None
        self.remote_packages: dict[str, int] | None = None
        super().__init__(basedir, local_db, sync_packages)

    def fetch_remote_versions(self) -> tuple[int, dict[str, int]]:
        self.last_serial = self.pypi.changelog_last_serial()
        self.remote_packages = self.pypi.list_packages_with_serial()
        logger.info("Remote has %s packages", len(self.remote_packages))
        with overwrite(self.basedir / "remote.json") as f:
            json.dump(self.remote_packages, f)
            logger.info("File saved to remote.json.")
        return self.last_serial, self.remote_packages

    def get_package_metadata(self, package_name: str) -> dict:
        return self.pypi.get_package_metadata(package_name)

    def get_package_simple(self, package_name: str) -> dict:
        return self.pypi.get_package_simple(package_name)

    def do_update(
        self,
        package_name: str,
        file_inclusion_checker: FileInclusionChecker,
        package_files_included: bool,
        use_db: bool = True,
    ) -> int | None:
        logger.info("updating %s", package_name)
        package_simple_path = self.simple_dir / package_name
        exists = package_simple_path.exists()
        try:
            meta_original = self.get_package_metadata(package_name)
            logger.debug("%s meta: %s", package_name, meta_original)
        except PackageNotFoundError:
            if (
                self.remote_packages is not None
                and package_name in self.remote_packages
            ):
                recorded_serial = self.remote_packages[package_name]
            else:
                recorded_serial = None
            if (
                not exists  # When it exists locally, PyPI probably removes it and we need to do removal work.
                and recorded_serial is not None
                and self.last_serial is not None
                and abs(recorded_serial - self.last_serial) < IGNORE_THRESHOLD
            ):
                logger.warning(
                    "%s missing from upstream (its serial %s, remote last serial %s), try next time...",
                    package_name,
                    recorded_serial,
                    self.last_serial,
                )
                return None

            logger.warning(
                "%s missing from upstream (its serial %s, remote last serial %s), remove and ignore in the future.",
                package_name,
                recorded_serial,
                self.last_serial,
            )
            # try remove it locally, if it does not exist upstream
            self.do_remove(package_name, use_db=False)
            if not use_db:
                return PACKAGE_NOT_FOUND_SERIAL
            self.local_db.set_with_file_serial(
                package_name,
                PACKAGE_NOT_FOUND_SERIAL,
                PACKAGE_FILES_METADATA_ONLY,
            )
            return None
        if not exists:
            package_simple_path.mkdir(exist_ok=True)

        core_metadata_map = {}
        try:
            simple = self.get_package_simple(package_name)
            core_metadata_map = self.get_core_metadata_map(simple)
        except PackageNotFoundError:
            # Some mirrors may not implement PEP 691 simple API, just go ahead
            pass
        # filter prerelease and wheel files, if necessary
        meta = file_inclusion_checker.get_filtered_meta(package_name, meta_original)

        if not package_files_included:
            self.remove_package_files(package_name)
        elif self.sync_packages:
            # sync packages first, then sync index
            existing_hrefs = get_existing_hrefs(package_simple_path)
            existing_hrefs = (
                {} if existing_hrefs is None else {p: m for p, m in existing_hrefs}
            )
            release_files = get_release_files(meta)
            # remove packages that no longer exist remotely
            remote_hrefs = [file_url_to_local_url(i["url"]) for i in release_files]
            should_remove = list(set(existing_hrefs) - set(remote_hrefs))
            for href in should_remove:
                p = unquote(href)
                logger.info("removing file %s (if exists)", p)
                package_path = Path(normpath(package_simple_path / p))
                package_path.unlink(missing_ok=True)
                # Also remove associated metadata file
                if existing_hrefs.get(href, False):
                    metadata_path = package_path.with_name(
                        package_path.name + ".metadata"
                    )
                    metadata_path.unlink(missing_ok=True)
            for i in release_files:
                url = i["url"]
                dest = Path(
                    normpath(package_simple_path / file_url_to_local_path(i["url"]))
                )
                has_metadata = core_metadata_map.get(i["filename"], False)
                logger.info("downloading file %s -> %s", url, dest)
                if self.skip_this_package(i, dest, has_metadata):
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)
                success, _resp = download(self.session, url, dest)
                if not success:
                    logger.warning("skipping %s as it fails downloading", package_name)
                    return None

                # PEP 658: Download metadata file if available
                if has_metadata:
                    m_url = url + ".metadata"
                    m_dest = dest.with_name(dest.name + ".metadata")
                    logger.info("downloading metadata %s -> %s", m_url, m_dest)
                    m_success, _m_resp = download(self.session, m_url, m_dest)
                    if not m_success:
                        logger.warning(
                            "ignoring %s metadata as it fails downloading", package_name
                        )

        last_serial: int = meta["last_serial"]

        self.write_meta_to_simple(package_simple_path, meta, core_metadata_map)
        json_meta_path = self.jsonmeta_dir / package_name
        with overwrite(json_meta_path) as f:
            json.dump(meta_original, f)

        if use_db:
            self.record_local_update(package_name, last_serial, package_files_included)

        return last_serial
