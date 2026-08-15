import json
import logging
from os.path import normpath
from pathlib import Path
from urllib.parse import unquote, urljoin

import requests

from ..constants import LOCAL_DB_SERIAL_NAME, LOCAL_JSON_NAME
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


class SyncPlainHTTP(SyncBase):
    def __init__(
        self,
        upstream: str,
        basedir: Path,
        local_db: LocalVersionKV,
        sync_packages: bool = False,
        use_pypi_index: bool = False,
    ) -> None:
        self.upstream = upstream
        self.session = create_requests_session()
        self.pypi: PyPI | None
        if use_pypi_index:
            self.pypi = PyPI()
        else:
            self.pypi = None
        super().__init__(basedir, local_db, sync_packages)

    def fetch_remote_versions(self) -> tuple[int, dict[str, int]]:
        remote_pkgs: dict[str, int]
        if not self.pypi:
            remote_pkg_db_url = urljoin(self.upstream, LOCAL_JSON_NAME)
            resp = self.session.get(remote_pkg_db_url)
            resp.raise_for_status()
            remote_pkgs = resp.json()
            # first fallback to max serial in remote_pkgs
            serial = max(remote_pkgs.values()) if remote_pkgs else -1
            # then try to get last serial from remote
            remote_last_serial_url = urljoin(self.upstream, LOCAL_DB_SERIAL_NAME)
            try:
                resp = self.session.get(remote_last_serial_url)
                resp.raise_for_status()
                serial = int(resp.text.strip())
            except (requests.RequestException, ValueError):
                logger.warning(
                    f"cannot get last_serial from upstream, fallback to max package serial in {LOCAL_JSON_NAME}",
                    exc_info=True,
                )
        else:
            serial = self.pypi.changelog_last_serial()
            remote_pkgs = self.pypi.list_packages_with_serial()
        logger.info("Remote has %s packages", len(remote_pkgs))
        with overwrite(self.basedir / "remote.json") as f:
            json.dump(remote_pkgs, f)
            logger.info("File saved to remote.json.")
        return serial, remote_pkgs

    def get_package_metadata(self, package_name: str) -> dict:
        file_url = urljoin(self.upstream, f"json/{package_name}")
        success, resp = download(
            self.session, file_url, self.jsonmeta_dir / (package_name + ".new")
        )
        if not success:
            logger.error(
                "download %s JSON meta fails with code %s",
                package_name,
                resp.status_code if resp else None,
            )
            raise PackageNotFoundError
        assert resp
        return resp.json()

    def get_package_simple(self, package_name: str) -> dict:
        if not self.pypi:
            # Use shadowmire static file first for less consumption
            req = self.session.get(
                urljoin(self.upstream, f"simple/{package_name}/index.v1_json")
            )
            if req.status_code == 404:
                raise PackageNotFoundError
            return req.json()
        else:
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
        package_simple_path.mkdir(exist_ok=True)
        # Download JSON meta
        try:
            meta_original = self.get_package_metadata(package_name)
        except PackageNotFoundError:
            return None
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
            hrefs = get_existing_hrefs(package_simple_path)
            existing_hrefs = {} if hrefs is None else {p: m for p, m in hrefs}
            release_files = get_release_files(meta)
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
            package_simple_url = urljoin(self.upstream, f"simple/{package_name}/")
            for i in release_files:
                href = file_url_to_local_url(i["url"])
                path = file_url_to_local_path(i["url"])
                url = urljoin(package_simple_url, href)
                dest = Path(normpath(package_simple_path / path))
                logger.info("downloading file %s -> %s", url, dest)
                has_metadata = core_metadata_map.get(i["filename"], False)
                if self.skip_this_package(i, dest, has_metadata):
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                success, resp = download(self.session, url, dest)
                if not success:
                    if resp and resp.status_code == 404:
                        # handle special case: upstream filters out some files
                        logger.warning(
                            "cannot find %s at upstream, fallback to pypi", url
                        )
                        url = i["url"]  # original pypi URL
                        success, resp = download(self.session, url, dest)
                        if not success:
                            logger.warning(
                                "skipping %s as it fails downloading (from pypi)",
                                package_name,
                            )
                            return None
                    else:
                        logger.warning(
                            "skipping %s as it fails downloading", package_name
                        )
                        return None

                # PEP 658: Download metadata file if available
                if has_metadata:
                    # Try from upstream first, then fallback to PyPI if needed
                    m_url = url + ".metadata"
                    m_dest = dest.with_name(dest.name + ".metadata")
                    logger.info("downloading metadata %s -> %s", m_url, m_dest)
                    m_success, m_resp = download(self.session, m_url, m_dest)
                    if not m_success:
                        if m_resp and m_resp.status_code == 404:
                            pypi_m_url = i["url"] + ".metadata"
                            logger.warning(
                                "cannot find metadata %s at upstream, fallback to pypi",
                                m_url,
                            )
                            m_success, m_resp = download(
                                self.session, pypi_m_url, m_dest
                            )
                            if not m_success:
                                logger.warning(
                                    "ignoring %s metadata as it fails downloading (from pypi)",
                                    package_name,
                                )
                        else:
                            logger.warning(
                                "ignoring %s metadata as it fails downloading",
                                package_name,
                            )

        # OK, now it's safe to rename
        (self.jsonmeta_dir / (package_name + ".new")).rename(
            self.jsonmeta_dir / package_name
        )
        # generate indexes
        self.write_meta_to_simple(package_simple_path, meta_original, core_metadata_map)

        last_serial: int = meta["last_serial"]
        if use_db:
            self.record_local_update(package_name, last_serial, package_files_included)

        return last_serial
