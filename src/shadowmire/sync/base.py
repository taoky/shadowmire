import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import islice
from os.path import normpath
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote

from tqdm import tqdm

from ..constants import (
    IOWORKERS,
    LOCAL_DB_SERIAL_NAME,
    MAX_DELETION,
    PACKAGE_FILES_METADATA_ONLY,
    PACKAGE_FILES_PENDING,
    PACKAGE_NOT_FOUND_SERIAL,
    WORKERS,
)
from ..database import LocalVersionKV
from ..errors import ExitProgramException, exit_with_futures
from ..filesystem import overwrite, remove_dir_with_files
from ..filters import FileInclusionChecker, PackageInclusionChecker
from ..simple import (
    file_url_to_local_url,
    generate_html_simple_page,
    generate_json_simple_page,
    get_existing_hrefs,
    get_package_urls_from_index_html,
    get_package_urls_from_index_json,
    get_package_urls_size_from_index_json,
    get_release_files,
)

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    """Operations required to make the local mirror match the current policy."""

    # Remove the entire project, including simple/JSON metadata and package files.
    remove: list[str]
    # Refresh project metadata and, when enabled by policy, its package files.
    update: list[str]
    # Remove only package files while retaining project metadata (metadata-only).
    package_remove: list[str]
    # Persist already-verified file states without performing package IO.
    package_state_update: dict[str, int]
    # Upstream serial written to the root simple indexes after applying the plan.
    remote_last_serial: int


class SyncBase:
    def __init__(
        self, basedir: Path, local_db: LocalVersionKV, sync_packages: bool = False
    ) -> None:
        self.basedir = basedir
        self.local_db = local_db
        self.simple_dir = basedir / "simple"
        self.packages_dir = basedir / "packages"
        self.jsonmeta_dir = basedir / "json"
        # create the dirs, if not exist
        self.simple_dir.mkdir(parents=True, exist_ok=True)
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        self.jsonmeta_dir.mkdir(parents=True, exist_ok=True)
        self.sync_packages = sync_packages

    def filter_remote(
        self, remote: dict[str, int], package_inclusion_checker: PackageInclusionChecker
    ) -> dict[str, int]:
        if not package_inclusion_checker.has_rules():
            return remote
        res = {}
        for k, v in remote.items():
            if package_inclusion_checker.includes_metadata(k):
                res[k] = v
        return res

    def inspect_package_files(
        self, package_name: str, package_files_included: bool
    ) -> Literal["remove", "update"] | None:
        """
        Return the package-file operation required for a package, if any.
        """
        if package_files_included and not self.sync_packages:
            return None

        package_simple_path = self.simple_dir / package_name
        hrefs = get_existing_hrefs(package_simple_path)
        if hrefs is None:
            return "update" if package_files_included else None

        for href, has_metadata in hrefs:
            relative_path = unquote(href)
            package_path = Path(normpath(package_simple_path / relative_path))
            metadata_path = package_path.with_name(package_path.name + ".metadata")
            has_existing = package_path.exists() or (
                has_metadata and metadata_path.exists()
            )
            has_missing = not package_path.exists() or (
                has_metadata and not metadata_path.exists()
            )
            if package_files_included and has_missing:
                return "update"
            if not package_files_included and has_existing:
                return "remove"
        return None

    def determine_package_file_plan(
        self,
        package_serials: dict[str, int],
        package_inclusion_checker: PackageInclusionChecker,
    ) -> tuple[list[str], list[str], dict[str, int]]:
        if not package_inclusion_checker.filters_package_files():
            return [], [], {}

        package_remove = []
        package_update = []
        package_state_update = {}
        with ThreadPoolExecutor(max_workers=IOWORKERS) as executor:
            package_iter = iter(package_serials)
            with tqdm(
                total=len(package_serials), desc="Checking package filter"
            ) as pbar:
                while batch := list(islice(package_iter, IOWORKERS * 100)):
                    futures = {
                        executor.submit(
                            self.inspect_package_files,
                            package_name,
                            package_inclusion_checker.includes_package_files(
                                package_name
                            ),
                        ): package_name
                        for package_name in batch
                    }
                    for future in as_completed(futures):
                        package_name = futures[future]
                        action = future.result()
                        if action == "remove":
                            package_remove.append(package_name)
                        elif action == "update":
                            package_update.append(package_name)
                        elif package_inclusion_checker.includes_package_files(
                            package_name
                        ):
                            package_state_update[package_name] = (
                                package_serials[package_name]
                                if self.sync_packages
                                else PACKAGE_FILES_PENDING
                            )
                        else:
                            package_state_update[package_name] = (
                                PACKAGE_FILES_METADATA_ONLY
                            )
                        pbar.update(1)

        return package_remove, package_update, package_state_update

    def determine_sync_plan(
        self,
        local: dict[str, int],
        package_inclusion_checker: PackageInclusionChecker,
        reconcile_package_files: bool = False,
        local_file_serials: dict[str, int | None] | None = None,
    ) -> Plan:
        """
        local should NOT skip PACKAGE_NOT_FOUND_SERIAL entries
        """
        remote_sn, remote_pkgs = self.fetch_remote_versions()
        remote_pkgs = self.filter_remote(remote_pkgs, package_inclusion_checker)
        with open(self.basedir / "remote_excluded.json", "w") as f:
            json.dump(remote_pkgs, f)

        to_remove = []
        local_keys = set(local.keys())
        remote_keys = set(remote_pkgs.keys())
        for i in local_keys - remote_keys:
            to_remove.append(i)
            local_keys.remove(i)
        # There are always some packages in PyPI's list_packages_with_serial() but actually not there
        # Don't count them when comparing len(to_remove) with MAX_DELETION
        if len(to_remove) > MAX_DELETION:
            logger.error(
                "Too many packages to remove (%d > %d)", len(to_remove), MAX_DELETION
            )
            logger.info("Some packages that would be removed:")
            for p in to_remove[:100]:
                logger.info("- %s", p)
            for p in to_remove[100:]:
                logger.debug("- %s", p)
            logger.error(
                "Use SHADOWMIRE_MAX_DELETION env to adjust the threshold if you really want to proceed"
            )
            sys.exit(2)
        to_update = list(remote_keys - local_keys)
        for i in local_keys:
            local_serial = local[i]
            remote_serial = remote_pkgs[i]
            if local_serial != remote_serial:
                if local_serial == PACKAGE_NOT_FOUND_SERIAL:
                    logger.info("skip %s, as it's marked as not exist at upstream", i)
                    to_remove.append(i)
                else:
                    to_update.append(i)
        package_remove = []
        package_state_update = {}
        retained_packages = local_keys & remote_keys
        retained_unchanged = retained_packages - set(to_update) - set(to_remove)
        if reconcile_package_files:
            package_serials = {
                package_name: remote_pkgs[package_name]
                for package_name in retained_unchanged
            }
            (
                package_remove,
                package_update,
                package_state_update,
            ) = self.determine_package_file_plan(
                package_serials, package_inclusion_checker
            )
            to_update.extend(package_update)
        elif local_file_serials is not None:
            for package_name in retained_unchanged:
                package_files_included = (
                    package_inclusion_checker.includes_package_files(package_name)
                )
                file_serial = local_file_serials.get(package_name)
                # NULL means the old Shadowmire behavior: package files are
                # assumed to match the metadata serial. This avoids an IO spike
                # solely because the schema gained a new nullable column.
                effective_file_serial = (
                    local[package_name] if file_serial is None else file_serial
                )
                if package_files_included:
                    if (
                        self.sync_packages
                        and effective_file_serial != remote_pkgs[package_name]
                    ):
                        to_update.append(package_name)
                elif (
                    file_serial is not None
                    and file_serial != PACKAGE_FILES_METADATA_ONLY
                ):
                    package_remove.append(package_name)
        output = Plan(
            remove=sorted(set(to_remove)),
            update=sorted(set(to_update)),
            package_remove=sorted(package_remove),
            package_state_update=package_state_update,
            remote_last_serial=remote_sn,
        )
        return output

    def fetch_remote_versions(self) -> tuple[int, dict[str, int]]:
        # returns (last_serial, {package_name: serial, ...})
        raise NotImplementedError

    def get_package_metadata(self, package_name: str) -> dict:
        raise NotImplementedError

    def get_package_simple(self, package_name: str) -> dict:
        raise NotImplementedError

    def get_core_metadata_map(self, simple: dict) -> dict:
        """
        get a filename to core-metadata map from simple API info for PEP 658 implementation.
        """
        files = simple.get("files", [])
        if not files:
            return {}

        file_map = {
            f["filename"]: f.get(
                "core-metadata",
                # Fallback for legacy PEP 714 attribute
                f.get("data-dist-info-metadata", False),
            )
            for f in files
        }
        return file_map

    def check_and_update(
        self,
        package_names: list[str],
        package_inclusion_checker: PackageInclusionChecker,
        file_inclusion_checker: FileInclusionChecker,
        json_files: set[str],
        packages_pathcache: set[str],
        compare_size: bool,
    ) -> bool:
        def is_consistent(package_name: str) -> bool:
            if package_name not in json_files:
                # save a newfstatat() when name already in json_files
                logger.info("add %s as it does not have json API file", package_name)
                return False
            package_simple_path = self.simple_dir / package_name
            html_simple = package_simple_path / "index.html"
            htmlv1_simple = package_simple_path / "index.v1_html"
            json_simple = package_simple_path / "index.v1_json"
            try:
                # always create index.html symlink, if not exists or not a symlink
                if not html_simple.is_symlink():
                    html_simple.unlink(missing_ok=True)
                    html_simple.symlink_to("index.v1_html")
                hrefs_html = get_package_urls_from_index_html(htmlv1_simple)
                hrefsize_json = get_package_urls_size_from_index_json(json_simple)
                href_metadata_json = dict(get_package_urls_from_index_json(json_simple))
            except FileNotFoundError:
                logger.info(
                    "add %s as it does not have index.v1_html or index.v1_json",
                    package_name,
                )
                return False
            if (
                hrefs_html is None
                or hrefsize_json is None
                or hrefs_html != [i[0] for i in hrefsize_json]
            ):
                # something unexpected happens...
                logger.info("add %s as its indexes are not consistent", package_name)
                return False
            # Check with JSON meta, ensuring that package file list is consistent
            json_meta_path = self.jsonmeta_dir / package_name
            try:
                with open(json_meta_path, "r") as f:
                    meta = json.load(f)
                meta = file_inclusion_checker.get_filtered_meta(package_name, meta)
                release_files = get_release_files(meta)
                hrefs_from_meta = {
                    file_url_to_local_url(i["url"]) for i in release_files
                }
            except (json.JSONDecodeError, FileNotFoundError, KeyError):
                logger.info(
                    "add %s as its JSON meta is not valid",
                    package_name,
                )
                return False
            for href in hrefs_html:
                if href not in hrefs_from_meta:
                    logger.info(
                        "add %s as its HTML index has href %s not in JSON meta",
                        package_name,
                        href,
                    )
                    return False

            package_files_included = package_inclusion_checker.includes_package_files(
                package_name
            )
            if not package_files_included:
                self.remove_package_files_from_hrefs(
                    package_simple_path, list(href_metadata_json.items())
                )

            # OK, check if all hrefs have corresponding files
            if self.sync_packages and package_files_included:
                for href, size in hrefsize_json:
                    relative_path = unquote(href)
                    dest_pathstr = normpath(package_simple_path / relative_path)
                    try:
                        # Fast shortcut to avoid stat() it
                        if dest_pathstr not in packages_pathcache:
                            raise FileNotFoundError
                        if href_metadata_json.get(href, False):
                            metadata_pathstr = dest_pathstr + ".metadata"
                            if metadata_pathstr not in packages_pathcache:
                                raise FileNotFoundError
                        if compare_size and size != -1:
                            dest = Path(dest_pathstr)
                            # So, do stat() for real only when we need to do so,
                            # have a size, and it really exists in pathcache.
                            dest_stat = dest.stat()
                            dest_size = dest_stat.st_size
                            if dest_size != size:
                                logger.info(
                                    "add %s as its local size %s != %s",
                                    package_name,
                                    dest_size,
                                    size,
                                )
                                return False
                    except FileNotFoundError:
                        logger.info("add %s as it's missing packages", package_name)
                        return False

            return True

        to_update = []
        package_state_update: dict[str, int] = {}
        local_serials = self.local_db.dump()
        with ThreadPoolExecutor(max_workers=IOWORKERS) as executor:
            futures = {
                executor.submit(is_consistent, package_name): package_name
                for package_name in package_names
            }
            try:
                for future in tqdm(
                    as_completed(futures),
                    total=len(package_names),
                    desc="Checking consistency",
                ):
                    package_name = futures[future]
                    try:
                        consistent = future.result()
                        if not consistent:
                            to_update.append(package_name)
                        elif not package_inclusion_checker.includes_package_files(
                            package_name
                        ):
                            package_state_update[package_name] = (
                                PACKAGE_FILES_METADATA_ONLY
                            )
                        elif self.sync_packages:
                            package_state_update[package_name] = local_serials[
                                package_name
                            ]
                    except Exception:
                        logger.warning(
                            "%s generated an exception", package_name, exc_info=True
                        )
                        raise
            except:
                exit_with_futures(futures)

        logger.info("%s packages to update in check_and_update()", len(to_update))
        if package_state_update:
            self.local_db.batch_set_file_serials(package_state_update)
        return self.parallel_update(
            to_update, package_inclusion_checker, file_inclusion_checker
        )

    def parallel_update(
        self,
        package_names: list[str],
        package_inclusion_checker: PackageInclusionChecker,
        file_inclusion_checker: FileInclusionChecker,
    ) -> bool:
        success = True
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(
                    self.do_update,
                    package_name,
                    file_inclusion_checker,
                    package_inclusion_checker.includes_package_files(package_name),
                    False,
                ): (
                    idx,
                    package_name,
                )
                for idx, package_name in enumerate(package_names)
            }
            try:
                for future in tqdm(
                    as_completed(futures), total=len(package_names), desc="Updating"
                ):
                    idx, package_name = futures[future]
                    try:
                        serial = future.result()
                        if serial:
                            self.record_local_update(
                                package_name,
                                serial,
                                package_inclusion_checker.includes_package_files(
                                    package_name
                                ),
                            )
                    except Exception as e:
                        if isinstance(e, (KeyboardInterrupt)):
                            raise
                        logger.warning(
                            "%s generated an exception", package_name, exc_info=True
                        )
                        success = False
                    if idx % 100 == 0:
                        logger.info("dumping local db...")
                        self.local_db.dump_json()
            except (ExitProgramException, KeyboardInterrupt):
                exit_with_futures(futures)
        return success

    def record_local_update(
        self, package_name: str, serial: int, package_files_included: bool
    ) -> None:
        if package_files_included:
            file_serial = serial if self.sync_packages else PACKAGE_FILES_PENDING
        else:
            file_serial = PACKAGE_FILES_METADATA_ONLY
        self.local_db.set_with_file_serial(package_name, serial, file_serial)

    def do_sync_plan(
        self,
        plan: Plan,
        package_inclusion_checker: PackageInclusionChecker,
        file_inclusion_checker: FileInclusionChecker,
    ) -> bool:
        to_remove = plan.remove
        to_update = plan.update

        for package_name in to_remove:
            self.do_remove(package_name)

        logger.info(
            "%s packages have files excluded by package filters",
            len(plan.package_remove),
        )
        for package_name in plan.package_remove:
            self.remove_package_files(package_name)
            self.local_db.set_file_serial(package_name, PACKAGE_FILES_METADATA_ONLY)

        if plan.package_state_update:
            self.local_db.batch_set_file_serials(plan.package_state_update)

        return self.parallel_update(
            to_update, package_inclusion_checker, file_inclusion_checker
        )

    def remove_package_files(self, package_name: str) -> None:
        package_simple_dir = self.simple_dir / package_name
        package_files = get_existing_hrefs(package_simple_dir)
        if not package_files:
            return

        self.remove_package_files_from_hrefs(package_simple_dir, package_files)

    def remove_package_files_from_hrefs(
        self,
        package_simple_dir: Path,
        package_files: list[tuple[str, bool]],
    ) -> None:
        for href, has_metadata in package_files:
            relative_path = unquote(href)
            package_path = Path(normpath(package_simple_dir / relative_path))
            if package_path.exists():
                package_path.unlink()
                logger.info("Removed package file %s", package_path)
            if has_metadata:
                metadata_path = package_path.with_name(package_path.name + ".metadata")
                if metadata_path.exists():
                    metadata_path.unlink()
                    logger.info("Removed package metadata file %s", metadata_path)

    def do_remove(
        self, package_name: str, use_db: bool = True, remove_packages: bool = True
    ) -> None:
        metajson_path = self.jsonmeta_dir / package_name
        package_simple_dir = self.simple_dir / package_name
        if metajson_path.exists() or package_simple_dir.exists():
            # To make this less noisy...
            logger.info("Removing package %s", package_name)
        packages_to_remove = get_existing_hrefs(package_simple_dir)
        if remove_packages and packages_to_remove:
            paths_to_remove = []
            for p, has_metadata in packages_to_remove:
                path = package_simple_dir / p
                paths_to_remove.append(path)
                if has_metadata:
                    paths_to_remove.append(path.with_name(path.name + ".metadata"))
            for p in paths_to_remove:
                if p.exists():
                    p.unlink()
                    logger.info("Removed file %s", p)
        remove_dir_with_files(package_simple_dir)
        metajson_path = self.jsonmeta_dir / package_name
        metajson_path.unlink(missing_ok=True)
        if use_db:
            old_serial = self.local_db.get(package_name)
            if old_serial != PACKAGE_NOT_FOUND_SERIAL:
                self.local_db.remove(package_name)

    def do_update(
        self,
        package_name: str,
        file_inclusion_checker: FileInclusionChecker,
        package_files_included: bool,
        use_db: bool = True,
    ) -> int | None:
        raise NotImplementedError

    def write_meta_to_simple(
        self, package_simple_path: Path, meta: dict, core_metadata_map: dict
    ) -> None:
        simple_html_contents = generate_html_simple_page(meta, core_metadata_map)
        simple_json_contents = generate_json_simple_page(meta, core_metadata_map)
        for html_filename in ("index.v1_html",):
            html_path = package_simple_path / html_filename
            with overwrite(html_path) as f:
                f.write(simple_html_contents)
        for json_filename in ("index.v1_json",):
            json_path = package_simple_path / json_filename
            with overwrite(json_path) as f:
                f.write(simple_json_contents)
        index_html_path = package_simple_path / "index.html"
        if not index_html_path.is_symlink():
            if index_html_path.exists():
                index_html_path.unlink()
            index_html_path.symlink_to("index.v1_html")

    def finalize(self, index_serial: int) -> None:
        local_names = self.local_db.keys()
        # generate v1_html index
        v1_html_index_path = self.basedir / "simple" / "index.v1_html"
        # modified from bandersnatch
        with overwrite(v1_html_index_path) as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html>\n")
            f.write("  <head>\n")
            f.write('    <meta name="pypi:repository-version" content="1.0">\n')
            f.write("    <title>Simple Index</title>\n")
            f.write("  </head>\n")
            f.write("  <body>\n")
            # This will either be the simple dir, or if we are using index
            # directory hashing, a list of subdirs to process.
            for pkg in local_names:
                # We're really trusty that this is all encoded in UTF-8. :/
                f.write(f'    <a href="{pkg}/">{pkg}</a><br/>\n')
            f.write("  </body>\n</html>")
        # always link index.html to index.v1_html
        html_simple_path = self.basedir / "simple" / "index.html"
        if not html_simple_path.is_symlink():
            html_simple_path.unlink(missing_ok=True)
            html_simple_path.symlink_to("index.v1_html")

        # generate v1_json index and local.db{,.serial} for downstream use
        v1_json_index_path = self.basedir / "simple" / "index.v1_json"
        with overwrite(v1_json_index_path) as f:
            index_json: dict[str, Any] = {
                "meta": {
                    "api-version": "1.1",
                    "_last-serial": index_serial,
                },
                "projects": [{"name": n} for n in sorted(local_names)],
            }
            json.dump(index_json, f)
        with overwrite(self.basedir / LOCAL_DB_SERIAL_NAME) as f:
            f.write(str(index_serial))
        self.local_db.dump_json()

    def skip_this_package(self, i: dict, dest: Path, has_metadata: bool) -> bool:
        """
        A helper function for subclasses implementing do_update().
        As existence check is also done with stat(), this would not bring extra I/O overhead.
        Returns if skip this package or not.
        """
        try:
            if has_metadata:
                m_dest = dest.with_name(dest.name + ".metadata")
                if not m_dest.exists():
                    logger.warning(
                        "metadata %s not exists locally, so the package would still be downloaded.",
                        dest,
                    )
                    return False
            dest_size = dest.stat().st_size
            i_size = i.get("size", -1)
            if i_size == -1:
                return True
            if dest_size == i_size:
                return True
            logger.warning(
                "file %s exists locally, but size does not match with upstream, so it would still be downloaded.",
                dest,
            )
            return False
        except FileNotFoundError:
            return False
