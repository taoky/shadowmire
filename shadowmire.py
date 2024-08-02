import sys
from typing import Any, Optional
import xmlrpc.client
from dataclasses import dataclass
import re
import json
from urllib.parse import urljoin, urlparse, urlunparse
from pathlib import Path
from html.parser import HTMLParser
import logging
import html
import argparse
import os
from contextlib import contextmanager
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

USER_AGENT = "Shadowmire (https://github.com/taoky/shadowmire)"

# Note that it's suggested to use only 3 workers for PyPI.
WORKERS = int(os.environ.get("SHADOWMIRE_WORKERS", "3"))
if WORKERS > 10:
    logger.warning(
        "You have set a worker value larger than 10, which is forbidden by PyPI maintainers."
    )
    logger.warning("Don't blame me if you were banned!")


class PackageNotFoundError(Exception):
    pass


class ExitProgramException(Exception):
    pass


def exit_handler(signum, frame):
    raise ExitProgramException


signal.signal(signal.SIGTERM, exit_handler)


class LocalVersionKV:
    """
    A key-value database wrapper over sqlite3.

    As it would have consistency issue if it's writing while downstream is downloading the database.
    An extra "jsonpath" is used, to store kv results when necessary.
    """

    def __init__(self, dbpath: Path, jsonpath: Path) -> None:
        self.conn = sqlite3.connect(dbpath)
        self.jsonpath = jsonpath
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS local(key TEXT PRIMARY KEY, value INT NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[int]:
        cur = self.conn.cursor()
        res = cur.execute("SELECT key, value FROM local WHERE key = ?", (key,))
        row = res.fetchone()
        return row[0] if row else None

    INSERT_SQL = "INSERT INTO local (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value"

    def set(self, key: str, value: int) -> None:
        cur = self.conn.cursor()
        cur.execute(self.INSERT_SQL, (key, value))
        self.conn.commit()

    def batch_set(self, d: dict[str, int]) -> None:
        cur = self.conn.cursor()
        kvs = [(k, v) for k, v in d.items()]
        cur.executemany(self.INSERT_SQL, kvs)
        self.conn.commit()

    def remove(self, key: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM local WHERE key = ?", (key,))
        self.conn.commit()

    def nuke(self, commit: bool = True) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM local")
        if commit:
            self.conn.commit()

    def keys(self) -> list[str]:
        cur = self.conn.cursor()
        res = cur.execute("SELECT key FROM local")
        rows = res.fetchall()
        return [row[0] for row in rows]

    def dump(self) -> dict[str, int]:
        cur = self.conn.cursor()
        res = cur.execute("SELECT key, value FROM local")
        rows = res.fetchall()
        return {row[0]: row[1] for row in rows}

    def dump_json(self) -> None:
        res = self.dump()
        with overwrite(self.jsonpath) as f:
            json.dump(res, f)


@contextmanager
def overwrite(file_path: Path, mode: str = "w", tmp_suffix: str = ".tmp"):
    tmp_path = file_path.parent / (file_path.name + tmp_suffix)
    try:
        with open(tmp_path, mode) as tmp_file:
            yield tmp_file
        tmp_path.rename(file_path)
    except Exception:
        # well, just keep the tmp_path in error case.
        raise


def normalize(name: str) -> str:
    """
    See https://peps.python.org/pep-0503/#normalized-names
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def remove_dir_with_files(directory: Path) -> None:
    """
    Remove dir in a safer (non-recursive) way, which means that the directory should have no child directories.
    """
    assert directory.is_dir()
    for item in directory.iterdir():
        item.unlink()
    directory.rmdir()
    logger.info("Removed dir %s", directory)


def get_packages_from_index_html(contents: str) -> list[str]:
    """
    Get all <a> href (fragments removed) from given simple/<package>/index.html contents
    """

    class ATagHTMLParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.hrefs: list[Optional[str]] = []

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            if tag == "a":
                for attr in attrs:
                    if attr[0] == "href":
                        self.hrefs.append(attr[1])

    p = ATagHTMLParser()
    p.feed(contents)

    ret = []
    for href in p.hrefs:
        if href:
            parsed_url = urlparse(href)
            clean_url = urlunparse(parsed_url._replace(fragment=""))
            ret.append(clean_url)
    return ret


def get_existing_hrefs(package_simple_path: Path) -> list[str]:
    existing_hrefs = []
    try:
        with open(package_simple_path / "index.html") as f:
            contents = f.read()
        existing_hrefs = get_packages_from_index_html(contents)
    except FileNotFoundError:
        pass
    return existing_hrefs


class CustomXMLRPCTransport(xmlrpc.client.Transport):
    """
    Set user-agent for xmlrpc.client
    """

    user_agent = USER_AGENT


def create_requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


class PyPI:
    """
    Upstream which implements full PyPI APIs
    """

    host = "https://pypi.org"
    # Let's assume that only sha256 exists...
    digest_name = "sha256"

    def __init__(self) -> None:
        self.xmlrpc_client = xmlrpc.client.ServerProxy(
            urljoin(self.host, "pypi"), transport=CustomXMLRPCTransport()
        )
        self.session = create_requests_session()

    def list_packages_with_serial(self) -> dict[str, int]:
        return self.xmlrpc_client.list_packages_with_serial()  # type: ignore

    def get_package_metadata(self, package_name: str) -> dict:
        req = self.session.get(urljoin(self.host, f"pypi/{package_name}/json"))
        if req.status_code == 404:
            raise PackageNotFoundError
        return req.json()

    def get_release_files_from_meta(self, package_meta: dict) -> list[dict]:
        release_files = []
        for release in package_meta["releases"].values():
            release_files.extend(release)
        release_files.sort(key=lambda x: x["filename"])
        return release_files

    def file_url_to_local_url(self, url: str) -> str:
        parsed = urlparse(url)
        assert parsed.path.startswith("/packages")
        prefix = "../.."
        return prefix + parsed.path

    # Func modified from bandersnatch
    def generate_html_simple_page(self, package_meta: dict) -> str:
        package_rawname = package_meta["info"]["name"]
        simple_page_content = (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "  <head>\n"
            '    <meta name="pypi:repository-version" content="{0}">\n'
            "    <title>Links for {1}</title>\n"
            "  </head>\n"
            "  <body>\n"
            "    <h1>Links for {1}</h1>\n"
        ).format("1.0", package_rawname)

        release_files = self.get_release_files_from_meta(package_meta)

        def gen_html_file_tags(release: dict) -> str:
            file_tags = ""

            # data-requires-python: requires_python
            if "requires_python" in release and release["requires_python"] is not None:
                file_tags += (
                    f' data-requires-python="{html.escape(release["requires_python"])}"'
                )

            # data-yanked: yanked_reason
            if "yanked" in release and release["yanked"]:
                if "yanked_reason" in release and release["yanked_reason"]:
                    file_tags += (
                        f' data-yanked="{html.escape(release["yanked_reason"])}"'
                    )
                else:
                    file_tags += ' data-yanked=""'

            return file_tags

        simple_page_content += "\n".join(
            [
                '    <a href="{}#{}={}"{}>{}</a><br/>'.format(
                    self.file_url_to_local_url(r["url"]),
                    self.digest_name,
                    r["digests"][self.digest_name],
                    gen_html_file_tags(r),
                    r["filename"],
                )
                for r in release_files
            ]
        )

        simple_page_content += (
            f"\n  </body>\n</html>\n<!--SERIAL {package_meta['last_serial']}-->"
        )

        return simple_page_content

    # Func modified from bandersnatch
    def generate_json_simple_page(self, package_meta: dict) -> str:
        package_json: dict[str, Any] = {
            "files": [],
            "meta": {
                "api-version": "1.1",
                "_last-serial": str(package_meta["last_serial"]),
            },
            "name": package_meta["info"]["name"],
            # TODO: Just sorting by default sort - Maybe specify order in future PEP
            "versions": sorted(package_meta["releases"].keys()),
        }

        release_files = self.get_release_files_from_meta(package_meta)

        # Add release files into the JSON dict
        for r in release_files:
            package_json["files"].append(
                {
                    "filename": r["filename"],
                    "hashes": {
                        self.digest_name: r["digests"][self.digest_name],
                    },
                    "requires-python": r.get("requires_python", ""),
                    "size": r["size"],
                    "upload-time": r.get("upload_time_iso_8601", ""),
                    "url": self.file_url_to_local_url(r["url"]),
                    "yanked": r.get("yanked", False),
                }
            )

        return json.dumps(package_json)


# (normalized_name as key, value)
ShadowmirePackageItem = tuple[str, int]


@dataclass
class Plan:
    remove: list[str]
    update: list[str]


class SyncBase:
    def __init__(
        self, basedir: Path, local_db: LocalVersionKV, sync_packages: bool = False
    ) -> None:
        self.basedir = basedir
        self.local_db = local_db
        self.simple_dir = basedir / "simple"
        self.packages_dir = basedir / "packages"
        # create the dirs, if not exist
        self.simple_dir.mkdir(parents=True, exist_ok=True)
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        self.sync_packages = sync_packages
    
    def filter_remote_with_excludes(self, remote: dict[str, int], excludes: list[re.Pattern]) -> dict[str, int]:
        if not excludes:
            return remote
        res = {}
        for k, v in remote.items():
            matched = False
            for exclude in excludes:
                if exclude.match(k):
                    matched = True
                    break
            if not matched:
                res[k] = v
        return res

    def determine_sync_plan(self, local: dict[str, int], excludes: list[re.Pattern]) -> Plan:
        remote = self.fetch_remote_versions()
        remote = self.filter_remote_with_excludes(remote, excludes)
        # store remote to remote.json
        with overwrite(self.basedir / "remote.json") as f:
            json.dump(remote, f)
        to_remove = []
        to_update = []
        local_keys = set(local.keys())
        remote_keys = set(remote.keys())
        for i in local_keys - remote_keys:
            to_remove.append(i)
            local_keys.remove(i)
        for i in remote_keys - local_keys:
            to_update.append(i)
        for i in local_keys:
            local_serial = local[i]
            remote_serial = remote[i]
            if local_serial != remote_serial:
                to_update.append(i)
        output = Plan(remove=to_remove, update=to_update)
        return output

    def fetch_remote_versions(self) -> dict[str, int]:
        raise NotImplementedError

    def parallel_update(self, package_names: list) -> None:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(self.do_update, package_name, False): (
                    idx,
                    package_name,
                )
                for idx, package_name in enumerate(package_names)
            }
            try:
                for future in tqdm(as_completed(futures), total=len(package_names)):
                    idx, package_name = futures[future]
                    try:
                        serial = future.result()
                        if serial:
                            self.local_db.set(package_name, serial)
                    except Exception as e:
                        if e is ExitProgramException or e is KeyboardInterrupt:
                            raise
                        logger.warning(
                            "%s generated an exception", package_name, exc_info=True
                        )
                    if idx % 1000 == 0:
                        self.local_db.dump_json()
            except (ExitProgramException, KeyboardInterrupt):
                logger.info("Get ExitProgramException or KeyboardInterrupt, exiting...")
                for future in futures:
                    future.cancel()
                sys.exit(1)

    def do_sync_plan(self, plan: Plan) -> None:
        to_remove = plan.remove
        to_update = plan.update

        for package_name in to_remove:
            self.do_remove(package_name)

        self.parallel_update(to_update)

    def do_remove(self, package_name: str) -> None:
        logger.info("removing %s", package_name)
        meta_dir = self.simple_dir / package_name
        index_html = meta_dir / "index.html"
        try:
            with open(index_html) as f:
                packages_to_remove = get_packages_from_index_html(f.read())
                for p in packages_to_remove:
                    p_path = meta_dir / p
                    try:
                        p_path.unlink()
                        logger.info("Removed file %s", p_path)
                    except FileNotFoundError:
                        pass
        except FileNotFoundError:
            pass
        # remove all files inside meta_dir
        self.local_db.remove(package_name)
        remove_dir_with_files(meta_dir)

    def do_update(self, package_name: str, write_db: bool = True) -> Optional[int]:
        raise NotImplementedError

    def finalize(self) -> None:
        local_names = self.local_db.keys()
        # generate index.html at basedir
        index_path = self.basedir / "simple" / "index.html"
        # modified from bandersnatch
        with overwrite(index_path) as f:
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
        self.local_db.dump_json()


def download(session: requests.Session, url: str, dest: Path) -> bool:
    resp = session.get(url, allow_redirects=True)
    if resp.status_code >= 400:
        logger.warning("download %s failed, skipping this package", url)
        return False
    with overwrite(dest, "wb") as f:
        f.write(resp.content)
    return True


class SyncPyPI(SyncBase):
    def __init__(
        self, basedir: Path, local_db: LocalVersionKV, sync_packages: bool = False
    ) -> None:
        self.pypi = PyPI()
        self.session = create_requests_session()
        super().__init__(basedir, local_db, sync_packages)

    def fetch_remote_versions(self) -> dict[str, int]:
        remote_serials = self.pypi.list_packages_with_serial()
        ret = {}
        for key in remote_serials:
            ret[normalize(key)] = remote_serials[key]
        return ret

    def do_update(self, package_name: str, write_db: bool = True) -> Optional[int]:
        logger.info("updating %s", package_name)
        package_simple_path = self.simple_dir / package_name
        package_simple_path.mkdir(exist_ok=True)
        try:
            meta = self.pypi.get_package_metadata(package_name)
            logger.debug("%s meta: %s", package_name, meta)
        except PackageNotFoundError:
            logger.warning("%s missing from upstream, skip.", package_name)
            return None

        if self.sync_packages:
            # sync packages first, then sync index
            existing_hrefs = get_existing_hrefs(package_simple_path)
            release_files = self.pypi.get_release_files_from_meta(meta)
            # remove packages that no longer exist remotely
            remote_hrefs = [
                self.pypi.file_url_to_local_url(i["url"]) for i in release_files
            ]
            should_remove = list(set(existing_hrefs) - set(remote_hrefs))
            for p in should_remove:
                logger.info("removing file %s (if exists)", p)
                package_path = (package_simple_path / p).resolve()
                package_path.unlink(missing_ok=True)
            for i in release_files:
                url = i["url"]
                dest = (
                    package_simple_path / self.pypi.file_url_to_local_url(i["url"])
                ).resolve()
                logger.info("downloading file %s -> %s", url, dest)
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                download(self.session, url, dest)

        last_serial: int = meta["last_serial"]
        simple_html_contents = self.pypi.generate_html_simple_page(meta)
        simple_json_contents = self.pypi.generate_json_simple_page(meta)

        for html_filename in ("index.html", "index.v1_html"):
            html_path = package_simple_path / html_filename
            with overwrite(html_path) as f:
                f.write(simple_html_contents)
        for json_filename in ("index.v1_json",):
            json_path = package_simple_path / json_filename
            with overwrite(json_path) as f:
                f.write(simple_json_contents)

        if write_db:
            self.local_db.set(package_name, last_serial)

        return last_serial


class SyncPlainHTTP(SyncBase):
    def __init__(
        self,
        upstream: str,
        basedir: Path,
        local_db: LocalVersionKV,
        sync_packages: bool = False,
    ) -> None:
        self.upstream = upstream
        self.session = create_requests_session()
        super().__init__(basedir, local_db, sync_packages)

    def fetch_remote_versions(self) -> dict[str, int]:
        remote_url = urljoin(self.upstream, "local.json")
        resp = self.session.get(remote_url)
        resp.raise_for_status()
        remote: dict[str, int] = resp.json()
        return remote

    def do_update(self, package_name: str, write_db: bool = True) -> Optional[int]:
        logger.info("updating %s", package_name)
        package_simple_path = self.simple_dir / package_name
        package_simple_path.mkdir(exist_ok=True)
        if self.sync_packages:
            existing_hrefs = get_existing_hrefs(package_simple_path)
        # directly fetch remote files
        for filename in ("index.html", "index.v1_html", "index.v1_json"):
            file_url = urljoin(self.upstream, f"/simple/{package_name}/{filename}")
            success = download(self.session, file_url, package_simple_path / filename)
            if not success:
                if filename != "index.html":
                    logger.warning("%s does not exist", file_url)
                    continue
                else:
                    logger.error("%s does not exist. Stop with this.", file_url)
                    return None

        if self.sync_packages:
            current_hrefs = get_existing_hrefs(package_simple_path)
            should_remove = list(set(existing_hrefs) - set(current_hrefs))
            for p in should_remove:
                logger.info("removing file %s (if exists)", p)
                package_path = (package_simple_path / p).resolve()
                package_path.unlink(missing_ok=True)
            package_simple_url = urljoin(self.upstream, f"/simple/{package_name}/")
            for href in current_hrefs:
                url = urljoin(package_simple_url, href)
                dest = (package_simple_path / href).resolve()
                logger.info("downloading file %s -> %s", url, dest)
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                download(self.session, url, dest)

        last_serial = get_local_serial(package_simple_path)
        if not last_serial:
            logger.warning("cannot get valid package serial from %s", package_name)
        else:
            if write_db:
                self.local_db.set(package_name, last_serial)

        return last_serial


def get_local_serial(package_simple_path: Path) -> Optional[int]:
    package_name = package_simple_path.name
    package_index_path = package_simple_path / "index.html"
    try:
        with open(package_index_path) as f:
            contents = f.read()
    except FileNotFoundError:
        logger.warning("%s does not have index.html, skipping", package_name)
        return None
    try:
        serial_comment = contents.splitlines()[-1].strip()
        serial = int(serial_comment.removeprefix("<!--SERIAL ").removesuffix("-->"))
        return serial
    except Exception:
        logger.warning("cannot parse %s index.html", package_name, exc_info=True)
        return None


def main(args: argparse.Namespace) -> None:
    log_level = logging.DEBUG if os.environ.get("DEBUG") else logging.INFO
    logging.basicConfig(level=log_level)
    logger.debug(args)

    basedir = Path(".")
    local_db = LocalVersionKV(basedir / "local.db", basedir / "local.json")

    if args.command == "sync":
        sync = SyncPyPI(
            basedir=basedir, local_db=local_db, sync_packages=args.sync_packages
        )
        local = local_db.dump()
        plan = sync.determine_sync_plan(local, args.excludes)
        # save plan for debugging
        with overwrite(basedir / "plan.json") as f:
            json.dump(plan, f, default=vars)
        sync.do_sync_plan(plan)
        sync.finalize()
    elif args.command == "genlocal":
        local = {}
        for package_path in (basedir / "simple").iterdir():
            package_name = package_path.name
            serial = get_local_serial(package_path)
            if serial:
                local[package_name] = serial
        local_db.nuke(commit=False)
        local_db.batch_set(local)
        local_db.dump_json()
    elif args.command == "verify":
        sync = SyncPyPI(
            basedir=basedir, local_db=local_db, sync_packages=args.sync_packages
        )
        local_names = set(local_db.keys())
        simple_dirs = set([i.name for i in (basedir / "simple").iterdir() if i.is_dir()])
        for package_name in simple_dirs - local_names:
            sync.do_remove(package_name)
        sync.parallel_update(list(local_names))
        sync.finalize()
        # clean up unreferenced package files
        ref_set = set()
        for sname in simple_dirs:
            sd = basedir / "simple" / sname
            hrefs = get_existing_hrefs(sd)
            for i in hrefs:
                ref_set.add(str((sd / i).resolve()))
        for file in (basedir / "packages").glob("*/*/*/*"):
            file = file.resolve()
            if str(file) not in ref_set:
                logger.info("removing unreferenced %s", file)
                file.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("shadowmire: lightweight PyPI syncing tool")
    subparsers = parser.add_subparsers(dest="command")

    parser_sync = subparsers.add_parser("sync", help="Sync from upstream")
    parser_sync.add_argument(
        "--sync-packages",
        help="Sync packages instead of just indexes",
        action="store_true",
    )
    parser_sync.add_argument(
        "--exclude", help="Remote package names to exclude. Regex.", nargs="*"
    )
    parser_genlocal = subparsers.add_parser(
        "genlocal", help="(Re)generate local db and json from simple/"
    )
    parser_verify = subparsers.add_parser(
        "verify",
        help="Verify existing sync from local db, download missing things, remove unreferenced packages",
    )
    parser_verify.add_argument(
        "--sync-packages",
        help="Sync packages instead of just indexes",
        action="store_true",
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    if args.command == "sync" and args.exclude:
        args.excludes = [re.compile(i) for i in args.exclude]
    main(args)
