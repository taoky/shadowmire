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
import requests

logger = logging.getLogger(__name__)

USER_AGENT = "Shadowmire (https://github.com/taoky/shadowmire)"


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
        return self.session.get(urljoin(self.host, f"pypi/{package_name}/json")).json()

    def get_release_files_from_meta(self, package_meta: dict) -> list[dict]:
        release_files = []
        for release in package_meta["releases"].values():
            release_files.extend(release)
        release_files.sort(key=lambda x: x["filename"])
        return release_files

    def _file_url_to_local_url(self, url: str) -> str:
        parsed = urlparse(url)
        assert parsed.path.startswith("/packages")
        prefix = "../.."
        return prefix + parsed.path

    # Func modified from bandersnatch
    def generate_html_simple_page(
        self, package_meta: dict, package_rawname: str
    ) -> str:
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
                    self._file_url_to_local_url(r["url"]),
                    self.digest_name,
                    r["digests"][self.digest_name],
                    gen_html_file_tags(r),
                    r["filename"],
                )
                for r in release_files
            ]
        )

        simple_page_content += (
            f"\n  </body>\n</html>\n<!--SERIAL {package_meta["last_serial"]}-->"
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
                    "url": self._file_url_to_local_url(r["url"]),
                    "yanked": r.get("yanked", False),
                }
            )

        return json.dumps(package_json)


@dataclass
class ShadowmirePackageValue:
    serial: int
    raw_name: str


# (normalized_name as key, value)
ShadowmirePackageItem = tuple[str, ShadowmirePackageValue]


@dataclass
class Plan:
    remove: list[str]
    update: list[str]


class SyncBase:
    def __init__(self, basedir: Path, sync_packages: bool = False) -> None:
        self.basedir = basedir
        self.simple_dir = basedir / "simple"
        self.packages_dir = basedir / "packages"
        # create the dirs, if not exist
        self.simple_dir.mkdir(parents=True, exist_ok=True)
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        self.sync_packages = sync_packages
        self.remote: Optional[dict[str, ShadowmirePackageValue]] = None

    def determine_sync_plan(self, local: dict[str, ShadowmirePackageValue]) -> Plan:
        remote = self.fetch_remote_versions()
        self.remote = remote
        # store remote to remote.json
        with open(self.basedir / "remote.json", "w") as f:
            json.dump(remote, f)
        to_remove = []
        to_update = []
        local_keys = set(local.keys())
        remote_keys = set(remote.keys())
        for i in local_keys - remote_keys:
            to_remove.append(i)
        for i in remote_keys - local_keys:
            to_update.append(i)
        for i in local_keys:
            local_serial = local[i]
            remote_serial = remote[i]
            if local_serial != remote_serial:
                to_update.append(i)
        output = Plan(remove=to_remove, update=to_update)
        return output

    def fetch_remote_versions(self) -> dict[str, ShadowmirePackageValue]:
        raise NotImplementedError

    def do_sync_plan(self, plan: Plan) -> None:
        assert self.remote
        to_remove = plan.remove
        to_update = plan.update

        for package in to_remove:
            logger.info("Removing %s", package)
            meta_dir = self.simple_dir / package
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
            remove_dir_with_files(meta_dir)
        for package in to_update:
            self.do_update((package, self.remote[package]))

    def do_update(self, package: ShadowmirePackageItem) -> None:
        raise NotImplementedError

    def finalize(self) -> None:
        assert self.remote
        # generate index.html at basedir
        index_path = self.basedir / "simple" / "index.html"
        # modified from bandersnatch
        with open(index_path, "w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html>\n")
            f.write("  <head>\n")
            f.write('    <meta name="pypi:repository-version" content="1.0">\n')
            f.write("    <title>Simple Index</title>\n")
            f.write("  </head>\n")
            f.write("  <body>\n")
            # This will either be the simple dir, or if we are using index
            # directory hashing, a list of subdirs to process.
            for pkg in self.remote:
                # We're really trusty that this is all encoded in UTF-8. :/
                f.write(f'    <a href="{pkg}/">{pkg}</a><br/>\n')
            f.write("  </body>\n</html>")
        remote_json_path = self.basedir / "remote.json"
        local_json_path = self.basedir / "local.json"
        remote_json_path.rename(local_json_path)


class SyncPyPI(SyncBase):
    def __init__(self, basedir: Path, sync_packages: bool = False) -> None:
        self.pypi = PyPI()
        self.session = create_requests_session()
        super().__init__(basedir, sync_packages)

    def fetch_remote_versions(self) -> dict[str, ShadowmirePackageValue]:
        remote_serials = self.pypi.list_packages_with_serial()
        ret = {}
        for key in remote_serials:
            ret[normalize(key)] = ShadowmirePackageValue(
                serial=remote_serials[key], raw_name=key
            )
        return ret

    def do_update(self, package: ShadowmirePackageItem) -> None:
        package_name = package[0]
        # The serial get from metadata now might be newer than package_serial...
        # package_serial = package[1].serial
        package_rawname = package[1].raw_name

        package_simple_dir = self.simple_dir / package_name
        package_simple_dir.mkdir(exist_ok=True)
        meta = self.pypi.get_package_metadata(package_name)

        simple_html_contents = self.pypi.generate_html_simple_page(
            meta, package_rawname
        )
        simple_json_contents = self.pypi.generate_json_simple_page(meta)

        for html_filename in ("index.html", "index.v1_html"):
            html_path = package_simple_dir / html_filename
            with open(html_path, "w") as f:
                f.write(simple_html_contents)
        for json_filename in ("index.v1_json",):
            json_path = package_simple_dir / json_filename
            with open(json_path, "w") as f:
                f.write(simple_json_contents)

        if self.sync_packages:
            raise NotImplementedError


class SyncPlainHTTP(SyncBase):
    def __init__(
        self, upstream: str, basedir: Path, sync_packages: bool = False
    ) -> None:
        self.upstream = upstream
        self.session = create_requests_session()
        super().__init__(basedir, sync_packages)

    def fetch_remote_versions(self) -> dict[str, ShadowmirePackageValue]:
        remote_url = urljoin(self.upstream, "local.json")
        resp = self.session.get(remote_url)
        resp.raise_for_status()
        remote: dict[str, ShadowmirePackageValue] = resp.json()
        return remote

    def do_update(self, package: tuple[str, ShadowmirePackageValue]) -> None:
        package_name = package[0]
        package_simple_dir = self.simple_dir / package_name
        package_simple_dir.mkdir(exist_ok=True)
        # directly fetch remote files
        for filename in ("index.html", "index.v1_html", "index.v1_json"):
            file_url = urljoin(self.upstream, f"/simple/{package_name}/{filename}")
            resp = self.session.get(file_url)
            if resp.status_code == 404:
                if filename != "index.html":
                    logger.warning("%s does not exist", file_url)
                    continue
                else:
                    logger.error("%s does not exist. Stop with this.", file_url)
                    # TODO: error handling
                    break
            else:
                resp.raise_for_status()
            content = resp.content
            with open(package_simple_dir / filename, "wb") as f:
                f.write(content)

        if self.sync_packages:
            raise NotImplementedError


def main() -> None:
    logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    main()
