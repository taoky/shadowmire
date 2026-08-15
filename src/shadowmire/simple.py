import html
import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal, overload
from urllib.parse import unquote, urlparse, urlunparse

from .filesystem import fast_readall

DIGEST_NAME = "sha256"


@overload
def get_package_urls_from_index_html(
    html_path: Path, with_metadata: Literal[True]
) -> list[tuple[str, bool]]: ...


@overload
def get_package_urls_from_index_html(
    html_path: Path, with_metadata: Literal[False] = False
) -> list[str]: ...


def get_package_urls_from_index_html(html_path: Path, with_metadata: bool = False):
    """Return fragment-free links from a project HTML simple index."""

    class ATagHTMLParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.data: list[tuple[str, bool]] = []

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            if tag != "a":
                return
            href = None
            has_metadata = False
            for name, value in attrs:
                if name == "href":
                    href = value
                elif name in {"data-dist-info-metadata", "data-core-metadata"}:
                    has_metadata = True
            if href:
                self.data.append((href, has_metadata))

    parser = ATagHTMLParser()
    parser.feed(fast_readall(html_path).decode())

    result = []
    for href, has_metadata in parser.data:
        parsed_url = urlparse(href)
        clean_url = urlunparse(parsed_url._replace(fragment=""))
        result.append((clean_url, has_metadata) if with_metadata else clean_url)
    return result


def get_package_urls_from_index_json(json_path: Path) -> list[tuple[str, bool]]:
    """Return URLs and core-metadata flags from a PEP 691 project index."""
    contents_dict = json.loads(fast_readall(json_path))

    def metadata(file: dict) -> bool:
        return file.get(
            "core-metadata",
            # Fallback for the legacy PEP 714 attribute.
            file.get("data-dist-info-metadata", False),
        )

    return [(file["url"], metadata(file)) for file in contents_dict["files"]]


def get_package_urls_size_from_index_json(json_path: Path) -> list[tuple[str, int]]:
    """Return URLs and sizes from a PEP 691 project index; unknown size is -1."""
    contents_dict = json.loads(fast_readall(json_path))
    return [(file["url"], file.get("size", -1)) for file in contents_dict["files"]]


def get_existing_hrefs(package_simple_path: Path) -> list[tuple[str, bool]] | None:
    """Read an existing project index, preferring JSON over HTML."""
    json_file = package_simple_path / "index.v1_json"
    html_file = package_simple_path / "index.html"
    try:
        return get_package_urls_from_index_json(json_file)
    except FileNotFoundError:
        try:
            return get_package_urls_from_index_html(html_file, with_metadata=True)
        except FileNotFoundError:
            return None


def get_release_files(package_meta: dict) -> list[dict]:
    release_files = []
    for release in package_meta["releases"].values():
        release_files.extend(release)
    release_files.sort(key=lambda file: file["filename"])
    return release_files


def file_url_to_local_url(url: str) -> str:
    """Convert an upstream distribution URL to a URL relative to simple/<name>."""
    parsed = urlparse(url)
    assert parsed.path.startswith("/packages")
    return "../.." + parsed.path


def file_url_to_local_path(url: str) -> Path:
    """Convert an upstream distribution URL to a decoded local relative path."""
    path = unquote(urlparse(url).path)
    assert path.startswith("/packages")
    return Path("../..") / path[1:]


# Modified from bandersnatch.
def generate_html_simple_page(package_meta: dict, core_metadata_map: dict) -> str:
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

    def gen_html_file_tags(release: dict) -> str:
        file_tags = ""
        if release.get("requires_python") is not None:
            file_tags += (
                f' data-requires-python="{html.escape(release["requires_python"])}"'
            )
        if release.get("yanked"):
            if release.get("yanked_reason"):
                file_tags += f' data-yanked="{html.escape(release["yanked_reason"])}"'
            else:
                file_tags += ' data-yanked=""'
        if core_metadata_map.get(release["filename"], False):
            metadata = core_metadata_map[release["filename"]]
            if metadata.get(DIGEST_NAME):
                digest = html.escape(metadata[DIGEST_NAME])
                file_tags += (
                    f' data-dist-info-metadata="{DIGEST_NAME}={digest}"'
                    f' data-core-metadata="{DIGEST_NAME}={digest}"'
                )
            else:
                file_tags += ' data-dist-info-metadata="true" data-core-metadata="true"'
        return file_tags

    simple_page_content += "\n".join(
        '    <a href="{}#{}={}"{}>{}</a><br/>'.format(
            file_url_to_local_url(release["url"]),
            DIGEST_NAME,
            release["digests"][DIGEST_NAME],
            gen_html_file_tags(release),
            release["filename"],
        )
        for release in get_release_files(package_meta)
    )
    return (
        simple_page_content
        + f"\n  </body>\n</html>\n<!--SERIAL {package_meta['last_serial']}-->"
    )


# Modified from bandersnatch.
def generate_json_simple_page(package_meta: dict, core_metadata_map: dict) -> str:
    package_json: dict[str, Any] = {
        "files": [],
        "meta": {
            "api-version": "1.1",
            "_last-serial": str(package_meta["last_serial"]),
        },
        "name": package_meta["info"]["name"],
        "versions": sorted(package_meta["releases"].keys()),
    }
    for release in get_release_files(package_meta):
        package_json["files"].append(
            {
                "core-metadata": core_metadata_map.get(release["filename"], False),
                "data-dist-info-metadata": core_metadata_map.get(
                    release["filename"], False
                ),
                "filename": release["filename"],
                "hashes": {DIGEST_NAME: release["digests"][DIGEST_NAME]},
                "requires-python": release.get("requires_python", ""),
                "size": release["size"],
                "upload-time": release.get("upload_time_iso_8601", ""),
                "url": file_url_to_local_url(release["url"]),
                "yanked": release.get("yanked", False),
            }
        )
    return json.dumps(package_json)
