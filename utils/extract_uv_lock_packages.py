#!/usr/bin/env python3

"""Extract index package names from a uv lock file."""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from packaging.utils import InvalidName, canonicalize_name

SUPPORTED_LOCK_VERSION = 1


@dataclass(frozen=True)
class ExtractionResult:
    packages: list[str]
    non_index: list[tuple[str, str]]


def _canonicalize(name: object, entry_number: int) -> str:
    if not isinstance(name, str):
        raise TypeError(f"package entry {entry_number}: name must be a string")
    try:
        return str(canonicalize_name(name, validate=True))
    except InvalidName as error:
        raise ValueError(
            f"package entry {entry_number}: invalid project name {name!r}"
        ) from error


def _non_index_source(source: Mapping[str, object]) -> str:
    for key in ("editable", "virtual", "directory", "git", "url", "path"):
        if key in source:
            return key
    return "unknown"


def extract_packages(document: Mapping[str, object]) -> ExtractionResult:
    if document.get("version") != SUPPORTED_LOCK_VERSION:
        raise ValueError(
            f"unsupported uv lock version {document.get('version')!r}; "
            f"expected {SUPPORTED_LOCK_VERSION}"
        )
    entries = document.get("package")
    if not isinstance(entries, list):
        raise TypeError("uv lock file must contain a [[package]] array")

    packages: set[str] = set()
    non_index: set[tuple[str, str]] = set()
    for entry_number, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise TypeError(f"package entry {entry_number}: expected a table")
        package = _canonicalize(entry.get("name"), entry_number)
        source = entry.get("source")
        if not isinstance(source, dict):
            raise TypeError(f"package entry {entry_number}: source must be a table")
        if "registry" in source:
            if not isinstance(source["registry"], str):
                raise TypeError(
                    f"package entry {entry_number}: registry must be a string"
                )
            packages.add(package)
        else:
            non_index.add((package, _non_index_source(source)))
    return ExtractionResult(sorted(packages), sorted(non_index))


def read_uv_lock(path: Path) -> ExtractionResult:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    return extract_packages(document)


def _format_non_index(entries: Sequence[tuple[str, str]]) -> str:
    return ", ".join(f"{name} ({source})" for name, source in entries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract registry package names from uv.lock. Outputs one normalized "
            "project name per line."
        )
    )
    parser.add_argument("lock_file", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="defaults to stdout")
    parser.add_argument(
        "--strict-non-index",
        action="store_true",
        help="fail instead of skipping editable, path, Git, URL, and virtual sources",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="allow a lock file with no registry packages",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = read_uv_lock(args.lock_file)
        if result.non_index and args.strict_non_index:
            raise ValueError(
                "lock contains non-index packages: "
                f"{_format_non_index(result.non_index)}"
            )
        if not result.packages and not args.allow_empty:
            raise ValueError(
                "lock contains no registry packages; use --allow-empty if intentional"
            )

        contents = "".join(f"{package}\n" for package in result.packages)
        if args.output:
            args.output.write_text(contents, encoding="utf-8")
        else:
            sys.stdout.write(contents)

        print(
            f"Extracted {len(result.packages)} registry projects from "
            f"{args.lock_file}.",
            file=sys.stderr,
        )
        if result.non_index:
            print(
                "Skipped non-index packages; handle them separately: "
                f"{_format_non_index(result.non_index)}.",
                file=sys.stderr,
            )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
