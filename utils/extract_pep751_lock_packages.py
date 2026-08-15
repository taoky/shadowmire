#!/usr/bin/env python3

"""Extract index package names from a PEP 751 pylock.toml file."""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from packaging.utils import InvalidName, canonicalize_name

SUPPORTED_LOCK_VERSION = "1.0"
DIRECT_SOURCE_KEYS = ("vcs", "directory", "archive")


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


def _has_distribution_files(entry: Mapping[str, object]) -> bool:
    has_distributions = False
    if "sdist" in entry:
        if not isinstance(entry["sdist"], dict):
            raise TypeError("sdist must be a table")
        has_distributions = True
    if "wheels" in entry:
        wheels = entry["wheels"]
        if not isinstance(wheels, list):
            raise TypeError("wheels must be an array of tables")
        if any(not isinstance(wheel, dict) for wheel in wheels):
            raise TypeError("wheels must be an array of tables")
        has_distributions = has_distributions or bool(wheels)
    return has_distributions


def extract_packages(document: Mapping[str, object]) -> ExtractionResult:
    if document.get("lock-version") != SUPPORTED_LOCK_VERSION:
        raise ValueError(
            f"unsupported PEP 751 lock version {document.get('lock-version')!r}; "
            f"expected {SUPPORTED_LOCK_VERSION!r}"
        )
    entries = document.get("packages")
    if not isinstance(entries, list):
        raise TypeError("PEP 751 lock file must contain a [[packages]] array")

    packages: set[str] = set()
    non_index: set[tuple[str, str]] = set()
    for entry_number, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise TypeError(f"package entry {entry_number}: expected a table")
        package = _canonicalize(entry.get("name"), entry_number)
        direct_sources = [key for key in DIRECT_SOURCE_KEYS if key in entry]
        try:
            has_distributions = _has_distribution_files(entry)
        except TypeError as error:
            raise TypeError(f"package entry {entry_number}: {error}") from error
        if len(direct_sources) > 1 or (direct_sources and has_distributions):
            raise ValueError(
                f"package entry {entry_number}: conflicting package sources"
            )
        if direct_sources:
            if not isinstance(entry[direct_sources[0]], dict):
                raise TypeError(
                    f"package entry {entry_number}: {direct_sources[0]} must be a table"
                )
            non_index.add((package, direct_sources[0]))
        elif has_distributions:
            index = entry.get("index")
            if index is not None and not isinstance(index, str):
                raise TypeError(f"package entry {entry_number}: index must be a string")
            packages.add(package)
        else:
            raise ValueError(
                f"package entry {entry_number}: no supported package source"
            )
    return ExtractionResult(sorted(packages), sorted(non_index))


def read_pep751_lock(path: Path) -> ExtractionResult:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    return extract_packages(document)


def _format_non_index(entries: Sequence[tuple[str, str]]) -> str:
    return ", ".join(f"{name} ({source})" for name, source in entries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract index package names from a PEP 751 lock file. Outputs one "
            "normalized project name per line."
        )
    )
    parser.add_argument("lock_file", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="defaults to stdout")
    parser.add_argument(
        "--strict-non-index",
        action="store_true",
        help="fail instead of skipping VCS, directory, and direct archive sources",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="allow a lock file with no index packages",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = read_pep751_lock(args.lock_file)
        if result.non_index and args.strict_non_index:
            raise ValueError(
                "lock contains non-index packages: "
                f"{_format_non_index(result.non_index)}"
            )
        if not result.packages and not args.allow_empty:
            raise ValueError(
                "lock contains no index packages; use --allow-empty if intentional"
            )

        contents = "".join(f"{package}\n" for package in result.packages)
        if args.output:
            args.output.write_text(contents, encoding="utf-8")
        else:
            sys.stdout.write(contents)

        print(
            f"Extracted {len(result.packages)} index projects from {args.lock_file}.",
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
