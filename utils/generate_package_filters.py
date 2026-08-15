#!/usr/bin/env python3

"""Generate TOML package_filters entries from a project-name list."""

import argparse
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path

from packaging.utils import InvalidName, canonicalize_name

ACTIONS = ("include", "metadata-only", "exclude")
DEFAULT_MAX_PATTERN_LENGTH = 32 * 1024


def read_package_names(lines: Iterable[str]) -> list[str]:
    packages: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        try:
            packages.add(str(canonicalize_name(name, validate=True)))
        except InvalidName as e:
            raise ValueError(
                f"line {line_number}: invalid project name {name!r}"
            ) from e
    return sorted(packages)


def build_patterns(package_names: Iterable[str], max_pattern_length: int) -> list[str]:
    prefix = "^(?:"
    suffix = ")$"
    if max_pattern_length <= len(prefix) + len(suffix):
        raise ValueError("max pattern length is too small")

    patterns = []
    chunk: list[str] = []
    chunk_length = len(prefix) + len(suffix)
    for package_name in package_names:
        escaped = re.escape(package_name)
        added_length = len(escaped) + (1 if chunk else 0)
        if chunk and chunk_length + added_length > max_pattern_length:
            patterns.append(prefix + "|".join(chunk) + suffix)
            chunk = []
            chunk_length = len(prefix) + len(suffix)
            added_length = len(escaped)
        if chunk_length + added_length > max_pattern_length:
            raise ValueError(
                f"project name {package_name!r} does not fit in max pattern length"
            )
        chunk.append(escaped)
        chunk_length += added_length
    if chunk:
        patterns.append(prefix + "|".join(chunk) + suffix)
    return patterns


def generate_toml_entries(action: str, patterns: Iterable[str]) -> str:
    if action not in ACTIONS:
        raise ValueError(f"invalid action {action!r}")
    return "".join(
        f"{{ action = \"{action}\", pattern = '{pattern}' }},\n" for pattern in patterns
    )


def atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w") as output:
            output.write(contents)
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate package_filters TOML entries from project names."
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--action", choices=ACTIONS, default="include")
    parser.add_argument(
        "--max-pattern-length", type=int, default=DEFAULT_MAX_PATTERN_LENGTH
    )
    args = parser.parse_args()

    with args.input_file.open() as input_file:
        package_names = read_package_names(input_file)
    patterns = build_patterns(package_names, args.max_pattern_length)
    atomic_write(args.output_file, generate_toml_entries(args.action, patterns))


if __name__ == "__main__":
    main()
