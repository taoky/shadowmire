import re
import tomllib

import pytest

from utils.generate_package_filters import (
    build_patterns,
    generate_toml_entries,
    read_package_names,
)


def test_names_are_validated_normalized_sorted_and_deduplicated():
    names = read_package_names(["Requests\n", "requests\n", "foo_bar\n", "# x\n"])

    assert names == ["foo-bar", "requests"]


def test_invalid_name_reports_its_line():
    with pytest.raises(ValueError, match="line 2"):
        read_package_names(["requests\n", "not a requirement>=1\n"])


def test_patterns_are_exact_and_chunked():
    patterns = build_patterns(["alpha", "bravo", "long-name"], max_pattern_length=18)

    assert len(patterns) > 1
    combined = re.compile("|".join(patterns))
    assert combined.fullmatch("alpha")
    assert combined.fullmatch("long-name")
    assert not combined.fullmatch("long-name-extra")


def test_generated_entries_are_valid_toml():
    patterns = build_patterns(["foo-bar", "requests"], 1024)
    entries = generate_toml_entries("include", patterns)
    parsed = tomllib.loads(f"[options]\npackage_filters = [\n{entries}]\n")

    assert parsed["options"]["package_filters"][0]["action"] == "include"
