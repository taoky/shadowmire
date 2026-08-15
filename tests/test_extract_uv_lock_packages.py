import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from utils.extract_uv_lock_packages import extract_packages, main, read_uv_lock


def test_extracts_normalized_registry_packages_and_skips_other_sources():
    result = extract_packages(
        tomllib.loads(
            """
            version = 1

            [[package]]
            name = "Requests"
            source = { registry = "https://pypi.org/simple" }

            [[package]]
            name = "requests"
            source = { registry = "https://example.invalid/simple" }

            [[package]]
            name = "My_Project"
            source = { editable = "." }

            [[package]]
            name = "git-project"
            source = { git = "https://example.invalid/repository.git" }
            """
        )
    )

    assert result.packages == ["requests"]
    assert result.non_index == [
        ("git-project", "git"),
        ("my-project", "editable"),
    ]


def test_reads_the_repository_uv_lock_and_excludes_editable_project():
    lock_file = Path(__file__).parents[1] / "uv.lock"

    result = read_uv_lock(lock_file)

    assert "google-cloud-bigquery" in result.packages
    assert "shadowmire" not in result.packages
    assert ("shadowmire", "editable") in result.non_index


def test_rejects_unsupported_lock_version():
    with pytest.raises(ValueError, match="unsupported uv lock version"):
        extract_packages({"version": 2, "package": []})


def test_strict_non_index_does_not_replace_output(tmp_path):
    lock_file = tmp_path / "uv.lock"
    output = tmp_path / "packages.txt"
    lock_file.write_text(
        'version = 1\n[[package]]\nname = "local"\nsource = { editable = "." }\n'
    )
    output.write_text("keep-me\n")

    with pytest.raises(SystemExit) as error:
        main(
            [
                str(lock_file),
                "--strict-non-index",
                "--allow-empty",
                "--output",
                str(output),
            ]
        )

    assert error.value.code == 2
    assert output.read_text() == "keep-me\n"


def test_cli_writes_sorted_names_and_reports_skipped_sources(tmp_path, capsys):
    lock_file = tmp_path / "uv.lock"
    output = tmp_path / "packages.txt"
    lock_file.write_text(
        """
        version = 1
        [[package]]
        name = "Bravo"
        source = { registry = "https://pypi.org/simple" }
        [[package]]
        name = "alpha"
        source = { registry = "https://pypi.org/simple" }
        [[package]]
        name = "local"
        source = { virtual = "." }
        """
    )

    result = main([str(lock_file), "--output", str(output)])

    assert result == 0
    assert output.read_text() == "alpha\nbravo\n"
    stderr = capsys.readouterr().err
    assert "Extracted 2 registry projects" in stderr
    assert "local (virtual)" in stderr


def test_script_can_run_directly_outside_the_checkout(tmp_path):
    script = Path(__file__).parents[1] / "utils" / "extract_uv_lock_packages.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--strict-non-index" in result.stdout
