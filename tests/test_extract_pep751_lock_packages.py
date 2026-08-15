import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from utils.extract_pep751_lock_packages import extract_packages, main


def test_extracts_index_distributions_and_skips_direct_sources():
    result = extract_packages(
        tomllib.loads(
            """
            lock-version = "1.0"
            created-by = "tests"

            [[packages]]
            name = "Requests"
            version = "2.0"
            index = "https://pypi.org/simple/"
            [[packages.wheels]]
            name = "requests-2.0-py3-none-any.whl"
            url = "https://files.pythonhosted.org/requests.whl"
            [packages.wheels.hashes]
            sha256 = "abc"

            [[packages]]
            name = "requests"
            version = "1.0"
            [packages.sdist]
            name = "requests-1.0.tar.gz"
            url = "https://files.pythonhosted.org/requests.tar.gz"
            [packages.sdist.hashes]
            sha256 = "def"

            [[packages]]
            name = "Local_Project"
            [packages.directory]
            path = "."
            editable = true

            [[packages]]
            name = "git-project"
            [packages.vcs]
            type = "git"
            url = "https://example.invalid/repository.git"
            commit-id = "deadbeef"
            """
        )
    )

    assert result.packages == ["requests"]
    assert result.non_index == [
        ("git-project", "vcs"),
        ("local-project", "directory"),
    ]


def test_index_is_optional_for_distribution_entries():
    result = extract_packages(
        {
            "lock-version": "1.0",
            "packages": [
                {
                    "name": "example",
                    "version": "1.0",
                    "wheels": [{"name": "example.whl"}],
                }
            ],
        }
    )

    assert result.packages == ["example"]


def test_rejects_unsupported_lock_version():
    with pytest.raises(ValueError, match="unsupported PEP 751 lock version"):
        extract_packages({"lock-version": "2.0", "packages": []})


def test_rejects_conflicting_sources():
    with pytest.raises(ValueError, match="conflicting package sources"):
        extract_packages(
            {
                "lock-version": "1.0",
                "packages": [
                    {
                        "name": "example",
                        "archive": {"url": "https://example.invalid/a.whl"},
                        "wheels": [{"name": "a.whl"}],
                    }
                ],
            }
        )


def test_cli_writes_names_and_reports_direct_sources(tmp_path, capsys):
    lock_file = tmp_path / "pylock.toml"
    output = tmp_path / "packages.txt"
    lock_file.write_text(
        """
        lock-version = "1.0"
        [[packages]]
        name = "example"
        [[packages.wheels]]
        name = "example.whl"
        url = "https://example.invalid/example.whl"
        [packages.wheels.hashes]
        sha256 = "abc"
        [[packages]]
        name = "direct"
        [packages.archive]
        url = "https://example.invalid/direct.whl"
        [packages.archive.hashes]
        sha256 = "def"
        """
    )

    result = main([str(lock_file), "--output", str(output)])

    assert result == 0
    assert output.read_text() == "example\n"
    stderr = capsys.readouterr().err
    assert "Extracted 1 index projects" in stderr
    assert "direct (archive)" in stderr


def test_script_can_run_directly_outside_the_checkout(tmp_path):
    script = Path(__file__).parents[1] / "utils" / "extract_pep751_lock_packages.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "PEP 751" in result.stdout
