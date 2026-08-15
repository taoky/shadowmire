import subprocess
import sys

from click.testing import CliRunner

from shadowmire.cli import cli


def test_cli_reports_package_version():
    result = CliRunner().invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert result.output == "cli, version 2.0.0.dev0\n"


def test_module_entrypoint_works_outside_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "shadowmire", "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Usage: python -m shadowmire" in result.stdout


def test_import_does_not_install_signal_handler(tmp_path):
    script = """
import signal

calls = []
signal.signal = lambda *args: calls.append(args)
import shadowmire.cli
assert calls == []
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
