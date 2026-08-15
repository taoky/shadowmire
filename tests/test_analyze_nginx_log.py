import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from utils.analyze_nginx_log import (
    PackageTraffic,
    analyze_lines,
    decompressor_command,
    extract_project,
    main,
    open_log,
    parse_combined_line,
    parse_coverage,
    select_by_coverage,
    select_recent_logs,
)


def log_line(
    target: str,
    *,
    remote_addr: str = "127.0.0.1",
    timestamp: str = "10/Oct/2000:13:55:36 +0000",
    method: str = "GET",
    status: int = 200,
    bytes_sent: int = 100,
) -> str:
    return (
        f"{remote_addr} - - [{timestamp}] "
        f'"{method} {target} HTTP/1.1" {status} {bytes_sent} '
        '"-" "pip/26.0"\n'
    )


def test_script_can_run_directly_outside_the_checkout(tmp_path):
    script = Path(__file__).parents[1] / "utils" / "analyze_nginx_log.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--glob GLOB" in result.stdout


def test_parse_combined_line_extracts_required_fields():
    request = parse_combined_line(log_line("/simple/requests/?x=1", bytes_sent=42))

    assert request is not None
    assert request.client_network == "127.0.0.0/24"
    assert request.method == "GET"
    assert request.target == "/simple/requests/?x=1"
    assert request.status == 200
    assert request.bytes_sent == 42
    assert parse_combined_line("not a combined log line") is None


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("/simple/Foo_Bar/", "foo-bar"),
        ("/prefix/simple/Foo%2EBar/index.html", "foo-bar"),
        ("/simple/", None),
        ("/packages/aa/bb/hash/Requests-2.32.4-py3-none-any.whl", None),
    ],
)
def test_extract_project(target, expected):
    assert extract_project(target) == expected


def test_analyze_lines_aggregates_and_classifies_skipped_requests():
    lines = [
        log_line("/simple/Requests/", bytes_sent=10),
        log_line("/simple/requests/", bytes_sent=20),
        log_line("/simple/missing/", status=404),
        log_line("/simple/head-only/", method="HEAD"),
        log_line("/robots.txt"),
        "broken\n",
    ]

    traffic, stats = analyze_lines(lines)

    assert traffic == {"requests": PackageTraffic("requests", 1, 10)}
    assert stats.lines == 6
    assert stats.matched == 1
    assert stats.duplicate == 1
    assert stats.unsuccessful == 2
    assert stats.unattributed == 1
    assert stats.malformed == 1


def test_ipv4_votes_are_deduplicated_by_24_for_five_minutes():
    lines = [
        log_line("/simple/requests/", remote_addr="192.0.2.1"),
        log_line(
            "/simple/requests/",
            remote_addr="192.0.3.1",
            timestamp="10/Oct/2000:13:56:00 +0000",
        ),
        log_line(
            "/simple/requests/",
            remote_addr="192.0.2.254",
            timestamp="10/Oct/2000:13:59:35 +0000",
        ),
        log_line(
            "/simple/requests/",
            remote_addr="192.0.2.2",
            timestamp="10/Oct/2000:14:00:36 +0000",
        ),
    ]

    traffic, stats = analyze_lines(lines)

    assert traffic["requests"].requests == 3
    assert stats.duplicate == 1


def test_ipv6_votes_are_deduplicated_by_48():
    lines = [
        log_line("/simple/requests/", remote_addr="2001:db8:abcd:1::1"),
        log_line("/simple/requests/", remote_addr="2001:db8:abcd:ffff::1"),
        log_line("/simple/requests/", remote_addr="2001:db8:abce::1"),
    ]

    traffic, stats = analyze_lines(lines)

    assert traffic["requests"].requests == 2
    assert stats.duplicate == 1


def test_ipv4_mapped_ipv6_uses_ipv4_network_prefix():
    traffic, stats = analyze_lines(
        [
            log_line("/simple/requests/", remote_addr="::ffff:192.0.2.1"),
            log_line("/simple/requests/", remote_addr="192.0.2.254"),
        ]
    )

    assert traffic["requests"].requests == 1
    assert stats.duplicate == 1


def test_different_projects_from_same_network_are_independent():
    traffic, stats = analyze_lines(
        [
            log_line("/simple/alpha/", remote_addr="192.0.2.1"),
            log_line("/simple/bravo/", remote_addr="192.0.2.2"),
        ]
    )

    assert set(traffic) == {"alpha", "bravo"}
    assert stats.duplicate == 0


def test_strict_mode_rejects_malformed_input():
    with pytest.raises(ValueError, match="line 2"):
        analyze_lines([log_line("/simple/requests/"), "broken\n"], strict=True)


def test_select_by_coverage_uses_deterministic_traffic_order():
    traffic = [
        PackageTraffic("charlie", requests=10, bytes=100),
        PackageTraffic("alpha", requests=60, bytes=10),
        PackageTraffic("bravo", requests=30, bytes=50),
    ]

    selected, selected_weight, total = select_by_coverage(traffic, "0.8", "requests")

    assert [record.package for record in selected] == ["alpha", "bravo"]
    assert selected_weight == 90
    assert total == 100


def test_select_by_coverage_can_include_boundary_ties():
    traffic = [
        PackageTraffic("alpha", requests=50),
        PackageTraffic("bravo", requests=25),
        PackageTraffic("charlie", requests=25),
    ]

    selected, selected_weight, total = select_by_coverage(
        traffic, "0.7", "requests", include_ties=True
    )

    assert [record.package for record in selected] == ["alpha", "bravo", "charlie"]
    assert selected_weight == total == 100


@pytest.mark.parametrize("coverage", ["0", "1.1", "invalid"])
def test_invalid_coverage_is_rejected(coverage):
    with pytest.raises(ValueError):
        parse_coverage(coverage)


def test_recent_logs_are_sorted_by_mtime_after_glob_filtering(tmp_path):
    old = tmp_path / "pypi.log.2.gz"
    middle = tmp_path / "pypi.log.1.xz"
    newest = tmp_path / "pypi.log"
    generated_output = tmp_path / "popular.log.txt"
    ignored = tmp_path / "README"
    for timestamp, path in enumerate(
        (old, middle, newest, ignored, generated_output), start=1
    ):
        path.touch()
        os.utime(path, ns=(timestamp, timestamp))

    selected = select_recent_logs(str(tmp_path / "*.log*"), 2, exclude=generated_output)

    assert selected == [newest, middle]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("access.log", None),
        ("access.log.gz", ["gzip", "-dc", "--"]),
        ("access.log.xz", ["xz", "-dc", "--"]),
        ("access.log.zst", ["zstd", "-dc", "--"]),
        ("access.log.zstd", ["zstd", "-dc", "--"]),
    ],
)
def test_decompressor_command_is_selected_from_suffix(filename, expected):
    command = decompressor_command(Path(filename))

    if expected is None:
        assert command is None
    else:
        assert command == [*expected, filename]


@pytest.mark.parametrize(
    ("suffix", "executable"),
    [(".gz", "gzip"), (".xz", "xz"), (".zst", "zstd")],
)
def test_open_log_streams_external_decompressors(tmp_path, suffix, executable):
    if shutil.which(executable) is None:
        pytest.skip(f"{executable} is not installed")
    contents = log_line("/simple/requests/")
    compressed = subprocess.run(
        [executable, "-c"],
        input=contents.encode(),
        check=True,
        capture_output=True,
    ).stdout
    log_path = tmp_path / f"access.log{suffix}"
    log_path.write_bytes(compressed)

    with open_log(log_path) as stream:
        assert stream.read() == contents


def test_missing_decompressor_is_reported(tmp_path, monkeypatch):
    from utils import analyze_nginx_log

    log_path = tmp_path / "access.log.gz"
    log_path.touch()
    monkeypatch.setitem(analyze_nginx_log.DECOMPRESSORS, ".gz", "not-a-command")

    with (
        pytest.raises(OSError, match="not-a-command.*required"),
        open_log(log_path) as stream,
    ):
        list(stream)


def test_cli_selects_recent_glob_matches_and_uses_gzip_binary(tmp_path, capsys):
    old_log = tmp_path / "access.log.2"
    compressed_log = tmp_path / "access.log.1.gz"
    newest_log = tmp_path / "access.log"
    output_path = tmp_path / "packages.txt"
    output_path.write_text("old\n")
    old_log.write_text(log_line("/simple/charlie/"))
    with gzip.open(compressed_log, "wt") as log:
        log.writelines(
            [
                log_line("/simple/alpha/"),
                log_line("/simple/alpha/"),
                log_line("/simple/alpha/"),
            ]
        )
    newest_log.write_text(log_line("/simple/bravo/") + log_line("/simple/alpha/"))
    os.utime(old_log, ns=(1, 1))
    os.utime(compressed_log, ns=(2, 2))
    os.utime(newest_log, ns=(3, 3))

    result = main(
        [
            "--glob",
            str(tmp_path / "access.log*"),
            "-k",
            "2",
            "--coverage",
            "1",
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    assert output_path.read_text() == "alpha\nbravo\n"
    stderr = capsys.readouterr().err
    assert "Selected 2/2 projects" in stderr
    assert "deduplicated 3 repeated requests" in stderr


def test_empty_analysis_does_not_replace_output(tmp_path):
    log_path = tmp_path / "access.log"
    output_path = tmp_path / "packages.txt"
    log_path.write_text(log_line("/robots.txt"))
    output_path.write_text("keep-me\n")

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--glob",
                str(tmp_path / "access.log*"),
                "--output",
                str(output_path),
            ]
        )

    assert error.value.code == 2
    assert output_path.read_text() == "keep-me\n"


def test_invalid_recent_count_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="greater than 0"):
        select_recent_logs(str(tmp_path / "*"), 0)
