import json
import subprocess
import sys
from datetime import UTC, date, datetime

import pytest

from shadowmire_utils import analyze_clickhouse
from shadowmire_utils.analyze_clickhouse import (
    PackageTraffic,
    QueryWindow,
    analyze_rows,
    build_query,
    build_query_window,
    main,
    parse_table,
    query_rows,
)


class FakeResponse:
    def __init__(self, rows=(), *, status_error=None, text=""):
        self.rows = rows
        self.status_error = status_error
        self.text = text
        self.closed = False

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    def iter_lines(self, *, decode_unicode):
        assert decode_unicode is True
        for row in self.rows:
            if isinstance(row, str):
                yield row
            else:
                yield json.dumps(row)

    def close(self):
        self.closed = True


def row(
    url: str,
    *,
    timestamp: float = 1_597_930_536.0,
    clientip: str = "127.0.0.1",
    size: int = 100,
):
    return {
        "timestamp": timestamp,
        "clientip": clientip,
        "url": url,
        "size": size,
    }


def test_installed_module_can_run_outside_the_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "shadowmire_utils.analyze_clickhouse", "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--table TABLE" in result.stdout


def test_query_window_uses_complete_utc_days():
    window = build_query_window(7, date(2026, 8, 16))

    assert window.start == datetime(2026, 8, 9, tzinfo=UTC)
    assert window.end == datetime(2026, 8, 16, tzinfo=UTC)


def test_query_uses_schema_fields_and_parameterized_optional_filters():
    query = build_query("mirrors.access_log", source="edge", repo="pypi")

    assert "FROM mirrors.access_log" in query
    assert "event_time >= {start_time:DateTime64(3, 'UTC')}" in query
    assert "method = 'GET'" in query
    assert "status >= 200 AND status < 400" in query
    assert "toString(clientip) AS clientip" in query
    assert "source = {source:String}" in query
    assert "repo = {repo:String}" in query
    assert "ORDER BY event_time, request_id" in query
    assert "SELECT *" not in query


@pytest.mark.parametrize(
    "value", ["access_log", "mirrors.access-log", "mirrors.access_log FINAL", "x;DROP"]
)
def test_table_must_be_a_qualified_identifier(value):
    with pytest.raises(Exception, match="database-qualified"):
        parse_table(value)


def test_analyze_rows_matches_nginx_deduplication_and_metrics():
    traffic, stats = analyze_rows(
        [
            row("/pypi/web/simple/Foo_Bar/", size=10),
            row("/pypi/web/simple/foo.bar/", timestamp=1_597_930_550, size=20),
            row(
                "/simple/foo-bar/",
                timestamp=1_597_930_550,
                clientip="192.0.2.1",
                size=30,
            ),
            row("/robots.txt"),
            {
                "timestamp": "bad",
                "clientip": "127.0.0.1",
                "url": "/simple/x/",
                "size": 1,
            },
        ]
    )

    assert traffic == {"foo-bar": PackageTraffic("foo-bar", 2, 40)}
    assert stats.lines == 5
    assert stats.matched == 2
    assert stats.duplicate == 1
    assert stats.unattributed == 1
    assert stats.malformed == 1


def test_query_rows_sends_credentials_and_parameters_without_putting_them_in_sql(
    monkeypatch,
):
    response = FakeResponse([row("/simple/requests/")])
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(analyze_clickhouse.requests, "post", fake_post)
    window = QueryWindow(
        datetime(2026, 8, 9, tzinfo=UTC), datetime(2026, 8, 16, tzinfo=UTC)
    )

    rows = list(
        query_rows(
            url="https://clickhouse.example/",
            user="reader",
            password="secret",
            table="mirrors.access_log",
            window=window,
            source="edge's source",
            repo="pypi",
            timeout=30,
            verify="ca.pem",
        )
    )

    assert rows == [row("/simple/requests/")]
    assert response.closed is True
    url, kwargs = calls[0]
    assert url == "https://clickhouse.example/"
    assert kwargs["headers"] == {
        "X-ClickHouse-User": "reader",
        "X-ClickHouse-Key": "secret",
    }
    assert kwargs["params"]["param_source"] == "edge's source"
    assert kwargs["verify"] == "ca.pem"
    assert b"secret" not in kwargs["data"]
    assert b"edge's source" not in kwargs["data"]


def test_cli_writes_sorted_selection_after_complete_query(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "packages.txt"
    observed = []

    def fake_query_rows(**kwargs):
        observed.append(kwargs)
        return iter(
            [
                row("/simple/bravo/", clientip="192.0.2.1", size=40),
                row("/simple/alpha/", clientip="198.51.100.1", size=60),
            ]
        )

    monkeypatch.setattr(analyze_clickhouse, "query_rows", fake_query_rows)

    result = main(
        [
            "--url",
            "http://clickhouse:8123",
            "--end-date",
            "2026-08-16",
            "--coverage",
            "1",
            "--source",
            "edge",
            "--repo",
            "pypi",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert output.read_text() == "alpha\nbravo\n"
    assert observed[0]["source"] == "edge"
    assert observed[0]["repo"] == "pypi"
    assert observed[0]["window"].start.date() == date(2026, 8, 9)
    assert "Selected 2/2 projects" in capsys.readouterr().err


def test_cli_empty_result_does_not_replace_output(tmp_path, monkeypatch):
    output = tmp_path / "packages.txt"
    output.write_text("keep-me\n")
    monkeypatch.setattr(analyze_clickhouse, "query_rows", lambda **kwargs: iter(()))

    with pytest.raises(SystemExit) as error:
        main(["--output", str(output)])

    assert error.value.code == 2
    assert output.read_text() == "keep-me\n"
