from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from google.api_core.exceptions import GoogleAPICallError
from google.auth.exceptions import GoogleAuthError

from utils import analyze_bigquery
from utils.analyze_bigquery import (
    BYTES_PER_GIB,
    BYTES_PER_TIB,
    REQUESTS_QUERY,
    PackageRequests,
    aggregate_requests,
    build_query_window,
    estimated_cost_usd,
    format_query_estimate,
    main,
    select_by_coverage,
)


class FakeQueryJob:
    def __init__(
        self,
        *,
        processed: int,
        billed: int = 0,
        rows=(),
        cache_hit: bool = False,
    ):
        self.total_bytes_processed = processed
        self.total_bytes_billed = billed
        self.cache_hit = cache_hit
        self._rows = rows

    def result(self):
        return self._rows


class FakeClient:
    def __init__(self, dry_run_job, query_job=None, query_error=None):
        self.dry_run_job = dry_run_job
        self.query_job = query_job
        self.query_error = query_error
        self.calls = []

    def query(self, query, *, job_config, location):
        self.calls.append((query, job_config, location))
        if job_config.dry_run:
            return self.dry_run_job
        if self.query_error is not None:
            raise self.query_error
        assert self.query_job is not None
        return self.query_job


def install_fake_client(monkeypatch, client):
    constructed = []

    def factory(*, project, location):
        constructed.append((project, location))
        return client

    monkeypatch.setattr(analyze_bigquery.bigquery, "Client", factory)
    return constructed


def test_query_only_reads_partition_and_project_columns():
    assert "pypi.simple_requests" in REQUESTS_QUERY
    assert "file.project" not in REQUESTS_QUERY
    assert "timestamp >= @start_time" in REQUESTS_QUERY
    assert "timestamp < @end_time" in REQUESTS_QUERY
    assert "details." not in REQUESTS_QUERY
    assert "SELECT *" not in REQUESTS_QUERY


def test_query_window_uses_complete_utc_days():
    window = build_query_window(7, date(2026, 8, 16))

    assert window.start == datetime(2026, 8, 9, tzinfo=UTC)
    assert window.end == datetime(2026, 8, 16, tzinfo=UTC)


def test_aggregate_requests_normalizes_merges_and_skips_invalid_rows():
    traffic, invalid = aggregate_requests(
        [
            {"project": "Foo_Bar", "requests": 10},
            {"project": "foo.bar", "requests": 5},
            {"project": "invalid name", "requests": 3},
            {"project": "empty", "requests": 0},
        ]
    )

    assert traffic == [PackageRequests("foo-bar", 15)]
    assert invalid == 2


def test_select_by_coverage_matches_request_boundary_and_ties():
    traffic = [
        PackageRequests("alpha", 50),
        PackageRequests("bravo", 25),
        PackageRequests("charlie", 25),
    ]

    selected, selected_weight, total = select_by_coverage(
        traffic, "0.7", include_ties=True
    )

    assert [record.package for record in selected] == ["alpha", "bravo", "charlie"]
    assert selected_weight == total == 100


def test_cost_estimate_uses_current_us_on_demand_rate():
    assert estimated_cost_usd(100 * BYTES_PER_GIB) == Decimal("0.6103515625")
    assert estimated_cost_usd(BYTES_PER_TIB) == Decimal("6.25")
    assert "9.77%" in format_query_estimate(100 * BYTES_PER_GIB)
    assert "US$0.61" in format_query_estimate(100 * BYTES_PER_GIB)


def test_cli_dry_runs_before_query_and_writes_sorted_output(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "packages.txt"
    dry_run_job = FakeQueryJob(processed=100 * BYTES_PER_GIB)
    query_job = FakeQueryJob(
        processed=100 * BYTES_PER_GIB,
        billed=100 * BYTES_PER_GIB,
        rows=[
            {"project": "Bravo", "requests": 40},
            {"project": "alpha", "requests": 60},
        ],
    )
    client = FakeClient(dry_run_job, query_job)
    constructed = install_fake_client(monkeypatch, client)

    result = main(
        [
            "--project",
            "billing-project",
            "--end-date",
            "2026-08-16",
            "--coverage",
            "1",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert constructed == [("billing-project", "US")]
    assert [call[1].dry_run for call in client.calls] == [True, False]
    assert client.calls[1][1].maximum_bytes_billed == BYTES_PER_TIB
    assert output.read_text() == "alpha\nbravo\n"
    stderr = capsys.readouterr().err
    assert "9.77% of the 1 TiB" in stderr
    assert "Selected 2/2 projects" in stderr


def test_cli_estimate_over_limit_fails_without_replacing_output(tmp_path, monkeypatch):
    output = tmp_path / "packages.txt"
    output.write_text("keep-me\n")
    client = FakeClient(FakeQueryJob(processed=101))
    install_fake_client(monkeypatch, client)

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--project",
                "billing-project",
                "--maximum-bytes-billed",
                "100",
                "--output",
                str(output),
            ]
        )

    assert error.value.code == 2
    assert len(client.calls) == 1
    assert output.read_text() == "keep-me\n"


def test_cli_explicit_dry_run_does_not_query_or_write(tmp_path, monkeypatch):
    output = tmp_path / "packages.txt"
    output.write_text("keep-me\n")
    client = FakeClient(FakeQueryJob(processed=BYTES_PER_GIB))
    install_fake_client(monkeypatch, client)

    result = main(
        [
            "--project",
            "billing-project",
            "--dry-run",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert len(client.calls) == 1
    assert output.read_text() == "keep-me\n"


def test_cli_query_error_does_not_replace_output(tmp_path, monkeypatch):
    output = tmp_path / "packages.txt"
    output.write_text("keep-me\n")
    client = FakeClient(
        FakeQueryJob(processed=BYTES_PER_GIB),
        query_error=GoogleAPICallError("query failed"),
    )
    install_fake_client(monkeypatch, client)

    with pytest.raises(SystemExit) as error:
        main(["--project", "billing-project", "--output", str(output)])

    assert error.value.code == 2
    assert len(client.calls) == 2
    assert output.read_text() == "keep-me\n"


def test_cli_empty_result_does_not_replace_output(tmp_path, monkeypatch):
    output = tmp_path / "packages.txt"
    output.write_text("keep-me\n")
    client = FakeClient(
        FakeQueryJob(processed=BYTES_PER_GIB),
        FakeQueryJob(processed=BYTES_PER_GIB, rows=[]),
    )
    install_fake_client(monkeypatch, client)

    with pytest.raises(SystemExit) as error:
        main(["--project", "billing-project", "--output", str(output)])

    assert error.value.code == 2
    assert output.read_text() == "keep-me\n"


def test_cli_reports_authentication_error(monkeypatch):
    def fail_client(*, project, location):
        raise GoogleAuthError(f"no credentials for {project} in {location}")

    monkeypatch.setattr(analyze_bigquery.bigquery, "Client", fail_client)

    with pytest.raises(SystemExit) as error:
        main(["--project", "billing-project", "--dry-run"])

    assert error.value.code == 2
