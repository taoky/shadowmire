#!/usr/bin/env python3

"""Select popular PyPI projects from the public BigQuery simple-request data."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

from google.api_core.exceptions import GoogleAPICallError
from google.auth.exceptions import GoogleAuthError
from google.cloud import bigquery
from packaging.utils import InvalidName, canonicalize_name

PYPI_SIMPLE_REQUESTS_TABLE = "bigquery-public-data.pypi.simple_requests"
BIGQUERY_LOCATION = "US"
DEFAULT_DAYS = 1  # ~130 GiB per execution for now
DEFAULT_COVERAGE = "0.99"
BYTES_PER_GIB = 1 << 30
BYTES_PER_TIB = 1 << 40
MONTHLY_FREE_QUERY_BYTES = BYTES_PER_TIB
DEFAULT_MAXIMUM_BYTES_BILLED = MONTHLY_FREE_QUERY_BYTES
ON_DEMAND_USD_PER_TIB = Decimal("6.25")
# Display-only estimate based on the US on-demand rate on 2026-08-16. BigQuery's
# maximum_bytes_billed setting, rather than this value, enforces the safety cap.

REQUESTS_QUERY = f"""
SELECT
  project,
  COUNT(*) AS requests
FROM `{PYPI_SIMPLE_REQUESTS_TABLE}`
WHERE timestamp >= @start_time
  AND timestamp < @end_time
GROUP BY project
""".strip()


@dataclass(frozen=True)
class PackageRequests:
    package: str
    requests: int


@dataclass(frozen=True)
class QueryWindow:
    start: datetime
    end: datetime


class RequestRow(Protocol):
    def __getitem__(self, key: str, /) -> str | int | None: ...


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer {value!r}") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}; expected YYYY-MM-DD"
        ) from error


def parse_coverage(value: str | Decimal) -> Decimal:
    try:
        coverage = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"invalid coverage {value!r}") from error
    if not Decimal(0) < coverage <= Decimal(1):
        raise ValueError("coverage must be greater than 0 and at most 1")
    return coverage


def build_query_window(days: int, end_date: date) -> QueryWindow:
    if days <= 0:
        raise ValueError("days must be greater than 0")
    start_date = end_date - timedelta(days=days)
    return QueryWindow(
        start=datetime.combine(start_date, time.min, tzinfo=UTC),
        end=datetime.combine(end_date, time.min, tzinfo=UTC),
    )


def aggregate_requests(
    rows: Iterable[RequestRow],
) -> tuple[list[PackageRequests], int]:
    """Normalize BigQuery rows and merge counts for equivalent project names."""
    requests_by_package: dict[str, int] = {}
    invalid = 0
    for row in rows:
        try:
            raw_project = row["project"]
            raw_requests = row["requests"]
            if not isinstance(raw_project, str) or not isinstance(
                raw_requests, (str, int)
            ):
                raise TypeError
            requests = int(raw_requests)
            package = str(canonicalize_name(raw_project, validate=True))
        except (InvalidName, KeyError, TypeError, ValueError):
            invalid += 1
            continue
        if requests <= 0:
            invalid += 1
            continue
        requests_by_package[package] = requests_by_package.get(package, 0) + requests
    return (
        [
            PackageRequests(package, requests)
            for package, requests in requests_by_package.items()
        ],
        invalid,
    )


def select_by_coverage(
    traffic: Iterable[PackageRequests],
    coverage: str | Decimal,
    *,
    include_ties: bool = False,
) -> tuple[list[PackageRequests], int, int]:
    """Select the heaviest projects until cumulative coverage is reached."""
    coverage = parse_coverage(coverage)
    eligible = [record for record in traffic if record.requests > 0]
    eligible.sort(key=lambda record: (-record.requests, record.package))
    total = sum(record.requests for record in eligible)
    if total == 0:
        return [], 0, 0

    target = int((Decimal(total) * coverage).to_integral_value(rounding=ROUND_CEILING))
    selected: list[PackageRequests] = []
    selected_weight = 0
    boundary_weight: int | None = None
    for record in eligible:
        if selected_weight >= target and (
            not include_ties or record.requests != boundary_weight
        ):
            break
        selected.append(record)
        selected_weight += record.requests
        if selected_weight >= target and boundary_weight is None:
            boundary_weight = record.requests
    return selected, selected_weight, total


def query_config(
    window: QueryWindow,
    *,
    dry_run: bool,
    maximum_bytes_billed: int | None = None,
) -> bigquery.QueryJobConfig:
    config = bigquery.QueryJobConfig(
        dry_run=dry_run,
        use_query_cache=not dry_run,
        query_parameters=[
            bigquery.ScalarQueryParameter("start_time", "TIMESTAMP", window.start),
            bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", window.end),
        ],
    )
    if maximum_bytes_billed is not None:
        config.maximum_bytes_billed = maximum_bytes_billed
    return config


def estimate_query_bytes(client: bigquery.Client, window: QueryWindow) -> int:
    job = client.query(
        REQUESTS_QUERY,
        job_config=query_config(window, dry_run=True),
        location=BIGQUERY_LOCATION,
    )
    return int(job.total_bytes_processed or 0)


def query_requests(
    client: bigquery.Client,
    window: QueryWindow,
    maximum_bytes_billed: int,
) -> tuple[Iterable[RequestRow], bigquery.QueryJob]:
    job = client.query(
        REQUESTS_QUERY,
        job_config=query_config(
            window,
            dry_run=False,
            maximum_bytes_billed=maximum_bytes_billed,
        ),
        location=BIGQUERY_LOCATION,
    )
    return job.result(), job


def estimated_cost_usd(processed_bytes: int) -> Decimal:
    return Decimal(processed_bytes) / Decimal(BYTES_PER_TIB) * ON_DEMAND_USD_PER_TIB


def format_query_estimate(processed_bytes: int) -> str:
    gib = Decimal(processed_bytes) / Decimal(BYTES_PER_GIB)
    free_percentage = (
        Decimal(processed_bytes) / Decimal(MONTHLY_FREE_QUERY_BYTES) * Decimal(100)
    )
    cost = estimated_cost_usd(processed_bytes)
    return (
        f"{gib:.2f} GiB, {free_percentage:.2f}% of the 1 TiB monthly free "
        f"query allowance, up to US${cost:.2f} if no free allowance remains"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select popular PyPI projects from the public BigQuery simple-request "
            "data. Outputs one normalized project name per line."
        )
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Google Cloud project used to run and bill the query",
    )
    parser.add_argument(
        "--days",
        type=parse_positive_int,
        default=DEFAULT_DAYS,
        help=f"number of complete UTC days to query (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        help="exclusive UTC end date in YYYY-MM-DD form (default: today)",
    )
    parser.add_argument(
        "--coverage",
        default=DEFAULT_COVERAGE,
        help="cumulative request ratio to retain, greater than 0 and at most 1",
    )
    parser.add_argument(
        "--include-ties",
        action="store_true",
        help="include all projects tied at the coverage boundary",
    )
    parser.add_argument(
        "--maximum-bytes-billed",
        type=parse_positive_int,
        default=DEFAULT_MAXIMUM_BYTES_BILLED,
        help=(
            "hard per-query billing limit in bytes "
            f"(default: {DEFAULT_MAXIMUM_BYTES_BILLED}, or 1 TiB)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only validate the query and report its estimated scan cost",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="allow an empty result instead of treating it as an error",
    )
    parser.add_argument("-o", "--output", type=Path, help="defaults to stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        coverage = parse_coverage(args.coverage)
        end_date = args.end_date or datetime.now(UTC).date()
        window = build_query_window(args.days, end_date)
        client = bigquery.Client(project=args.project, location=BIGQUERY_LOCATION)

        estimated_bytes = estimate_query_bytes(client, window)
        print(
            f"Estimated query scan for {window.start.date()} through "
            f"{window.end.date()} (exclusive): "
            f"{format_query_estimate(estimated_bytes)}.",
            file=sys.stderr,
        )
        if estimated_bytes > args.maximum_bytes_billed:
            raise ValueError(
                f"estimated query size {estimated_bytes} bytes exceeds "
                f"--maximum-bytes-billed={args.maximum_bytes_billed}"
            )
        if args.dry_run:
            return 0

        rows, job = query_requests(client, window, args.maximum_bytes_billed)
        traffic, invalid_rows = aggregate_requests(rows)
        selected, selected_requests, total_requests = select_by_coverage(
            traffic, coverage, include_ties=args.include_ties
        )
        if not selected and not args.allow_empty:
            raise ValueError(
                "query returned no valid requests; check the date range or use "
                "--allow-empty"
            )

        contents = "".join(
            f"{record.package}\n"
            for record in sorted(selected, key=lambda record: record.package)
        )
        if args.output:
            args.output.write_text(contents, encoding="utf-8")
        else:
            sys.stdout.write(contents)

        achieved = (
            Decimal(selected_requests) / Decimal(total_requests)
            if total_requests
            else Decimal(0)
        )
        billed_bytes = int(job.total_bytes_billed or 0)
        cache_note = "; result cache hit" if job.cache_hit else ""
        print(
            f"Selected {len(selected)}/{len(traffic)} projects; covered "
            f"{selected_requests}/{total_requests} requests ({achieved:.2%}). "
            f"BigQuery billed {billed_bytes} bytes{cache_note}; skipped "
            f"{invalid_rows} invalid aggregate rows.",
            file=sys.stderr,
        )
    except (GoogleAPICallError, GoogleAuthError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
