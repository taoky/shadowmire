"""Select popular PyPI projects from the mirrors.access_log ClickHouse table."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import requests

from .analyze_nginx_log import (
    AnalysisStats,
    LogRequest,
    PackageTraffic,
    VoteDeduplicator,
    client_network,
    extract_project,
    parse_coverage,
    select_by_coverage,
)

DEFAULT_CLICKHOUSE_URL = "http://localhost:8123"
DEFAULT_TABLE = "mirrors.access_log"
DEFAULT_DAYS = 7
DEFAULT_COVERAGE = "0.99"
DEFAULT_TIMEOUT = 300.0
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class QueryWindow:
    start: datetime
    end: datetime


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer {value!r}") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid number {value!r}") from error
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


def parse_table(value: str) -> str:
    if IDENTIFIER_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "table must be a database-qualified ClickHouse identifier"
        )
    return value


def build_query_window(days: int, end_date: date) -> QueryWindow:
    if days <= 0:
        raise ValueError("days must be greater than 0")
    start_date = end_date - timedelta(days=days)
    return QueryWindow(
        start=datetime.combine(start_date, time.min, tzinfo=UTC),
        end=datetime.combine(end_date, time.min, tzinfo=UTC),
    )


def build_query(table: str, *, source: str | None, repo: str | None) -> str:
    """Build a parameterized query for rows eligible for nginx-style counting."""
    table = parse_table(table)
    filters = [
        "event_time >= {start_time:DateTime64(3, 'UTC')}",
        "event_time < {end_time:DateTime64(3, 'UTC')}",
        "method = 'GET'",
        "status >= 200 AND status < 400",
        "match(url, '(^|/)simple/[^/?]+')",
    ]
    if source is not None:
        filters.append("source = {source:String}")
    if repo is not None:
        filters.append("repo = {repo:String}")
    where = "\n  AND ".join(filters)
    return f"""
SELECT
  timestamp,
  toString(clientip) AS clientip,
  url,
  size
FROM {table}
PREWHERE {where}
ORDER BY event_time, request_id
FORMAT JSONEachRow
""".strip()


def _format_query_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def query_parameters(
    window: QueryWindow, *, source: str | None, repo: str | None
) -> dict[str, str]:
    parameters = {
        "param_start_time": _format_query_time(window.start),
        "param_end_time": _format_query_time(window.end),
    }
    if source is not None:
        parameters["param_source"] = source
    if repo is not None:
        parameters["param_repo"] = repo
    return parameters


def query_rows(
    *,
    url: str,
    user: str,
    password: str,
    table: str,
    window: QueryWindow,
    source: str | None,
    repo: str | None,
    timeout: float,
    verify: bool | str,
) -> Iterator[Mapping[str, Any]]:
    """Stream JSONEachRow results from ClickHouse's HTTP interface."""
    query = build_query(table, source=source, repo=repo)
    headers = {"X-ClickHouse-User": user}
    if password:
        headers["X-ClickHouse-Key"] = password

    response = requests.post(
        url,
        params=query_parameters(window, source=source, repo=repo),
        headers=headers,
        data=query.encode(),
        stream=True,
        timeout=timeout,
        verify=verify,
    )
    try:
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            detail = response.text.strip()
            if len(detail) > 1000:
                detail = f"{detail[:1000]}..."
            suffix = f": {detail}" if detail else ""
            raise ValueError(f"ClickHouse query failed{suffix}") from error

        for line_number, line in enumerate(
            response.iter_lines(decode_unicode=True), start=1
        ):
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(
                    f"ClickHouse returned invalid JSON on result line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(
                    f"ClickHouse returned a non-object on result line {line_number}"
                )
            yield row
    finally:
        response.close()


def analyze_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    strict: bool = False,
) -> tuple[dict[str, PackageTraffic], AnalysisStats]:
    """Apply the nginx analyzer's project extraction and vote deduplication."""
    traffic: dict[str, PackageTraffic] = {}
    stats = AnalysisStats()
    deduplicator = VoteDeduplicator()

    for row_number, row in enumerate(rows, start=1):
        stats.lines += 1
        try:
            raw_timestamp = row["timestamp"]
            raw_clientip = row["clientip"]
            target = row["url"]
            raw_size = row["size"]
            if (
                isinstance(raw_timestamp, bool)
                or not isinstance(raw_timestamp, (int, float, str))
                or not isinstance(raw_clientip, str)
                or not isinstance(target, str)
                or isinstance(raw_size, bool)
                or not isinstance(raw_size, (int, str))
            ):
                raise TypeError
            timestamp = int(float(raw_timestamp))
            bytes_sent = int(raw_size)
            network = client_network(raw_clientip)
            if network is None or bytes_sent < 0:
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError):
            stats.malformed += 1
            if strict:
                raise ValueError(f"row {row_number}: invalid ClickHouse result row")
            continue

        project = extract_project(target)
        if project is None:
            stats.unattributed += 1
            continue
        request = LogRequest(
            client_network=network,
            timestamp=timestamp,
            method="GET",
            target=target,
            status=200,
            bytes_sent=bytes_sent,
        )
        if deduplicator.is_duplicate(request, project):
            stats.duplicate += 1
            continue
        record = traffic.setdefault(project, PackageTraffic(project))
        record.requests += 1
        record.bytes += bytes_sent
        stats.matched += 1

    return traffic, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select popular PyPI projects from a ClickHouse access-log table. "
            "Outputs one normalized project name per line."
        )
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("CLICKHOUSE_URL", DEFAULT_CLICKHOUSE_URL),
        help=(
            "ClickHouse HTTP endpoint (default: CLICKHOUSE_URL or "
            f"{DEFAULT_CLICKHOUSE_URL})"
        ),
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("CLICKHOUSE_USER", "default"),
        help="ClickHouse user (default: CLICKHOUSE_USER or default)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("CLICKHOUSE_PASSWORD", ""),
        help="ClickHouse password (default: CLICKHOUSE_PASSWORD)",
    )
    parser.add_argument(
        "--table",
        type=parse_table,
        default=DEFAULT_TABLE,
        help=f"database-qualified access-log table (default: {DEFAULT_TABLE})",
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
    parser.add_argument("--source", help="only include rows with this source value")
    parser.add_argument("--repo", help="only include rows with this repo value")
    parser.add_argument("--metric", choices=("requests", "bytes"), default="requests")
    parser.add_argument(
        "--coverage",
        default=DEFAULT_COVERAGE,
        help="cumulative traffic ratio to retain, greater than 0 and at most 1",
    )
    parser.add_argument(
        "--include-ties",
        action="store_true",
        help="include all projects tied at the coverage boundary",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail on the first malformed result row instead of skipping it",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="allow an empty result instead of treating it as an error",
    )
    parser.add_argument(
        "--timeout",
        type=parse_positive_float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP query timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    tls_group = parser.add_mutually_exclusive_group()
    tls_group.add_argument(
        "--ca-cert", type=Path, help="CA bundle used to verify HTTPS"
    )
    tls_group.add_argument(
        "--insecure",
        action="store_true",
        help="disable HTTPS certificate verification",
    )
    parser.add_argument("-o", "--output", type=Path, help="defaults to stdout")
    parser.add_argument(
        "--quiet", action="store_true", help="do not print an analysis summary"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        coverage = parse_coverage(args.coverage)
        end_date = args.end_date or datetime.now(UTC).date()
        window = build_query_window(args.days, end_date)
        verify: bool | str = str(args.ca_cert) if args.ca_cert else not args.insecure

        rows = query_rows(
            url=args.url,
            user=args.user,
            password=args.password,
            table=args.table,
            window=window,
            source=args.source,
            repo=args.repo,
            timeout=args.timeout,
            verify=verify,
        )
        traffic, stats = analyze_rows(rows, strict=args.strict)
        selected, selected_weight, total_weight = select_by_coverage(
            traffic.values(),
            coverage,
            args.metric,
            include_ties=args.include_ties,
        )
        if not selected and not args.allow_empty:
            raise ValueError(
                "query returned no attributable traffic with a positive selected "
                "metric; check the filters or use --allow-empty"
            )

        contents = "".join(
            f"{record.package}\n"
            for record in sorted(selected, key=lambda record: record.package)
        )
        if args.output:
            args.output.write_text(contents, encoding="utf-8")
        else:
            sys.stdout.write(contents)

        if not args.quiet:
            achieved = selected_weight / total_weight if total_weight else 0
            print(
                f"Selected {len(selected)}/{len(traffic)} projects for "
                f"{window.start.date()} through {window.end.date()} (exclusive); "
                f"covered {selected_weight}/{total_weight} {args.metric} "
                f"({achieved:.2%}). Skipped {stats.malformed} malformed and "
                f"{stats.unattributed} unattributed rows; deduplicated "
                f"{stats.duplicate} repeated requests.",
                file=sys.stderr,
            )
    except (OSError, requests.RequestException, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
