#!/usr/bin/env python3

"""
A reference implementation of
selecting popular PyPI projects from nginx combined access logs.
"""

from __future__ import annotations

import argparse
import glob
import heapq
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from ipaddress import IPv6Address, ip_address, ip_network
from pathlib import Path
from typing import IO, Literal
from urllib.parse import unquote, urlsplit

from packaging.utils import InvalidName, canonicalize_name

Metric = Literal["requests", "bytes"]

DEFAULT_RECENT_LOGS = 7
DEDUPLICATION_WINDOW_SECONDS = 5 * 60
DECOMPRESSORS = {
    ".gz": "gzip",
    ".xz": "xz",
    ".zst": "zstd",
    ".zstd": "zstd",
}

COMBINED_LOG_RE = re.compile(
    r"^(?P<remote_addr>\S+) \S+ \S+ \[(?P<time_local>[^]]+)\] "
    r'"(?P<request>(?:[^"\\]|\\.)*)" '
    r"(?P<status>\d{3}) (?P<bytes>\d+|-) "
    r'"(?:[^"\\]|\\.)*" "(?:[^"\\]|\\.)*"(?: .*)?$'
)

NGINX_TIME_RE = re.compile(
    r"^(?P<day>\d{2})/(?P<month>[A-Za-z]{3})/(?P<year>\d{4}):"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}) "
    r"(?P<offset>[+-]\d{4})$"
)
MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


@dataclass(frozen=True)
class LogRequest:
    client_network: str
    timestamp: int
    method: str
    target: str
    status: int
    bytes_sent: int


@dataclass
class PackageTraffic:
    package: str
    requests: int = 0
    bytes: int = 0

    def value(self, metric: Metric) -> int:
        return self.requests if metric == "requests" else self.bytes


@dataclass
class AnalysisStats:
    lines: int = 0
    malformed: int = 0
    unsuccessful: int = 0
    unattributed: int = 0
    duplicate: int = 0
    matched: int = 0


class VoteDeduplicator:
    """Deduplicate project votes by client network in a rolling time window."""

    def __init__(self, window_seconds: int = DEDUPLICATION_WINDOW_SECONDS) -> None:
        self.window_seconds = window_seconds
        self._watermark = 0
        self._last_counted: dict[tuple[str, str], int] = {}
        self._expirations: list[tuple[int, tuple[str, str], int]] = []

    def is_duplicate(self, request: LogRequest, project: str) -> bool:
        self._watermark = max(self._watermark, request.timestamp)
        while self._expirations and self._expirations[0][0] <= self._watermark:
            _, key, timestamp = heapq.heappop(self._expirations)
            if self._last_counted.get(key) == timestamp:
                del self._last_counted[key]

        key = (request.client_network, project)
        previous = self._last_counted.get(key)
        if previous is not None and request.timestamp - previous < self.window_seconds:
            return True

        self._last_counted[key] = request.timestamp
        heapq.heappush(
            self._expirations,
            (request.timestamp + self.window_seconds, key, request.timestamp),
        )
        return False


def parse_nginx_timestamp(value: str) -> int | None:
    match = NGINX_TIME_RE.fullmatch(value)
    if match is None:
        return None
    try:
        month = MONTHS[match.group("month").title()]
        offset = match.group("offset")
        offset_minutes = int(offset[1:3]) * 60 + int(offset[3:5])
        if offset[0] == "-":
            offset_minutes = -offset_minutes
        parsed = datetime(
            int(match.group("year")),
            month,
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
            tzinfo=timezone(timedelta(minutes=offset_minutes)),
        )
    except (KeyError, ValueError):
        return None
    return int(parsed.timestamp())


def client_network(remote_addr: str) -> str | None:
    try:
        address = ip_address(remote_addr)
        if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        prefix_length = 24 if address.version == 4 else 48
        return ip_network(f"{address}/{prefix_length}", strict=False).with_prefixlen
    except ValueError:
        return None


def parse_combined_line(line: str) -> LogRequest | None:
    """Parse the fields needed from one nginx combined-format log line."""
    match = COMBINED_LOG_RE.match(line.rstrip("\r\n"))
    if match is None:
        return None
    request_parts = match.group("request").split(" ", 2)
    if len(request_parts) < 2:
        return None
    network = client_network(match.group("remote_addr"))
    timestamp = parse_nginx_timestamp(match.group("time_local"))
    if network is None or timestamp is None:
        return None
    try:
        bytes_sent = 0 if match.group("bytes") == "-" else int(match.group("bytes"))
        return LogRequest(
            client_network=network,
            timestamp=timestamp,
            method=request_parts[0],
            target=request_parts[1],
            status=int(match.group("status")),
            bytes_sent=bytes_sent,
        )
    except ValueError:
        return None


def _canonicalize_project(name: str) -> str | None:
    try:
        return str(canonicalize_name(unquote(name), validate=True))
    except InvalidName:
        return None


def extract_project(target: str) -> str | None:
    """Extract a normalized project name from a /simple/<project>/ URL."""
    try:
        segments = [segment for segment in urlsplit(target).path.split("/") if segment]
    except ValueError:
        return None

    for index, segment in enumerate(segments[:-1]):
        if segment == "simple":
            return _canonicalize_project(segments[index + 1])

    return None


def analyze_lines(
    lines: Iterable[str],
    *,
    strict: bool = False,
    deduplicator: VoteDeduplicator | None = None,
) -> tuple[dict[str, PackageTraffic], AnalysisStats]:
    if deduplicator is None:
        deduplicator = VoteDeduplicator()
    traffic: dict[str, PackageTraffic] = {}
    stats = AnalysisStats()
    for line_number, line in enumerate(lines, start=1):
        stats.lines += 1
        request = parse_combined_line(line)
        if request is None:
            stats.malformed += 1
            if strict:
                raise ValueError(f"line {line_number}: invalid combined log entry")
            continue
        if request.method != "GET" or not 200 <= request.status < 400:
            stats.unsuccessful += 1
            continue
        project = extract_project(request.target)
        if project is None:
            stats.unattributed += 1
            continue
        if deduplicator.is_duplicate(request, project):
            stats.duplicate += 1
            continue
        record = traffic.setdefault(project, PackageTraffic(project))
        record.requests += 1
        record.bytes += request.bytes_sent
        stats.matched += 1
    return traffic, stats


def parse_coverage(value: str | Decimal) -> Decimal:
    try:
        coverage = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"invalid coverage {value!r}") from error
    if not Decimal(0) < coverage <= Decimal(1):
        raise ValueError("coverage must be greater than 0 and at most 1")
    return coverage


def select_by_coverage(
    traffic: Iterable[PackageTraffic],
    coverage: str | Decimal,
    metric: Metric,
    *,
    include_ties: bool = False,
) -> tuple[list[PackageTraffic], int, int]:
    """Select the heaviest projects until cumulative coverage is reached."""
    coverage = parse_coverage(coverage)
    eligible = [record for record in traffic if record.value(metric) > 0]
    eligible.sort(key=lambda record: (-record.value(metric), record.package))
    total = sum(record.value(metric) for record in eligible)
    if total == 0:
        return [], 0, 0

    target = int((Decimal(total) * coverage).to_integral_value(rounding=ROUND_CEILING))
    selected: list[PackageTraffic] = []
    selected_weight = 0
    boundary_weight: int | None = None
    for record in eligible:
        weight = record.value(metric)
        if selected_weight >= target and (
            not include_ties or weight != boundary_weight
        ):
            break
        selected.append(record)
        selected_weight += weight
        if selected_weight >= target and boundary_weight is None:
            boundary_weight = weight
    return selected, selected_weight, total


def select_recent_logs(
    pattern: str,
    recent: int,
    *,
    exclude: Path | None = None,
) -> list[Path]:
    """Return the most recently modified matching regular files."""
    if recent <= 0:
        raise ValueError("recent log count must be greater than 0")
    excluded_path = exclude.resolve() if exclude is not None else None
    files = [
        Path(match)
        for match in glob.iglob(pattern, recursive=True)
        if (path := Path(match)).is_file()
        and (excluded_path is None or path.resolve() != excluded_path)
    ]
    files.sort(key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    return files[:recent]


def decompressor_command(path: Path) -> list[str] | None:
    executable = DECOMPRESSORS.get(path.suffix.lower())
    return [executable, "-dc", "--", str(path)] if executable else None


@contextmanager
def open_log(path: Path) -> Iterator[IO[str]]:
    """Open a plain log or stream it through its external decompressor."""
    command = decompressor_command(path)
    if command is None:
        with path.open(encoding="utf-8", errors="replace") as stream:
            yield stream
        return

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as error:
        raise OSError(
            f"decompressor {command[0]!r} is required to read {path.name}"
        ) from error
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        yield process.stdout
    except BaseException:
        process.stdout.close()
        process.terminate()
        process.wait()
        raise
    else:
        process.stdout.close()
        stderr = process.stderr.read().strip()
        returncode = process.wait()
        if returncode != 0:
            detail = f": {stderr}" if stderr else ""
            raise OSError(
                f"{command[0]} failed to decompress {path.name} "
                f"with exit status {returncode}{detail}"
            )
    finally:
        process.stderr.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select popular PyPI projects from nginx combined access logs. "
            "Outputs one normalized project name per line."
        )
    )
    parser.add_argument(
        "--glob",
        required=True,
        help="full path glob for access logs, for example /var/log/nginx/access.log*",
    )
    parser.add_argument(
        "-k",
        "--recent",
        type=int,
        default=DEFAULT_RECENT_LOGS,
        help=f"analyze the most recently modified matching files (default: {DEFAULT_RECENT_LOGS})",
    )
    parser.add_argument("--metric", choices=("requests", "bytes"), default="requests")
    parser.add_argument(
        "--coverage",
        default="0.99",
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
        help="fail on the first malformed log line instead of skipping it",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="allow an empty result instead of treating it as an error",
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
        log_files = select_recent_logs(args.glob, args.recent, exclude=args.output)
        if not log_files:
            raise ValueError(f"no log files match {args.glob!r}")
        combined_traffic: dict[str, PackageTraffic] = {}
        combined_stats = AnalysisStats()
        deduplicator = VoteDeduplicator()
        # Rotated nginx logs are normally chronological internally. Processing
        # the selected files oldest-first keeps the rolling deduplication window
        # bounded even though selection returns newest-first.
        for log_path in reversed(log_files):
            with open_log(log_path) as stream:
                traffic, stats = analyze_lines(
                    stream, strict=args.strict, deduplicator=deduplicator
                )
                for record in traffic.values():
                    combined = combined_traffic.setdefault(
                        record.package, PackageTraffic(record.package)
                    )
                    combined.requests += record.requests
                    combined.bytes += record.bytes
                for field in AnalysisStats.__dataclass_fields__:
                    setattr(
                        combined_stats,
                        field,
                        getattr(combined_stats, field) + getattr(stats, field),
                    )
                if not args.quiet:
                    print(
                        f"Read {stats.lines} lines from {log_path}; "
                        f"matched {stats.matched} requests.",
                        file=sys.stderr,
                    )

        selected, selected_weight, total_weight = select_by_coverage(
            combined_traffic.values(),
            coverage,
            args.metric,
            include_ties=args.include_ties,
        )
        if not selected and not args.allow_empty:
            raise ValueError(
                "no attributable traffic with a positive selected metric; "
                "check --glob or use --allow-empty"
            )
        contents = "".join(
            f"{record.package}\n"
            for record in sorted(selected, key=lambda r: r.package)
        )
        if args.output:
            args.output.write_text(contents, encoding="utf-8")
        else:
            sys.stdout.write(contents)

        if not args.quiet:
            achieved = (
                Decimal(selected_weight) / Decimal(total_weight)
                if total_weight
                else Decimal(0)
            )
            print(
                f"Selected {len(selected)}/{len(combined_traffic)} projects; "
                f"covered {selected_weight}/{total_weight} {args.metric} "
                f"({achieved:.2%}). Skipped {combined_stats.malformed} malformed, "
                f"{combined_stats.unsuccessful} unsuccessful/non-GET, and "
                f"{combined_stats.unattributed} unattributed requests; "
                f"deduplicated {combined_stats.duplicate} repeated requests.",
                file=sys.stderr,
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
