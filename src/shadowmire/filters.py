import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import click

from .constants import PRERELEASE_PATTERNS


class PackageFilterAction(StrEnum):
    INCLUDE = "include"
    METADATA_ONLY = "metadata-only"
    EXCLUDE = "exclude"

    @property
    def includes_metadata(self) -> bool:
        return self is not PackageFilterAction.EXCLUDE

    @property
    def includes_package_files(self) -> bool:
        return self is PackageFilterAction.INCLUDE


@dataclass(frozen=True)
class PackageFilterRule:
    action: PackageFilterAction
    pattern: re.Pattern[str]


class PackageFilterParamType(click.ParamType):
    name = "package-filter"

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> PackageFilterRule:
        if isinstance(value, PackageFilterRule):
            return value

        if isinstance(value, str):
            action, separator, pattern = value.partition(":")
            if not separator:
                self.fail(
                    "must use ACTION:PATTERN format (for example, exclude:^django)",
                    param,
                    ctx,
                )
        elif isinstance(value, dict):
            required_keys = {"action", "pattern"}
            if set(value) != required_keys:
                self.fail(
                    "TOML rule must contain exactly 'action' and 'pattern'",
                    param,
                    ctx,
                )
            action = value["action"]
            pattern = value["pattern"]
        else:
            self.fail(
                "must be an ACTION:PATTERN string or a TOML inline table",
                param,
                ctx,
            )

        try:
            filter_action = PackageFilterAction(action)
        except (TypeError, ValueError):
            self.fail(
                "action must be 'include', 'metadata-only', or 'exclude'", param, ctx
            )
        if not isinstance(pattern, str) or not pattern:
            self.fail("pattern must be a non-empty string", param, ctx)

        try:
            compiled_pattern: re.Pattern[str] = re.compile(str(pattern))
        except re.error as e:
            self.fail(f"invalid regular expression {pattern!r}: {e}", param, ctx)

        return PackageFilterRule(action=filter_action, pattern=compiled_pattern)


PACKAGE_FILTER = PackageFilterParamType()


def compile_regexes(patterns: tuple[str, ...]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def match_patterns(
    s: str, ps: list[re.Pattern[str]] | tuple[re.Pattern[str], ...]
) -> bool:
    """
    Search if any of the patterns match the string `s`.

    Uses re.search(), matching anywhere in the string.
    """
    for p in ps:
        if p.search(s):
            return True
    return False


class PackageInclusionChecker:
    """
    A class for handling packages inclusion/exclusion based on regex patterns.
    """

    def __init__(
        self,
        exclude: tuple[str, ...],
        include: tuple[str, ...],
        package_filters: tuple[PackageFilterRule, ...] = (),
    ) -> None:
        if package_filters and (exclude or include):
            raise ValueError(
                "package_filters cannot be used together with include or exclude"
            )
        self.excludes = compile_regexes(exclude)
        self.includes = compile_regexes(include)
        self.package_filters = package_filters

    def has_rules(self) -> bool:
        return bool(self.excludes or self.includes or self.package_filters)

    def filters_package_files(self) -> bool:
        return bool(self.package_filters)

    def classify(self, package_name: str) -> PackageFilterAction:
        if self.package_filters:
            for rule in self.package_filters:
                if rule.pattern.search(package_name):
                    return rule.action
            return PackageFilterAction.INCLUDE

        if self.includes and match_patterns(package_name, self.includes):
            return PackageFilterAction.INCLUDE

        if self.excludes and match_patterns(package_name, self.excludes):
            return PackageFilterAction.EXCLUDE

        if not self.includes or self.excludes:
            return PackageFilterAction.INCLUDE
        return PackageFilterAction.EXCLUDE

    def includes_metadata(self, package_name: str) -> bool:
        return self.classify(package_name).includes_metadata

    def includes_package_files(self, package_name: str) -> bool:
        return self.classify(package_name).includes_package_files


class FileInclusionChecker:
    """
    A class for filtering package releases and files based on various criteria:

    - Shall this package exclude pre-releases?
    - Is this file excluded by given filename patterns?
    - Is this release yanked?
    - Is this release too old?
    """

    def __init__(
        self,
        prerelease_exclude: tuple[str],
        excluded_wheel_filename: tuple[str],
        filter_meta: bool,
        skip_yanked: bool,
        skip_old_packages_days: int | None,
        least_releases_to_keep: int,
    ) -> None:
        self.prerelease_excludes = compile_regexes(prerelease_exclude)
        self.excluded_wheel_filenames = compile_regexes(excluded_wheel_filename)
        self.filter_meta = filter_meta
        self.skip_yanked = skip_yanked
        self.skip_old_packages_days = skip_old_packages_days
        # Treat 0 as None...
        if self.skip_old_packages_days == 0:
            self.skip_old_packages_days = None
        self.least_releases_to_keep = least_releases_to_keep

    def has_rules(self) -> bool:
        return bool(
            self.prerelease_excludes
            or self.excluded_wheel_filenames
            or self.skip_yanked
            or self.skip_old_packages_days is not None
        )

    def get_filtered_meta(self, package_name: str, meta: dict) -> dict:
        """
        If filter_meta is True, modifies meta in place and returns it.
        Otherwise the original meta is not modified, and a filtered copy is returned.
        """
        if not self.has_rules():
            return meta
        if self.filter_meta:
            new_meta = meta
        else:
            new_meta = deepcopy(meta)

        if match_patterns(package_name, self.prerelease_excludes):
            for release in list(new_meta["releases"].keys()):
                if match_patterns(release, PRERELEASE_PATTERNS):
                    del new_meta["releases"][release]
        if self.excluded_wheel_filenames:
            for release_infos in new_meta["releases"].values():
                for release_idx in range(len(release_infos) - 1, -1, -1):
                    release_info = release_infos[release_idx]
                    filename = release_info["filename"]
                    if match_patterns(filename, self.excluded_wheel_filenames):
                        del release_infos[release_idx]
        if self.skip_yanked:
            for release_infos in new_meta["releases"].values():
                for release_idx in range(len(release_infos) - 1, -1, -1):
                    release_info = release_infos[release_idx]
                    if release_info.get("yanked", False):
                        del release_infos[release_idx]
        removed_old_release_infos: dict[str, list[tuple[datetime, dict]]] = {}
        if self.skip_old_packages_days is not None:
            threshold_date = datetime.now(UTC) - timedelta(
                days=self.skip_old_packages_days
            )
            releases = new_meta["releases"]
            for release, release_infos in releases.items():
                for release_idx in range(len(release_infos) - 1, -1, -1):
                    release_info = release_infos[release_idx]
                    upload_time_str = release_info.get("upload_time_iso_8601", None)
                    if upload_time_str is None:
                        continue
                    upload_time = datetime.fromisoformat(upload_time_str)
                    if upload_time < threshold_date:
                        removed_old_release_infos.setdefault(release, []).append(
                            (upload_time, release_info)
                        )
                        del release_infos[release_idx]
            if self.least_releases_to_keep > 0:
                remaining_releases = sum(1 for infos in releases.values() if infos)
                missing = self.least_releases_to_keep - remaining_releases
                if missing > 0:
                    # Re-add the newest releases that were removed due to age.
                    candidates: list[tuple[datetime, str]] = []
                    for release, removed_infos in removed_old_release_infos.items():
                        release_infos = releases.get(release)
                        if release_infos is None or release_infos:
                            continue
                        latest_upload = max(ts for ts, _ in removed_infos)
                        candidates.append((latest_upload, release))
                    candidates.sort(reverse=True)
                    for _, release in candidates[:missing]:
                        release_infos = releases.get(release)
                        if release_infos is None:
                            continue
                        to_restore = removed_old_release_infos.get(release, [])
                        for _, info in sorted(to_restore, key=lambda x: x[0]):
                            release_infos.append(info)

        return new_meta
