import json
from unittest.mock import Mock

import click
import pytest
from click.testing import CliRunner

from shadowmire import (
    PACKAGE_FILTER,
    PackageFilterRule,
    PackageInclusionChecker,
    SyncBase,
    cli,
)


def parse_rule(value: str | dict) -> PackageFilterRule:
    return PACKAGE_FILTER.convert(value, None, None)


class TestOrderedPackageFilters:
    def test_first_matching_rule_wins(self):
        checker = PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(
                parse_rule("include:^django-ninja$"),
                parse_rule("exclude:^django"),
            ),
        )

        assert checker.is_included("django-ninja") is True
        assert checker.is_included("django") is False
        assert checker.is_included("django-rest-framework") is False
        assert checker.is_included("flask") is True

    def test_reversing_rules_changes_the_result(self):
        checker = PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(
                parse_rule("exclude:^django"),
                parse_rule("include:^django-ninja$"),
            ),
        )

        assert checker.is_included("django-ninja") is False

    def test_unmatched_packages_are_included(self):
        checker = PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(parse_rule("include:^requests$"),),
        )

        assert checker.is_included("requests") is True
        assert checker.is_included("flask") is True

    def test_catch_all_exclude_creates_a_whitelist(self):
        checker = PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(
                parse_rule("include:^requests$"),
                parse_rule("include:^flask$"),
                parse_rule("exclude:.*"),
            ),
        )

        assert checker.is_included("requests") is True
        assert checker.is_included("flask") is True
        assert checker.is_included("django") is False

    def test_new_and_legacy_rules_cannot_be_mixed(self):
        with pytest.raises(ValueError, match="cannot be used together"):
            PackageInclusionChecker(
                include=("^requests$",),
                exclude=(),
                package_filters=(parse_rule("exclude:.*"),),
            )

    def test_rules_are_evaluated_independently_by_target(self):
        checker = PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(
                parse_rule("exclude@package:^large-package$"),
                parse_rule("exclude@metadata:^broken-package$"),
            ),
        )

        assert checker.is_included("large-package", "metadata") is True
        assert checker.is_included("large-package", "package") is False
        assert checker.is_included("broken-package", "metadata") is False
        # A metadata exclusion always prevents syncing package files.
        assert checker.is_included("broken-package", "package") is False

    def test_rules_for_other_targets_do_not_stop_matching(self):
        checker = PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(
                parse_rule("include@metadata:^demo$"),
                parse_rule("exclude@both:.*"),
            ),
        )

        assert checker.is_included("demo", "metadata") is True
        assert checker.is_included("demo", "package") is False


class TestPackageFilterParsing:
    def test_cli_pattern_may_contain_colons(self):
        rule = parse_rule("exclude:^demo::package$")

        assert rule.action == "exclude"
        assert rule.pattern.pattern == "^demo::package$"
        assert rule.target == "both"

    def test_cli_target(self):
        rule = parse_rule("exclude@package:^demo:@package$")

        assert rule.action == "exclude"
        assert rule.pattern.pattern == "^demo:@package$"
        assert rule.target == "package"

    def test_toml_inline_table(self):
        rule = parse_rule({"action": "include", "pattern": "^requests$"})

        assert rule.action == "include"
        assert rule.pattern.pattern == "^requests$"
        assert rule.target == "both"

    @pytest.mark.parametrize("target", ["package", "metadata", "both"])
    def test_toml_target(self, target):
        rule = parse_rule(
            {"action": "exclude", "pattern": "^requests$", "target": target}
        )

        assert rule.target == target

    @pytest.mark.parametrize(
        "value, message",
        [
            ("exclude", "TARGET.*PATTERN"),
            ("allow:^requests$", "action must be"),
            ("exclude@invalid:^requests$", "target must be"),
            ("exclude:", "non-empty string"),
            ("exclude:[", "invalid regular expression"),
            ({"action": "exclude"}, "must contain 'action' and 'pattern'"),
            (
                {"action": "exclude", "pattern": ".*", "extra": True},
                "must contain 'action' and 'pattern'",
            ),
            ({"action": 1, "pattern": ".*"}, "action must be"),
            (
                {"action": "exclude", "pattern": ".*", "target": 1},
                "target must be",
            ),
            ({"action": "exclude", "pattern": 1}, "non-empty string"),
        ],
    )
    def test_invalid_rule(self, value, message):
        with pytest.raises(click.BadParameter, match=message):
            parse_rule(value)


class TestPackageFilterCLI:
    def test_structured_toml_rules_are_accepted(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text(
            """
[options]
package_filters = [
    { action = "include", pattern = "^django-ninja$", target = "package" },
    { action = "exclude", pattern = "^django", target = "package" },
]
"""
        )

        result = CliRunner().invoke(
            cli,
            [
                "--config",
                str(config),
                "--repo",
                str(tmp_path / "repo"),
                "do-remove",
                "example",
            ],
        )

        assert result.exit_code == 0, result.output

    @pytest.mark.parametrize(
        "config_body, command_options",
        [
            (
                '[options]\ninclude = ["^requests$"]\n',
                ["--package-filter", "exclude:.*"],
            ),
            (
                """
[options]
package_filters = [{ action = "exclude", pattern = ".*" }]
""",
                ["--exclude", "^django$"],
            ),
            (
                """
[options]
include = ["^requests$"]
package_filters = [{ action = "exclude", pattern = ".*" }]
""",
                [],
            ),
        ],
    )
    def test_new_and_legacy_options_cannot_be_mixed(
        self, tmp_path, config_body, command_options
    ):
        config = tmp_path / "config.toml"
        config.write_text(config_body)

        result = CliRunner().invoke(
            cli,
            [
                "--config",
                str(config),
                "--repo",
                str(tmp_path / "repo"),
                "do-remove",
                *command_options,
                "example",
            ],
        )

        assert result.exit_code == 2
        assert "cannot be used together" in result.output


class StubSync(SyncBase):
    def __init__(self, basedir, remote, sync_packages):
        self.remote = remote
        self.updates = []
        self.local_db_mock = Mock()
        super().__init__(basedir, self.local_db_mock, sync_packages)

    def fetch_remote_versions(self):
        return 10, self.remote

    def do_update(
        self,
        package_name,
        file_inclusion_checker,
        package_files_included,
        use_db=True,
    ):
        self.updates.append((package_name, package_files_included))
        return self.remote[package_name]


def write_simple_project(basedir, package_name, filename, has_metadata=True):
    package_simple_dir = basedir / "simple" / package_name
    package_simple_dir.mkdir(parents=True)
    relative_url = f"../../packages/{filename}"
    (package_simple_dir / "index.v1_json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "url": relative_url,
                        "core-metadata": has_metadata,
                    }
                ]
            }
        )
    )
    return basedir / "packages" / filename


class TestPackageFilePlan:
    def checker(self):
        return PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(
                parse_rule("include@package:^popular$"),
                parse_rule("exclude@package:.*"),
            ),
        )

    def test_missing_included_files_are_updated_and_excluded_files_are_removed(
        self, tmp_path
    ):
        popular_path = write_simple_project(tmp_path, "popular", "popular.whl")
        other_path = write_simple_project(tmp_path, "other", "other.whl")
        other_path.parent.mkdir(parents=True, exist_ok=True)
        other_path.write_bytes(b"package")
        other_path.with_name("other.whl.metadata").write_bytes(b"metadata")
        syncer = StubSync(tmp_path, {"popular": 1, "other": 1}, sync_packages=True)

        plan = syncer.determine_sync_plan({"popular": 1, "other": 1}, self.checker())

        assert plan.remove == []
        assert plan.update == ["popular"]
        assert plan.package_remove == ["other"]
        assert not popular_path.exists()

    def test_no_sync_packages_still_cleans_but_does_not_schedule_downloads(
        self, tmp_path
    ):
        write_simple_project(tmp_path, "popular", "popular.whl")
        other_path = write_simple_project(tmp_path, "other", "other.whl")
        other_path.parent.mkdir(parents=True, exist_ok=True)
        other_path.write_bytes(b"package")
        syncer = StubSync(tmp_path, {"popular": 1, "other": 1}, sync_packages=False)

        plan = syncer.determine_sync_plan({"popular": 1, "other": 1}, self.checker())

        assert plan.update == []
        assert plan.package_remove == ["other"]

    def test_removing_package_files_preserves_project_metadata(self, tmp_path):
        package_path = write_simple_project(tmp_path, "other", "other.whl")
        package_path.parent.mkdir(parents=True, exist_ok=True)
        package_path.write_bytes(b"package")
        metadata_path = package_path.with_name("other.whl.metadata")
        metadata_path.write_bytes(b"metadata")
        json_path = tmp_path / "json" / "other"
        json_path.parent.mkdir(parents=True)
        json_path.write_text("{}")
        syncer = StubSync(tmp_path, {"other": 1}, sync_packages=False)

        syncer.remove_package_files("other")

        assert not package_path.exists()
        assert not metadata_path.exists()
        assert (tmp_path / "simple" / "other" / "index.v1_json").exists()
        assert json_path.exists()
        assert syncer.local_db_mock.mock_calls == []

    def test_updates_receive_the_package_target_decision(self, tmp_path):
        write_simple_project(tmp_path, "popular", "popular.whl")
        write_simple_project(tmp_path, "other", "other.whl")
        syncer = StubSync(tmp_path, {"popular": 2, "other": 2}, sync_packages=True)
        checker = self.checker()
        plan = syncer.determine_sync_plan({"popular": 1, "other": 1}, checker)

        success = syncer.do_sync_plan(plan, checker, Mock())

        assert success is True
        assert sorted(syncer.updates) == [("other", False), ("popular", True)]

    def test_metadata_target_removes_the_whole_project(self, tmp_path):
        syncer = StubSync(tmp_path, {"broken": 1, "healthy": 1}, False)
        checker = PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(parse_rule("exclude@metadata:^broken$"),),
        )

        plan = syncer.determine_sync_plan({"broken": 1, "healthy": 1}, checker)

        assert plan.remove == ["broken"]
        assert plan.update == []
        assert plan.package_remove == []
