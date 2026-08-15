import json
from unittest.mock import Mock

import click
import pytest
from click.testing import CliRunner

from shadowmire.cli import cli
from shadowmire.filters import (
    PACKAGE_FILTER,
    PackageFilterAction,
    PackageFilterRule,
    PackageInclusionChecker,
)
from shadowmire.sync.base import Plan, SyncBase
from shadowmire.sync.pypi import SyncPyPI


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

        assert checker.includes_metadata("django-ninja") is True
        assert checker.includes_metadata("django") is False
        assert checker.includes_metadata("django-rest-framework") is False
        assert checker.includes_metadata("flask") is True

    def test_reversing_rules_changes_the_result(self):
        checker = PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(
                parse_rule("exclude:^django"),
                parse_rule("include:^django-ninja$"),
            ),
        )

        assert checker.includes_metadata("django-ninja") is False

    def test_unmatched_packages_are_included(self):
        checker = PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(parse_rule("include:^requests$"),),
        )

        assert checker.includes_metadata("requests") is True
        assert checker.includes_metadata("flask") is True

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

        assert checker.includes_metadata("requests") is True
        assert checker.includes_metadata("flask") is True
        assert checker.includes_metadata("django") is False

    def test_new_and_legacy_rules_cannot_be_mixed(self):
        with pytest.raises(ValueError, match="cannot be used together"):
            PackageInclusionChecker(
                include=("^requests$",),
                exclude=(),
                package_filters=(parse_rule("exclude:.*"),),
            )

    def test_metadata_only_keeps_metadata_without_package_files(self):
        checker = PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(
                parse_rule("metadata-only:^large-package$"),
                parse_rule("exclude:^broken-package$"),
            ),
        )

        assert checker.includes_metadata("large-package") is True
        assert checker.includes_package_files("large-package") is False
        assert checker.includes_metadata("broken-package") is False
        assert checker.includes_package_files("broken-package") is False

    def test_first_match_selects_one_state(self):
        checker = PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(
                parse_rule("metadata-only:^demo$"),
                parse_rule("exclude:.*"),
            ),
        )

        assert checker.classify("demo") is PackageFilterAction.METADATA_ONLY
        assert checker.includes_metadata("demo") is True
        assert checker.includes_package_files("demo") is False


class TestPackageFilterParsing:
    def test_cli_pattern_may_contain_colons(self):
        rule = parse_rule("exclude:^demo::package$")

        assert rule.action == "exclude"
        assert rule.pattern.pattern == "^demo::package$"

    def test_metadata_only_action(self):
        rule = parse_rule("metadata-only:^demo:@package$")

        assert rule.action == "metadata-only"
        assert rule.pattern.pattern == "^demo:@package$"

    def test_toml_inline_table(self):
        rule = parse_rule({"action": "include", "pattern": "^requests$"})

        assert rule.action == "include"
        assert rule.pattern.pattern == "^requests$"

    def test_toml_metadata_only(self):
        rule = parse_rule({"action": "metadata-only", "pattern": "^requests$"})

        assert rule.action == "metadata-only"

    @pytest.mark.parametrize(
        "value, message",
        [
            ("exclude", "ACTION:PATTERN"),
            ("allow:^requests$", "action must be"),
            ("exclude@package:^requests$", "action must be"),
            ("exclude:", "non-empty string"),
            ("exclude:[", "invalid regular expression"),
            ({"action": "exclude"}, "must contain exactly"),
            (
                {"action": "exclude", "pattern": ".*", "target": "package"},
                "must contain exactly",
            ),
            ({"action": 1, "pattern": ".*"}, "action must be"),
            ({"action": "exclude", "pattern": 1}, "non-empty string"),
        ],
    )
    def test_invalid_rule(self, value, message):
        with pytest.raises(click.BadParameter, match=message):
            parse_rule(value)


class TestPackageFilterCLI:
    def test_sync_exposes_reconcile_option(self, tmp_path):
        result = CliRunner().invoke(
            cli,
            ["--repo", str(tmp_path / "repo"), "sync", "--help"],
        )

        assert result.exit_code == 0, result.output
        assert "--reconcile-package-files" in result.output

    def test_sync_dry_run_only_prints_plan(self, tmp_path, monkeypatch):
        syncer = Mock()
        syncer.determine_sync_plan.return_value = Plan(
            remove=["removed"],
            update=["updated"],
            package_remove=["metadata-only"],
            package_state_update={"verified": 1},
            remote_last_serial=42,
        )
        monkeypatch.setattr("shadowmire.cli.get_syncer", Mock(return_value=syncer))
        repo = tmp_path / "repo"

        result = CliRunner().invoke(
            cli,
            ["--repo", str(repo), "sync", "--dry-run"],
        )

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {
            "remove": ["removed"],
            "update": ["updated"],
            "package_remove": ["metadata-only"],
            "package_state_update": {"verified": 1},
            "remote_last_serial": 42,
        }
        syncer.do_sync_plan.assert_not_called()
        syncer.finalize.assert_not_called()
        assert not (repo / "plan.json").exists()

    def test_structured_toml_rules_are_accepted(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text(
            """
[options]
package_filters = [
    { action = "include", pattern = "^django-ninja$" },
    { action = "metadata-only", pattern = "^django" },
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
                parse_rule("include:^popular$"),
                parse_rule("metadata-only:.*"),
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

        plan = syncer.determine_sync_plan(
            {"popular": 1, "other": 1},
            self.checker(),
            reconcile_package_files=True,
        )

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

        plan = syncer.determine_sync_plan(
            {"popular": 1, "other": 1},
            self.checker(),
            reconcile_package_files=True,
        )

        assert plan.update == []
        assert plan.package_remove == ["other"]

    def test_no_sync_packages_does_not_inspect_included_files(
        self, tmp_path, monkeypatch
    ):
        get_existing_hrefs = Mock(
            side_effect=AssertionError("unexpected filesystem IO")
        )
        monkeypatch.setattr(
            "shadowmire.sync.base.get_existing_hrefs", get_existing_hrefs
        )
        syncer = StubSync(tmp_path, {"popular": 1}, sync_packages=False)

        action = syncer.inspect_package_files("popular", package_files_included=True)

        assert action is None
        get_existing_hrefs.assert_not_called()

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

    def test_normal_sync_does_not_scan_unchanged_package_files(self, tmp_path):
        write_simple_project(tmp_path, "popular", "popular.whl")
        other_path = write_simple_project(tmp_path, "other", "other.whl")
        other_path.parent.mkdir(parents=True, exist_ok=True)
        other_path.write_bytes(b"package")
        syncer = StubSync(tmp_path, {"popular": 1, "other": 1}, sync_packages=True)

        plan = syncer.determine_sync_plan({"popular": 1, "other": 1}, self.checker())

        assert plan.update == []
        assert plan.package_remove == []
        assert other_path.exists()

    def test_reconcile_handles_ordered_rules(self, tmp_path):
        write_simple_project(tmp_path, "requests", "requests.whl")
        syncer = StubSync(tmp_path, {"requests": 1}, sync_packages=True)
        checker = PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(
                parse_rule("include:^requests$"),
                parse_rule("exclude:.*"),
            ),
        )

        plan = syncer.determine_sync_plan(
            {"requests": 1}, checker, reconcile_package_files=True
        )

        assert plan.update == ["requests"]
        assert plan.package_remove == []

    def test_incremental_update_cleans_excluded_package_files(self, tmp_path):
        package_path = write_simple_project(tmp_path, "other", "other.whl")
        package_path.parent.mkdir(parents=True, exist_ok=True)
        package_path.write_bytes(b"package")
        metadata_path = package_path.with_name("other.whl.metadata")
        metadata_path.write_bytes(b"metadata")
        syncer = object.__new__(SyncPyPI)
        SyncBase.__init__(syncer, tmp_path, Mock(), sync_packages=False)
        syncer.get_package_metadata = Mock(
            return_value={
                "info": {"name": "other"},
                "last_serial": 2,
                "releases": {},
            }
        )
        syncer.get_package_simple = Mock(return_value={"files": []})
        file_checker = Mock()
        file_checker.get_filtered_meta.side_effect = lambda _name, meta: meta

        serial = syncer.do_update(
            "other",
            file_checker,
            package_files_included=False,
            use_db=False,
        )

        assert serial == 2
        assert not package_path.exists()
        assert not metadata_path.exists()
        assert (tmp_path / "simple" / "other" / "index.v1_json").exists()

    def test_updates_receive_the_package_file_decision(self, tmp_path):
        write_simple_project(tmp_path, "popular", "popular.whl")
        write_simple_project(tmp_path, "other", "other.whl")
        syncer = StubSync(tmp_path, {"popular": 2, "other": 2}, sync_packages=True)
        checker = self.checker()
        plan = syncer.determine_sync_plan({"popular": 1, "other": 1}, checker)

        success = syncer.do_sync_plan(plan, checker, Mock())

        assert success is True
        assert sorted(syncer.updates) == [("other", False), ("popular", True)]

    def test_exclude_removes_the_whole_project(self, tmp_path):
        syncer = StubSync(tmp_path, {"broken": 1, "healthy": 1}, False)
        checker = PackageInclusionChecker(
            include=(),
            exclude=(),
            package_filters=(parse_rule("exclude:^broken$"),),
        )

        plan = syncer.determine_sync_plan({"broken": 1, "healthy": 1}, checker)

        assert plan.remove == ["broken"]
        assert plan.update == []
        assert plan.package_remove == []
