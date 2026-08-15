import click
import pytest
from click.testing import CliRunner

from shadowmire import (
    PACKAGE_FILTER,
    PackageFilterRule,
    PackageInclusionChecker,
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


class TestPackageFilterParsing:
    def test_cli_pattern_may_contain_colons(self):
        rule = parse_rule("exclude:^demo::package$")

        assert rule.action == "exclude"
        assert rule.pattern.pattern == "^demo::package$"

    def test_toml_inline_table(self):
        rule = parse_rule({"action": "include", "pattern": "^requests$"})

        assert rule.action == "include"
        assert rule.pattern.pattern == "^requests$"

    @pytest.mark.parametrize(
        "value, message",
        [
            ("exclude", "ACTION:PATTERN"),
            ("allow:^requests$", "action must be"),
            ("exclude:", "non-empty string"),
            ("exclude:[", "invalid regular expression"),
            ({"action": "exclude"}, "exactly 'action' and 'pattern'"),
            (
                {"action": "exclude", "pattern": ".*", "extra": True},
                "exactly 'action' and 'pattern'",
            ),
            ({"action": 1, "pattern": ".*"}, "action must be"),
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
    { action = "include", pattern = "^django-ninja$" },
    { action = "exclude", pattern = "^django" },
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
