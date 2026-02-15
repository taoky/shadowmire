"""
Unit tests for PackageInclusionChecker.is_included() method.

Tests cover the following scenarios:
1. Both includes and excludes are empty - all packages should be included
2. Both includes and excludes exist - includes has higher priority (whitelist + blacklist mode)
3. Only includes exists - whitelist mode
4. Only excludes exists - blacklist mode
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shadowmire import PackageInclusionChecker


# =============================================================================
# Test Data: Package Names
# =============================================================================

PACKAGE_NAMES = [
    "requests",
    "flask",
    "django",
    "numpy",
    "pandas",
    "flask-restful",
    "django-rest-framework",
    "fastapi",
    "pytest",
    "pytest-cov",
    "black",
    "mypy",
    "package-rc1",
    "package-a1",
    "package-b1",
    "package-dev1",
]

# =============================================================================
# Scenario 1: Both includes and excludes are empty
# =============================================================================


class TestBothEmpty:
    """Test when both includes and excludes are empty (no rules)."""

    @pytest.mark.parametrize("package_name", PACKAGE_NAMES)
    def test_empty_rules_all_included(self, package_name):
        """All packages should be included when no rules are specified."""
        checker = PackageInclusionChecker(exclude=(), include=())
        assert checker.is_included(package_name) is True


# =============================================================================
# Scenario 2: Both includes and excludes exist
# =============================================================================


class TestBothExist:
    """
    Test when both includes and excludes exist.

    Behavior: includes (whitelist) has higher priority than excludes (blacklist).
    - If matched by includes → included (True), regardless of excludes match
    - If not matched by includes but matched by excludes → excluded (False)
    - If matched by neither → default included (True)
    """

    def test_both_include_exclude(self):
        """Test when both include and exclude exist (include has priority)."""
        checker = PackageInclusionChecker(
            include=(r"flask",), exclude=(r"django", r"flask", r"pytest")
        )

        # Matches both include and exclude → included (include wins)
        assert checker.is_included("flask") is True
        assert checker.is_included("flask-restful") is True
        # Matches exclude only → excluded
        assert checker.is_included("django") is False
        assert checker.is_included("django-rest-framework") is False
        assert checker.is_included("pytest") is False
        assert checker.is_included("pytest-cov") is False
        # Matches neither → included (default)
        assert checker.is_included("requests") is True
        assert checker.is_included("numpy") is True
        assert checker.is_included("pandas") is True


# =============================================================================
# Scenario 3: Only includes exist (whitelist mode)
# =============================================================================


class TestOnlyIncludes:
    """
    Test when only includes exist (whitelist mode).

    Behavior:
    - If matched by includes → included (True)
    - If not matched by includes → excluded (False)
    """

    def test_whitelist_mode(self):
        """Test whitelist mode: matched packages included, unmatched excluded."""
        checker = PackageInclusionChecker(include=(r"flask",), exclude=())

        # Matched by whitelist - should be included
        assert checker.is_included("flask") is True
        assert checker.is_included("flask-restful") is True
        # Not matched by whitelist - should be excluded
        assert checker.is_included("django") is False
        assert checker.is_included("requests") is False
        assert checker.is_included("pytest") is False

# =============================================================================
# Scenario 4: Only excludes exist (blacklist mode)
# =============================================================================


class TestOnlyExcludes:
    """
    Test when only excludes exist (blacklist mode).

    Behavior:
    - If matched by excludes → excluded (False)
    - If not matched by excludes → included (True)
    """

    def test_blacklist_mode(self):
        """Test blacklist mode: matched packages excluded, unmatched included."""
        # Exclude patterns: (r"django", r"flask", r"pytest")
        checker = PackageInclusionChecker(include=(), exclude=(r"django", r"flask", r"pytest"))

        # Matched by blacklist - should be excluded
        assert checker.is_included("django") is False
        assert checker.is_included("flask") is False
        assert checker.is_included("pytest") is False
        assert checker.is_included("django-rest-framework") is False
        # Not matched by blacklist - should be included
        assert checker.is_included("requests") is True
        assert checker.is_included("numpy") is True
        assert checker.is_included("pandas") is True

# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_string_package_name(self):
        """Test with empty string as package name."""
        checker = PackageInclusionChecker(include=(r"flask",), exclude=())
        assert checker.is_included("") is False

    def test_empty_string_no_rules(self):
        """Test empty string package name with no rules."""
        checker = PackageInclusionChecker(exclude=(), include=())
        assert checker.is_included("") is True

    def test_anchored_pattern(self):
        """Test pattern with anchors (^ and $)."""
        checker = PackageInclusionChecker(include=(r"^flask$",), exclude=())

        assert checker.is_included("flask") is True
        assert checker.is_included("flask-restful") is False

    def test_case_sensitivity(self):
        """Test that pattern matching is case-sensitive."""
        checker = PackageInclusionChecker(include=(r"Flask",), exclude=())

        assert checker.is_included("Flask") is True
        assert checker.is_included("flask") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
