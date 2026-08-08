"""Scene-specific version gating (``at_least``, ``detect_tool_version``).

The generic probe (``parse_semver`` / ``detect_binary_version``) moved to
``crossby.utils.versioning``; its coverage now lives in
``tests/unit/test_utils/test_versioning.py``. This module keeps the
scene-specific gating and a re-export smoke test guarding source compatibility.
"""

from __future__ import annotations

from crossby.scenes import versioning


class TestReExport:
    def test_generic_helpers_re_exported(self) -> None:
        # Source compatibility: existing callers reach the generic probe through
        # the scenes module, so both names must still resolve here.
        from crossby.utils import versioning as utils_versioning

        assert versioning.parse_semver is utils_versioning.parse_semver
        assert versioning.detect_binary_version is utils_versioning.detect_binary_version


class TestAtLeast:
    def test_meets_floor(self) -> None:
        assert versioning.at_least((2, 1, 218), (2, 1, 129)) is True

    def test_below_floor(self) -> None:
        assert versioning.at_least((2, 1, 100), (2, 1, 129)) is False

    def test_unknown_fails_closed(self) -> None:
        assert versioning.at_least(None, (2, 1, 129)) is False
