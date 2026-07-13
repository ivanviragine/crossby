"""Tests for config defaults — tier mapping correctness."""

from __future__ import annotations

from crossby.config.defaults import TOOL_DEFAULTS, get_defaults
from crossby.models.ai import AIToolID


class TestMediumTierDefaults:
    """Medium tier must map to balanced-tier (not fast-tier) for all tools."""

    def test_medium_differs_from_easy_for_all_tools(self) -> None:
        for tool_id, mapping in TOOL_DEFAULTS.items():
            assert mapping.medium != mapping.easy, (
                f"{tool_id}: medium ({mapping.medium}) should differ from easy ({mapping.easy})"
            )

    def test_medium_equals_complex_for_all_tools(self) -> None:
        for tool_id, mapping in TOOL_DEFAULTS.items():
            assert mapping.medium == mapping.complex, (
                f"{tool_id}: medium ({mapping.medium}) should equal "
                f"complex ({mapping.complex}) — both balanced-tier"
            )

    def test_claude_medium_is_sonnet(self) -> None:
        mapping = get_defaults(AIToolID.CLAUDE)
        assert mapping.medium == "claude-sonnet-5"

    def test_cursor_medium_is_balanced(self) -> None:
        mapping = get_defaults(AIToolID.CURSOR)
        assert mapping.medium == "sonnet-4.6"


class TestClaudeTierDefaults:
    """Claude fallback tiers track the current model generation (WADE #309 port)."""

    def test_claude_tiers_resolve_current_models(self) -> None:
        mapping = get_defaults(AIToolID.CLAUDE)
        assert mapping.easy == "claude-haiku-4.5"
        assert mapping.medium == "claude-sonnet-5"
        assert mapping.complex == "claude-sonnet-5"
        assert mapping.very_complex == "claude-opus-4.8"
