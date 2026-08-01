"""Tests for config defaults — tier mapping correctness."""

from __future__ import annotations

import pytest

from crossby.ai_tools.model_utils import classify_tier_universal
from crossby.config.defaults import TOOL_DEFAULTS, get_defaults
from crossby.data import get_models_for_tool
from crossby.models.ai import AIToolID, ModelTier

_TIERS = ("easy", "medium", "complex", "very_complex")


def _iter_tier_defaults() -> list[tuple[str, str, str]]:
    """Flatten TOOL_DEFAULTS into (tool_id, tier, model_id) tuples."""
    return [
        (str(tool_id), tier, getattr(mapping, tier))
        for tool_id, mapping in TOOL_DEFAULTS.items()
        for tier in _TIERS
    ]


_ALL_DEFAULT_MODEL_IDS = sorted({model_id for _, _, model_id in _iter_tier_defaults()})


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
        assert mapping.medium == "composer-2.5"


class TestDefaultsRegistryGuard:
    """Every TOOL_DEFAULTS value must be a valid, classifiable registry entry.

    This is the linchpin against registry/defaults drift: `get_defaults()` seeds
    the model config written by `wade init` downstream, and every fallback pick
    must be a member of that tool's `get_models_for_tool()` list (data/models.json).
    """

    @pytest.mark.parametrize(("tool_id", "tier", "model_id"), _iter_tier_defaults())
    def test_default_is_registry_member(self, tool_id: str, tier: str, model_id: str) -> None:
        registry = get_models_for_tool(tool_id)
        assert model_id in registry, (
            f"{tool_id}.{tier} default '{model_id}' is not in get_models_for_tool('{tool_id}')"
        )

    @pytest.mark.parametrize("model_id", _ALL_DEFAULT_MODEL_IDS)
    def test_default_resolves_through_classifier(self, model_id: str) -> None:
        # Cheap insurance: novel effort-encoded IDs (composer-2.5-fast,
        # claude-opus-5-high, gemini-3.6-flash-*) must parse without raising.
        assert isinstance(classify_tier_universal(model_id), ModelTier)


class TestClaudeTierDefaults:
    """Claude fallback tiers track the current model generation (WADE #309 port)."""

    def test_claude_tiers_resolve_current_models(self) -> None:
        mapping = get_defaults(AIToolID.CLAUDE)
        assert mapping.easy == "claude-haiku-4.5"
        assert mapping.medium == "claude-sonnet-5"
        assert mapping.complex == "claude-sonnet-5"
        assert mapping.very_complex == "claude-opus-5"
