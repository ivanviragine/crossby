"""Tests for config defaults — tier mapping correctness."""

from __future__ import annotations

import pytest

from crossby.ai_tools.model_utils import classify_tier_universal
from crossby.config.defaults import TOOL_DEFAULTS, get_defaults
from crossby.data import get_models_for_tool
from crossby.models.ai import AIToolID, ModelTier

_TIERS = ("easy", "medium", "complex", "very_complex")

# Effort suffixes a default model ID may encode (antigravity-cli bakes effort
# into the model ID, so a suffixed default is launchable when its base is a
# registry member).
_EFFORT_SUFFIXES = ("-low", "-medium", "-high", "-xhigh", "-max")


def _strip_effort_suffix(model_id: str) -> str:
    for suffix in _EFFORT_SUFFIXES:
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def _iter_tier_defaults() -> list[tuple[str, str, str]]:
    """Flatten TOOL_DEFAULTS into (tool_id, tier, model_id) tuples."""
    return [
        (str(tool_id), tier, getattr(mapping, tier))
        for tool_id, mapping in TOOL_DEFAULTS.items()
        for tier in _TIERS
    ]


_ALL_DEFAULT_MODEL_IDS = sorted({model_id for _, _, model_id in _iter_tier_defaults()})

# Expected tier for every distinct TOOL_DEFAULTS model ID, hand-derived from
# classify_tier_universal's documented keyword rules (haiku/flash/mini/luna ->
# FAST; opus/fable/pro/sol/max -> POWERFUL; sonnet/terra or no keyword ->
# BALANCED). Pins how the novel effort-encoded IDs
# (composer-2.5-fast, claude-opus-5-high, gemini-3.8-flash-*)
# parse, so a regex/keyword regression fails the test instead of slipping through.
_EXPECTED_TIERS: dict[str, ModelTier] = {
    "anthropic/claude-haiku-4.5": ModelTier.FAST,
    "anthropic/claude-opus-4.7": ModelTier.POWERFUL,
    "anthropic/claude-sonnet-4.6": ModelTier.BALANCED,
    "claude-haiku-4.5": ModelTier.FAST,
    "claude-opus-5": ModelTier.POWERFUL,
    "claude-opus-5-high": ModelTier.POWERFUL,
    "claude-sonnet-4.6": ModelTier.BALANCED,
    "claude-sonnet-5": ModelTier.BALANCED,
    "composer-2.5": ModelTier.BALANCED,  # no keyword -> BALANCED fallback
    # "fast" is not a FAST keyword (haiku/flash/spark/mini/luna are), so this
    # falls through to the BALANCED default despite how the name reads.
    "composer-2.5-fast": ModelTier.BALANCED,
    "gemini-3.8-flash-high": ModelTier.FAST,
    "gemini-3.8-flash-low": ModelTier.FAST,
    "gemini-3.8-flash-medium": ModelTier.FAST,
    "gpt-5.4": ModelTier.POWERFUL,
    "gpt-5.6-luna": ModelTier.FAST,
    "gpt-5.6-sol": ModelTier.POWERFUL,
    "gpt-5.6-terra": ModelTier.BALANCED,
}


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
        if model_id in registry:
            return
        # A default may encode an effort suffix that the catalog stores only in
        # its base form (antigravity-cli). It's launchable when the base resolves
        # to a registry member — the adapter bakes the effort back in at launch.
        base = _strip_effort_suffix(model_id)
        assert base in registry, (
            f"{tool_id}.{tier} default '{model_id}' (base '{base}') is not in "
            f"get_models_for_tool('{tool_id}')"
        )

    def test_expected_tiers_cover_all_defaults(self) -> None:
        # Guards the map below against drift: adding a new default model ID
        # without an expected tier fails here instead of silently skipping it.
        assert set(_EXPECTED_TIERS) == set(_ALL_DEFAULT_MODEL_IDS)

    @pytest.mark.parametrize("model_id", _ALL_DEFAULT_MODEL_IDS)
    def test_default_classifies_to_expected_tier(self, model_id: str) -> None:
        # Genuine regression guard: the novel effort-encoded IDs
        # (composer-2.5-fast, claude-opus-5-high, gemini-3.6-flash-*) must keep
        # classifying by their family keyword rather than shifting tier.
        assert classify_tier_universal(model_id) == _EXPECTED_TIERS[model_id]

    @pytest.mark.parametrize(
        "tool_id",
        [AIToolID.CLAUDE, AIToolID.COPILOT, AIToolID.CODEX, AIToolID.OPENCODE],
    )
    def test_provider_with_three_roles_assigns_each_slot_semantically(self, tool_id: str) -> None:
        mapping = get_defaults(tool_id)
        assert classify_tier_universal(mapping.easy) == ModelTier.FAST
        assert classify_tier_universal(mapping.medium) == ModelTier.BALANCED
        assert classify_tier_universal(mapping.complex) == ModelTier.BALANCED
        assert classify_tier_universal(mapping.very_complex) == ModelTier.POWERFUL


class TestDocumentedFamilyRoles:
    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("gpt-5.6-luna-high", ModelTier.FAST),
            ("gpt-5.6-terra-low", ModelTier.BALANCED),
            ("gpt-5.6-sol-none", ModelTier.POWERFUL),
            ("claude-fable-5.1-low", ModelTier.POWERFUL),
        ],
    )
    def test_explicit_family_component_controls_tier(
        self, model_id: str, expected: ModelTier
    ) -> None:
        assert classify_tier_universal(model_id) == expected

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            # A variant marker upgrades the family it is attached to: luna-pro
            # is a more capable model than luna, not the fast tier. Regression
            # for the registered openai/gpt-5.6-luna-pro catalog entry.
            ("openai/gpt-5.6-luna-pro", ModelTier.POWERFUL),
            ("gpt-5.6-luna-pro", ModelTier.POWERFUL),
            # ...but "max" is an effort level far more often than a family, so
            # any family keyword outranks it: luna at max effort is still fast
            # and terra at max effort is still balanced. Both are registered
            # catalog IDs, and every other effort variant of them already
            # classifies by family.
            ("gpt-5.6-luna-max", ModelTier.FAST),
            ("gpt-5.6-luna-max-fast", ModelTier.FAST),
            ("gpt-5.6-terra-max", ModelTier.BALANCED),
            ("gpt-5.6-terra-max-fast", ModelTier.BALANCED),
            ("claude-sonnet-4.6-max", ModelTier.BALANCED),
            # With no family keyword to outrank it, max still reads as powerful.
            ("some-model-max", ModelTier.POWERFUL),
            # Plain family markers keep their tier when no variant is present.
            ("openai/gpt-5.6-luna", ModelTier.FAST),
            ("gemini-3.1-pro", ModelTier.POWERFUL),
            ("claude-haiku-4.5", ModelTier.FAST),
        ],
    )
    def test_variant_marker_outranks_family_but_effort_does_not(
        self, model_id: str, expected: ModelTier
    ) -> None:
        assert classify_tier_universal(model_id) == expected


class TestClaudeTierDefaults:
    """Claude fallback tiers track the current model generation (WADE #309 port)."""

    def test_claude_tiers_resolve_current_models(self) -> None:
        mapping = get_defaults(AIToolID.CLAUDE)
        assert mapping.easy == "claude-haiku-4.5"
        assert mapping.medium == "claude-sonnet-5"
        assert mapping.complex == "claude-sonnet-5"
        assert mapping.very_complex == "claude-opus-5"


@pytest.mark.parametrize(
    ("tool_id", "expected"),
    [
        (
            AIToolID.COPILOT,
            ("claude-haiku-4.5", "claude-sonnet-4.6", "claude-sonnet-4.6", "gpt-5.4"),
        ),
        (
            AIToolID.ANTIGRAVITY_CLI,
            (
                "gemini-3.8-flash-low",
                "gemini-3.8-flash-medium",
                "gemini-3.8-flash-medium",
                "gemini-3.8-flash-high",
            ),
        ),
        (
            AIToolID.CODEX,
            ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-terra", "gpt-5.6-sol"),
        ),
    ],
)
def test_refreshed_provider_defaults(tool_id: AIToolID, expected: tuple[str, ...]) -> None:
    mapping = get_defaults(tool_id)
    assert tuple(getattr(mapping, tier) for tier in _TIERS) == expected
