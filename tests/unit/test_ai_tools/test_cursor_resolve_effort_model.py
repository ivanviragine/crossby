"""Tests for CursorAdapter.resolve_effort_model: effort selects a Cursor catalog ID."""

from __future__ import annotations

import warnings

import pytest

import crossby.ai_tools.cursor as cursor_module
from crossby.ai_tools.cursor import CursorAdapter
from crossby.data import get_models_for_tool
from crossby.models.ai import EffortLevel

_BASE = "example-model-1.0"


@pytest.fixture
def thinking_registry(monkeypatch: pytest.MonkeyPatch) -> str:
    """Serve a registry where ``<base>-thinking`` exists and return the base.

    Cursor's live catalog no longer carries a bare base with a ``-thinking``
    twin (its thinking variants all encode an effort level), so the upgrade
    path is exercised against a fixed registry instead of being skipped.
    """
    monkeypatch.setattr(
        cursor_module, "get_models_for_tool", lambda _tool: [_BASE, f"{_BASE}-thinking"]
    )
    return _BASE


class TestResolveEffortModel:
    def test_unknown_model_passes_through_for_high_effort(self) -> None:
        """Unknown models: the -thinking variant isn't in the registry, so
        resolver must return the original unchanged rather than fabricate an
        invalid ID that the Cursor CLI would reject."""
        adapter = CursorAdapter()
        unknown = "fake-future-model-9.9"
        assert unknown not in get_models_for_tool("cursor")

        for effort in (EffortLevel.HIGH, EffortLevel.XHIGH, EffortLevel.MAX):
            assert adapter.resolve_effort_model(unknown, effort) == unknown

    def test_known_model_with_thinking_variant_is_upgraded(self, thinking_registry: str) -> None:
        """Sanity check the happy path: when `<base>-thinking` exists in the
        registry, high effort should upgrade to the thinking variant."""
        adapter = CursorAdapter()
        base = thinking_registry
        assert adapter.resolve_effort_model(base, EffortLevel.HIGH) == f"{base}-thinking"

    def test_no_thinking_models_pass_through(self) -> None:
        adapter = CursorAdapter()
        assert adapter.resolve_effort_model("auto", EffortLevel.HIGH) == "auto"

    def test_low_effort_does_not_modify(self, thinking_registry: str) -> None:
        adapter = CursorAdapter()
        base = thinking_registry
        assert adapter.resolve_effort_model(base, EffortLevel.LOW) == base

    def test_none_model_passes_through(self) -> None:
        adapter = CursorAdapter()
        assert adapter.resolve_effort_model(None, EffortLevel.HIGH) is None

    def test_already_thinking_not_double_suffixed(self) -> None:
        adapter = CursorAdapter()
        already = "claude-4.6-sonnet-medium-thinking"
        assert adapter.resolve_effort_model(already, EffortLevel.HIGH) == already


class TestCatalogEffortVariants:
    """Effort selects a real sibling from the bundled Cursor catalog.

    Every ID below comes from a logged-in ``agent --list-models``; the
    ``test_ids_come_from_catalog`` guard fails if the catalog drifts away from
    them, rather than letting these cases pass vacuously.
    """

    @pytest.mark.parametrize(
        ("model", "effort", "expected"),
        [
            # <family>-<effort>: the requested effort replaces the one in the ID.
            ("claude-opus-5-5-medium", EffortLevel.HIGH, "claude-opus-5-5-high"),
            ("claude-opus-5-5-high", EffortLevel.LOW, "claude-opus-5-5-low"),
            ("claude-opus-5-5-medium", EffortLevel.MAX, "claude-opus-5-5-max"),
            # -fast is kept.
            ("claude-opus-5-5-medium-fast", EffortLevel.XHIGH, "claude-opus-5-5-xhigh-fast"),
            # <family>-thinking-<effort>: thinking is kept.
            ("claude-opus-4-8-thinking-low", EffortLevel.MAX, "claude-opus-4-8-thinking-max"),
            (
                "claude-opus-4-8-thinking-high-fast",
                EffortLevel.MEDIUM,
                "claude-opus-4-8-thinking-medium-fast",
            ),
            # <family>-<effort>-thinking (older Claude naming).
            ("claude-4.6-opus-high-thinking", EffortLevel.MAX, "claude-4.6-opus-max-thinking"),
            # GPT-5.5 spells xhigh as extra-high.
            ("gpt-5.5-medium", EffortLevel.XHIGH, "gpt-5.5-extra-high"),
            ("gpt-5.5-extra-high-fast", EffortLevel.LOW, "gpt-5.5-low-fast"),
            # A none-effort ID still has an effort position to fill.
            ("gpt-5.6-sol-none", EffortLevel.HIGH, "gpt-5.6-sol-high"),
            # A bare ID whose family ships explicit efforts is the medium tier.
            ("gpt-5.3-codex", EffortLevel.HIGH, "gpt-5.3-codex-high"),
            ("gpt-5.3-codex-fast", EffortLevel.XHIGH, "gpt-5.3-codex-xhigh-fast"),
            ("gpt-5.3-codex-high", EffortLevel.MEDIUM, "gpt-5.3-codex"),
            ("gpt-5.2", EffortLevel.LOW, "gpt-5.2-low"),
            # "mini" is part of the family, not an effort.
            ("gpt-5.4-mini-medium", EffortLevel.HIGH, "gpt-5.4-mini-high"),
        ],
    )
    def test_effort_selects_catalog_sibling(
        self, model: str, effort: EffortLevel, expected: str
    ) -> None:
        assert CursorAdapter().resolve_effort_model(model, effort) == expected

    @pytest.mark.parametrize(
        ("model", "effort", "expected"),
        [
            # Non-thinking Opus 5 stops at high; max snaps down and keeps the ID.
            ("claude-opus-5-high", EffortLevel.MAX, "claude-opus-5-high"),
            # Kimi K3 has no medium; the tie between low and high goes higher.
            ("kimi-k3-low", EffortLevel.MEDIUM, "kimi-k3-high"),
            # Claude 4.6 Opus ships only high and max.
            ("claude-4.6-opus-max", EffortLevel.LOW, "claude-4.6-opus-high"),
        ],
    )
    def test_missing_tier_falls_back_to_nearest_with_warning(
        self, model: str, effort: EffortLevel, expected: str
    ) -> None:
        with pytest.warns(UserWarning, match="Cursor offers no"):
            assert CursorAdapter().resolve_effort_model(model, effort) == expected

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-3.1-pro",  # no effort variants at all
            "composer-2.5",
            "composer-2.5-fast",
            "claude-opus-4-8[context=1m,effort=high,fast=false]",  # explicit overrides
        ],
    )
    def test_models_without_effort_variants_pass_through(self, model: str) -> None:
        for effort in EffortLevel:
            assert CursorAdapter().resolve_effort_model(model, effort) == model

    def test_matching_effort_keeps_model(self) -> None:
        adapter = CursorAdapter()
        assert adapter.resolve_effort_model("claude-opus-5-5-high", EffortLevel.HIGH) == (
            "claude-opus-5-5-high"
        )
        assert adapter.resolve_effort_model("gpt-5.5-extra-high", EffortLevel.XHIGH) == (
            "gpt-5.5-extra-high"
        )

    def test_no_effort_leaves_model_unchanged(self) -> None:
        assert CursorAdapter().resolve_effort_model("claude-opus-5-5-medium", None) == (
            "claude-opus-5-5-medium"
        )

    def test_results_are_always_catalog_ids(self) -> None:
        # Exhaustive over the real catalog: whatever comes back is either the
        # input unchanged or a registered Cursor ID, never a fabricated one.
        adapter = CursorAdapter()
        registry = set(get_models_for_tool("cursor"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for model in sorted(registry):
                for effort in EffortLevel:
                    result = adapter.resolve_effort_model(model, effort)
                    assert result == model or result in registry, (model, effort, result)

    def test_launch_command_carries_the_resolved_model(self) -> None:
        cmd = CursorAdapter().build_launch_command(
            model="claude-opus-5-5-medium", effort=EffortLevel.HIGH
        )
        assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5-high"

    def test_ids_come_from_catalog(self) -> None:
        registry = set(get_models_for_tool("cursor"))
        used = {
            "claude-opus-5-5-medium",
            "claude-opus-5-5-high",
            "claude-opus-5-5-low",
            "claude-opus-5-5-max",
            "claude-opus-5-5-medium-fast",
            "claude-opus-5-5-xhigh-fast",
            "claude-opus-4-8-thinking-low",
            "claude-opus-4-8-thinking-max",
            "claude-opus-4-8-thinking-high-fast",
            "claude-opus-4-8-thinking-medium-fast",
            "claude-4.6-opus-high-thinking",
            "claude-4.6-opus-max-thinking",
            "claude-4.6-opus-max",
            "claude-4.6-opus-high",
            "gpt-5.5-medium",
            "gpt-5.5-extra-high",
            "gpt-5.5-extra-high-fast",
            "gpt-5.5-low-fast",
            "gpt-5.6-sol-none",
            "gpt-5.6-sol-high",
            "gpt-5.3-codex",
            "gpt-5.3-codex-high",
            "gpt-5.3-codex-fast",
            "gpt-5.3-codex-xhigh-fast",
            "gpt-5.2",
            "gpt-5.2-low",
            "gpt-5.4-mini-medium",
            "gpt-5.4-mini-high",
            "claude-opus-5-high",
            "kimi-k3-low",
            "kimi-k3-high",
            "gemini-3.1-pro",
            "composer-2.5",
            "composer-2.5-fast",
        }
        assert used <= registry, sorted(used - registry)
