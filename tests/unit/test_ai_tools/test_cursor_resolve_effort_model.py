"""Tests for CursorAdapter.resolve_effort_model registry validation."""

from __future__ import annotations

import pytest

from crossby.ai_tools.cursor import CursorAdapter
from crossby.data import get_models_for_tool
from crossby.models.ai import EffortLevel


def _pick_known_base_with_thinking() -> str:
    """Pick a registry entry where ``<base>-thinking`` also exists."""
    known = set(get_models_for_tool("cursor"))
    for model in sorted(known):
        if not model.endswith("-thinking") and f"{model}-thinking" in known:
            return model
    pytest.skip("No cursor model with a '-thinking' variant in the registry")


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

    def test_known_model_with_thinking_variant_is_upgraded(self) -> None:
        """Sanity check the happy path: when `<base>-thinking` exists in the
        registry, high effort should upgrade to the thinking variant."""
        adapter = CursorAdapter()
        base = _pick_known_base_with_thinking()
        assert adapter.resolve_effort_model(base, EffortLevel.HIGH) == f"{base}-thinking"

    def test_no_thinking_models_pass_through(self) -> None:
        adapter = CursorAdapter()
        assert adapter.resolve_effort_model("auto", EffortLevel.HIGH) == "auto"

    def test_low_effort_does_not_modify(self) -> None:
        adapter = CursorAdapter()
        base = _pick_known_base_with_thinking()
        assert adapter.resolve_effort_model(base, EffortLevel.LOW) == base

    def test_none_model_passes_through(self) -> None:
        adapter = CursorAdapter()
        assert adapter.resolve_effort_model(None, EffortLevel.HIGH) is None

    def test_already_thinking_not_double_suffixed(self) -> None:
        adapter = CursorAdapter()
        already = "claude-4.6-sonnet-medium-thinking"
        assert adapter.resolve_effort_model(already, EffortLevel.HIGH) == already
