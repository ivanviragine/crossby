"""Tests for CursorAdapter.resolve_effort_model registry validation."""

from __future__ import annotations

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
