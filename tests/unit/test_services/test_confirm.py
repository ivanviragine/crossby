"""Tests for the generic ``confirm_defaults`` helper.

Mocks ``crossby.ui.prompts.is_tty`` and ``prompts.select`` at the module
where they are defined so caller-side lazy imports resolve to the patches.
"""

from __future__ import annotations

from typing import Any

import pytest

from crossby.services.confirm import ConfirmField, confirm_defaults


class _PromptRecorder:
    """Tiny stand-in for ``prompts.select`` that plays back a scripted queue.

    Each scripted entry is a 0-based index to return for the next call.
    Assertions about invocation order are made by reading ``self.calls``.
    """

    def __init__(self, responses: list[int]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, title: str, items: list[str], default: int = 0) -> int:
        self.calls.append((title, list(items)))
        if not self.responses:
            raise AssertionError(
                f"prompts.select called beyond scripted responses: title={title!r}"
            )
        return self.responses.pop(0)


@pytest.fixture()
def tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: True)


@pytest.fixture()
def no_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)


def _install_select(monkeypatch: pytest.MonkeyPatch, recorder: _PromptRecorder) -> None:
    monkeypatch.setattr("crossby.ui.prompts.select", recorder)


class TestFastPaths:
    def test_non_tty_returns_current_values(
        self, no_tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _PromptRecorder([])
        _install_select(monkeypatch, recorder)

        fields = [
            ConfirmField(
                name="tool",
                label="Tool",
                current_value="claude",
                explicit=False,
                change_fn=lambda _v, _s: {"tool": "other"},
            ),
        ]
        result = confirm_defaults(fields)

        assert result == {"tool": "claude"}
        assert recorder.calls == []

    def test_all_explicit_returns_current_values(
        self, tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _PromptRecorder([])
        _install_select(monkeypatch, recorder)

        fields = [
            ConfirmField(
                name="a",
                label="A",
                current_value=1,
                explicit=True,
                change_fn=lambda _v, _s: {"a": 999},
            ),
            ConfirmField(
                name="b",
                label="B",
                current_value=2,
                explicit=True,
                change_fn=lambda _v, _s: {"b": 999},
            ),
        ]
        result = confirm_defaults(fields)

        assert result == {"a": 1, "b": 2}
        assert recorder.calls == []

    def test_all_visible_fields_explicit_still_returns_fast(
        self, tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hidden fields with explicit=False should not keep the loop alive."""
        recorder = _PromptRecorder([])
        _install_select(monkeypatch, recorder)

        fields = [
            ConfirmField(
                name="a",
                label="A",
                current_value="x",
                explicit=True,
                change_fn=lambda _v, _s: {"a": "y"},
            ),
            ConfirmField(
                name="hidden",
                label="Hidden",
                current_value=None,
                explicit=False,
                change_fn=lambda _v, _s: {"hidden": "set"},
                visible_when=lambda _s: False,
            ),
        ]
        result = confirm_defaults(fields)

        assert result == {"a": "x", "hidden": None}
        assert recorder.calls == []


class TestMenuAndChange:
    def test_proceed_exits_without_change(
        self, tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _PromptRecorder([0])
        _install_select(monkeypatch, recorder)

        fields = [
            ConfirmField(
                name="tool",
                label="Tool",
                current_value="claude",
                explicit=False,
                change_fn=lambda _v, _s: {"tool": "other"},
            ),
        ]
        result = confirm_defaults(fields)

        assert result == {"tool": "claude"}
        assert len(recorder.calls) == 1
        title, items = recorder.calls[0]
        assert title == "Confirm selection"
        assert items == ["Proceed", "Change tool"]

    def test_explicit_fields_have_no_change_entry(
        self, tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _PromptRecorder([0])
        _install_select(monkeypatch, recorder)

        fields = [
            ConfirmField(
                name="a",
                label="A",
                current_value=1,
                explicit=True,
                change_fn=lambda _v, _s: {"a": 2},
            ),
            ConfirmField(
                name="b",
                label="B",
                current_value=3,
                explicit=False,
                change_fn=lambda _v, _s: {"b": 4},
            ),
        ]
        confirm_defaults(fields)

        _title, items = recorder.calls[0]
        assert items == ["Proceed", "Change b"]

    def test_change_fn_updates_value(
        self, tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # First prompt: pick "Change tool" (index 1). Second prompt: Proceed (0).
        recorder = _PromptRecorder([1, 0])
        _install_select(monkeypatch, recorder)

        seen_states: list[dict[str, Any]] = []

        def _change(current: Any, state: dict[str, Any]) -> dict[str, Any]:
            seen_states.append(dict(state))
            return {"tool": "cursor"}

        fields = [
            ConfirmField(
                name="tool",
                label="Tool",
                current_value="claude",
                explicit=False,
                change_fn=_change,
            ),
        ]
        result = confirm_defaults(fields)

        assert result == {"tool": "cursor"}
        assert seen_states == [{"tool": "claude"}]

    def test_cascade_update_returns_multiple_keys(
        self, tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pick "Change tool" (index 1 of ["Proceed","Change tool","Change model"]),
        # then Proceed (0).
        recorder = _PromptRecorder([1, 0])
        _install_select(monkeypatch, recorder)

        def _tool_change(_current: Any, _state: dict[str, Any]) -> dict[str, Any]:
            return {"tool": "cursor", "model": "gpt-5"}

        fields = [
            ConfirmField(
                name="tool",
                label="Tool",
                current_value="claude",
                explicit=False,
                change_fn=_tool_change,
            ),
            ConfirmField(
                name="model",
                label="Model",
                current_value="sonnet",
                explicit=False,
                change_fn=lambda _v, _s: {"model": "other"},
            ),
        ]
        result = confirm_defaults(fields)

        assert result == {"tool": "cursor", "model": "gpt-5"}

    def test_visible_when_hides_field(
        self, tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _PromptRecorder([0])
        _install_select(monkeypatch, recorder)

        fields = [
            ConfirmField(
                name="a",
                label="A",
                current_value="x",
                explicit=False,
                change_fn=lambda _v, _s: {"a": "y"},
            ),
            ConfirmField(
                name="b",
                label="B",
                current_value="z",
                explicit=False,
                change_fn=lambda _v, _s: {"b": "w"},
                visible_when=lambda _s: False,
            ),
        ]
        confirm_defaults(fields)

        _title, items = recorder.calls[0]
        assert items == ["Proceed", "Change a"]

    def test_visible_when_reacts_to_state_change(
        self, tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Changing field ``a`` should toggle visibility of field ``b``."""
        # Loop:
        #  1st select: change a  (index 1)
        #  2nd select: now that a == "enable", b is visible. Proceed (0).
        recorder = _PromptRecorder([1, 0])
        _install_select(monkeypatch, recorder)

        fields = [
            ConfirmField(
                name="a",
                label="A",
                current_value="disable",
                explicit=False,
                change_fn=lambda _v, _s: {"a": "enable"},
            ),
            ConfirmField(
                name="b",
                label="B",
                current_value="off",
                explicit=False,
                change_fn=lambda _v, _s: {"b": "on"},
                visible_when=lambda s: s["a"] == "enable",
            ),
        ]
        result = confirm_defaults(fields)

        assert result == {"a": "enable", "b": "off"}
        assert recorder.calls[0][1] == ["Proceed", "Change a"]
        assert recorder.calls[1][1] == ["Proceed", "Change a", "Change b"]

    def test_custom_menu_label(
        self, tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _PromptRecorder([0])
        _install_select(monkeypatch, recorder)

        fields = [
            ConfirmField(
                name="yolo",
                label="YOLO mode",
                current_value=False,
                explicit=False,
                change_fn=lambda v, _s: {"yolo": not v},
                menu_label=lambda v: "Turn off YOLO mode" if v else "Turn on YOLO mode",
            ),
        ]
        confirm_defaults(fields)

        _title, items = recorder.calls[0]
        assert items == ["Proceed", "Turn on YOLO mode"]

    def test_custom_title_passed_through(
        self, tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _PromptRecorder([0])
        _install_select(monkeypatch, recorder)

        fields = [
            ConfirmField(
                name="x",
                label="X",
                current_value=1,
                explicit=False,
                change_fn=lambda _v, _s: {"x": 2},
            ),
        ]
        confirm_defaults(fields, title="Confirm handoff")

        title, _items = recorder.calls[0]
        assert title == "Confirm handoff"
