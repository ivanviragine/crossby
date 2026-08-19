"""Tests for ``crossby.ui.prompts.multi_select``, focused on ``select_all``.

Two layers are covered:

- The pure helpers backing the mutual-exclusion toggle and index resolution,
  tested directly (fast, no terminal needed).
- The actual interactive key bindings in ``_select_all_checkbox``, driven
  end-to-end through a real ``prompt_toolkit`` ``Application`` via a piped
  input, so the ~100-line custom keybinding implementation is exercised for
  real rather than only through its stubbed-out pure logic.
"""

from __future__ import annotations

import pytest
import typer
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from crossby.ui import prompts


def _run_checkbox(items: list[str], keys: str) -> list[int | str] | None:
    """Drive ``_select_all_checkbox`` with real key presses via a pipe input."""
    with create_pipe_input() as pipe_input:
        pipe_input.send_text(keys)
        with create_app_session(input=pipe_input, output=DummyOutput()):
            return prompts._select_all_checkbox("Select tools", items).ask()  # type: ignore[no-any-return]


class TestBuildSelectAllChoices:
    def test_select_all_choice_is_first_and_checked(self) -> None:
        choices = prompts._build_select_all_choices(["Claude", "Codex"])

        assert choices[0].value == prompts._SELECT_ALL_VALUE
        assert choices[0].checked is True

    def test_individual_choices_use_index_values_and_start_unchecked(self) -> None:
        choices = prompts._build_select_all_choices(["Claude", "Codex"])

        individual = choices[1:]
        assert [c.value for c in individual] == [0, 1]
        assert all(c.checked is False for c in individual)


class TestToggleSelectAll:
    def test_toggling_select_all_on_clears_individual_choices(self) -> None:
        result = prompts._toggle_select_all([0], prompts._SELECT_ALL_VALUE)
        assert result == [prompts._SELECT_ALL_VALUE]

    def test_toggling_select_all_off_leaves_nothing_selected(self) -> None:
        result = prompts._toggle_select_all([prompts._SELECT_ALL_VALUE], prompts._SELECT_ALL_VALUE)
        assert result == []

    def test_toggling_individual_choice_on_clears_select_all(self) -> None:
        result = prompts._toggle_select_all([prompts._SELECT_ALL_VALUE], 0)
        assert result == [0]

    def test_toggling_individual_choice_off_keeps_others(self) -> None:
        result = prompts._toggle_select_all([0, 1], 0)
        assert result == [1]

    def test_toggling_individual_choice_on_keeps_other_individuals(self) -> None:
        result = prompts._toggle_select_all([0], 1)
        assert result == [0, 1]


class TestResolveSelectAllIndices:
    def test_select_all_value_resolves_to_every_index(self) -> None:
        indices = prompts._resolve_select_all_indices(
            [prompts._SELECT_ALL_VALUE], ["Claude", "Codex"]
        )
        assert indices == [0, 1]

    def test_specific_items_resolve_to_their_indices(self) -> None:
        indices = prompts._resolve_select_all_indices([1], ["Claude", "Codex"])
        assert indices == [1]

    def test_nothing_selected_resolves_to_empty_list(self) -> None:
        assert prompts._resolve_select_all_indices([], ["Claude", "Codex"]) == []

    def test_all_individual_items_selected_resolves_to_every_index(self) -> None:
        indices = prompts._resolve_select_all_indices([0, 1], ["Claude", "Codex"])
        assert indices == [0, 1]

    def test_item_literally_named_select_all_sentinel_stays_distinguishable(self) -> None:
        # A real item happens to be labelled "__all__"; its value is its
        # index (1), never the string sentinel, so it can be selected alone.
        indices = prompts._resolve_select_all_indices([1], ["Claude", "__all__"])
        assert indices == [1]


class TestMultiSelectNonTty:
    def test_non_tty_returns_all_indices_with_select_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(prompts, "is_tty", lambda: False)
        result = prompts.multi_select("Select tools", ["Claude", "Codex"], select_all=True)
        assert result == [0, 1]

    def test_non_tty_returns_all_indices_without_select_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(prompts, "is_tty", lambda: False)
        result = prompts.multi_select("Select tools", ["Claude", "Codex"])
        assert result == [0, 1]


class _FakeQuestion:
    def __init__(self, answer: list[int | str] | None) -> None:
        self._answer = answer

    def ask(self) -> list[int | str] | None:
        return self._answer


class TestMultiSelectInteractiveSelectAll:
    def test_default_confirmation_selects_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(prompts, "is_tty", lambda: True)
        monkeypatch.setattr(
            prompts,
            "_select_all_checkbox",
            lambda _title, _items: _FakeQuestion([prompts._SELECT_ALL_VALUE]),
        )
        result = prompts.multi_select("Select tools", ["Claude", "Codex"], select_all=True)
        assert result == [0, 1]

    def test_specific_selection_returns_matching_indices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(prompts, "is_tty", lambda: True)
        monkeypatch.setattr(
            prompts, "_select_all_checkbox", lambda _title, _items: _FakeQuestion([1])
        )
        result = prompts.multi_select("Select tools", ["Claude", "Codex"], select_all=True)
        assert result == [1]

    def test_nothing_selected_returns_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(prompts, "is_tty", lambda: True)
        monkeypatch.setattr(
            prompts, "_select_all_checkbox", lambda _title, _items: _FakeQuestion([])
        )
        result = prompts.multi_select("Select tools", ["Claude", "Codex"], select_all=True)
        assert result == []

    def test_ctrl_c_raises_typer_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(prompts, "is_tty", lambda: True)
        monkeypatch.setattr(
            prompts, "_select_all_checkbox", lambda _title, _items: _FakeQuestion(None)
        )
        with pytest.raises(typer.Exit):
            prompts.multi_select("Select tools", ["Claude", "Codex"], select_all=True)


class TestSelectAllCheckboxKeyBindings:
    """Drives the real Application/KeyBindings wiring, not the stubbed pure logic."""

    def test_enter_alone_confirms_select_all(self) -> None:
        result = _run_checkbox(["Claude", "Codex", "Cursor"], "\r")
        assert result == [prompts._SELECT_ALL_VALUE]

    def test_selecting_one_item_deselects_select_all(self) -> None:
        # Down arrow (ESC O B) to the first item, space to toggle it, enter.
        result = _run_checkbox(["Claude", "Codex", "Cursor"], "\x1bOB \r")
        assert result == [0]

    def test_selecting_multiple_items_accumulates(self) -> None:
        result = _run_checkbox(["Claude", "Codex", "Cursor"], "\x1bOB \x1bOB \r")
        assert result == [0, 1]

    def test_reselecting_select_all_clears_individual_items(self) -> None:
        # Select an item, move back up to "Select all", toggle it back on.
        result = _run_checkbox(["Claude", "Codex", "Cursor"], "\x1bOB \x1bOA \r")
        assert result == [prompts._SELECT_ALL_VALUE]

    def test_deselecting_select_all_alone_leaves_nothing_selected(self) -> None:
        result = _run_checkbox(["Claude", "Codex"], " \r")
        assert result == []

    def test_ctrl_c_aborts_with_no_result(self) -> None:
        # Question.ask() swallows KeyboardInterrupt and returns None; the
        # caller (multi_select) is what turns that into typer.Exit.
        assert _run_checkbox(["Claude", "Codex"], "\x03") is None
