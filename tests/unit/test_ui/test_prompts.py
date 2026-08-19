"""Tests for ``crossby.ui.prompts.multi_select``, focused on ``select_all``.

The interactive checkbox UI itself isn't driven here (that would mean
simulating a real terminal); instead the pure helpers backing the
mutual-exclusion toggle and index resolution are tested directly, plus the
public ``multi_select`` entrypoints for non-TTY and the interactive path
(with ``_select_all_checkbox`` stubbed out).
"""

from __future__ import annotations

import pytest
import typer

from crossby.ui import prompts


class TestBuildSelectAllChoices:
    def test_select_all_choice_is_first_and_checked(self) -> None:
        choices = prompts._build_select_all_choices(["Claude", "Codex"])

        assert choices[0].value == prompts._SELECT_ALL_VALUE
        assert choices[0].checked is True

    def test_individual_choices_start_unchecked(self) -> None:
        choices = prompts._build_select_all_choices(["Claude", "Codex"])

        individual = choices[1:]
        assert [c.value for c in individual] == ["Claude", "Codex"]
        assert all(c.checked is False for c in individual)


class TestToggleSelectAll:
    def test_toggling_select_all_on_clears_individual_choices(self) -> None:
        result = prompts._toggle_select_all(["Claude"], prompts._SELECT_ALL_VALUE)
        assert result == [prompts._SELECT_ALL_VALUE]

    def test_toggling_select_all_off_leaves_nothing_selected(self) -> None:
        result = prompts._toggle_select_all([prompts._SELECT_ALL_VALUE], prompts._SELECT_ALL_VALUE)
        assert result == []

    def test_toggling_individual_choice_on_clears_select_all(self) -> None:
        result = prompts._toggle_select_all([prompts._SELECT_ALL_VALUE], "Claude")
        assert result == ["Claude"]

    def test_toggling_individual_choice_off_keeps_others(self) -> None:
        result = prompts._toggle_select_all(["Claude", "Codex"], "Claude")
        assert result == ["Codex"]

    def test_toggling_individual_choice_on_keeps_other_individuals(self) -> None:
        result = prompts._toggle_select_all(["Claude"], "Codex")
        assert result == ["Claude", "Codex"]


class TestResolveSelectAllIndices:
    def test_select_all_value_resolves_to_every_index(self) -> None:
        indices = prompts._resolve_select_all_indices(
            [prompts._SELECT_ALL_VALUE], ["Claude", "Codex"]
        )
        assert indices == [0, 1]

    def test_specific_items_resolve_to_their_indices(self) -> None:
        indices = prompts._resolve_select_all_indices(["Codex"], ["Claude", "Codex"])
        assert indices == [1]

    def test_nothing_selected_resolves_to_empty_list(self) -> None:
        assert prompts._resolve_select_all_indices([], ["Claude", "Codex"]) == []

    def test_all_individual_items_selected_resolves_to_every_index(self) -> None:
        indices = prompts._resolve_select_all_indices(["Claude", "Codex"], ["Claude", "Codex"])
        assert indices == [0, 1]


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
    def __init__(self, answer: list[str] | None) -> None:
        self._answer = answer

    def ask(self) -> list[str] | None:
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
            prompts, "_select_all_checkbox", lambda _title, _items: _FakeQuestion(["Codex"])
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
