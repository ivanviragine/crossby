"""Tests for prompt helpers and TTY-aware defaults."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from crossby.ui.prompts import confirm, multi_select, select


def _mock_questionary_result(return_value: object) -> MagicMock:
    question = MagicMock()
    question.ask.return_value = return_value
    return question


class TestSelect:
    def test_first_item_returns_zero(self) -> None:
        with (
            patch("crossby.ui.prompts.is_tty", return_value=True),
            patch("questionary.select", return_value=_mock_questionary_result("a")),
        ):
            result = select("Pick one", ["a", "b"])
        assert result == 0

    def test_second_item_returns_one(self) -> None:
        with (
            patch("crossby.ui.prompts.is_tty", return_value=True),
            patch("questionary.select", return_value=_mock_questionary_result("b")),
        ):
            result = select("Pick one", ["a", "b"])
        assert result == 1

    def test_returns_default_when_not_tty(self) -> None:
        with patch("crossby.ui.prompts.is_tty", return_value=False):
            result = select("Pick one", ["a", "b", "c"], default=2)
        assert result == 2


class TestConfirm:
    def test_returns_default_when_not_tty(self) -> None:
        with patch("crossby.ui.prompts.is_tty", return_value=False):
            assert confirm("Continue?", default=True) is True


class TestMultiSelect:
    def test_returns_all_items_when_not_tty(self) -> None:
        with patch("crossby.ui.prompts.is_tty", return_value=False):
            assert multi_select("Pick", ["a", "b", "c"]) == [0, 1, 2]
