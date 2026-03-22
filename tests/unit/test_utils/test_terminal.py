"""Focused tests for public terminal helper behavior."""

from __future__ import annotations

import io
from unittest.mock import Mock

from crossby.utils import terminal


class TestTerminalTitle:
    def test_set_terminal_title_writes_escape_when_tty(self, monkeypatch) -> None:
        fake_stderr = io.StringIO()
        monkeypatch.setattr("crossby.utils.terminal.is_tty", lambda: True)
        monkeypatch.setattr("crossby.utils.terminal.sys.stderr", fake_stderr)

        terminal.set_terminal_title("crossby test")

        assert "\033]0;crossby test\007" in fake_stderr.getvalue()

    def test_stop_title_keeper_joins_existing_thread(self, monkeypatch) -> None:
        thread = Mock()
        monkeypatch.setattr("crossby.utils.terminal._title_keeper_running", True)
        monkeypatch.setattr("crossby.utils.terminal._title_keeper_thread", thread)

        terminal.stop_title_keeper()

        thread.join.assert_called_once_with(timeout=3.0)
        assert terminal._title_keeper_running is False
        assert terminal._title_keeper_thread is None
