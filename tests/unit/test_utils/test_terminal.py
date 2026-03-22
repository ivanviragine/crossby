"""Tests for terminal title helpers and temp-script creation."""

from __future__ import annotations

import io
from unittest.mock import Mock

from crossby.utils import terminal


class TestTerminalTitle:
    def test_truncate_terminal_title(self) -> None:
        long = "x" * 80
        result = terminal._truncate_terminal_title(long)
        assert result.endswith("...")
        assert len(result) == 53

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


class TestCreateTempScript:
    def test_create_temp_script_content(self, monkeypatch) -> None:
        written: list[str] = []

        class _Tmp:
            name = "/tmp/crossby-test.sh"

            def __enter__(self) -> _Tmp:
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def write(self, data: str) -> int:
                written.append(data)
                return len(data)

        chmod_mock = Mock()
        monkeypatch.setattr("tempfile.NamedTemporaryFile", lambda **_: _Tmp())
        monkeypatch.setattr("crossby.utils.terminal.os.chmod", chmod_mock)

        path = terminal._create_temp_script(["python", "-V"], cwd="/tmp")

        assert path == "/tmp/crossby-test.sh"
        assert chmod_mock.call_args.args[1] == 0o700
        script = written[0]
        assert script.startswith("#!/usr/bin/env bash")
        assert "cd /tmp" in script
        assert "exec python -V" in script
