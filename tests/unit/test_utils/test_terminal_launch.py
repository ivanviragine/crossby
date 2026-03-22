"""Tests for terminal detection and launcher command construction."""

from __future__ import annotations

import subprocess
from unittest.mock import Mock

import pytest

from crossby.utils.terminal import (
    detect_terminal,
    launch_batch_in_terminals,
    launch_in_new_terminal,
)


def test_iterm2_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.delenv("TMUX", raising=False)
    assert detect_terminal() == "iterm2"


def test_gnome_terminal_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(
        "crossby.utils.terminal.shutil.which",
        lambda name: "/usr/bin/gnome-terminal" if name == "gnome-terminal" else None,
    )
    assert detect_terminal() == "gnome-terminal"


def test_iterm2_launch_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crossby.utils.terminal.sys.platform", "darwin")
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr("crossby.utils.terminal.shutil.which", lambda _: None)

    run_mock = Mock(return_value=Mock())
    monkeypatch.setattr("crossby.utils.terminal.subprocess.run", run_mock)

    result = launch_in_new_terminal(["python", "-V"], cwd="/tmp")

    assert result is True
    args = run_mock.call_args.args[0]
    assert args[0] == "osascript"
    assert 'tell application "iTerm2"' in args[2]
    assert "create window with default profile command" in args[2]


def test_gnome_terminal_launch_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr("crossby.utils.terminal.sys.platform", "linux")
    monkeypatch.setattr(
        "crossby.utils.terminal.shutil.which",
        lambda name: "/usr/bin/gnome-terminal" if name == "gnome-terminal" else None,
    )

    popen_mock = Mock(return_value=Mock())
    monkeypatch.setattr("crossby.utils.terminal.subprocess.Popen", popen_mock)

    result = launch_in_new_terminal(["python", "-V"], cwd="/tmp")

    assert result is True
    args = popen_mock.call_args.args[0]
    assert args[0] == "gnome-terminal"
    assert args[1:4] == ["--", "bash", "-c"]
    assert "; exec bash" in args[4]


def test_ghostty_macos_uses_open_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crossby.utils.terminal.sys.platform", "darwin")
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr("crossby.utils.terminal.shutil.which", lambda _: "/opt/bin/ghostty")
    monkeypatch.setattr(
        "crossby.utils.terminal._create_temp_script",
        lambda command, cwd=None: "/tmp/crossby-test.sh",
    )

    run_mock = Mock(return_value=Mock())
    timer_mock = Mock()
    timer_mock.return_value.start = Mock()
    monkeypatch.setattr("crossby.utils.terminal.subprocess.run", run_mock)
    monkeypatch.setattr("crossby.utils.terminal.threading.Timer", timer_mock)

    result = launch_in_new_terminal(["python", "-V"], cwd="/tmp")

    assert result is True
    assert run_mock.call_args.args[0] == [
        "open",
        "-na",
        "Ghostty",
        "--args",
        "-e",
        "/tmp/crossby-test.sh",
    ]


def test_ghostty_macos_failure_cleans_temp_script(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crossby.utils.terminal.sys.platform", "darwin")
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr("crossby.utils.terminal.shutil.which", lambda _: "/opt/bin/ghostty")
    monkeypatch.setattr(
        "crossby.utils.terminal._create_temp_script",
        lambda command, cwd=None: "/tmp/crossby-test.sh",
    )

    def _run_side_effect(*args, **kwargs):
        cmd = args[0]
        raise subprocess.CalledProcessError(1, cmd[0])

    unlink_mock = Mock()
    monkeypatch.setattr("crossby.utils.terminal.subprocess.run", _run_side_effect)
    monkeypatch.setattr("crossby.utils.terminal._safe_unlink", unlink_mock)

    result = launch_in_new_terminal(["python", "-V"], cwd="/tmp")

    assert result is False
    unlink_mock.assert_any_call("/tmp/crossby-test.sh")


def test_batch_single_delegates_to_launch_in_new_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_mock = Mock(return_value=True)
    monkeypatch.setattr("crossby.utils.terminal.launch_in_new_terminal", launch_mock)

    result = launch_batch_in_terminals([(["python", "-V"], "/tmp", "Crossby")])

    assert result is True
    launch_mock.assert_called_once_with(["python", "-V"], cwd="/tmp", title="Crossby")
