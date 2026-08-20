"""Tests for the ``crossby tools update`` backend service.

Every function is failure-safe: an updater that exits non-zero, is missing,
times out, or hits an ``OSError`` becomes report data, never an exception. These
tests monkeypatch ``detect_installed``, the version probe, and
``utils.process.run`` to drive each path deterministically.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence

import pytest

from crossby.ai_tools.base import AbstractAITool
from crossby.models.ai import AIToolCapabilities, AIToolID, AIToolType
from crossby.services import tool_update
from crossby.services.tool_update import probe_version, run_update, updatable_tools
from crossby.utils.process import CommandError


class _FakeAdapter:
    """Minimal adapter stand-in exposing a fixed ``capabilities()``."""

    def __init__(self, caps: AIToolCapabilities) -> None:
        self._caps = caps

    def capabilities(self) -> AIToolCapabilities:
        return self._caps


class _VersionSequence:
    """Return probe results in order across successive calls (before, after, …)."""

    def __init__(self, values: Sequence[tuple[int, int, int] | None]) -> None:
        self.values = list(values)
        self.calls = 0

    def __call__(self, _binary: str) -> tuple[int, int, int] | None:
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


def _completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _gui_caps_with_command() -> AIToolCapabilities:
    # A GUI adapter that *accidentally* declares an update_command — the
    # structural TERMINAL invariant must still exclude/reject it.
    return AIToolCapabilities(
        tool_id=AIToolID.VSCODE,
        display_name="VS Code",
        binary="code",
        tool_type=AIToolType.GUI,
        update_command=("code", "update"),
    )


def _pin_installed(monkeypatch: pytest.MonkeyPatch, ids: Iterable[AIToolID]) -> None:
    monkeypatch.setattr(AbstractAITool, "detect_installed", classmethod(lambda _cls: list(ids)))


class TestUpdatableTools:
    def test_filters_installed_command_and_terminal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # vscode is installed but GUI/no-command → excluded; claude and
        # antigravity-cli are terminal with a command → included.
        _pin_installed(monkeypatch, [AIToolID.VSCODE, AIToolID.CLAUDE, AIToolID.ANTIGRAVITY_CLI])
        assert updatable_tools() == [AIToolID.ANTIGRAVITY_CLI, AIToolID.CLAUDE]

    def test_uninstalled_tools_excluded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _pin_installed(monkeypatch, [AIToolID.CLAUDE])
        assert updatable_tools() == [AIToolID.CLAUDE]

    def test_deterministic_registry_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The result order follows the registry, not detect_installed()'s order.
        _pin_installed(monkeypatch, [AIToolID.CURSOR, AIToolID.CLAUDE, AIToolID.CODEX])
        assert updatable_tools() == [AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.CURSOR]

    def test_gui_defense_non_terminal_with_command_excluded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        caps = _gui_caps_with_command()
        monkeypatch.setattr(
            AbstractAITool, "available_tools", classmethod(lambda _cls: [AIToolID.VSCODE])
        )
        _pin_installed(monkeypatch, [AIToolID.VSCODE])
        monkeypatch.setattr(
            AbstractAITool, "get", classmethod(lambda _cls, _tid: _FakeAdapter(caps))
        )
        assert updatable_tools() == []


class TestProbeVersion:
    def test_formats_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tool_update, "detect_binary_version", lambda _b: (2, 1, 7))
        assert probe_version("claude") == "2.1.7"

    def test_unknown_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tool_update, "detect_binary_version", lambda _b: None)
        assert probe_version("nope") is None


class TestRunUpdateSuccess:
    def test_success_with_before_after_versions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            tool_update, "detect_binary_version", _VersionSequence([(2, 1, 0), (2, 2, 0)])
        )
        monkeypatch.setattr(tool_update.process, "run", lambda *_a, **_k: _completed(0, "updated"))
        result = run_update(AIToolID.CLAUDE)
        assert result.success is True
        assert result.exit_code == 0
        assert result.before_version == "2.1.0"
        assert result.after_version == "2.2.0"
        assert result.unchanged is False
        assert result.updated is True
        assert result.error is None
        assert result.command == ("claude", "update")

    def test_unknown_version_does_not_fail_update(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tool_update, "detect_binary_version", lambda _b: None)
        monkeypatch.setattr(tool_update.process, "run", lambda *_a, **_k: _completed(0, "ok"))
        result = run_update(AIToolID.CLAUDE)
        assert result.success is True
        assert result.before_version is None
        assert result.after_version is None
        assert result.unchanged is False
        assert result.updated is False

    def test_unchanged_when_version_static(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tool_update, "detect_binary_version", lambda _b: (2, 1, 0))
        monkeypatch.setattr(tool_update.process, "run", lambda *_a, **_k: _completed(0, "ok"))
        result = run_update(AIToolID.CLAUDE)
        assert result.success is True
        assert result.before_version == result.after_version == "2.1.0"
        assert result.unchanged is True
        assert result.updated is False


class TestRunUpdateFailures:
    def test_nonzero_exit_populates_error_even_when_output_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tool_update, "detect_binary_version", lambda _b: None)
        monkeypatch.setattr(tool_update.process, "run", lambda *_a, **_k: _completed(3))
        result = run_update(AIToolID.CLAUDE)
        assert result.success is False
        assert result.exit_code == 3
        # No bare ✗: error is always set, even with empty updater output.
        assert result.error == "exited with code 3"
        assert result.output_tail == ""

    def test_command_error_preserves_returncode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tool_update, "detect_binary_version", lambda _b: None)

        def boom(*_a: object, **_k: object) -> object:
            raise CommandError(["claude", "update"], 127, "command not found: claude")

        monkeypatch.setattr(tool_update.process, "run", boom)
        result = run_update(AIToolID.CLAUDE)
        assert result.success is False
        assert result.exit_code == 127
        assert result.error is not None and "not found" in result.error

    def test_timeout_retains_partial_bytes_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tool_update, "detect_binary_version", lambda _b: None)

        def boom(*_a: object, **_k: object) -> object:
            raise subprocess.TimeoutExpired(
                cmd="claude", timeout=600, output=b"partial-stdout", stderr=b"partial-stderr"
            )

        monkeypatch.setattr(tool_update.process, "run", boom)
        result = run_update(AIToolID.CLAUDE)
        assert result.success is False
        assert result.exit_code is None
        assert "partial-stdout" in result.output_tail
        assert "partial-stderr" in result.output_tail
        assert result.error is not None and "timed out" in result.error

    def test_timeout_with_none_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tool_update, "detect_binary_version", lambda _b: None)

        def boom(*_a: object, **_k: object) -> object:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=600)

        monkeypatch.setattr(tool_update.process, "run", boom)
        result = run_update(AIToolID.CLAUDE)
        assert result.success is False
        assert result.exit_code is None
        assert result.output_tail == ""
        assert result.error is not None and "timed out" in result.error

    def test_oserror_is_recorded_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tool_update, "detect_binary_version", lambda _b: None)

        def boom(*_a: object, **_k: object) -> object:
            raise PermissionError("permission denied")

        monkeypatch.setattr(tool_update.process, "run", boom)
        result = run_update(AIToolID.CLAUDE)
        assert result.success is False
        assert result.exit_code is None
        assert result.error is not None and "permission denied" in result.error


class TestRunUpdateGuards:
    def test_rejects_non_terminal_with_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        caps = _gui_caps_with_command()
        monkeypatch.setattr(
            AbstractAITool, "get", classmethod(lambda _cls, _tid: _FakeAdapter(caps))
        )
        with pytest.raises(ValueError, match="not updatable"):
            run_update(AIToolID.VSCODE)

    def test_rejects_none_command_even_for_terminal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        caps = AIToolCapabilities(
            tool_id=AIToolID.CLAUDE,
            display_name="Claude Code",
            binary="claude",
            tool_type=AIToolType.TERMINAL,
            update_command=None,
        )
        monkeypatch.setattr(
            AbstractAITool, "get", classmethod(lambda _cls, _tid: _FakeAdapter(caps))
        )
        with pytest.raises(ValueError, match="not updatable"):
            run_update(AIToolID.CLAUDE)
