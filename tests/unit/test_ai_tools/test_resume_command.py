"""Tests for build_resume_command() and supports_resume capability across adapters."""

from __future__ import annotations

from pathlib import Path

from crossby.ai_tools.antigravity_cli import AntigravityCLIAdapter
from crossby.ai_tools.claude import ClaudeAdapter
from crossby.ai_tools.codex import CodexAdapter
from crossby.ai_tools.copilot import CopilotAdapter
from crossby.ai_tools.opencode import OpenCodeAdapter


class TestResumeCommandSupported:
    """Adapters that support resume should return the correct command."""

    def test_claude_resume_command(self) -> None:
        adapter = ClaudeAdapter()
        cmd = adapter.build_resume_command("abc-123")
        assert cmd == ["claude", "--resume", "abc-123"]

    def test_claude_supports_resume_capability(self) -> None:
        adapter = ClaudeAdapter()
        assert adapter.capabilities().supports_resume is True

    def test_copilot_resume_command(self) -> None:
        adapter = CopilotAdapter()
        cmd = adapter.build_resume_command("sess-456")
        assert cmd == ["copilot", "--resume=sess-456"]

    def test_copilot_supports_resume_capability(self) -> None:
        adapter = CopilotAdapter()
        assert adapter.capabilities().supports_resume is True

    def test_codex_resume_command(self) -> None:
        adapter = CodexAdapter()
        cmd = adapter.build_resume_command("xyz-789")
        assert cmd == ["codex", "resume", "xyz-789"]

    def test_codex_supports_resume_capability(self) -> None:
        adapter = CodexAdapter()
        assert adapter.capabilities().supports_resume is True

    def test_opencode_resume_command(self) -> None:
        adapter = OpenCodeAdapter()
        cmd = adapter.build_resume_command("oc-111")
        assert cmd == ["opencode", "-s", "oc-111"]

    def test_opencode_supports_resume_capability(self) -> None:
        adapter = OpenCodeAdapter()
        assert adapter.capabilities().supports_resume is True

    def test_antigravity_cli_resume_command(self) -> None:
        adapter = AntigravityCLIAdapter()
        cmd = adapter.build_resume_command("sess-1")
        assert cmd == ["agy", "--conversation", "sess-1"]

    def test_antigravity_cli_supports_resume_capability(self) -> None:
        adapter = AntigravityCLIAdapter()
        assert adapter.capabilities().supports_resume is True


class TestResumeCommandUnsupported:
    """Adapters without resume support should return None."""

    def test_base_default_returns_none(self) -> None:
        # Use an adapter that doesn't override build_resume_command
        from crossby.ai_tools.base import AbstractAITool
        from crossby.models.ai import AIToolID

        # VSCode doesn't override build_resume_command
        adapter = AbstractAITool.get(AIToolID.VSCODE)
        assert adapter.build_resume_command("any-id") is None

    def test_vscode_does_not_support_resume(self) -> None:
        from crossby.ai_tools.base import AbstractAITool
        from crossby.models.ai import AIToolID

        adapter = AbstractAITool.get(AIToolID.VSCODE)
        assert adapter.capabilities().supports_resume is False

    def test_cursor_does_not_support_resume(self) -> None:
        from crossby.ai_tools.base import AbstractAITool
        from crossby.models.ai import AIToolID

        adapter = AbstractAITool.get(AIToolID.CURSOR)
        assert adapter.capabilities().supports_resume is False


class TestResumeContextNoTypeError:
    """Every resume override accepts the new keyword-only sandbox context.

    The context threads through polymorphic dispatch; a missing keyword param on
    any override would raise TypeError at the call site. Non-Codex overrides
    ignore it and stay byte-identical.
    """

    def test_all_overrides_accept_context(self, tmp_path: Path) -> None:
        from crossby.ai_tools.base import AbstractAITool

        expected = {
            "claude": ["claude", "--resume", "sid"],
            "copilot": ["copilot", "--resume=sid"],
            "opencode": ["opencode", "-s", "sid"],
            "antigravity-cli": ["agy", "--conversation", "sid"],
        }
        for tool, want in expected.items():
            adapter = AbstractAITool.get(tool)
            # Passing working_dir (even a real worktree) + network must not raise
            # and must not change the non-Codex command.
            cmd = adapter.build_resume_command(
                "sid", working_dir=tmp_path, network_access=True
            )
            assert cmd == want

    def test_base_default_accepts_context(self) -> None:
        from crossby.ai_tools.base import AbstractAITool
        from crossby.models.ai import AIToolID

        adapter = AbstractAITool.get(AIToolID.VSCODE)
        assert adapter.build_resume_command("x", working_dir=None, network_access=True) is None


class TestGuiLaunchAcceptsNetwork:
    """GUI launch() overrides accept and ignore network_access (no TypeError)."""

    def test_gui_launch_ignores_network(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from crossby.ai_tools.base import AbstractAITool
        from crossby.models.ai import AIToolID

        for tool in (AIToolID.VSCODE, AIToolID.ANTIGRAVITY):
            adapter = AbstractAITool.get(tool)
            module = type(adapter).__module__
            with patch(f"{module}.run_with_transcript", return_value=0) as mock_run:
                rc = adapter.launch(working_dir=tmp_path, network_access=True)
            assert rc == 0
            mock_run.assert_called_once()
