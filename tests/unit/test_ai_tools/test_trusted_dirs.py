"""Probe tests for trusted_dirs in build_launch_command() across adapters."""

from __future__ import annotations

from crossby.ai_tools.claude import ClaudeAdapter
from crossby.ai_tools.codex import CodexAdapter
from crossby.ai_tools.gemini import GeminiAdapter


class TestTrustedDirsLaunchCommand:
    """build_launch_command() includes the correct per-tool trusted-dir flags."""

    def test_claude_single_trusted_dir(self) -> None:
        cmd = ClaudeAdapter().build_launch_command(trusted_dirs=["/tmp/plan"])
        assert "--add-dir" in cmd
        assert "/tmp/plan" in cmd
        assert cmd[cmd.index("/tmp/plan") - 1] == "--add-dir"

    def test_claude_multiple_trusted_dirs(self) -> None:
        cmd = ClaudeAdapter().build_launch_command(trusted_dirs=["/a", "/b"])
        pairs = [(cmd[i], cmd[i + 1]) for i in range(len(cmd) - 1) if cmd[i] == "--add-dir"]
        paths = [p for _, p in pairs]
        assert "/a" in paths
        assert "/b" in paths

    def test_codex_trusted_dir_prepends_sandbox(self) -> None:
        cmd = CodexAdapter().build_launch_command(trusted_dirs=["/tmp/plan"])
        assert "--sandbox" in cmd
        assert "workspace-write" in cmd
        assert "--add-dir" in cmd
        assert "/tmp/plan" in cmd
        # sandbox flag must appear before --add-dir
        assert cmd.index("--sandbox") < cmd.index("--add-dir")

    def test_gemini_trusted_dir_uses_include_directories(self) -> None:
        cmd = GeminiAdapter().build_launch_command(trusted_dirs=["/tmp/plan"])
        assert "--include-directories" in cmd
        assert "/tmp/plan" in cmd
        assert cmd[cmd.index("/tmp/plan") - 1] == "--include-directories"
