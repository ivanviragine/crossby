"""Probe tests for trusted_dirs in build_launch_command() across adapters."""

from __future__ import annotations

from crossby.ai_tools.antigravity_cli import AntigravityCLIAdapter
from crossby.ai_tools.claude import ClaudeAdapter
from crossby.ai_tools.codex import CodexAdapter
from crossby.ai_tools.copilot import CopilotAdapter


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

    def test_codex_accept_edits_trusted_dir_no_duplicate_sandbox(self) -> None:
        # accept-edits already sets ``-s workspace-write``; trusted-dir handling
        # must not re-emit ``--sandbox workspace-write`` (Codex/clap rejects the
        # duplicate flag).
        cmd = CodexAdapter().build_launch_command(
            accept_edits=True, trusted_dirs=["/tmp/plan"]
        )
        assert "--sandbox" not in cmd
        assert cmd.count("workspace-write") == 1
        assert "--add-dir" in cmd
        assert "/tmp/plan" in cmd

    def test_codex_auto_trusted_dir_no_duplicate_sandbox(self) -> None:
        # auto downgrades to accept-edits on Codex, which sets the sandbox; the
        # trusted-dir append must not duplicate it.
        cmd = CodexAdapter().build_launch_command(auto=True, trusted_dirs=["/tmp/plan"])
        assert "--sandbox" not in cmd
        assert cmd.count("workspace-write") == 1

    def test_codex_yolo_trusted_dir_still_sets_sandbox(self) -> None:
        # yolo (``-a never``) does not set workspace-write, so trusted dirs must
        # still emit the sandbox flag they require to take effect.
        cmd = CodexAdapter().build_launch_command(yolo=True, trusted_dirs=["/tmp/plan"])
        assert "--sandbox" in cmd
        assert "workspace-write" in cmd
        assert cmd.index("--sandbox") < cmd.index("--add-dir")

    def test_antigravity_cli_trusted_dir_uses_add_dir(self) -> None:
        cmd = AntigravityCLIAdapter().build_launch_command(trusted_dirs=["/tmp/plan"])
        assert "--add-dir" in cmd
        assert "/tmp/plan" in cmd
        assert cmd[cmd.index("/tmp/plan") - 1] == "--add-dir"

    def test_copilot_trusted_dir_uses_add_dir(self) -> None:
        cmd = CopilotAdapter().build_launch_command(trusted_dirs=["/tmp/plan"])
        assert "--add-dir" in cmd
        assert "/tmp/plan" in cmd
        assert cmd[cmd.index("/tmp/plan") - 1] == "--add-dir"
