"""Tests for allowed_commands_args() across AI tool adapters."""

from __future__ import annotations

from crossby.ai_tools.antigravity_cli import AntigravityCLIAdapter
from crossby.ai_tools.claude import ClaudeAdapter
from crossby.ai_tools.codex import CodexAdapter
from crossby.ai_tools.copilot import CopilotAdapter
from crossby.ai_tools.cursor import CursorAdapter
from crossby.ai_tools.opencode import OpenCodeAdapter


class TestClaudeAllowedCommands:
    """Tests for ClaudeAdapter.allowed_commands_args()."""

    def test_single_pattern(self) -> None:
        adapter = ClaudeAdapter()
        result = adapter.allowed_commands_args(["crossby:*"])
        assert result == ["--allowedTools", "Bash(crossby:*)"]

    def test_multiple_patterns(self) -> None:
        adapter = ClaudeAdapter()
        result = adapter.allowed_commands_args(["crossby:*", "./scripts/check.sh:*"])
        assert result == [
            "--allowedTools",
            "Bash(crossby:*)",
            "Bash(./scripts/check.sh:*)",
        ]

    def test_pattern_without_args(self) -> None:
        adapter = ClaudeAdapter()
        result = adapter.allowed_commands_args(["./scripts/fmt.sh"])
        assert result == ["--allowedTools", "Bash(./scripts/fmt.sh)"]

    def test_empty_list(self) -> None:
        adapter = ClaudeAdapter()
        result = adapter.allowed_commands_args([])
        assert result == []


class TestCopilotAllowedCommands:
    """Tests for CopilotAdapter.allowed_commands_args()."""

    def test_single_pattern(self) -> None:
        adapter = CopilotAdapter()
        result = adapter.allowed_commands_args(["crossby:*"])
        assert result == ["--allow-tool", "shell(crossby:*)"]

    def test_multiple_patterns(self) -> None:
        adapter = CopilotAdapter()
        result = adapter.allowed_commands_args(["crossby:*", "./scripts/check.sh:*"])
        assert result == [
            "--allow-tool",
            "shell(crossby:*)",
            "--allow-tool",
            "shell(./scripts/check.sh:*)",
        ]

    def test_pattern_without_args(self) -> None:
        adapter = CopilotAdapter()
        result = adapter.allowed_commands_args(["./scripts/fmt.sh"])
        assert result == ["--allow-tool", "shell(./scripts/fmt.sh)"]


class TestAntigravityCLIAllowedCommands:
    """Tests for AntigravityCLIAdapter — returns empty (no support)."""

    def test_returns_empty(self) -> None:
        adapter = AntigravityCLIAdapter()
        result = adapter.allowed_commands_args(["crossby:*"])
        assert result == []

    def test_returns_empty_with_multiple_patterns(self) -> None:
        adapter = AntigravityCLIAdapter()
        result = adapter.allowed_commands_args(["crossby:*", "./scripts/check.sh:*"])
        assert result == []


class TestCursorAllowedCommands:
    """Tests for CursorAdapter — returns empty (config-file permissions)."""

    def test_returns_empty(self) -> None:
        adapter = CursorAdapter()
        result = adapter.allowed_commands_args(["crossby:*"])
        assert result == []

    def test_returns_empty_with_multiple_patterns(self) -> None:
        adapter = CursorAdapter()
        result = adapter.allowed_commands_args(["crossby:*", "./scripts/check.sh:*"])
        assert result == []


class TestCodexAllowedCommands:
    """Tests for CodexAdapter — returns empty (sandbox mode)."""

    def test_returns_empty(self) -> None:
        adapter = CodexAdapter()
        result = adapter.allowed_commands_args(["crossby:*"])
        assert result == []


class TestOpenCodeAllowedCommands:
    """Tests for OpenCodeAdapter — returns empty (no support)."""

    def test_returns_empty(self) -> None:
        adapter = OpenCodeAdapter()
        result = adapter.allowed_commands_args(["crossby:*"])
        assert result == []


class TestBuildLaunchCommandWithAllowedCommands:
    """Tests that build_launch_command passes allowed_commands through."""

    def test_claude_includes_allowed_commands_in_command(self) -> None:
        adapter = ClaudeAdapter()
        cmd = adapter.build_launch_command(
            allowed_commands=["crossby:*", "./scripts/test.sh:*"],
        )
        assert "--allowedTools" in cmd
        assert "Bash(crossby:*)" in cmd
        assert "Bash(./scripts/test.sh:*)" in cmd

    def test_no_allowed_commands_omits_flags(self) -> None:
        adapter = ClaudeAdapter()
        cmd = adapter.build_launch_command()
        assert "--allowedTools" not in cmd

    def test_copilot_includes_allow_tool_flags(self) -> None:
        adapter = CopilotAdapter()
        cmd = adapter.build_launch_command(
            allowed_commands=["crossby:*"],
        )
        assert "--allow-tool" in cmd
        assert "shell(crossby:*)" in cmd
