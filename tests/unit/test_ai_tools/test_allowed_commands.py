"""Focused tests for allowed-command propagation through adapters."""

from __future__ import annotations

import pytest

from crossby.ai_tools.claude import ClaudeAdapter
from crossby.ai_tools.codex import CodexAdapter
from crossby.ai_tools.copilot import CopilotAdapter
from crossby.ai_tools.gemini import GeminiAdapter


@pytest.mark.parametrize(
    ("adapter", "expected"),
    [
        (
            ClaudeAdapter(),
            ["--allowedTools", "Bash(crossby:*)", "Bash(./scripts/check.sh:*)"],
        ),
        (
            CopilotAdapter(),
            [
                "--allow-tool",
                "shell(crossby:*)",
                "--allow-tool",
                "shell(./scripts/check.sh:*)",
            ],
        ),
        (
            GeminiAdapter(),
            [
                "--allowed-tools",
                "shell(crossby:*)",
                "--allowed-tools",
                "shell(./scripts/check.sh:*)",
            ],
        ),
    ],
    ids=["claude", "copilot", "gemini"],
)
def test_supported_adapters_include_allowed_command_flags(adapter, expected: list[str]) -> None:
    cmd = adapter.build_launch_command(
        allowed_commands=["crossby:*", "./scripts/check.sh:*"],
    )
    assert expected == cmd[1:]


def test_unsupported_adapters_omit_allowed_command_flags() -> None:
    cmd = CodexAdapter().build_launch_command(
        allowed_commands=["crossby:*", "./scripts/check.sh:*"],
    )
    assert "crossby:*" not in " ".join(cmd)
    assert "shell(" not in " ".join(cmd)
    assert "Bash(" not in " ".join(cmd)
