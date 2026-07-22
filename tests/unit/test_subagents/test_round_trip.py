"""Round-trip and cross-tool conversion integration tests."""

from __future__ import annotations

import pytest

from crossby.subagents import convert
from crossby.subagents.api import parse

CLAUDE_AGENT = """\
---
name: researcher
description: Research and summarize
tools:
  - Read
  - Grep
  - Bash
model: sonnet
---
You are a research assistant.
"""

CODEX_AGENT = """\
name = "worker"
description = "Does work"
developer_instructions = "Be concise."
model = "gpt-5"
sandbox_mode = "read-only"
"""

COPILOT_AGENT = """\
---
name: helper
description: General help
target: vscode
tools: [read, search]
---
Helper body.
"""


@pytest.mark.parametrize(
    "from_tool,source",
    [
        ("claude", CLAUDE_AGENT),
        ("codex", CODEX_AGENT),
        ("copilot", COPILOT_AGENT),
    ],
)
@pytest.mark.parametrize("to_tool", ["claude", "copilot", "cursor", "codex"])
def test_convert_does_not_raise_for_any_pair(from_tool: str, source: str, to_tool: str) -> None:
    """Every (source, target) pair should produce a parseable output."""
    result = convert(from_tool, to_tool, source)
    assert result.payload is not None
    assert result.target_tool == to_tool


def test_claude_round_trip_preserves_tools() -> None:
    """claude → claude should preserve the canonical Claude tool names exactly."""
    result = convert("claude", "claude", CLAUDE_AGENT)
    ir2, _ = parse("claude", result.payload)
    assert ir2.tools == ["read_file", "grep", "bash"]
    assert ir2.name == "researcher"
    assert ir2.model == "sonnet"


def test_codex_to_claude_recovers_body() -> None:
    """Codex's developer_instructions becomes Claude's body."""
    result = convert("codex", "claude", CODEX_AGENT)
    assert "Be concise." in result.payload
    # No tools allowlist in Codex, so none in output.
    assert "tools:" not in result.payload


def test_claude_to_codex_emits_two_artifacts() -> None:
    result = convert("claude", "codex", CLAUDE_AGENT)
    payload = result.payload
    assert hasattr(payload, "agent_toml")
    assert hasattr(payload, "config_fragment")
    assert payload.suggested_filename == "researcher.toml"


def test_cursor_drops_write_tools_with_warning() -> None:
    """Sending a write-capable Claude agent to Cursor warns and drops tools."""
    text = "---\nname: a\ndescription: D\ntools: [Edit, Write]\n---\nbody\n"
    result = convert("claude", "cursor", text)
    assert any(w.severity.value == "dropped" for w in result.warnings)


def test_extras_only_round_trip_to_same_tool() -> None:
    """Tool-specific extras should not leak into a foreign target."""
    text = "---\nname: a\ndescription: D\ncustomFoo: 7\n---\nbody\n"
    cross = convert("claude", "cursor", text)
    assert "customFoo" not in cross.payload
    same = convert("claude", "claude", text)
    assert "customFoo" in same.payload
