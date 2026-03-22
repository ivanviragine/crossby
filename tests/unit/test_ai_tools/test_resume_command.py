"""Focused tests for adapter resume-command behavior."""

from __future__ import annotations

import pytest

from crossby.ai_tools.base import AbstractAITool
from crossby.models.ai import AIToolID


@pytest.mark.parametrize(
    ("tool_id", "session_id", "expected"),
    [
        (AIToolID.CLAUDE, "abc-123", ["claude", "--resume", "abc-123"]),
        (AIToolID.COPILOT, "sess-456", ["copilot", "--resume=sess-456"]),
        (AIToolID.CODEX, "xyz-789", ["codex", "resume", "xyz-789"]),
        (AIToolID.OPENCODE, "oc-111", ["opencode", "-s", "oc-111"]),
        (AIToolID.GEMINI, "sess-1", ["gemini", "--resume", "sess-1"]),
    ],
)
def test_supported_adapters_build_resume_command(
    tool_id: AIToolID,
    session_id: str,
    expected: list[str],
) -> None:
    adapter = AbstractAITool.get(tool_id)
    assert adapter.build_resume_command(session_id) == expected


@pytest.mark.parametrize("tool_id", [AIToolID.VSCODE, AIToolID.CURSOR, AIToolID.ANTIGRAVITY])
def test_unsupported_adapters_return_none_for_resume(tool_id: AIToolID) -> None:
    adapter = AbstractAITool.get(tool_id)
    assert adapter.build_resume_command("any-id") is None
