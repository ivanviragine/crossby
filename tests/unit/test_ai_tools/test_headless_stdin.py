"""Tests for ``AbstractAITool.headless_prompt_stdin_args`` and its overrides.

The summarizer selects stdin vs argv delivery from this method, so the default
(``None`` — argv path) and the two shipped overrides (Claude, Codex) are pinned
here. This is separate from the managed-session transports: OpenCode's verified
``run`` stdin wire is assembled in its headless adapter, not by the generic
interactive-launch builder used by the summarizer.
"""

from __future__ import annotations

import pytest

from crossby.ai_tools.antigravity_cli import AntigravityCLIAdapter
from crossby.ai_tools.claude import ClaudeAdapter
from crossby.ai_tools.codex import CodexAdapter
from crossby.ai_tools.copilot import CopilotAdapter
from crossby.ai_tools.cursor import CursorAdapter
from crossby.ai_tools.opencode import OpenCodeAdapter


def test_claude_returns_print() -> None:
    assert ClaudeAdapter().headless_prompt_stdin_args() == ["--print"]


def test_codex_returns_exec() -> None:
    assert CodexAdapter().headless_prompt_stdin_args() == ["exec"]


@pytest.mark.parametrize(
    "adapter_cls",
    [CopilotAdapter, CursorAdapter, OpenCodeAdapter, AntigravityCLIAdapter],
)
def test_tools_without_a_summarizer_stdin_contract_default_to_none(adapter_cls: type) -> None:
    """The legacy summarizer path keeps using argv unless it has its own contract."""
    assert adapter_cls().headless_prompt_stdin_args() is None
