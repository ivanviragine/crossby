"""Tests for ``AbstractAITool.headless_prompt_stdin_args`` and its overrides.

The summarizer selects stdin vs argv delivery from this method, so the default
(``None`` — argv path) and the two shipped overrides (Claude, Codex) are pinned
here. Undocumented-stdin adapters (Cursor, OpenCode, antigravity-cli) and
Copilot must stay on the ``None`` default until their stdin contract is verified.
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
def test_undocumented_stdin_tools_default_to_none(adapter_cls: type) -> None:
    """Tools without a verified stdin contract inherit the ``None`` default,
    keeping them on the byte-ceiling argv path this issue."""
    assert adapter_cls().headless_prompt_stdin_args() is None
