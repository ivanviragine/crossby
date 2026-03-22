"""Tests for prompt delivery clipboard fallback behavior."""

from __future__ import annotations

from unittest.mock import patch

import crossby.ai_tools  # noqa: F401 - register adapters
from crossby.ai_tools import AbstractAITool
from crossby.services.prompt_delivery import deliver_prompt_if_needed


class TestDeliverPromptIfNeeded:
    def test_noop_for_tools_with_initial_message_support(self) -> None:
        adapter = AbstractAITool.get("claude")
        with patch("crossby.services.prompt_delivery.copy_to_clipboard") as mock_clip:
            deliver_prompt_if_needed(adapter, "test prompt")
        mock_clip.assert_not_called()

    def test_clipboard_success_for_unsupported_tool(self) -> None:
        adapter = AbstractAITool.get("vscode")
        with (
            patch("crossby.services.prompt_delivery.copy_to_clipboard", return_value=True),
            patch("crossby.services.prompt_delivery.console") as mock_console,
        ):
            deliver_prompt_if_needed(adapter, "test prompt")
        mock_console.hint.assert_called_once()
        assert "clipboard" in mock_console.hint.call_args.args[0].lower()
        assert "VS Code" in mock_console.hint.call_args.args[0]

    def test_clipboard_failure_shows_prompt_panel(self) -> None:
        adapter = AbstractAITool.get("antigravity")
        with (
            patch("crossby.services.prompt_delivery.copy_to_clipboard", return_value=False),
            patch("crossby.services.prompt_delivery.console") as mock_console,
        ):
            deliver_prompt_if_needed(adapter, "my full prompt text")
        mock_console.warn.assert_called_once()
        mock_console.panel.assert_called_once_with("my full prompt text", title="Prompt")

    def test_all_terminal_cli_tools_are_noop(self) -> None:
        for tool_id in ("claude", "copilot", "gemini", "codex", "cursor", "opencode"):
            adapter = AbstractAITool.get(tool_id)
            with patch("crossby.services.prompt_delivery.copy_to_clipboard") as mock_clip:
                deliver_prompt_if_needed(adapter, "test")
            mock_clip.assert_not_called()
