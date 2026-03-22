"""Unit tests for shared live-lane helpers."""

from __future__ import annotations

import pytest

from tests.live_support import is_prerequisite_failure, parse_selected_tools

_TOOLS = ("claude", "copilot", "gemini", "codex", "cursor", "opencode")


def test_parse_selected_tools_uses_fallback_when_env_missing() -> None:
    assert parse_selected_tools(None, _TOOLS, fallback=("claude", "codex")) == {"claude", "codex"}


def test_parse_selected_tools_parses_explicit_env_value() -> None:
    assert parse_selected_tools("claude, codex", _TOOLS) == {"claude", "codex"}


def test_parse_selected_tools_rejects_unknown_tool() -> None:
    with pytest.raises(ValueError, match="Unknown tool selection"):
        parse_selected_tools("claude, madeup", _TOOLS)


def test_is_prerequisite_failure_detects_auth_and_workspace_prompts() -> None:
    assert is_prerequisite_failure("Authentication required. Please log in.") is True
    assert is_prerequisite_failure("Please trust this workspace before continuing.") is True


def test_is_prerequisite_failure_does_not_match_unrelated_auth_substrings() -> None:
    assert is_prerequisite_failure("The word author appears here.") is False
    assert is_prerequisite_failure("This output mentions authority, not login.") is False
