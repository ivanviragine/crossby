"""Tests for HandoffSummarizer parsing + behavior."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crossby.ai_tools.base import AbstractAITool
from crossby.handoff.models import (
    ConversationTranscript,
    ConversationTurn,
    SessionRef,
)
from crossby.handoff.summarizer import (
    HandoffSummarizer,
    SummarizerParseError,
    SummarizerToolNotInstalled,
    parse_markdown_sections,
)
from crossby.models.ai import AIToolID


def _ref() -> SessionRef:
    return SessionRef(
        tool_id=AIToolID.CLAUDE,
        session_id="s",
        path=Path("/tmp/s.jsonl"),
        started_at=datetime(2026, 3, 1),
        cwd=Path("/Users/tester/proj"),
    )


def _transcript(n_turns: int = 2) -> ConversationTranscript:
    turns = [
        ConversationTurn(role="user", content=f"turn-{i}")
        for i in range(n_turns)
    ]
    return ConversationTranscript(session_ref=_ref(), turns=turns)


def _make_summarizer_tool(
    *, json_schema_args: list[str] | None = None, tool_id: AIToolID = AIToolID.CLAUDE
) -> MagicMock:
    tool = MagicMock(spec=AbstractAITool)
    tool.TOOL_ID = tool_id
    tool.structured_output_args = MagicMock(return_value=json_schema_args or [])
    tool.build_launch_command = MagicMock(return_value=["fake", "--prompt", "X"])
    return tool


def test_ensure_installed_raises_when_tool_missing() -> None:
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool)
    with patch.object(AbstractAITool, "detect_installed", return_value=[]):
        with pytest.raises(SummarizerToolNotInstalled):
            summarizer.ensure_installed()


def test_summarize_parses_json_payload() -> None:
    tool = _make_summarizer_tool(json_schema_args=["--output-format", "json"])
    summarizer = HandoffSummarizer(tool)
    json_stdout = (
        '{"current_task": "Refactor auth", '
        '"key_decisions": ["drop cache"], '
        '"modified_files": ["auth.py"], '
        '"blockers": [], '
        '"next_steps": ["write migration"], '
        '"critical_context": "cache is load-bearing"}'
    )
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=json_stdout, stderr="")

    with patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]):
        with patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc) as run:
            doc = summarizer.summarize(
                _transcript(), source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
            )

    assert doc.current_task == "Refactor auth"
    assert doc.key_decisions == ["drop cache"]
    assert doc.modified_files == [Path("auth.py")]
    assert doc.next_steps == ["write migration"]
    assert doc.critical_context == "cache is load-bearing"
    assert doc.source_tool == AIToolID.CLAUDE
    assert doc.target_tool == AIToolID.CODEX
    # build_launch_command must have been called with a json_schema.
    _, kwargs = tool.build_launch_command.call_args
    assert kwargs["json_schema"] is not None
    run.assert_called_once()


def test_summarize_falls_back_to_markdown_when_no_json_support() -> None:
    tool = _make_summarizer_tool(json_schema_args=[])  # no JSON flags
    summarizer = HandoffSummarizer(tool)
    markdown = (
        "## Current Task\nShip the thing.\n\n"
        "## Key Decisions\n- use stripe\n- drop cache\n\n"
        "## Modified Files\n- foo.py\n\n"
        "## Blockers\n\n"
        "## Next Steps\n- run tests\n\n"
        "## Critical Context\nDon't break billing.\n"
    )
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=markdown, stderr="")

    with patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]):
        with patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc):
            doc = summarizer.summarize(
                _transcript(), source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
            )

    assert doc.current_task == "Ship the thing."
    assert doc.key_decisions == ["use stripe", "drop cache"]
    assert doc.modified_files == [Path("foo.py")]
    assert doc.blockers == []
    assert doc.next_steps == ["run tests"]
    assert "billing" in doc.critical_context
    # JSON schema must NOT have been passed.
    _, kwargs = tool.build_launch_command.call_args
    assert kwargs["json_schema"] is None


def test_summarize_raises_when_output_is_unparseable() -> None:
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool)
    fake_proc = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="plain text with no structure", stderr=""
    )

    with patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]):
        with patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc):
            with pytest.raises(SummarizerParseError):
                summarizer.summarize(
                    _transcript(), source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
                )


def test_summarize_raises_when_tool_exits_nonzero() -> None:
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool)
    fake_proc = subprocess.CompletedProcess(
        args=[], returncode=2, stdout="", stderr="boom"
    )

    with patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]):
        with patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc):
            with pytest.raises(SummarizerParseError, match="boom"):
                summarizer.summarize(
                    _transcript(), source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
                )


def test_summarize_calls_on_truncate_when_transcript_trimmed() -> None:
    tool = _make_summarizer_tool()
    # Tiny budget forces truncation.
    summarizer = HandoffSummarizer(tool, token_budget=5)
    big_turns = [
        ConversationTurn(role="user", content="x" * 400) for _ in range(4)
    ]
    transcript = ConversationTranscript(session_ref=_ref(), turns=big_turns)
    markdown = (
        "## Current Task\nt\n## Key Decisions\n## Modified Files\n"
        "## Blockers\n## Next Steps\n## Critical Context\n"
    )
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=markdown, stderr="")

    captured: list[tuple[int, int]] = []

    def on_trunc(original: int, kept: int) -> None:
        captured.append((original, kept))

    with patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]):
        with patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc):
            summarizer.summarize(
                transcript,
                source_tool=AIToolID.CLAUDE,
                target_tool=AIToolID.CODEX,
                on_truncate=on_trunc,
            )

    assert captured
    original, kept = captured[0]
    assert original == 4
    assert kept < original


def test_parse_markdown_sections_returns_none_without_headings() -> None:
    assert parse_markdown_sections("plain text") is None


def test_parse_markdown_sections_extracts_bullets_and_text() -> None:
    text = (
        "## Current Task\nDo the work.\n"
        "## Key Decisions\n- a\n- b\n"
        "## Modified Files\n"
        "## Blockers\n* legacy dep\n"
        "## Next Steps\n- ship\n"
        "## Critical Context\nmind the cache\n"
    )
    payload = parse_markdown_sections(text)
    assert payload is not None
    assert payload["current_task"] == "Do the work."
    assert payload["key_decisions"] == ["a", "b"]
    assert payload["modified_files"] == []
    assert payload["blockers"] == ["legacy dep"]
    assert payload["next_steps"] == ["ship"]
    assert payload["critical_context"] == "mind the cache"


# Suppress unused-import linter complaints in some configs.
_ = Any
