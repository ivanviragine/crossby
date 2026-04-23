"""Tests for transcript truncation helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from crossby.handoff.models import (
    ConversationTranscript,
    ConversationTurn,
    SessionRef,
    ToolCall,
)
from crossby.handoff.truncate import approx_tokens, truncate_transcript, turn_tokens
from crossby.models.ai import AIToolID


def _ref() -> SessionRef:
    return SessionRef(
        tool_id=AIToolID.CLAUDE,
        session_id="s",
        path=Path("/tmp/s.jsonl"),
        started_at=datetime(2026, 3, 1),
        cwd=Path("/Users/tester/proj"),
    )


def _turn(role: str, content: str, tool_calls: list[ToolCall] | None = None) -> ConversationTurn:
    return ConversationTurn(role=role, content=content, tool_calls=tool_calls or [])  # type: ignore[arg-type]


def test_approx_tokens_floor_is_one() -> None:
    assert approx_tokens("") == 1
    assert approx_tokens("abcd") == 1
    assert approx_tokens("a" * 40) == 10


def test_turn_tokens_counts_tool_calls() -> None:
    turn = _turn(
        "assistant",
        "hi",
        tool_calls=[ToolCall(name="shell", arguments={"cmd": "ls -la"})],
    )
    # content is 2 chars → 1 token; tool call adds name + kv costs
    assert turn_tokens(turn) > approx_tokens("hi")


def test_truncate_keeps_newest_turns_and_marks_truncated() -> None:
    # Each content 400 chars → ~100 tokens. Budget of 250 should keep only last 2.
    turns = [_turn("user", "x" * 400) for _ in range(5)]
    transcript = ConversationTranscript(session_ref=_ref(), turns=turns)

    result = truncate_transcript(transcript, token_budget=250)

    assert result.truncated is True
    assert len(result.turns) == 2
    assert result.turns == turns[-2:]


def test_truncate_does_nothing_when_budget_large_enough() -> None:
    turns = [_turn("user", "hello"), _turn("assistant", "world")]
    transcript = ConversationTranscript(session_ref=_ref(), turns=turns)

    result = truncate_transcript(transcript, token_budget=1_000)

    assert result.truncated is False
    assert result.turns == turns


def test_truncate_always_keeps_at_least_one_turn() -> None:
    # A single oversized turn still survives — "and kept" guard in loop.
    huge = _turn("user", "y" * 10_000)
    transcript = ConversationTranscript(session_ref=_ref(), turns=[huge])

    result = truncate_transcript(transcript, token_budget=10)

    assert len(result.turns) == 1


def test_truncate_preserves_prior_truncated_flag() -> None:
    turns = [_turn("user", "short")]
    transcript = ConversationTranscript(session_ref=_ref(), turns=turns, truncated=True)

    result = truncate_transcript(transcript, token_budget=1_000)

    assert result.truncated is True


@pytest.mark.parametrize("bad_budget", [0, -1, -500])
def test_truncate_rejects_non_positive_budget(bad_budget: int) -> None:
    transcript = ConversationTranscript(session_ref=_ref(), turns=[_turn("user", "hi")])

    with pytest.raises(ValueError, match="token_budget"):
        truncate_transcript(transcript, token_budget=bad_budget)
