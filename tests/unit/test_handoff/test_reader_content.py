"""Conversation-only extraction shared by all handoff readers."""

from datetime import UTC, datetime

import pytest

from crossby.handoff._utils import visible_text
from crossby.handoff.models import ConversationTranscript


def test_readers_keep_visible_conversation_in_order(
    conversation_transcript: ConversationTranscript,
) -> None:
    assert [(t.role, t.content) for t in conversation_transcript.turns] == [
        ("user", "Fix the flaky test."),
        ("assistant", "Checking the retry loop."),
        ("assistant", "The retry loop has a race."),
    ]
    for turn in conversation_transcript.turns:
        assert turn.timestamp == datetime(2026, 3, 3, 16, 46, 30, tzinfo=UTC)
        assert not turn.tool_calls
        assert not turn.file_refs
        assert turn.tool_call_id is None


@pytest.mark.parametrize(
    "content",
    [
        "[reasoning] is a literal part of this user message",
        [{"text": "[reasoning] is a literal part of this user message"}],
        [{"content": "[reasoning] is a literal part of this user message"}],
        ["[reasoning] is a literal part of this user message"],
    ],
)
def test_legacy_text_and_literal_reasoning_labels_are_preserved(content: object) -> None:
    assert visible_text(content) == "[reasoning] is a literal part of this user message"


@pytest.mark.parametrize(
    "content", [None, 123, {}, [], [None, 123, {"text": 123}, {"type": [], "text": "skip"}]]
)
def test_non_text_content_is_skipped(content: object) -> None:
    assert visible_text(content) == ""
