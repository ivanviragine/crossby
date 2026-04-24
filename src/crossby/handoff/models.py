"""Pydantic models shared by every handoff reader, summarizer, and writer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from crossby.models.ai import AIToolID

Role = Literal["user", "assistant", "tool", "system"]


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class ToolCall(BaseModel):
    """A single tool/function call recorded in a transcript turn."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class SessionRef(BaseModel):
    """Pointer to an on-disk session for a specific tool.

    ``cwd`` is None when the source format does not record it.
    """

    tool_id: AIToolID
    session_id: str
    path: Path
    started_at: datetime
    cwd: Path | None = None


class ConversationTurn(BaseModel):
    """A single message/event in a conversation transcript."""

    role: Role
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    file_refs: list[Path] = Field(default_factory=list)
    timestamp: datetime | None = None


class ConversationTranscript(BaseModel):
    """The full transcript returned by every session reader."""

    session_ref: SessionRef
    turns: list[ConversationTurn] = Field(default_factory=list)
    truncated: bool = False


class HandoffDocument(BaseModel):
    """The structured summary written to ``HANDOFF-<timestamp>.md``."""

    source_tool: AIToolID
    target_tool: AIToolID
    session_ref: SessionRef
    current_task: str = ""
    key_decisions: list[str] = Field(default_factory=list)
    modified_files: list[Path] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    critical_context: str = ""
    created_at: datetime = Field(default_factory=_utc_now)


class RawHandoff(BaseModel):
    """An unparsed handoff — the summarizer's raw output under a custom prompt."""

    source_tool: AIToolID
    target_tool: AIToolID
    session_ref: SessionRef
    body: str
    prompt_source: str
    created_at: datetime = Field(default_factory=_utc_now)
