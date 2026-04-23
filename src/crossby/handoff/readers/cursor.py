"""Cursor session reader.

Cursor stores per-project files under ``~/.cursor/projects/<encoded>/``.
Path encoding strips the leading ``/`` and replaces remaining ``/`` with
``-`` (see ``ai_tools.cursor.preserve_session_data``). JSON files in that
directory that match a chat-like shape (``{"messages": [...]}``,
``{"chats": [{"messages": [...]}]}``, or a top-level list of messages)
are parsed; other files are ignored.

Cursor's on-disk chat format is not publicly documented and varies across
releases. The reader accepts the known shapes defensively; anything it
cannot recognize is skipped rather than crashing the handoff.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from crossby.handoff._utils import mtime_as_utc
from crossby.handoff.models import (
    ConversationTranscript,
    ConversationTurn,
    SessionRef,
)
from crossby.models.ai import AIToolID

logger = structlog.get_logger()


def _projects_root() -> Path:
    return Path.home() / ".cursor" / "projects"


def _encode(path: Path) -> str:
    return str(path.resolve()).lstrip("/").replace("/", "-")


def locate_sessions(project_path: Path) -> list[SessionRef]:
    encoded_dir = _projects_root() / _encode(project_path)
    if not encoded_dir.exists() or not encoded_dir.is_dir():
        return []

    refs: list[SessionRef] = []
    for json_file in sorted(encoded_dir.rglob("*.json")):
        ref = _session_ref_from_file(json_file, project_path.resolve())
        if ref is not None:
            refs.append(ref)
    return refs


def read_session(ref: SessionRef) -> ConversationTranscript:
    try:
        data = json.loads(ref.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("handoff.cursor.unreadable", path=str(ref.path), err=str(exc))
        return ConversationTranscript(session_ref=ref, turns=[])

    messages = _extract_messages(data)
    turns: list[ConversationTurn] = []
    for msg in messages:
        turn = _turn_from_message(msg)
        if turn is not None:
            turns.append(turn)
    return ConversationTranscript(session_ref=ref, turns=turns)


def _session_ref_from_file(path: Path, cwd: Path) -> SessionRef | None:
    try:
        started_at = mtime_as_utc(path)
    except OSError:
        return None
    return SessionRef(
        tool_id=AIToolID.CURSOR,
        session_id=path.stem,
        path=path,
        started_at=started_at,
        cwd=cwd,
    )


def _extract_messages(data: Any) -> list[dict[str, Any]]:
    """Tolerant message extractor — handles a few known Cursor shapes."""
    if isinstance(data, dict):
        if isinstance(data.get("messages"), list):
            return [m for m in data["messages"] if isinstance(m, dict)]
        if isinstance(data.get("chat"), dict) and isinstance(
            data["chat"].get("messages"), list
        ):
            return [m for m in data["chat"]["messages"] if isinstance(m, dict)]
        if isinstance(data.get("chats"), list):
            out: list[dict[str, Any]] = []
            for chat in data["chats"]:
                if isinstance(chat, dict) and isinstance(chat.get("messages"), list):
                    out.extend(m for m in chat["messages"] if isinstance(m, dict))
            return out
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict) and ("role" in m or "content" in m)]
    return []


def _turn_from_message(msg: dict[str, Any]) -> ConversationTurn | None:
    role = str(msg.get("role", "")).lower()
    if role not in {"user", "assistant", "tool", "system"}:
        return None
    content = msg.get("content") or msg.get("text") or ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
            elif isinstance(block, str):
                parts.append(block)
        content = "\n".join(parts)
    if not isinstance(content, str) or not content.strip():
        return None
    return ConversationTurn(
        role=role,
        content=content,
        timestamp=_parse_timestamp(msg.get("timestamp") or msg.get("created_at")),
    )


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, ValueError, OverflowError):
            return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None
