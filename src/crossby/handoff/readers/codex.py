"""Codex session reader.

Codex stores rollouts under
``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``. Every rollout starts with
a ``session_meta`` event that carries ``id``, ``timestamp``, and ``cwd``.
Subsequent events are either ``response_item`` (``message`` / ``reasoning``
/ ``function_call``) or ``event_msg`` (higher-level protocol events). We
extract user/assistant messages and function calls; everything else is
skipped.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from crossby.handoff._utils import mtime_as_utc, safe_resolve
from crossby.handoff.models import (
    ConversationTranscript,
    ConversationTurn,
    SessionRef,
    ToolCall,
)
from crossby.models.ai import AIToolID

logger = structlog.get_logger()


def _sessions_root() -> Path:
    return Path.home() / ".codex" / "sessions"


def locate_sessions(project_path: Path) -> list[SessionRef]:
    """Scan rollout files and return refs for those matching ``project_path``."""
    root = _sessions_root()
    if not root.exists():
        return []

    target = project_path.resolve()
    refs: list[SessionRef] = []
    for rollout in sorted(root.rglob("rollout-*.jsonl")):
        meta = _read_session_meta(rollout)
        if meta is None:
            continue
        cwd_value = meta.get("cwd")
        cwd = Path(cwd_value) if isinstance(cwd_value, str) and cwd_value else None
        if cwd is None or safe_resolve(cwd) != target:
            continue
        started_at = _parse_timestamp(meta.get("timestamp"))
        if started_at is None:
            started_at = mtime_as_utc(rollout)
        session_id = str(meta.get("id") or rollout.stem)
        refs.append(
            SessionRef(
                tool_id=AIToolID.CODEX,
                session_id=session_id,
                path=rollout,
                started_at=started_at,
                cwd=cwd,
            )
        )
    return refs


def read_session(ref: SessionRef) -> ConversationTranscript:
    turns: list[ConversationTurn] = []
    with ref.path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("handoff.codex.bad_jsonl_line", path=str(ref.path))
                continue
            turn = _turn_from_event(event)
            if turn is not None:
                turns.append(turn)
    return ConversationTranscript(session_ref=ref, turns=turns)


def _read_session_meta(rollout: Path) -> dict[str, Any] | None:
    try:
        with rollout.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    return None
                if event.get("type") == "session_meta":
                    payload = event.get("payload")
                    if isinstance(payload, dict):
                        return payload
                return None
    except OSError:
        return None
    return None


def _turn_from_event(event: dict[str, Any]) -> ConversationTurn | None:
    if event.get("type") != "response_item":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    timestamp = _parse_timestamp(event.get("timestamp"))
    payload_type = payload.get("type")
    if payload_type == "message":
        return _message_turn(payload, timestamp)
    if payload_type == "function_call":
        return _function_call_turn(payload, timestamp)
    if payload_type == "reasoning":
        return _reasoning_turn(payload, timestamp)
    return None


def _message_turn(payload: dict[str, Any], ts: datetime | None) -> ConversationTurn | None:
    role = str(payload.get("role", "")).lower()
    if role == "developer":
        role = "system"
    if role not in {"user", "assistant", "system", "tool"}:
        return None
    content = payload.get("content")
    text = _stringify_codex_content(content)
    if not text.strip():
        return None
    return ConversationTurn(role=role, content=text, timestamp=ts)


def _function_call_turn(payload: dict[str, Any], ts: datetime | None) -> ConversationTurn | None:
    name = str(payload.get("name") or "unknown")
    arguments_raw = payload.get("arguments")
    arguments: dict[str, Any] = {}
    if isinstance(arguments_raw, str):
        try:
            parsed = json.loads(arguments_raw)
        except json.JSONDecodeError:
            parsed = {"raw": arguments_raw}
        arguments = parsed if isinstance(parsed, dict) else {"value": parsed}
    elif isinstance(arguments_raw, dict):
        arguments = arguments_raw
    return ConversationTurn(
        role="assistant",
        content="",
        tool_calls=[ToolCall(name=name, arguments=arguments, call_id=payload.get("call_id"))],
        timestamp=ts,
    )


def _reasoning_turn(payload: dict[str, Any], ts: datetime | None) -> ConversationTurn | None:
    summaries = payload.get("summary")
    text_parts: list[str] = []
    if isinstance(summaries, list):
        for entry in summaries:
            if isinstance(entry, dict) and isinstance(entry.get("text"), str):
                text_parts.append(entry["text"])
    content = "\n".join(text_parts).strip()
    if not content:
        return None
    return ConversationTurn(
        role="assistant",
        content=f"[reasoning] {content}",
        timestamp=ts,
    )


def _stringify_codex_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                for key in ("text", "input_text", "output_text", "content"):
                    value = block.get(key)
                    if isinstance(value, str) and value:
                        parts.append(value)
                        break
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
