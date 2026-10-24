"""Codex session reader.

Codex stores rollouts under
``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``. Every rollout starts with
a ``session_meta`` event that carries ``id``, ``timestamp``, and ``cwd``.
Subsequent events are either ``response_item`` (``message`` / ``reasoning``
/ ``function_call`` / ``function_call_output`` / custom-tool variants) or
``event_msg`` (higher-level protocol events). We extract only user messages and
visible assistant responses. Reasoning, tool activity, source instructions,
and non-text blocks are skipped.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from crossby.handoff._utils import mtime_as_utc, safe_resolve, visible_text
from crossby.handoff.models import (
    ConversationTranscript,
    ConversationTurn,
    SessionRef,
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
    if not isinstance(payload, dict) or payload.get("type") != "message":
        return None
    role = str(payload.get("role", "")).lower()
    if role not in {"user", "assistant"} or payload.get("channel") == "analysis":
        return None
    text = visible_text(payload.get("content"))
    if not text.strip():
        return None
    return ConversationTurn(
        role=role,
        content=text,
        timestamp=_parse_timestamp(event.get("timestamp")),
    )


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
