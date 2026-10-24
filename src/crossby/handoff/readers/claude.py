"""Claude Code session reader.

Claude writes one JSONL file per session in
``~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl``. The filesystem
path is encoded by replacing ``/`` and ``.`` with ``-`` (see
``ai_tools.claude._encode_claude_path``). Each line is a JSON event; user /
assistant events carry the conversation turns.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from crossby.handoff._utils import mtime_as_utc, visible_text
from crossby.handoff.models import (
    ConversationTranscript,
    ConversationTurn,
    SessionRef,
)
from crossby.models.ai import AIToolID

logger = structlog.get_logger()


def _projects_root() -> Path:
    return Path.home() / ".claude" / "projects"


def _encode(path: Path) -> str:
    return str(path).replace("/", "-").replace(".", "-")


def locate_sessions(project_path: Path) -> list[SessionRef]:
    """Return every Claude session JSONL file associated with ``project_path``."""
    encoded_dir = _projects_root() / _encode(project_path.resolve())
    if not encoded_dir.exists() or not encoded_dir.is_dir():
        return []

    refs: list[SessionRef] = []
    for jsonl in sorted(encoded_dir.glob("*.jsonl")):
        ref = _session_ref_from_file(jsonl, project_path.resolve())
        if ref is not None:
            refs.append(ref)
    return refs


def read_session(ref: SessionRef) -> ConversationTranscript:
    """Parse Claude's JSONL event stream into a transcript."""
    turns: list[ConversationTurn] = []
    with ref.path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("handoff.claude.bad_jsonl_line", path=str(ref.path))
                continue
            turn = _turn_from_event(event)
            if turn is not None:
                turns.append(turn)
    return ConversationTranscript(session_ref=ref, turns=turns)


def _session_ref_from_file(path: Path, cwd: Path) -> SessionRef | None:
    session_id = path.stem
    started_at = _first_event_timestamp(path)
    if started_at is None:
        started_at = mtime_as_utc(path)
    return SessionRef(
        tool_id=AIToolID.CLAUDE,
        session_id=session_id,
        path=path,
        started_at=started_at,
        cwd=cwd,
    )


def _first_event_timestamp(path: Path) -> datetime | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = event.get("timestamp")
                if isinstance(ts, str):
                    parsed = _parse_timestamp(ts)
                    if parsed is not None:
                        return parsed
    except OSError:
        return None
    return None


def _turn_from_event(event: dict[str, Any]) -> ConversationTurn | None:
    event_type = event.get("type")
    if event_type not in ("user", "assistant") or event.get("isMeta"):
        return None
    message = event.get("message") or {}
    if not isinstance(message, dict):
        return None
    text = visible_text(message.get("content"))
    if not text.strip():
        return None
    return ConversationTurn(
        role=event_type,
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
