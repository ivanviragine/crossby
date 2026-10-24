"""GitHub Copilot CLI session reader.

Each Copilot session is a directory under ``~/.copilot/session-state/<uuid>/``
containing ``events.jsonl`` (the event stream) and ``workspace.yaml``
(session metadata including the working directory). ``workspace.yaml``
carries the cwd / branch / creation timestamp, so we use it for session
location; ``events.jsonl`` provides the actual turns.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import yaml

from crossby.handoff._utils import mtime_as_utc, safe_resolve, visible_text
from crossby.handoff.models import (
    ConversationTranscript,
    ConversationTurn,
    SessionRef,
)
from crossby.models.ai import AIToolID

logger = structlog.get_logger()


def _sessions_root() -> Path:
    return Path.home() / ".copilot" / "session-state"


def locate_sessions(project_path: Path) -> list[SessionRef]:
    root = _sessions_root()
    if not root.exists():
        return []
    target = project_path.resolve()

    refs: list[SessionRef] = []
    for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        workspace = session_dir / "workspace.yaml"
        events = session_dir / "events.jsonl"
        if not workspace.exists() or not events.exists():
            continue
        meta = _read_workspace(workspace)
        if meta is None:
            continue
        cwd_value = meta.get("cwd")
        cwd = Path(cwd_value) if isinstance(cwd_value, str) and cwd_value else None
        if cwd is None or safe_resolve(cwd) != target:
            continue
        started_at = _parse_timestamp(meta.get("created_at")) or mtime_as_utc(session_dir)
        session_id = str(meta.get("id") or session_dir.name)
        refs.append(
            SessionRef(
                tool_id=AIToolID.COPILOT,
                session_id=session_id,
                path=events,
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
                logger.warning("handoff.copilot.bad_jsonl_line", path=str(ref.path))
                continue
            turn = _turn_from_event(event)
            if turn is not None:
                turns.append(turn)
    return ConversationTranscript(session_ref=ref, turns=turns)


def _read_workspace(path: Path) -> dict[str, Any] | None:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("handoff.copilot.bad_workspace", path=str(path), err=str(exc))
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def _turn_from_event(event: dict[str, Any]) -> ConversationTurn | None:
    event_type = event.get("type")
    if event_type not in ("user.message", "assistant.message"):
        return None
    data = event.get("data") or {}
    if not isinstance(data, dict):
        return None
    content = visible_text(data.get("content"))
    if not content.strip():
        return None
    return ConversationTurn(
        role="user" if event_type == "user.message" else "assistant",
        content=content,
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
