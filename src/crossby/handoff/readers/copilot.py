"""GitHub Copilot CLI session reader.

Each Copilot session is a directory under ``~/.copilot/session-state/<uuid>/``
containing ``events.jsonl`` (the event stream) and ``workspace.yaml``
(session metadata including the working directory). ``workspace.yaml``
carries the cwd / branch / creation timestamp, so we use it for session
location; ``events.jsonl`` provides the actual turns.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
import yaml

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
    data = event.get("data") or {}
    if not isinstance(data, dict):
        return None
    timestamp = _parse_timestamp(event.get("timestamp"))
    if event_type == "user.message":
        content = _coerce_text(data.get("content"))
        if not content:
            return None
        return ConversationTurn(role="user", content=content, timestamp=timestamp)
    if event_type == "assistant.message":
        content = _coerce_text(data.get("content"))
        tool_calls = _extract_tool_requests(data.get("toolRequests"))
        file_refs = _extract_file_refs(data.get("toolRequests"))
        if not content and not tool_calls:
            return None
        return ConversationTurn(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            file_refs=file_refs,
            timestamp=timestamp,
        )
    if event_type == "tool.result":
        content = _coerce_text(data.get("output") or data.get("result"))
        if not content:
            return None
        return ConversationTurn(role="tool", content=content, timestamp=timestamp)
    return None


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _extract_tool_requests(value: Any) -> list[ToolCall]:
    if not isinstance(value, list):
        return []
    out: list[ToolCall] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "unknown")
        args = entry.get("arguments") or {}
        if isinstance(args, dict):
            out.append(ToolCall(name=name, arguments=args, call_id=entry.get("toolCallId")))
    return out


def _extract_file_refs(value: Any) -> list[Path]:
    if not isinstance(value, list):
        return []
    refs: list[Path] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        args = entry.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        for key in ("path", "file_path", "filePath"):
            candidate = args.get(key)
            if isinstance(candidate, str) and candidate:
                refs.append(Path(candidate))
    return refs


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


