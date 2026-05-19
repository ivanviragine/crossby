"""Pick the most relevant session for a project from a list of SessionRefs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from crossby.handoff._utils import safe_resolve
from crossby.handoff.models import SessionRef


def pick_latest_session(refs: list[SessionRef], project_root: Path) -> SessionRef | None:
    """Return the most recent session whose ``cwd`` resolves to ``project_root``.

    Sessions without a recorded ``cwd`` are excluded — a reader cannot prove
    they belong to this project so surfacing them would be a lie.
    """
    target = project_root.resolve()
    matches = [ref for ref in refs if ref.cwd is not None and safe_resolve(ref.cwd) == target]
    if not matches:
        return None
    matches.sort(key=lambda r: r.started_at, reverse=True)
    return matches[0]


def _relative_time(dt: datetime) -> str:
    """Return a human-readable relative time string for ``dt`` (e.g. '2 minutes ago')."""
    now = datetime.now(tz=UTC)
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    delta = now - aware
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def _format_session_row(ref: SessionRef) -> str:
    """Render one picker row: '<short_id>  ·  <relative>  ·  <iso_utc>'."""
    short_id = ref.session_id[:8] if len(ref.session_id) > 8 else ref.session_id
    aware = (
        ref.started_at
        if ref.started_at.tzinfo is not None
        else ref.started_at.replace(tzinfo=UTC)
    )
    iso_utc = aware.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    relative = _relative_time(ref.started_at)
    return f"{short_id}  ·  {relative}  ·  {iso_utc}"


def prompt_for_session(refs: list[SessionRef], project_root: Path) -> SessionRef | None:
    """Interactively prompt the user to pick a session from matching refs.

    Filters to refs whose ``cwd`` resolves to ``project_root``, sorts newest-first,
    presents an arrow-key picker via ``crossby.ui.prompts.select``, and returns
    the chosen ``SessionRef``. Returns ``None`` if no sessions match.
    """
    from crossby.ui import prompts

    target = project_root.resolve()
    matches = [ref for ref in refs if ref.cwd is not None and safe_resolve(ref.cwd) == target]
    if not matches:
        return None
    matches.sort(key=lambda r: r.started_at, reverse=True)
    rows = [_format_session_row(ref) for ref in matches]
    idx = prompts.select("Select a session to hand off:", items=rows, default=0)
    return matches[idx]
