"""Pick the most relevant session for a project from a list of SessionRefs."""

from __future__ import annotations

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
