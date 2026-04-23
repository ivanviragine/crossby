"""Tests for pick_latest_session."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from crossby.handoff.models import SessionRef
from crossby.handoff.picker import pick_latest_session
from crossby.models.ai import AIToolID


def _ref(session_id: str, started: datetime, cwd: Path | None) -> SessionRef:
    return SessionRef(
        tool_id=AIToolID.CLAUDE,
        session_id=session_id,
        path=Path(f"/tmp/{session_id}.jsonl"),
        started_at=started,
        cwd=cwd,
    )


def test_returns_most_recent_matching_cwd() -> None:
    root = Path("/Users/tester/proj")
    older = _ref("old", datetime(2026, 1, 1, 10), root)
    newer = _ref("new", datetime(2026, 3, 1, 10), root)
    stray = _ref("stray", datetime(2026, 4, 1, 10), Path("/Users/tester/other"))

    chosen = pick_latest_session([older, newer, stray], root)

    assert chosen is not None
    assert chosen.session_id == "new"


def test_returns_none_when_no_sessions_match() -> None:
    refs = [_ref("a", datetime(2026, 1, 1), Path("/elsewhere"))]
    assert pick_latest_session(refs, Path("/Users/tester/proj")) is None


def test_sessions_without_cwd_are_excluded() -> None:
    root = Path("/Users/tester/proj")
    no_cwd = _ref("no-cwd", datetime(2026, 5, 1), None)
    match = _ref("has-cwd", datetime(2026, 2, 1), root)
    chosen = pick_latest_session([no_cwd, match], root)
    assert chosen is not None
    assert chosen.session_id == "has-cwd"
