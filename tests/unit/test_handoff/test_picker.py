"""Tests for pick_latest_session and prompt_for_session."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from crossby.handoff.models import SessionRef
from crossby.handoff.picker import _format_session_row, pick_latest_session, prompt_for_session
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


# --- prompt_for_session tests -----------------------------------------------


def test_prompt_for_session_returns_user_choice() -> None:
    """Monkeypatching select to return index 1 should yield the 2nd-newest session."""
    root = Path("/Users/tester/proj")
    newer = _ref("019cb497-newer", datetime(2026, 5, 18, 14, 23), root)
    older = _ref("7a3f1d0c-older", datetime(2026, 5, 18, 13, 15), root)

    with patch("crossby.ui.prompts.select", return_value=1) as mock_select:
        chosen = prompt_for_session([newer, older], root)

    assert chosen is not None
    assert chosen.session_id == "7a3f1d0c-older"
    mock_select.assert_called_once()


def test_prompt_for_session_defaults_to_newest() -> None:
    """The picker must pass default=0 and order refs newest-first."""
    root = Path("/Users/tester/proj")
    older = _ref("old-session", datetime(2026, 5, 17, 9, 4), root)
    newer = _ref("new-session", datetime(2026, 5, 18, 14, 23), root)

    captured: dict[str, object] = {}

    def _fake_select(title: str, items: list[str], default: int = 0, **_kw: object) -> int:
        captured["default"] = default
        captured["items"] = items
        return 0

    with patch("crossby.ui.prompts.select", side_effect=_fake_select):
        chosen = prompt_for_session([older, newer], root)

    assert captured["default"] == 0
    items = captured["items"]
    assert isinstance(items, list)
    assert items[0].startswith("new-sess")
    assert chosen is not None
    assert chosen.session_id == "new-session"


def test_prompt_for_session_excludes_other_projects() -> None:
    """Sessions from a different project root must not appear in the picker."""
    root = Path("/Users/tester/proj")
    other = Path("/Users/tester/other")
    match = _ref("in-proj", datetime(2026, 5, 18, 12, 0), root)
    stray = _ref("in-other", datetime(2026, 5, 18, 14, 0), other)

    with patch("crossby.ui.prompts.select", return_value=0) as mock_select:
        chosen = prompt_for_session([match, stray], root)

    assert chosen is not None
    assert chosen.session_id == "in-proj"
    _call_args = mock_select.call_args
    items = _call_args[1]["items"] if "items" in _call_args[1] else _call_args[0][1]
    assert len(items) == 1


def test_prompt_for_session_no_matches_returns_none() -> None:
    """Returns None immediately when no refs match the project root."""
    root = Path("/Users/tester/proj")
    refs = [_ref("x", datetime(2026, 1, 1), Path("/elsewhere"))]

    with patch("crossby.ui.prompts.select") as mock_select:
        result = prompt_for_session(refs, root)

    assert result is None
    mock_select.assert_not_called()


def test_format_session_row_shape() -> None:
    """Row must have shape '<short_id>  ·  <relative>  ·  <iso_utc>'."""
    from datetime import UTC

    ref = _ref("019cb497eclong", datetime(2026, 5, 18, 14, 23, tzinfo=UTC), Path("/p"))
    row = _format_session_row(ref)
    parts = row.split("  ·  ")
    assert len(parts) == 3
    assert parts[0] == "019cb497"
    assert "2026-05-18" in parts[2]
    assert parts[2].endswith("Z")
