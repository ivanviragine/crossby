"""Tests for the Copilot session reader."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from crossby.handoff.models import SessionRef
from crossby.handoff.readers import copilot as copilot_reader
from crossby.models.ai import AIToolID


def _ref(path: Path, cwd: Path, session_id: str = "sess-1") -> SessionRef:
    return SessionRef(
        tool_id=AIToolID.COPILOT,
        session_id=session_id,
        path=path,
        started_at=datetime(2026, 2, 21, 14, 38, 1),
        cwd=cwd,
    )


def test_happy_path_parses_user_assistant_and_tool_turns(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    events = tmp_path / "events.jsonl"
    shutil.copy(fixtures_dir / "copilot_events.jsonl", events)
    ref = _ref(events, cwd=Path("/Users/tester/proj"))

    transcript = copilot_reader.read_session(ref)

    roles = [t.role for t in transcript.turns]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert transcript.turns[1].tool_calls[0].name == "view"
    assert transcript.turns[1].file_refs == [Path("/Users/tester/proj/PLAN.md")]


def test_empty_session_returns_no_turns(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text('{"type":"session.start","data":{}}\n', encoding="utf-8")
    ref = _ref(events, cwd=Path("/Users/tester/proj"), session_id="empty")

    transcript = copilot_reader.read_session(ref)

    assert transcript.turns == []


def test_malformed_line_is_skipped(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        '{"type":"user.message","data":{"content":"Before."}}\n'
        "not-json\n"
        '{"type":"assistant.message","data":{"content":"After."}}\n',
        encoding="utf-8",
    )
    ref = _ref(events, cwd=Path("/Users/tester/proj"), session_id="malformed")

    transcript = copilot_reader.read_session(ref)

    assert [t.role for t in transcript.turns] == ["user", "assistant"]


def test_locate_sessions_reads_workspace_yaml(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    session_dir = fake_home / ".copilot" / "session-state" / "sess-1"
    session_dir.mkdir(parents=True)
    shutil.copy(fixtures_dir / "copilot_workspace.yaml", session_dir / "workspace.yaml")
    shutil.copy(fixtures_dir / "copilot_events.jsonl", session_dir / "events.jsonl")

    refs = copilot_reader.locate_sessions(Path("/Users/tester/proj"))
    assert len(refs) == 1
    assert refs[0].session_id == "sess-1"
    assert refs[0].path.name == "events.jsonl"


def test_locate_sessions_skips_session_with_different_cwd(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    session_dir = fake_home / ".copilot" / "session-state" / "sess-other"
    session_dir.mkdir(parents=True)
    workspace = {
        "id": "sess-other",
        "cwd": "/Users/tester/different",
        "created_at": "2026-02-21T14:38:01.959Z",
    }
    (session_dir / "workspace.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in workspace.items()), encoding="utf-8"
    )
    shutil.copy(fixtures_dir / "copilot_events.jsonl", session_dir / "events.jsonl")

    assert copilot_reader.locate_sessions(Path("/Users/tester/proj")) == []
    # sanity: the event file is readable when pointed at directly
    _ = json.loads((session_dir / "events.jsonl").read_text().splitlines()[0])
