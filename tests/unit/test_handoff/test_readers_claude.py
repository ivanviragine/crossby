"""Tests for the Claude session reader."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pytest

from crossby.handoff.models import SessionRef
from crossby.handoff.readers import claude as claude_reader
from crossby.models.ai import AIToolID


def _ref(path: Path, cwd: Path, session_id: str = "session-happy") -> SessionRef:
    return SessionRef(
        tool_id=AIToolID.CLAUDE,
        session_id=session_id,
        path=path,
        started_at=datetime(2026, 3, 24, 18, 35, 33),
        cwd=cwd,
    )


def test_happy_path_returns_user_and_assistant_turns(fixtures_dir: Path, tmp_path: Path) -> None:
    session_file = tmp_path / "s1.jsonl"
    shutil.copy(fixtures_dir / "claude_happy.jsonl", session_file)
    ref = _ref(session_file, cwd=Path("/Users/tester/proj"))

    transcript = claude_reader.read_session(ref)

    assert [(t.role, t.content) for t in transcript.turns] == [
        ("user", "Let's refactor the auth module."),
        ("assistant", "The auth module has three routes; the /login one caches sessions."),
        ("user", "Great, proceed with the refactor."),
    ]
    assert all(not t.tool_calls and not t.file_refs for t in transcript.turns)


def test_empty_session_returns_no_turns(fixtures_dir: Path, tmp_path: Path) -> None:
    session_file = tmp_path / "s1.jsonl"
    shutil.copy(fixtures_dir / "claude_empty.jsonl", session_file)
    ref = _ref(session_file, cwd=Path("/Users/tester/proj"), session_id="empty")

    transcript = claude_reader.read_session(ref)

    assert transcript.turns == []


def test_malformed_line_is_skipped(fixtures_dir: Path, tmp_path: Path) -> None:
    session_file = tmp_path / "s1.jsonl"
    shutil.copy(fixtures_dir / "claude_malformed.jsonl", session_file)
    ref = _ref(session_file, cwd=Path("/Users/tester/proj"), session_id="malformed")

    transcript = claude_reader.read_session(ref)

    # Two turns survive; the bad line is dropped.
    assert len(transcript.turns) == 2
    assert transcript.turns[0].role == "user"
    assert transcript.turns[1].role == "assistant"


def test_locate_sessions_uses_claude_encoded_dir(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Project path must encode to the dir we create.
    project = Path("/Users/tester/proj")
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    encoded = "-Users-tester-proj"  # / → -, . → -; here no dots
    session_dir = fake_home / ".claude" / "projects" / encoded
    session_dir.mkdir(parents=True)
    shutil.copy(fixtures_dir / "claude_happy.jsonl", session_dir / "abc.jsonl")

    refs = claude_reader.locate_sessions(project)

    assert len(refs) == 1
    assert refs[0].session_id == "abc"
    assert refs[0].path.name == "abc.jsonl"
    assert refs[0].cwd == project


def test_locate_sessions_missing_dir_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    assert claude_reader.locate_sessions(Path("/nope/nowhere")) == []
