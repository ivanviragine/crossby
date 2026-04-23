"""Tests for the Cursor session reader."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pytest

from crossby.handoff.models import SessionRef
from crossby.handoff.readers import cursor as cursor_reader
from crossby.models.ai import AIToolID


def _ref(path: Path, cwd: Path, session_id: str = "chat") -> SessionRef:
    return SessionRef(
        tool_id=AIToolID.CURSOR,
        session_id=session_id,
        path=path,
        started_at=datetime(2026, 2, 10, 12, 0, 0),
        cwd=cwd,
    )


def test_happy_path_parses_messages(fixtures_dir: Path, tmp_path: Path) -> None:
    chat = tmp_path / "chat.json"
    shutil.copy(fixtures_dir / "cursor_chat.json", chat)
    ref = _ref(chat, cwd=Path("/Users/tester/proj"))

    transcript = cursor_reader.read_session(ref)

    roles = [t.role for t in transcript.turns]
    assert roles == ["user", "assistant", "assistant"]
    assert "payment" in transcript.turns[0].content.lower()


def test_empty_session_returns_no_turns(fixtures_dir: Path, tmp_path: Path) -> None:
    chat = tmp_path / "chat.json"
    shutil.copy(fixtures_dir / "cursor_empty.json", chat)
    ref = _ref(chat, cwd=Path("/Users/tester/proj"), session_id="empty")

    transcript = cursor_reader.read_session(ref)

    assert transcript.turns == []


def test_malformed_messages_are_skipped(fixtures_dir: Path, tmp_path: Path) -> None:
    chat = tmp_path / "chat.json"
    shutil.copy(fixtures_dir / "cursor_malformed.json", chat)
    ref = _ref(chat, cwd=Path("/Users/tester/proj"), session_id="malformed")

    transcript = cursor_reader.read_session(ref)

    # Valid turn + second valid turn; one without role + one without content dropped.
    assert [t.role for t in transcript.turns] == ["user", "assistant"]


def test_locate_sessions_uses_cursor_encoded_dir(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    project = Path("/Users/tester/proj")
    encoded = "Users-tester-proj"  # strip leading /, then / → -
    session_dir = fake_home / ".cursor" / "projects" / encoded
    session_dir.mkdir(parents=True)
    shutil.copy(fixtures_dir / "cursor_chat.json", session_dir / "chat.json")

    refs = cursor_reader.locate_sessions(project)
    assert len(refs) == 1
    assert refs[0].session_id == "chat"


def test_locate_sessions_missing_dir_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    assert cursor_reader.locate_sessions(Path("/nope")) == []
