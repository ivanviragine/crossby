"""Tests for the Codex session reader."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from crossby.handoff.models import ConversationTranscript, SessionRef
from crossby.handoff.readers import codex as codex_reader
from crossby.models.ai import AIToolID


def _ref(path: Path, cwd: Path, session_id: str = "019cb497") -> SessionRef:
    return SessionRef(
        tool_id=AIToolID.CODEX,
        session_id=session_id,
        path=path,
        started_at=datetime(2026, 3, 3, 16, 46, 21),
        cwd=cwd,
    )


def test_happy_path_keeps_only_user_and_visible_assistant_messages(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    session_file = tmp_path / "rollout-happy.jsonl"
    shutil.copy(fixtures_dir / "codex_happy.jsonl", session_file)
    ref = _ref(session_file, cwd=Path("/Users/tester/proj"))

    transcript = codex_reader.read_session(ref)

    assert [(t.role, t.content) for t in transcript.turns] == [
        ("user", "Help me fix the flaky test."),
        ("assistant", "Found it — retry loop has a race."),
    ]
    assert all(not t.tool_calls and not t.file_refs for t in transcript.turns)


def test_empty_session_only_meta_returns_no_turns(fixtures_dir: Path, tmp_path: Path) -> None:
    session_file = tmp_path / "rollout-empty.jsonl"
    shutil.copy(fixtures_dir / "codex_empty.jsonl", session_file)
    ref = _ref(session_file, cwd=Path("/Users/tester/proj"), session_id="empty")

    transcript = codex_reader.read_session(ref)

    assert transcript.turns == []


def test_malformed_line_is_skipped(fixtures_dir: Path, tmp_path: Path) -> None:
    session_file = tmp_path / "rollout-malformed.jsonl"
    shutil.copy(fixtures_dir / "codex_malformed.jsonl", session_file)
    ref = _ref(session_file, cwd=Path("/Users/tester/proj"), session_id="malformed")

    transcript = codex_reader.read_session(ref)

    assert [t.role for t in transcript.turns] == ["user", "assistant"]


def test_locate_sessions_filters_by_cwd(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    root = fake_home / ".codex" / "sessions" / "2026" / "03" / "03"
    root.mkdir(parents=True)
    shutil.copy(
        fixtures_dir / "codex_happy.jsonl",
        root / "rollout-2026-03-03T13-46-21-019cb497.jsonl",
    )

    refs = codex_reader.locate_sessions(Path("/Users/tester/proj"))
    assert len(refs) == 1
    assert refs[0].cwd == Path("/Users/tester/proj")
    assert refs[0].session_id == "019cb497-ec14-7453-9224-af8e8944b4c5"


def test_locate_sessions_skips_other_projects(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    root = fake_home / ".codex" / "sessions" / "2026" / "03" / "03"
    root.mkdir(parents=True)
    shutil.copy(
        fixtures_dir / "codex_happy.jsonl",
        root / "rollout-1.jsonl",
    )

    # Different project — the cwd in the fixture is /Users/tester/proj.
    assert codex_reader.locate_sessions(Path("/Users/tester/other-proj")) == []


def _read_payloads(tmp_path: Path, payloads: list[dict[str, Any]]) -> ConversationTranscript:
    session_file = tmp_path / "rollout-result.jsonl"
    session_file.write_text(
        "\n".join(
            json.dumps({"type": "response_item", "timestamp": "2026-03-03T16:46:30Z", "payload": p})
            for p in payloads
        ),
        encoding="utf-8",
    )
    return codex_reader.read_session(_ref(session_file, tmp_path))


@pytest.mark.parametrize("role", ["system", "developer", "tool"])
def test_source_instructions_and_tool_messages_are_skipped(tmp_path: Path, role: str) -> None:
    transcript = _read_payloads(
        tmp_path,
        [
            {"type": "message", "role": role, "content": "Source-only context"},
            {"type": "message", "role": "user", "content": "Continue the refactor"},
        ],
    )
    assert [(t.role, t.content) for t in transcript.turns] == [("user", "Continue the refactor")]


@pytest.mark.parametrize("channel", ["commentary", "final", None])
def test_visible_assistant_channels_are_preserved(tmp_path: Path, channel: str | None) -> None:
    transcript = _read_payloads(
        tmp_path,
        [{"type": "message", "role": "assistant", "channel": channel, "content": "Update"}],
    )
    assert [t.content for t in transcript.turns] == ["Update"]
