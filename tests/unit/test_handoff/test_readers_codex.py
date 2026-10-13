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
from crossby.handoff.truncate import truncate_transcript, turn_tokens
from crossby.models.ai import AIToolID


def _ref(path: Path, cwd: Path, session_id: str = "019cb497") -> SessionRef:
    return SessionRef(
        tool_id=AIToolID.CODEX,
        session_id=session_id,
        path=path,
        started_at=datetime(2026, 3, 3, 16, 46, 21),
        cwd=cwd,
    )


def test_happy_path_parses_messages_reasoning_and_function_calls(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    session_file = tmp_path / "rollout-happy.jsonl"
    shutil.copy(fixtures_dir / "codex_happy.jsonl", session_file)
    ref = _ref(session_file, cwd=Path("/Users/tester/proj"))

    transcript = codex_reader.read_session(ref)

    roles = [t.role for t in transcript.turns]
    assert roles == ["user", "assistant", "assistant", "assistant"]
    assert "flaky test" in transcript.turns[0].content
    assert transcript.turns[1].content.startswith("[reasoning]")
    assert transcript.turns[2].tool_calls[0].name == "shell"
    assert transcript.turns[2].tool_calls[0].arguments == {"command": "rg flaky"}
    assert "race" in transcript.turns[3].content


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


@pytest.mark.parametrize(
    "output", ["Process exited with code 0\n3 passed", "FAILED: payment charged twice"]
)
def test_function_output_without_assistant_restatement(tmp_path: Path, output: str) -> None:
    transcript = _read_payloads(
        tmp_path,
        [
            {"type": "message", "role": "user", "content": "Run the tests"},
            {
                "type": "function_call",
                "name": "shell",
                "arguments": '{"command":"pytest"}',
                "call_id": "c1",
            },
            {"type": "function_call_output", "call_id": "c1", "output": output},
        ],
    )
    assert [t.role for t in transcript.turns] == ["user", "assistant", "tool"]
    call, result = transcript.turns[1:]
    assert result.content == output
    assert result.tool_call_id == call.tool_calls[0].call_id == "c1"
    assert result.timestamp == call.timestamp
    kept = truncate_transcript(transcript, token_budget=turn_tokens(result))
    assert kept.turns == [result]
    assert kept.truncated
    assert kept.turns[0].tool_call_id == "c1"


@pytest.mark.parametrize("payload_type", ["function_call_output", "custom_tool_call_output"])
def test_output_blocks_preserve_text_and_ignore_images(payload_type: str) -> None:
    turn = codex_reader._turn_from_event(
        {
            "type": "response_item",
            "payload": {
                "type": payload_type,
                "call_id": "c2",
                "output": [
                    {"type": "input_text", "text": "FAILED: migration"},
                    {"type": "input_image", "image_url": "data:image/png;base64,opaque"},
                    {"type": "input_text", "text": "database locked"},
                ],
            },
        }
    )
    assert turn is not None
    assert turn.role == "tool"
    assert turn.content == "FAILED: migration\ndatabase locked"
    assert turn.tool_call_id == "c2"


@pytest.mark.parametrize("output", [None, 123, {}, [], "", "  ", [{"text": 123}], [None]])
def test_malformed_or_empty_output_is_skipped(output: object) -> None:
    assert (
        codex_reader._turn_from_event(
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": output,
                },
            }
        )
        is None
    )


@pytest.mark.parametrize("call_id", [None, 42, {}, []])
def test_output_without_valid_call_id_retains_evidence(call_id: object) -> None:
    turn = codex_reader._turn_from_event(
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": "command failed",
            },
        }
    )
    assert turn is not None
    assert turn.content == "command failed"
    assert turn.tool_call_id is None


def test_custom_call_preserves_raw_input_and_output(tmp_path: Path) -> None:
    raw_input = "*** Begin Patch\n*** End Patch"
    transcript = _read_payloads(
        tmp_path,
        [
            {
                "type": "custom_tool_call",
                "name": "apply_patch",
                "call_id": "patch1",
                "input": raw_input,
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "patch1",
                "output": "Success. Updated a.py",
            },
        ],
    )
    assert [t.role for t in transcript.turns] == ["assistant", "tool"]
    call = transcript.turns[0].tool_calls[0]
    assert call.name == "apply_patch"
    assert call.arguments == {"input": raw_input}
    assert call.call_id == transcript.turns[1].tool_call_id == "patch1"
    assert transcript.turns[1].content == "Success. Updated a.py"


def test_interleaved_outputs_keep_explicit_association(tmp_path: Path) -> None:
    transcript = _read_payloads(
        tmp_path,
        [
            {
                "type": "function_call",
                "name": "shell",
                "call_id": "a",
                "arguments": '{"cmd":"test a"}',
            },
            {
                "type": "function_call",
                "name": "shell",
                "call_id": "b",
                "arguments": '{"cmd":"test b"}',
            },
            {"type": "function_call_output", "call_id": "b", "output": "b failed"},
            {"type": "function_call_output", "call_id": "a", "output": "a passed"},
        ],
    )
    assert [(t.tool_call_id, t.content) for t in transcript.turns if t.role == "tool"] == [
        ("b", "b failed"),
        ("a", "a passed"),
    ]


@pytest.mark.parametrize("raw_input", [None, 123, {}, []])
def test_malformed_custom_input_is_skipped(raw_input: object) -> None:
    assert (
        codex_reader._turn_from_event(
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "name": "apply_patch",
                    "input": raw_input,
                },
            }
        )
        is None
    )
