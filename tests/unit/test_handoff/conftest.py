"""Shared fixtures for handoff reader tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from crossby.handoff.models import ConversationTranscript, SessionRef
from crossby.handoff.readers import claude, codex, copilot, cursor
from crossby.models.ai import AIToolID

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "handoff"


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture()
def project_path() -> Path:
    # A stable project root that matches the ``cwd`` fields in our fixtures.
    return Path("/Users/tester/proj")


@pytest.fixture(params=["claude", "codex", "copilot", "cursor"])
def conversation_transcript(
    request: pytest.FixtureRequest, tmp_path: Path
) -> ConversationTranscript:
    """Read native sessions containing much more tool/reasoning data than conversation."""
    noise = "DISCARD " * 10_000
    user = {"role": "user", "content": [{"type": "input_text", "text": "Fix the flaky test."}]}
    progress = {
        "role": "assistant",
        "channel": "commentary",
        "content": [
            {"type": "text", "text": "Checking the retry loop."},
            {"type": "thinking", "thinking": noise, "text": noise},
            {"type": "reasoning", "text": noise},
            {"type": "tool_result", "content": noise, "text": noise},
            {"type": "image", "text": noise},
        ],
    }
    final = {"role": "assistant", "channel": "final", "content": "The retry loop has a race."}
    call = {"type": "tool_use", "id": "call1", "name": "shell", "input": {"cmd": noise}}
    result = {"type": "tool_result", "tool_use_id": "call1", "content": noise}
    empty = {"role": "assistant", "content": "  "}
    instructions = {"role": "system", "content": noise}
    source = request.param
    if source == "claude":
        # Claude stores tool results in user events, including alongside real text.
        user["content"].append(result)
        progress["content"].append(call)
        messages = [
            user,
            progress,
            {"role": "user", "content": [result]},
            {"role": "assistant", "content": [call, {"type": "thinking", "thinking": noise}]},
            instructions,
            empty,
            final,
        ]
        events = [{"type": m["role"], "message": m} for m in messages]
        events.insert(1, {"type": "user", "isMeta": True, "message": {"content": noise}})
        reader = claude.read_session
    elif source == "codex":
        payloads = [
            {"type": "message", **user},
            {"type": "message", **progress},
            {"type": "function_call", "name": "shell", "arguments": noise, "call_id": "c1"},
            {"type": "function_call_output", "output": noise, "call_id": "c1"},
            {"type": "custom_tool_call", "name": "apply_patch", "input": noise},
            {"type": "custom_tool_call_output", "output": [result]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": noise}]},
            {"type": "message", "role": "assistant", "channel": "analysis", "content": noise},
            {"type": "message", **instructions},
            {"type": "message", **empty},
            {"type": "message", **final},
        ]
        events = [{"type": "response_item", "payload": p} for p in payloads]
        reader = codex.read_session
    elif source == "copilot":
        tool_request = {
            "toolCallId": "call1",
            "name": "shell",
            "arguments": {"cmd": noise},
            "type": "function",
        }
        events = [
            {"type": "user.message", "data": user},
            {"type": "assistant.message", "data": {**progress, "toolRequests": [tool_request]}},
            {"type": "tool.result", "data": {"output": noise}},
            {"type": "assistant.reasoning", "data": {"content": noise}},
            {"type": "system.message", "data": instructions},
            {"type": "assistant.message", "data": {**empty, "toolRequests": [tool_request]}},
            {"type": "assistant.message", "data": final},
        ]
        reader = copilot.read_session
    else:
        messages = [
            user,
            progress,
            {"role": "tool", "content": noise},
            instructions,
            {"role": "developer", "content": noise},
            {"role": "assistant", "channel": "analysis", "content": noise},
            {"role": "assistant", "content": [call, result]},
            empty,
            final,
        ]
        reader = cursor.read_session

    path = tmp_path / "conversation.jsonl"
    timestamp = "2026-03-03T16:46:30Z"
    if source == "cursor":
        path.write_text(json.dumps({"messages": [{**m, "timestamp": timestamp} for m in messages]}))
    else:
        path.write_text("\n".join(json.dumps({**e, "timestamp": timestamp}) for e in events))
    ref = SessionRef(
        tool_id=AIToolID(source),
        session_id="conversation",
        path=path,
        started_at=datetime(2026, 3, 3, tzinfo=UTC),
        cwd=tmp_path,
    )
    return reader(ref)
