"""Unattended headless coverage for the six terminal adapters.

The native payloads below are recorded from live runs of the exact CLI builds
each adapter declares as its ``verified_version``:

* Claude Code 2.1.263 — ``claude --print --output-format json|stream-json``
* Codex CLI 0.154.0 — ``codex exec --json``
* Cursor Agent 2026.09.10-fd3934a — ``agent --print --output-format json|stream-json``
* GitHub Copilot CLI 1.0.83 — ``copilot --prompt -s``
* OpenCode 1.18.31 — ``opencode run --format json``
* Antigravity CLI 1.2.6 — ``agy --print --output-format json|stream-json``

Every session here runs a real child process through the shared managed
transport, so process ownership, stdin delivery, and cleanup are exercised
rather than mocked.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.headless import HeadlessUnsupportedError
from crossby.models.ai import (
    AIToolID,
    HeadlessEventKind,
    HeadlessInteractionMode,
    HeadlessNativeOutput,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
)
from crossby.utils.versioning import BinaryVersion, parse_semver

PROMPT = "confidential unattended prompt"
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

# --- recorded native payloads ------------------------------------------------

CLAUDE_RESULT: dict[str, Any] = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "terminal_reason": "completed",
    "result": '{"answer":"OK"}',
    "structured_output": {"answer": "OK"},
    "session_id": "1492f340-106a-4548-b7ba-6c3aba132018",
    "permission_denials": [],
    "num_turns": 2,
    "duration_ms": 3643,
    "usage": {
        "input_tokens": 2,
        "output_tokens": 63,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 26256,
    },
}
# Recorded on 2.1.263: ``subtype`` stays "success" on a failed turn, so only
# ``is_error`` and ``terminal_reason`` describe the real outcome.
CLAUDE_ERROR_RESULT: dict[str, Any] = {
    **CLAUDE_RESULT,
    "is_error": True,
    "terminal_reason": "api_error",
    "result": "Not logged in · Please run /login",
    "structured_output": None,
    "permission_denials": [{"tool_name": "Bash", "tool_use_id": "toolu_01"}],
}
CLAUDE_STREAM = (
    {
        "type": "system",
        "subtype": "init",
        "session_id": CLAUDE_RESULT["session_id"],
        "model": "claude-sonnet-5",
    },
    {
        "type": "assistant",
        "session_id": CLAUDE_RESULT["session_id"],
        "message": {"role": "assistant", "content": [{"type": "text", "text": PROMPT}]},
    },
    CLAUDE_RESULT,
)

CODEX_EVENTS = (
    {"type": "thread.started", "thread_id": "01a0b474-607c-77f3-a090-9139c3e1128f"},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": "OK"}},
    {"type": "turn.completed", "usage": {"input_tokens": 9810, "output_tokens": 1}},
)
CODEX_FAILED_EVENTS = (
    {"type": "thread.started", "thread_id": "01a0b474-607c-77f3-a090-9139c3e1128f"},
    {"type": "turn.started"},
    {"type": "error", "message": "unexpected status 401 Unauthorized"},
    {"type": "turn.failed", "error": {"message": "unexpected status 401 Unauthorized"}},
)

CURSOR_RESULT: dict[str, Any] = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "duration_ms": 3197,
    "duration_api_ms": 3197,
    "result": "OK",
    "session_id": "d7af02cf-bfa9-4aa2-951c-3944de7a7ce7",
    "request_id": "7e11401c-a2c2-4cdd-a7c1-d26252a0c1b2",
    "usage": {
        "inputTokens": 9118,
        "outputTokens": 27,
        "cacheReadTokens": 512,
        "cacheWriteTokens": 0,
    },
}
CURSOR_STREAM = (
    {
        "type": "system",
        "subtype": "init",
        "session_id": CURSOR_RESULT["session_id"],
        "cwd": "/private/tmp",
    },
    {
        "type": "user",
        "session_id": CURSOR_RESULT["session_id"],
        "message": {"role": "user", "content": [{"type": "text", "text": PROMPT}]},
    },
    {
        "type": "assistant",
        "session_id": CURSOR_RESULT["session_id"],
        "message": {"role": "assistant", "content": [{"type": "text", "text": "OK"}]},
    },
    CURSOR_RESULT,
)

OPENCODE_SESSION = "ses_f4b8d194cffejRZaI9Gvpw6FE8"
OPENCODE_EVENTS = (
    {
        "type": "step_start",
        "sessionID": OPENCODE_SESSION,
        "part": {"id": "prt_0", "type": "step-start", "sessionID": OPENCODE_SESSION},
    },
    {
        "type": "text",
        "sessionID": OPENCODE_SESSION,
        "part": {"id": "prt_1", "type": "text", "text": "OK", "sessionID": OPENCODE_SESSION},
    },
    {
        "type": "step_finish",
        "sessionID": OPENCODE_SESSION,
        "part": {
            "id": "prt_2",
            "type": "step-finish",
            "reason": "stop",
            "sessionID": OPENCODE_SESSION,
            "tokens": {
                "total": 9811,
                "input": 9810,
                "output": 1,
                "reasoning": 0,
                "cache": {"write": 0, "read": 0},
            },
        },
    },
)

AGY_CONVERSATION = "3b1adb15-893a-40e5-8a48-77ba6946f5f5"
AGY_RESULT: dict[str, Any] = {
    "conversation_id": AGY_CONVERSATION,
    "status": "SUCCESS",
    "response": '{"answer":"OK"}\n',
    "duration_seconds": 3.858326,
    "num_turns": 1,
    "structured_output": {"answer": "OK"},
    "json_schema": SCHEMA,
    "usage": {
        "input_tokens": 13415,
        "output_tokens": 486,
        "thinking_tokens": 455,
        "cache_read_tokens": 0,
        "total_tokens": 13901,
    },
}
AGY_STREAM = (
    {"event": "init", "conversation_id": AGY_CONVERSATION, "init": {"cwd": "/private/tmp"}},
    {
        "event": "step_update",
        "step_update": {
            "conversation_id": AGY_CONVERSATION,
            "step_index": 1,
            "state": "ACTIVE",
            "step_type": "agent_response",
            "text_delta": "OK",
        },
    },
    {"event": "result", "result": AGY_RESULT},
)
# Synthetic, derived from agy's collected-plan waiting contract rather than a
# live capture: an unattended run must never be able to answer this.
AGY_WAITING: dict[str, Any] = {
    "conversation_id": AGY_CONVERSATION,
    "status": "WAITING",
    "question": {"id": "q1", "prompt": "Which package manager should I use?"},
}


# --- harness -----------------------------------------------------------------


def _jsonl(frames: tuple[dict[str, Any], ...]) -> str:
    return "".join(f"{json.dumps(frame)}\n" for frame in frames)


def _fake_cli(
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    stdin_record: Path | None = None,
    sleep_seconds: float = 0.0,
    child_pid_file: Path | None = None,
) -> list[str]:
    """Build a real child process that replays a recorded native payload."""
    program = [
        "import os,sys,time",
        f"stdin_record={str(stdin_record)!r}",
        f"child_pid_file={str(child_pid_file)!r}",
        "data=sys.stdin.read()",
        "open(stdin_record,'w').write(data) if stdin_record!='None' else None",
    ]
    if child_pid_file is not None:
        # A grandchild in the same owned process group proves tree cleanup.
        program += [
            "pid=os.fork()",
            "if pid==0:\n"
            "    open(child_pid_file,'w').write(str(os.getpid()))\n"
            "    time.sleep(60)\n"
            "    os._exit(0)",
        ]
    program += [
        f"sys.stdout.write({stdout!r})",
        f"sys.stderr.write({stderr!r})",
        "sys.stdout.flush()",
        "sys.stderr.flush()",
        f"time.sleep({sleep_seconds!r})",
        f"sys.exit({exit_code})",
    ]
    return [sys.executable, "-c", "\n".join(program)]


def _adapter(monkeypatch: pytest.MonkeyPatch, tool: AIToolID, argv: list[str]) -> AbstractAITool:
    """Return an adapter whose native CLI and version probe are deterministic."""
    adapter = AbstractAITool.get(tool)
    version = adapter.capabilities().headless.verified_version
    assert version is not None
    normalized = parse_semver(version)
    assert normalized is not None

    monkeypatch.setattr(
        type(adapter),
        "_headless_command",
        lambda self, request, **_kwargs: list(argv),
        raising=False,
    )
    monkeypatch.setattr(
        type(adapter),
        "_detect_headless_version",
        lambda self, **_kwargs: BinaryVersion(normalized, version),
        raising=True,
    )
    return adapter


def _request(tmp_path: Path, **updates: Any) -> HeadlessSessionRequest:
    values: dict[str, Any] = {
        "prompt": PROMPT,
        "working_dir": tmp_path,
        "timeout_seconds": 30.0,
        "interaction_mode": HeadlessInteractionMode.UNATTENDED,
    }
    values.update(updates)
    return HeadlessSessionRequest(**values)


def _assert_prompt_is_private(result: HeadlessSessionResult) -> None:
    """No normalized event or diagnostic may echo the caller's prompt."""
    for event in result.events:
        assert event.message is None or PROMPT not in event.message
        assert PROMPT not in json.dumps(event.payload or {})
    for diagnostic in (*result.warnings, *result.denials):
        assert PROMPT not in diagnostic


# --- per-adapter native envelope coverage ------------------------------------


def test_claude_json_envelope_yields_schema_validated_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delivered = tmp_path / "stdin.txt"
    adapter = _adapter(
        monkeypatch,
        AIToolID.CLAUDE,
        _fake_cli(stdout=json.dumps(CLAUDE_RESULT), stdin_record=delivered),
    )

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON, response_schema=SCHEMA)
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_json == {"answer": "OK"}
    assert result.session_id == CLAUDE_RESULT["session_id"]
    assert result.native_status == "completed"
    assert result.usage is not None and result.usage.output_tokens == 63
    # Claude reads its prompt from stdin, never from argv.
    assert delivered.read_text() == PROMPT
    _assert_prompt_is_private(result)


def test_claude_native_error_is_not_success_on_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(
        monkeypatch,
        AIToolID.CLAUDE,
        _fake_cli(stdout=json.dumps(CLAUDE_ERROR_RESULT), exit_code=0),
    )

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON)
    )

    assert result.exit_code == 0
    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.final_json is None and not result.final_json_present
    assert result.native_status == "api_error"
    assert any("Bash" in denial for denial in result.denials)


def test_claude_denial_reporting_is_bounded_and_drops_tool_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    noisy = json.loads(json.dumps(CLAUDE_RESULT))
    noisy["permission_denials"] = [
        {"tool_name": "Bash", "tool_use_id": f"toolu_{index}", "tool_input": {"command": PROMPT}}
        for index in range(200)
    ]
    adapter = _adapter(monkeypatch, AIToolID.CLAUDE, _fake_cli(stdout=json.dumps(noisy)))

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON)
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert len(result.denials) == 65
    assert "136 further tool calls" in result.denials[-1]
    # The native entry carries the proposed tool input; only the name survives.
    _assert_prompt_is_private(result)


def test_claude_stream_json_progress_never_leaks_native_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(monkeypatch, AIToolID.CLAUDE, _fake_cli(stdout=_jsonl(CLAUDE_STREAM)))

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSONL)
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_json == CLAUDE_RESULT
    assert result.session_id == CLAUDE_RESULT["session_id"]
    kinds = [event.kind for event in result.events]
    assert kinds[0] is HeadlessEventKind.STARTED
    assert HeadlessEventKind.PROGRESS in kinds
    _assert_prompt_is_private(result)


def test_claude_text_output_returns_the_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(monkeypatch, AIToolID.CLAUDE, _fake_cli(stdout="OK\n"))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "OK"


def test_codex_turn_events_are_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    delivered = tmp_path / "stdin.txt"
    adapter = _adapter(
        monkeypatch,
        AIToolID.CODEX,
        _fake_cli(stdout=_jsonl(CODEX_EVENTS), stdin_record=delivered),
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "OK"
    assert result.thread_id == CODEX_EVENTS[0]["thread_id"]
    assert result.native_status == "turn.completed"
    assert delivered.read_text() == PROMPT


def test_codex_turn_failed_is_failure_on_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(
        monkeypatch,
        AIToolID.CODEX,
        _fake_cli(stdout=_jsonl(CODEX_FAILED_EVENTS), exit_code=0),
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.exit_code == 0
    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.native_status == "turn.failed"
    assert any("401" in warning for warning in result.warnings)


def test_codex_without_a_terminal_turn_event_is_invalid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(monkeypatch, AIToolID.CODEX, _fake_cli(stdout=_jsonl(CODEX_EVENTS[:3])))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert any("terminal turn event" in warning for warning in result.warnings)


def test_cursor_json_envelope_reports_proposal_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(monkeypatch, AIToolID.CURSOR, _fake_cli(stdout=json.dumps(CURSOR_RESULT)))

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON)
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_json == CURSOR_RESULT
    assert result.native_status == "success"
    assert result.usage is not None and result.usage.input_tokens == 9118
    assert any("proposals" in warning for warning in result.warnings)


def test_cursor_stream_json_hides_the_echoed_user_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(monkeypatch, AIToolID.CURSOR, _fake_cli(stdout=_jsonl(CURSOR_STREAM)))

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSONL)
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.session_id == CURSOR_RESULT["session_id"]
    # Cursor's stream echoes the prompt in its ``user`` frame.
    _assert_prompt_is_private(result)


def test_copilot_text_response_is_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(monkeypatch, AIToolID.COPILOT, _fake_cli(stdout="OK\n"))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "OK"
    assert any("--allow-all-tools" in warning for warning in result.warnings)


def test_copilot_policy_failure_surfaces_bounded_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(
        monkeypatch,
        AIToolID.COPILOT,
        _fake_cli(stderr="Error: Access denied by policy settings", exit_code=1),
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.final_text is None
    assert any("Access denied by policy" in warning for warning in result.warnings)


def test_opencode_raw_events_are_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(monkeypatch, AIToolID.OPENCODE, _fake_cli(stdout=_jsonl(OPENCODE_EVENTS)))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "OK"
    assert result.session_id == OPENCODE_SESSION
    assert result.native_status == "stop"
    assert result.usage is not None and result.usage.total_tokens == 9811


def test_opencode_non_stop_finish_reason_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aborted = list(OPENCODE_EVENTS)
    aborted[-1] = json.loads(json.dumps(aborted[-1]))
    aborted[-1]["part"]["reason"] = "aborted"
    adapter = _adapter(
        monkeypatch, AIToolID.OPENCODE, _fake_cli(stdout=_jsonl(tuple(aborted)), exit_code=0)
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.exit_code == 0
    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.native_status == "aborted"


def test_antigravity_json_envelope_validates_the_schema_echo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(
        monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=json.dumps(AGY_RESULT))
    )

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON, response_schema=SCHEMA)
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_json == {"answer": "OK"}
    assert result.conversation_id == AGY_CONVERSATION
    assert result.native_status == "success"
    assert result.usage is not None and result.usage.total_tokens == 13901


def test_antigravity_rejects_a_mismatched_schema_echo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tampered = json.loads(json.dumps(AGY_RESULT))
    tampered["json_schema"] = {"type": "object"}
    adapter = _adapter(
        monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=json.dumps(tampered))
    )

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON, response_schema=SCHEMA)
    )

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert any("echo the requested schema" in warning for warning in result.warnings)


def test_antigravity_stream_json_events_are_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=_jsonl(AGY_STREAM)))

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSONL)
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_json == AGY_RESULT
    assert result.conversation_id == AGY_CONVERSATION


def test_antigravity_waiting_state_fails_an_unattended_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(
        monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=json.dumps(AGY_WAITING))
    )

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON)
    )

    assert result.status is HeadlessTerminalStatus.FAILED
    assert any("unexpected native question" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)
    assert all("package manager" not in (event.message or "") for event in result.events)


# --- shared unattended guarantees --------------------------------------------


@pytest.mark.parametrize(
    ("tool", "native_output"),
    [
        (AIToolID.CLAUDE, HeadlessNativeOutput.TEXT),
        (AIToolID.CODEX, HeadlessNativeOutput.TEXT),
        (AIToolID.CURSOR, HeadlessNativeOutput.JSON),
        (AIToolID.COPILOT, HeadlessNativeOutput.TEXT),
        (AIToolID.OPENCODE, HeadlessNativeOutput.TEXT),
        (AIToolID.ANTIGRAVITY_CLI, HeadlessNativeOutput.TEXT),
    ],
)
def test_unattended_sessions_never_inherit_parent_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: AIToolID,
    native_output: HeadlessNativeOutput,
) -> None:
    """A native CLI that reads stdin gets EOF, never the caller's terminal."""
    delivered = tmp_path / f"{tool.value}-stdin.txt"
    adapter = _adapter(monkeypatch, tool, _fake_cli(stdout="{}", stdin_record=delivered))

    adapter.run_headless_session(_request(tmp_path, native_output=native_output))

    prompt_transport = adapter.capabilities().headless.prompt_transport
    assert prompt_transport is not None
    if prompt_transport.value == "stdin":
        assert delivered.read_text() == PROMPT
    else:
        # /dev/null, so the child reads EOF immediately instead of blocking.
        assert delivered.read_text() == ""


@pytest.mark.parametrize(
    ("tool", "native_output"),
    [
        (AIToolID.CURSOR, HeadlessNativeOutput.JSON),
        (AIToolID.COPILOT, HeadlessNativeOutput.TEXT),
        (AIToolID.OPENCODE, HeadlessNativeOutput.JSONL),
    ],
)
def test_schema_requests_fail_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: AIToolID,
    native_output: HeadlessNativeOutput,
) -> None:
    started = tmp_path / "started.txt"
    adapter = _adapter(monkeypatch, tool, _fake_cli(stdout="{}", stdin_record=started))

    with pytest.raises(HeadlessUnsupportedError, match="response schema"):
        adapter.run_headless_session(
            _request(tmp_path, native_output=native_output, response_schema=SCHEMA)
        )

    assert not started.exists()


@pytest.mark.parametrize("tool", [AIToolID.CLAUDE, AIToolID.ANTIGRAVITY_CLI])
def test_text_output_rejects_a_schema_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: AIToolID
) -> None:
    started = tmp_path / "started.txt"
    adapter = _adapter(monkeypatch, tool, _fake_cli(stdout="{}", stdin_record=started))

    with pytest.raises(HeadlessUnsupportedError, match="structured_output"):
        adapter.run_headless_session(
            _request(tmp_path, native_output=HeadlessNativeOutput.TEXT, response_schema=SCHEMA)
        )

    assert not started.exists()


def test_schema_violating_structured_output_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    violating = json.loads(json.dumps(CLAUDE_RESULT))
    violating["structured_output"] = {"answer": 7}
    adapter = _adapter(monkeypatch, AIToolID.CLAUDE, _fake_cli(stdout=json.dumps(violating)))

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON, response_schema=SCHEMA)
    )

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert not result.final_json_present


@pytest.mark.parametrize(
    "tool",
    [
        AIToolID.CLAUDE,
        AIToolID.CODEX,
        AIToolID.CURSOR,
        AIToolID.COPILOT,
        AIToolID.OPENCODE,
        AIToolID.ANTIGRAVITY_CLI,
    ],
)
def test_malformed_native_output_is_invalid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: AIToolID
) -> None:
    outputs = AbstractAITool.get(tool).capabilities().headless.native_outputs
    machine_readable = [output for output in outputs if output is not HeadlessNativeOutput.TEXT]
    if not machine_readable:
        pytest.skip(f"{tool.value} declares no machine-readable native output")
    adapter = _adapter(monkeypatch, tool, _fake_cli(stdout="not json at all\n", exit_code=0))

    result = adapter.run_headless_session(_request(tmp_path, native_output=machine_readable[0]))

    assert result.exit_code == 0
    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert not result.final_json_present


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-only")
def test_timeout_kills_the_process_tree_and_keeps_partial_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_pid_file = tmp_path / "grandchild.pid"
    adapter = _adapter(
        monkeypatch,
        AIToolID.CODEX,
        _fake_cli(
            stdout=_jsonl(CODEX_EVENTS[:1]),
            sleep_seconds=60.0,
            child_pid_file=child_pid_file,
        ),
    )

    started = time.monotonic()
    result = adapter.run_headless_session(_request(tmp_path, timeout_seconds=1.5))
    elapsed = time.monotonic() - started

    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert elapsed < 20
    assert result.final_text is None and not result.final_json_present
    # Safe provenance streamed before the deadline survives the timeout.
    assert result.thread_id == CODEX_EVENTS[0]["thread_id"]
    assert HeadlessEventKind.PROGRESS in [event.kind for event in result.events]
    assert any("deadline" in warning for warning in result.warnings)
    assert result.events[-1].kind is HeadlessEventKind.TERMINAL
    _assert_prompt_is_private(result)

    grandchild = int(child_pid_file.read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except OSError:
            break
        time.sleep(0.05)
    else:  # pragma: no cover - only on a cleanup regression
        os.kill(grandchild, signal.SIGKILL)
        pytest.fail("the managed timeout left an owned descendant running")
