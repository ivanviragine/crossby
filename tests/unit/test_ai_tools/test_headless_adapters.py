"""Unattended headless coverage for the six terminal adapters.

The native payloads below combine live captures from the exact CLI builds each
adapter declares as its ``verified_version``, documented Codex success events,
and a synthetic Antigravity waiting state. Their exact provenance is recorded
in ``docs/unattended-headless-verification.md``:

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
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools import base as base_mod
from crossby.ai_tools import plan_process
from crossby.ai_tools.antigravity_cli import AntigravityCLIAdapter, _whole_run_timeout_seconds
from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.claude import ClaudeAdapter
from crossby.ai_tools.headless import HeadlessRequestError, HeadlessUnsupportedError
from crossby.models.ai import (
    AIToolID,
    EffortLevel,
    HeadlessEventKind,
    HeadlessInteractionMode,
    HeadlessNativeOutput,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
    PlanCommandPolicy,
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


def test_windows_managed_transport_claims_and_closes_the_complete_process_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pass

    process = FakeProcess()
    closed: list[int] = []
    resumed: list[FakeProcess] = []
    tree = plan_process._WindowsProcessTree(1234, lambda handle: closed.append(handle))
    monkeypatch.setattr(plan_process, "_is_windows", lambda: True)
    monkeypatch.setattr(plan_process, "_create_windows_process_tree", lambda _proc: tree)
    monkeypatch.setattr(plan_process, "_resume_windows_process", lambda proc: resumed.append(proc))

    assert plan_process.process_tree_popen_kwargs() == {"creationflags": 0x00000004}
    plan_process.own_process_tree(process)  # type: ignore[arg-type]
    plan_process.kill_process_group(process)  # type: ignore[arg-type]

    assert resumed == [process]
    assert closed == [1234]


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


@pytest.mark.parametrize(
    ("tool", "native_output", "envelope"),
    [
        (AIToolID.CLAUDE, HeadlessNativeOutput.JSON, CLAUDE_ERROR_RESULT),
        (AIToolID.CURSOR, HeadlessNativeOutput.JSON, {**CURSOR_RESULT, "is_error": True}),
    ],
)
def test_native_error_diagnostics_do_not_include_response_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: AIToolID,
    native_output: HeadlessNativeOutput,
    envelope: dict[str, Any],
) -> None:
    native_error_text = "untrusted native response text that must stay private"
    payload = {**envelope, "result": native_error_text}
    adapter = _adapter(monkeypatch, tool, _fake_cli(stdout=json.dumps(payload), exit_code=0))

    result = adapter.run_headless_session(_request(tmp_path, native_output=native_output))

    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.final_text is None and not result.final_json_present
    assert all(native_error_text not in warning for warning in result.warnings)


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


@pytest.mark.parametrize(
    ("tool", "frames"),
    [
        (AIToolID.CLAUDE, (*CLAUDE_STREAM, CLAUDE_ERROR_RESULT)),
        (AIToolID.CURSOR, (*CURSOR_STREAM, {**CURSOR_RESULT, "is_error": True})),
        (AIToolID.ANTIGRAVITY_CLI, (*AGY_STREAM, {"event": "result", "result": AGY_WAITING})),
    ],
)
def test_streaming_adapters_reject_repeated_final_result_envelopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: AIToolID,
    frames: tuple[dict[str, Any], ...],
) -> None:
    adapter = _adapter(monkeypatch, tool, _fake_cli(stdout=_jsonl(frames)))

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSONL)
    )

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("multiple final result envelopes" in warning for warning in result.warnings)
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
    failed_events = json.loads(json.dumps(CODEX_FAILED_EVENTS))
    failed_events[2]["message"] = f"failed command includes {PROMPT}"
    failed_events[3]["error"]["message"] = f"failed response includes {PROMPT}"
    adapter = _adapter(
        monkeypatch,
        AIToolID.CODEX,
        _fake_cli(stdout=_jsonl(tuple(failed_events)), exit_code=0),
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.exit_code == 0
    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.native_status == "turn.failed"
    assert "Codex CLI reported 1 native error event(s)." in result.warnings
    assert "Codex CLI turn failed." in result.warnings
    assert all("failed command" not in warning for warning in result.warnings)
    assert all("failed response" not in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


@pytest.mark.parametrize(
    "terminal_frames",
    [
        (
            {"type": "turn.completed"},
            {"type": "turn.completed"},
        ),
        (
            {"type": "turn.failed"},
            {"type": "turn.completed"},
        ),
    ],
    ids=["duplicated", "conflicting"],
)
def test_codex_multiple_terminal_turn_events_are_invalid_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_frames: tuple[dict[str, str], dict[str, str]],
) -> None:
    events = (*CODEX_EVENTS[:3], *terminal_frames)
    adapter = _adapter(monkeypatch, AIToolID.CODEX, _fake_cli(stdout=_jsonl(events), exit_code=0))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.exit_code == 0
    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None
    assert any("multiple terminal turn events" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


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


def test_copilot_policy_failure_withholds_native_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_diagnostic = f"Error: Access denied by policy settings for {PROMPT}"
    adapter = _adapter(
        monkeypatch,
        AIToolID.COPILOT,
        _fake_cli(stderr=native_diagnostic, exit_code=1),
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.final_text is None
    assert any("diagnostics that are withheld" in warning for warning in result.warnings)
    assert all(native_diagnostic not in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


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


def test_opencode_accumulates_usage_from_each_model_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_finish = json.loads(json.dumps(OPENCODE_EVENTS[-1]))
    first_finish["part"]["reason"] = "tool-calls"
    first_finish["part"]["tokens"] = {
        "total": 40,
        "input": 30,
        "output": None,
        "cache": {"read": 5},
    }
    final_finish = json.loads(json.dumps(OPENCODE_EVENTS[-1]))
    final_finish["part"]["tokens"] = {
        "total": 10,
        "input": None,
        "output": 4,
        "cache": {"read": None},
    }
    events = (*OPENCODE_EVENTS[:-1], first_finish, final_finish)
    adapter = _adapter(monkeypatch, AIToolID.OPENCODE, _fake_cli(stdout=_jsonl(events)))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.usage is not None
    assert result.usage.total_tokens == 50
    assert result.usage.input_tokens == 30
    assert result.usage.output_tokens == 4
    assert result.usage.cached_tokens == 5
    assert result.usage.session_id == OPENCODE_SESSION


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


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX execve limits")
def test_claude_oversized_schema_argument_fails_before_version_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = ClaudeAdapter()
    version_probes = 0

    def probe(**_kwargs: Any) -> BinaryVersion:
        nonlocal version_probes
        version_probes += 1
        raise AssertionError("oversized schema must fail before version probing")

    monkeypatch.setattr(adapter, "_detect_headless_version", probe)

    with pytest.raises(HeadlessRequestError, match="native argv argument"):
        adapter.run_headless_session(
            _request(
                tmp_path,
                native_output=HeadlessNativeOutput.JSON,
                response_schema={"type": "object", "description": "x" * 120_000},
            )
        )

    assert version_probes == 0


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX execve limits")
def test_antigravity_aggregate_argv_and_environment_overflow_fails_before_version_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = AntigravityCLIAdapter()
    version_probes = 0

    def probe(**_kwargs: Any) -> BinaryVersion:
        nonlocal version_probes
        version_probes += 1
        raise AssertionError("oversized argv and environment must fail before version probing")

    request = _request(
        tmp_path,
        prompt="p" * 80_000,
        native_output=HeadlessNativeOutput.JSON,
        response_schema={"type": "object", "description": "s" * 80_000},
    )
    argv = adapter._headless_argv_for_validation(request)
    exec_size = base_mod._headless_posix_exec_size(argv)
    monkeypatch.setattr(
        base_mod,
        "_headless_posix_exec_limit",
        lambda: exec_size + base_mod._HEADLESS_POSIX_EXEC_SAFETY_MARGIN - 1,
    )
    monkeypatch.setattr(adapter, "_detect_headless_version", probe)

    with pytest.raises(HeadlessRequestError, match="native argv and environment"):
        adapter.run_headless_session(request)

    assert (
        max(len(argument.encode("utf-8")) for argument in argv)
        < base_mod._MAX_HEADLESS_ARGUMENT_PROMPT
    )
    assert version_probes == 0


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX execve limits")
def test_opencode_oversized_model_argument_fails_before_version_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = AbstractAITool.get(AIToolID.OPENCODE)
    version_probes = 0

    def probe(**_kwargs: Any) -> BinaryVersion:
        nonlocal version_probes
        version_probes += 1
        raise AssertionError("oversized model must fail before version probing")

    monkeypatch.setattr(adapter, "_detect_headless_version", probe)

    with pytest.raises(HeadlessRequestError, match="native argv argument"):
        adapter.run_headless_session(_request(tmp_path, model="m" * 120_001))

    assert version_probes == 0


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX execve limits")
def test_codex_aggregate_argv_overflow_fails_before_version_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = AbstractAITool.get(AIToolID.CODEX)
    version_probes = 0

    def probe(**_kwargs: Any) -> BinaryVersion:
        nonlocal version_probes
        version_probes += 1
        raise AssertionError("oversized argv must fail before version probing")

    request = _request(
        tmp_path,
        trusted_dirs=tuple(tmp_path / f"trusted-{index}" for index in range(8)),
    )
    argv = adapter._headless_argv_for_validation(request)
    exec_size = base_mod._headless_posix_exec_size(argv)
    monkeypatch.setattr(
        base_mod,
        "_headless_posix_exec_limit",
        lambda: exec_size + base_mod._HEADLESS_POSIX_EXEC_SAFETY_MARGIN - 1,
    )
    monkeypatch.setattr(adapter, "_detect_headless_version", probe)

    with pytest.raises(HeadlessRequestError, match="native argv and environment"):
        adapter.run_headless_session(request)

    assert (
        max(len(argument.encode("utf-8")) for argument in argv)
        < base_mod._MAX_HEADLESS_ARGUMENT_PROMPT
    )
    assert version_probes == 0


def test_codex_oversized_model_argument_fails_before_version_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = AbstractAITool.get(AIToolID.CODEX)
    version_probes = 0

    def probe(**_kwargs: Any) -> BinaryVersion:
        nonlocal version_probes
        version_probes += 1
        raise AssertionError("oversized model must fail before version probing")

    monkeypatch.setattr(adapter, "_detect_headless_version", probe)

    with pytest.raises(HeadlessRequestError, match="native argv argument"):
        adapter.run_headless_session(_request(tmp_path, model="m" * 120_001))

    assert version_probes == 0


@pytest.mark.parametrize(
    ("tool", "native_output"),
    [
        (AIToolID.CURSOR, HeadlessNativeOutput.JSON),
        (AIToolID.ANTIGRAVITY_CLI, HeadlessNativeOutput.TEXT),
    ],
)
def test_model_encoded_effort_without_model_fails_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: AIToolID,
    native_output: HeadlessNativeOutput,
) -> None:
    started = tmp_path / "started.txt"
    adapter = _adapter(monkeypatch, tool, _fake_cli(stdout="{}", stdin_record=started))

    with pytest.raises(HeadlessRequestError, match="explicit model"):
        adapter.run_headless_session(
            _request(tmp_path, effort=EffortLevel.HIGH, native_output=native_output)
        )

    assert not started.exists()


@pytest.mark.parametrize(
    ("model", "effort", "message"),
    [
        ("auto", EffortLevel.HIGH, "selected model is not known"),
        ("fake-future-model-high", EffortLevel.HIGH, "unknown model"),
        ("claude-opus-5-high", EffortLevel.MAX, "it resolves to"),
        ("claude-opus-4-8[effort=invalid]", EffortLevel.HIGH, "invalid effort overrides"),
        ("gpt-5.4-low[effort=high]", EffortLevel.HIGH, "conflicting effort encodings"),
        ("claude-opus-4-8[effort=low]", EffortLevel.HIGH, "which encodes effort='low'"),
    ],
)
def test_cursor_unrepresentable_effort_fails_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    effort: EffortLevel,
    message: str,
) -> None:
    started = tmp_path / "started.txt"
    adapter = _adapter(monkeypatch, AIToolID.CURSOR, _fake_cli(stdout="{}", stdin_record=started))

    with pytest.raises(HeadlessRequestError, match=message):
        adapter.run_headless_session(
            _request(
                tmp_path,
                model=model,
                effort=effort,
                native_output=HeadlessNativeOutput.JSON,
            )
        )

    assert not started.exists()


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("claude-opus-4-8[effort=high]", EffortLevel.HIGH),
        ("gpt-5.3-codex", EffortLevel.MEDIUM),
    ],
)
def test_cursor_exactly_representable_effort_is_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    effort: EffortLevel,
) -> None:
    adapter = _adapter(monkeypatch, AIToolID.CURSOR, _fake_cli(stdout=json.dumps(CURSOR_RESULT)))

    result = adapter.run_headless_session(
        _request(
            tmp_path,
            model=model,
            effort=effort,
            native_output=HeadlessNativeOutput.JSON,
        )
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED


def test_antigravity_conflicting_model_encoded_effort_fails_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = tmp_path / "started.txt"
    adapter = _adapter(
        monkeypatch,
        AIToolID.ANTIGRAVITY_CLI,
        _fake_cli(stdout="{}", stdin_record=started),
    )

    with pytest.raises(HeadlessRequestError, match="conflicting reasoning effort"):
        adapter.run_headless_session(
            _request(
                tmp_path,
                model="gemini-3.8-flash-low",
                effort=EffortLevel.HIGH,
            )
        )

    assert not started.exists()


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("gemini-3.1-pro", EffortLevel.MEDIUM),
        ("claude-sonnet-4-6", EffortLevel.HIGH),
    ],
)
def test_antigravity_unrepresentable_effort_fails_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    effort: EffortLevel,
) -> None:
    started = tmp_path / "started.txt"
    adapter = _adapter(
        monkeypatch,
        AIToolID.ANTIGRAVITY_CLI,
        _fake_cli(stdout="{}", stdin_record=started),
    )

    with pytest.raises(
        HeadlessRequestError, match="cannot preserve the requested reasoning effort"
    ):
        adapter.run_headless_session(_request(tmp_path, model=model, effort=effort))

    assert not started.exists()


def test_codex_headless_argv_metadata_probe_receives_deadline_and_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[float | None, threading.Event | None]] = []
    cancelled = threading.Event()

    def metadata_dirs(
        _working_dir: Path,
        *,
        deadline: float | None,
        cancel_event: threading.Event | None,
    ) -> list[Path]:
        captured.append((deadline, cancel_event))
        return []

    monkeypatch.setattr("crossby.ai_tools.codex.outside_root_git_metadata_dirs", metadata_dirs)
    adapter = AbstractAITool.get(AIToolID.CODEX)

    argv = adapter._headless_argv_for_validation(  # type: ignore[attr-defined]
        _request(tmp_path),
        deadline=123.0,
        cancel_event=cancelled,
    )

    assert argv[-1] == "-"
    assert captured == [(123.0, cancelled)]


def test_antigravity_windows_command_overflow_fails_before_version_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = AntigravityCLIAdapter()
    version_probes = 0

    def probe(**_kwargs: Any) -> BinaryVersion:
        nonlocal version_probes
        version_probes += 1
        raise AssertionError("oversized command must fail before version probing")

    monkeypatch.setattr(base_mod.sys, "platform", "win32")
    monkeypatch.setattr(adapter, "_detect_headless_version", probe)

    with pytest.raises(HeadlessRequestError, match="native command line"):
        adapter.run_headless_session(
            _request(
                tmp_path,
                prompt="x" * 30_000,
                native_output=HeadlessNativeOutput.JSON,
                response_schema={"type": "object", "description": "x" * 3_000},
            )
        )

    assert version_probes == 0


def test_windows_command_limit_counts_terminating_nul_before_version_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 32,767-unit rendered command still needs its CreateProcess NUL."""
    adapter = AntigravityCLIAdapter()
    version_probes = 0

    def probe(**_kwargs: Any) -> BinaryVersion:
        nonlocal version_probes
        version_probes += 1
        raise AssertionError("full Windows command line must fail before version probing")

    monkeypatch.setattr(base_mod.sys, "platform", "win32")
    monkeypatch.setattr(
        base_mod.subprocess,
        "list2cmdline",
        lambda _argv: "x" * base_mod._MAX_HEADLESS_WINDOWS_COMMAND_LINE,
    )
    monkeypatch.setattr(adapter, "_detect_headless_version", probe)

    with pytest.raises(HeadlessRequestError, match="32768-unit native command line"):
        adapter.run_headless_session(_request(tmp_path, native_output=HeadlessNativeOutput.JSON))

    assert version_probes == 0


@pytest.mark.parametrize(
    ("tool", "native_output"),
    [
        (AIToolID.CURSOR, HeadlessNativeOutput.JSON),
        (AIToolID.COPILOT, HeadlessNativeOutput.TEXT),
    ],
)
def test_windows_complete_argument_argv_overflow_fails_before_version_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: AIToolID,
    native_output: HeadlessNativeOutput,
) -> None:
    adapter = AbstractAITool.get(tool)
    version_probes = 0

    def probe(**_kwargs: Any) -> BinaryVersion:
        nonlocal version_probes
        version_probes += 1
        raise AssertionError("oversized command must fail before version probing")

    monkeypatch.setattr(base_mod.sys, "platform", "win32")
    monkeypatch.setattr(adapter, "_detect_headless_version", probe)

    with pytest.raises(HeadlessRequestError, match="native command line"):
        adapter.run_headless_session(
            _request(
                tmp_path,
                prompt="x" * 30_000,
                model="m" * 3_000,
                native_output=native_output,
            )
        )

    assert version_probes == 0


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


@pytest.mark.skipif(os.name != "posix", reason="process cleanup verification uses POSIX PIDs")
def test_timeout_tears_down_a_child_started_after_popen_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deadline during Popen cannot strand a child returned after cleanup."""
    pid_file = tmp_path / "late-child.pid"
    program = [
        sys.executable,
        "-c",
        (
            "import os, time\n"
            "from pathlib import Path\n"
            f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(60)\n"
        ),
    ]
    adapter = _adapter(monkeypatch, AIToolID.CODEX, program)
    native_popen = subprocess.Popen

    def delayed_popen(*args: Any, **kwargs: Any) -> Any:
        child = native_popen(*args, **kwargs)
        deadline = time.monotonic() + 2
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        return child

    monkeypatch.setattr("crossby.ai_tools.headless_cli.subprocess.Popen", delayed_popen)

    result = adapter.run_headless_session(_request(tmp_path, timeout_seconds=0.05))

    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert pid_file.exists()
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.01)
    else:  # pragma: no cover - only on a cleanup regression
        os.kill(pid, signal.SIGKILL)
        pytest.fail("Popen returned after cleanup and left its child running")


# --- exact unattended argv assembly ------------------------------------------
#
# The suite above replaces ``_headless_command`` with a deterministic stub so it
# can drive a real child process. These cases cover the opposite half: the real
# flags each adapter hands the native CLI, including the ones that make a run
# unattended in the first place.


def _command(tool: AIToolID, request: HeadlessSessionRequest, **kwargs: Any) -> list[str]:
    """Return the argv one adapter would hand its native CLI for ``request``."""
    # _headless_command is defined per adapter, not on AbstractAITool.
    build: Callable[..., list[str]] = AbstractAITool.get(tool)._headless_command  # type: ignore[attr-defined]
    return build(request, **kwargs)


def test_claude_headless_argv_denies_prompts_and_keeps_the_prompt_off_argv(
    tmp_path: Path,
) -> None:
    command = _command(
        AIToolID.CLAUDE,
        _request(tmp_path, native_output=HeadlessNativeOutput.JSONL, response_schema=SCHEMA),
    )

    assert command == [
        "claude",
        "--print",
        "--output-format",
        "stream-json",
        "--permission-prompts",
        "none",
        "--verbose",
        "--json-schema",
        json.dumps(SCHEMA, separators=(",", ":")),
    ]
    assert PROMPT not in command


def test_claude_headless_argv_normalizes_the_model_to_dashed_form(tmp_path: Path) -> None:
    # Crossby's internal spelling is dotted; the Claude CLI only accepts dashed.
    command = _command(AIToolID.CLAUDE, _request(tmp_path, model="claude-haiku-4.5"))

    assert command[command.index("--model") + 1] == "claude-haiku-4-5"


def test_claude_headless_argv_scopes_the_command_policy_to_the_session(tmp_path: Path) -> None:
    command = _command(
        AIToolID.CLAUDE,
        _request(
            tmp_path,
            trusted_dirs=(tmp_path / "shared",),
            command_policy=PlanCommandPolicy(allowed_commands=["pytest:*"]),
        ),
    )

    assert command[command.index("--add-dir") + 1] == str(tmp_path / "shared")
    assert command[command.index("--allowedTools") + 1] == "Bash(pytest:*)"
    # An empty --setting-sources keeps ambient allowlists from widening the policy.
    assert command[command.index("--setting-sources") + 1] == ""


def test_codex_headless_argv_pins_sandbox_and_network(tmp_path: Path) -> None:
    command = _command(AIToolID.CODEX, _request(tmp_path), schema_path=None)

    assert command[:6] == ["codex", "exec", "--json", "--color", "never", "--sandbox"]
    assert command[6] == "workspace-write"
    assert "sandbox_workspace_write.network_access=false" in command
    # A bare trailing "-" is what keeps the prompt on stdin.
    assert command[-1] == "-"
    assert PROMPT not in command


def test_codex_headless_argv_opens_the_sandbox_only_when_asked(tmp_path: Path) -> None:
    command = _command(
        AIToolID.CODEX,
        _request(tmp_path, sandbox=False, network_access=True),
        schema_path=None,
    )

    assert command[command.index("--sandbox") + 1] == "danger-full-access"
    # Outside a sandbox there is no workspace-write policy to pin.
    assert not any(part.startswith("sandbox_workspace_write.") for part in command)


def test_codex_headless_argv_passes_the_schema_file(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    command = _command(
        AIToolID.CODEX, _request(tmp_path, response_schema=SCHEMA), schema_path=schema_path
    )

    assert command[command.index("--output-schema") + 1] == str(schema_path)


def test_cursor_headless_argv_trusts_the_workspace_without_forcing_edits(
    tmp_path: Path,
) -> None:
    command = _command(AIToolID.CURSOR, _request(tmp_path, native_output=HeadlessNativeOutput.JSON))

    assert command[:4] == ["agent", "--print", "--output-format", "json"]
    # --trust answers the workspace-trust gate; per-tool approval stays gated.
    assert "--trust" in command
    assert command[command.index("--sandbox") + 1] == "enabled"
    assert "--force" not in command and "--yolo" not in command
    # Cursor takes its prompt as the trailing positional argument.
    assert command[-1] == PROMPT


def test_copilot_headless_argv_removes_the_question_tool(tmp_path: Path) -> None:
    command = _command(AIToolID.COPILOT, _request(tmp_path))

    assert command[:3] == ["copilot", "--prompt", PROMPT]
    # -s keeps stdout to the response; --no-ask-user removes the ask_user tool
    # so the agent cannot stall on a question nobody can answer.
    assert "-s" in command
    assert "--no-ask-user" in command


def test_copilot_headless_argv_normalizes_the_model_to_dotted_form(tmp_path: Path) -> None:
    # The Copilot CLI rejects Claude IDs unless the version uses dotted notation.
    command = _command(AIToolID.COPILOT, _request(tmp_path, model="claude-haiku-4-5"))

    assert command[command.index("--model") + 1] == "claude-haiku-4.5"


def test_opencode_headless_argv_never_auto_approves_or_exposes_the_prompt(tmp_path: Path) -> None:
    command = _command(AIToolID.OPENCODE, _request(tmp_path, model="anthropic/claude-haiku-4-5"))

    assert command[:6] == ["opencode", "run", "--format", "json", "--log-level", "ERROR"]
    # --auto would auto-approve permissions; OpenCode's own noninteractive
    # refusal is the intended behavior instead.
    assert "--auto" not in command
    assert command[command.index("--model") + 1] == "anthropic/claude-haiku-4-5"
    assert PROMPT not in command


def test_antigravity_headless_argv_carries_the_schema_and_print_timeout(
    tmp_path: Path,
) -> None:
    command = _command(
        AIToolID.ANTIGRAVITY_CLI,
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON, response_schema=SCHEMA),
        print_timeout_seconds=120,
    )

    assert command[:4] == ["agy", "--print", PROMPT, "--output-format"]
    assert command[4] == "json"
    assert command[command.index("--print-timeout") + 1] == "120s"
    assert command[command.index("--json-schema") + 1] == json.dumps(SCHEMA, separators=(",", ":"))


def test_antigravity_print_timeout_uses_the_whole_run_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``agy --print-timeout`` bounds the whole run, not one idle stretch.

    Deriving it from the idle budget would abort a healthy streaming turn as
    soon as the shorter idle window elapsed.
    """
    recorded: list[int] = []
    adapter = _adapter(monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout="OK\n"))
    stub: Callable[..., list[str]] = type(adapter)._headless_command  # type: ignore[attr-defined]

    def record(self: AbstractAITool, request: HeadlessSessionRequest, **kwargs: Any) -> list[str]:
        recorded.append(int(kwargs["print_timeout_seconds"]))
        return list(stub(self, request, **kwargs))

    monkeypatch.setattr(type(adapter), "_headless_command", record, raising=True)

    result = adapter.run_headless_session(
        _request(tmp_path, timeout_seconds=120.0, idle_timeout_seconds=5.0)
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert recorded and recorded[0] > 5


def test_antigravity_whole_run_timeout_rounds_fractional_seconds_up() -> None:
    assert _whole_run_timeout_seconds(deadline=120.0, now=0.001) == 120
    assert _whole_run_timeout_seconds(deadline=120.0, now=120.0) == 1
