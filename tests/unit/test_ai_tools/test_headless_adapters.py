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
from crossby.ai_tools.antigravity_cli import (
    AntigravityCLIAdapter,
    _stream_input,
    _whole_run_timeout_seconds,
)
from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.claude import ClaudeAdapter
from crossby.ai_tools.headless import (
    HeadlessRequestError,
    HeadlessTransportError,
    HeadlessUnsupportedError,
)
from crossby.models.ai import (
    AIToolID,
    EffortLevel,
    HeadlessEventKind,
    HeadlessFailureReason,
    HeadlessInteractionMode,
    HeadlessNativeOutput,
    HeadlessPromptTransport,
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

# Recorded on 1.0.109 (and the same frames on 1.0.105) with
# ``--output-format json``, reduced to the frames the adapter reads plus the
# prompt echo it must keep private.
COPILOT_SESSION = "f3efa8dd-8d2e-4688-913b-90a34e98d90c"


def _copilot_events(
    *,
    session: str = COPILOT_SESSION,
    answer: str | None = "OK",
    exit_code: int = 0,
    premium: int = 1,
    denied: int = 0,
    error: str | None = None,
) -> str:
    """Copilot's JSONL wire for one prompt-mode run."""
    frames: list[dict[str, Any]] = [
        {"type": "session.start", "data": {"sessionId": session, "copilotVersion": "1.0.109"}},
        {"type": "user.message", "data": {"content": PROMPT, "transformedContent": PROMPT}},
        {"type": "assistant.turn_start", "data": {"turnId": "0"}},
    ]
    for index in range(denied):
        frames += [
            {
                "type": "assistant.message",
                "data": {
                    "model": "gpt-6-luna",
                    "content": "",
                    "toolRequests": [{"name": "apply_patch", "arguments": PROMPT}],
                },
            },
            {
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": f"call_{index}",
                    "success": False,
                    "error": {
                        "message": "Permission to run this tool was denied due to the following "
                        "rules: `write`",
                        "code": "denied",
                    },
                },
            },
        ]
    if error is not None:
        frames.append({"type": "session.error", "data": {"errorType": "query", "message": error}})
    if answer is not None:
        frames.append(
            {
                "type": "assistant.message",
                "data": {
                    "phase": "final_answer",
                    "model": "gpt-6-luna",
                    "content": answer,
                    "toolRequests": [],
                },
            }
        )
    frames += [
        {"type": "assistant.turn_end", "data": {"turnId": "0"}},
        {
            "type": "result",
            "sessionId": session,
            "exitCode": exit_code,
            "usage": {"premiumRequests": premium, "totalApiDurationMs": 1614},
        },
    ]
    return "".join(f"{json.dumps(frame)}\n" for frame in frames)


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


def _agy_result(payload: dict[str, Any]) -> str:
    """agy's stream-json wire carrying only its final result envelope."""
    return _jsonl(({"event": "result", "result": payload},))


def _fake_cli(
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    stdin_record: Path | None = None,
    sleep_seconds: float = 0.0,
    child_pid_file: Path | None = None,
    close_stdin_early: bool = False,
) -> list[str]:
    """Build a real child process that replays a recorded native payload."""
    program = [
        "import os,sys,time",
        f"stdin_record={str(stdin_record)!r}",
        f"child_pid_file={str(child_pid_file)!r}",
    ]
    if close_stdin_early:
        program.append("sys.stdin.close()")
    else:
        program += [
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


def _periodic_jsonl_cli(frame: dict[str, Any]) -> list[str]:
    """Build a child that keeps emitting one JSONL frame until idle expiry."""
    return [
        sys.executable,
        "-c",
        "\n".join(
            (
                "import json,sys,time",
                f"frame={frame!r}",
                "for _ in range(50):",
                "    sys.stdout.write(json.dumps(frame) + '\\n')",
                "    sys.stdout.flush()",
                "    time.sleep(0.01)",
            )
        ),
    ]


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
    if tool is AIToolID.CLAUDE:
        # The recorded fixtures carry a real session ID; act as if the fake CLI
        # honored the ``--session-id`` crossby pins on every new session.
        from crossby.ai_tools import claude as claude_module

        monkeypatch.setattr(claude_module, "_new_session_id", lambda: CLAUDE_RESULT["session_id"])
    if tool is AIToolID.COPILOT:
        # Likewise for the --session-id crossby pins on every new Copilot session.
        from crossby.ai_tools import copilot as copilot_module

        monkeypatch.setattr(copilot_module, "_new_session_id", lambda: COPILOT_SESSION)
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


def _assert_prompt_is_private(result: HeadlessSessionResult, prompt: str = PROMPT) -> None:
    """No untrusted normalized event or diagnostic may echo the caller's prompt."""
    for event in result.events:
        assert event.message is None or prompt not in event.message
        assert prompt not in json.dumps(event.payload or {})
    for diagnostic in (*result.warnings, *result.denials):
        assert prompt not in diagnostic


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
    assert result.denials == (
        "Claude Code reported 1 permission denials under --permission-prompts none",
    )


@pytest.mark.parametrize("invalid_is_error", [None, 0, "false"], ids=["null", "zero", "string"])
def test_claude_rejects_non_boolean_error_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_is_error: object
) -> None:
    adapter = _adapter(
        monkeypatch,
        AIToolID.CLAUDE,
        _fake_cli(stdout=json.dumps({**CLAUDE_RESULT, "is_error": invalid_is_error})),
    )

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON)
    )

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("authoritative boolean is_error" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


def test_claude_rejects_missing_error_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = {**CLAUDE_RESULT}
    del envelope["is_error"]
    adapter = _adapter(
        monkeypatch,
        AIToolID.CLAUDE,
        _fake_cli(stdout=json.dumps(envelope)),
    )

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON)
    )

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("authoritative boolean is_error" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


@pytest.mark.parametrize(
    "status_updates",
    [
        {"terminal_reason": PROMPT},
        {"terminal_reason": None, "subtype": PROMPT},
        {"terminal_reason": "x" * 513},
    ],
    ids=["terminal-reason-prompt", "subtype-prompt", "terminal-reason-oversized"],
)
def test_claude_rejects_unsafe_native_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status_updates: dict[str, str | None]
) -> None:
    adapter = _adapter(
        monkeypatch,
        AIToolID.CLAUDE,
        _fake_cli(stdout=json.dumps({**CLAUDE_RESULT, **status_updates})),
    )

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON)
    )

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.native_status is None
    assert result.final_text is None and not result.final_json_present
    assert any("unsafe native status" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


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


def test_claude_denial_reporting_keeps_only_the_native_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    noisy = json.loads(json.dumps(CLAUDE_RESULT))
    noisy["permission_denials"] = [
        {
            "tool_name": f"untrusted command with {PROMPT} and credential-{index}",
            "tool_use_id": f"toolu_{index}",
            "tool_input": {"command": PROMPT},
        }
        for index in range(200)
    ]
    adapter = _adapter(monkeypatch, AIToolID.CLAUDE, _fake_cli(stdout=json.dumps(noisy)))

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON)
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.denials == (
        "Claude Code reported 200 permission denials under --permission-prompts none",
    )
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
    ("tool", "native_output", "frames", "prompt"),
    [
        (AIToolID.CLAUDE, HeadlessNativeOutput.JSONL, CLAUDE_STREAM, "result"),
        (AIToolID.CODEX, HeadlessNativeOutput.TEXT, CODEX_EVENTS, "codex"),
        (AIToolID.CURSOR, HeadlessNativeOutput.JSONL, CURSOR_STREAM, "result"),
    ],
)
def test_streaming_progress_labels_do_not_reject_matching_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: AIToolID,
    native_output: HeadlessNativeOutput,
    frames: tuple[dict[str, Any], ...],
    prompt: str,
) -> None:
    adapter = _adapter(monkeypatch, tool, _fake_cli(stdout=_jsonl(frames)))

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=native_output, prompt=prompt)
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert any(event.kind is HeadlessEventKind.PROGRESS for event in result.events)


@pytest.mark.parametrize(
    ("tool", "native_output", "unknown_frame"),
    [
        (AIToolID.CLAUDE, HeadlessNativeOutput.JSONL, {"type": "unknown"}),
        (AIToolID.CODEX, HeadlessNativeOutput.TEXT, {"type": "unknown"}),
        (AIToolID.CURSOR, HeadlessNativeOutput.JSONL, {"type": "unknown"}),
        (AIToolID.OPENCODE, HeadlessNativeOutput.TEXT, {"type": "unknown"}),
        (AIToolID.ANTIGRAVITY_CLI, HeadlessNativeOutput.JSONL, {"event": "unknown"}),
    ],
)
def test_unrecognized_streaming_frames_do_not_extend_idle_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: AIToolID,
    native_output: HeadlessNativeOutput,
    unknown_frame: dict[str, str],
) -> None:
    adapter = _adapter(monkeypatch, tool, _periodic_jsonl_cli(unknown_frame))

    result = adapter.run_headless_session(
        _request(
            tmp_path,
            native_output=native_output,
            timeout_seconds=1.0,
            idle_timeout_seconds=0.05,
        )
    )

    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert any("idle deadline" in warning for warning in result.warnings)


@pytest.mark.parametrize("native_output", [HeadlessNativeOutput.JSON, HeadlessNativeOutput.JSONL])
def test_json_decoder_value_errors_are_invalid_native_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_output: HeadlessNativeOutput,
) -> None:
    max_digits = sys.get_int_max_str_digits()
    if max_digits == 0:
        pytest.skip("the interpreter's JSON integer digit limit is disabled")
    malformed = '{"number":' + "9" * (max_digits + 1) + "}"
    adapter = _adapter(monkeypatch, AIToolID.CLAUDE, _fake_cli(stdout=malformed))

    result = adapter.run_headless_session(_request(tmp_path, native_output=native_output))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    _assert_prompt_is_private(result)


def test_codex_schema_decoder_value_error_is_invalid_native_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    max_digits = sys.get_int_max_str_digits()
    if max_digits == 0:
        pytest.skip("the interpreter's JSON integer digit limit is disabled")
    events = list(CODEX_EVENTS)
    events[2] = {
        **events[2],
        "item": {
            **events[2]["item"],
            "text": '{"answer":' + "9" * (max_digits + 1) + "}",
        },
    }
    adapter = _adapter(monkeypatch, AIToolID.CODEX, _fake_cli(stdout=_jsonl(tuple(events))))

    result = adapter.run_headless_session(_request(tmp_path, response_schema=SCHEMA))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    _assert_prompt_is_private(result)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX filesystem encoding")
@pytest.mark.parametrize(
    ("tool", "native_output"),
    [
        (AIToolID.CLAUDE, HeadlessNativeOutput.TEXT),
        (AIToolID.CURSOR, HeadlessNativeOutput.JSON),
        (AIToolID.COPILOT, HeadlessNativeOutput.TEXT),
    ],
)
def test_a_surrogate_escaped_prompt_reaches_stdin_as_its_original_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: AIToolID,
    native_output: HeadlessNativeOutput,
) -> None:
    """A filename decoded by os.fsdecode is written back as the bytes it came from."""
    record = tmp_path / "stdin.bin"
    program = (
        f"import sys; open({str(record)!r}, 'wb').write(sys.stdin.buffer.read()); "
        "sys.stdout.write('{}')"
    )
    adapter = _adapter(monkeypatch, tool, [sys.executable, "-c", program])

    adapter.run_headless_session(
        _request(
            tmp_path, prompt=os.fsdecode(b"read filename-\xff now"), native_output=native_output
        )
    )

    assert record.read_bytes() == b"read filename-\xff now"


def test_antigravity_escapes_any_prompt_into_valid_json() -> None:
    prompt = os.fsdecode(b"read filename-\xff now") + " \U0001f680 and a lone \ud800"
    line = _stream_input(prompt)
    assert line.isascii() and line.endswith("\n")
    assert json.loads(line)["message"]["content"] == prompt


def test_an_unencodable_prompt_fails_the_transport_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = tmp_path / "started.txt"
    adapter = _adapter(monkeypatch, AIToolID.CURSOR, _fake_cli(stdout="{}", stdin_record=started))

    with pytest.raises(HeadlessTransportError, match="cannot be written"):
        adapter.run_headless_session(_request(tmp_path, prompt="a lone \ud800 surrogate"))

    assert not started.exists()


@pytest.mark.parametrize(
    ("tool", "stdout"),
    [
        (AIToolID.CLAUDE, "OK\n"),
        (AIToolID.CODEX, _jsonl(CODEX_EVENTS)),
        (AIToolID.OPENCODE, _jsonl(OPENCODE_EVENTS)),
    ],
)
def test_stdin_delivery_failure_rejects_valid_native_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: AIToolID,
    stdout: str,
) -> None:
    adapter = _adapter(
        monkeypatch,
        tool,
        _fake_cli(stdout=stdout, close_stdin_early=True),
    )

    with pytest.raises(HeadlessTransportError, match="deliver the complete prompt") as raised:
        adapter.run_headless_session(_request(tmp_path, prompt="p" * (1024 * 1024)))

    assert raised.value.partial_result.is_partial
    assert raised.value.partial_result.final_text is None
    assert not raised.value.partial_result.final_json_present


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


@pytest.mark.parametrize(
    ("tool", "frames"),
    [
        (
            AIToolID.CLAUDE,
            (
                *CLAUDE_STREAM,
                {
                    "type": "assistant",
                    "session_id": CLAUDE_RESULT["session_id"],
                    "message": {"role": "assistant", "content": []},
                },
            ),
        ),
        (
            AIToolID.CURSOR,
            (
                *CURSOR_STREAM,
                {"type": "system", "session_id": CURSOR_RESULT["session_id"], "subtype": "init"},
            ),
        ),
        (
            AIToolID.ANTIGRAVITY_CLI,
            (*AGY_STREAM, {"event": "step_update", "conversation_id": AGY_CONVERSATION}),
        ),
    ],
    ids=["claude", "cursor", "antigravity"],
)
def test_streaming_adapters_reject_frames_after_final_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: AIToolID,
    frames: tuple[dict[str, Any], ...],
) -> None:
    adapter = _adapter(
        monkeypatch,
        tool,
        _fake_cli(stdout=_jsonl(frames)),
    )

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSONL)
    )

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("frame after its final result envelope" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


@pytest.mark.parametrize(
    ("prompt", "metadata_name", "expected_metadata"),
    [
        ("stop", "native_status", "stop"),
        (OPENCODE_SESSION, "session_id", OPENCODE_SESSION),
    ],
    ids=["native-status", "provenance-id"],
)
def test_opencode_accepts_canonical_metadata_matching_the_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt: str,
    metadata_name: str,
    expected_metadata: str,
) -> None:
    adapter = _adapter(monkeypatch, AIToolID.OPENCODE, _fake_cli(stdout=_jsonl(OPENCODE_EVENTS)))

    result = adapter.run_headless_session(_request(tmp_path, prompt=prompt))

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert getattr(result, metadata_name) == expected_metadata
    _assert_prompt_is_private(result, prompt)


@pytest.mark.parametrize(
    ("tool", "frames"),
    [
        (
            AIToolID.CLAUDE,
            (
                CLAUDE_STREAM[0],
                {**CLAUDE_STREAM[1], "session_id": "different-session"},
                CLAUDE_STREAM[2],
            ),
        ),
        (
            AIToolID.CURSOR,
            (
                CURSOR_STREAM[0],
                {**CURSOR_STREAM[1], "session_id": "different-session"},
                CURSOR_STREAM[2],
                CURSOR_STREAM[3],
            ),
        ),
        (
            AIToolID.CODEX,
            (
                CODEX_EVENTS[0],
                {"type": "thread.started", "thread_id": "different-thread"},
                *CODEX_EVENTS[1:],
            ),
        ),
        (
            AIToolID.OPENCODE,
            (
                OPENCODE_EVENTS[0],
                {**OPENCODE_EVENTS[1], "sessionID": "different-session"},
                OPENCODE_EVENTS[2],
            ),
        ),
        (
            AIToolID.ANTIGRAVITY_CLI,
            (
                AGY_STREAM[0],
                {**AGY_STREAM[1], "conversation_id": "different-conversation"},
                AGY_STREAM[2],
            ),
        ),
    ],
)
def test_streaming_adapters_reject_conflicting_native_provenance(
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
    assert any("conflicting session provenance" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


def test_claude_rejects_conflicting_provenance_in_unterminated_final_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = {**CLAUDE_STREAM[0], "session_id": "session-a"}
    final = {**CLAUDE_RESULT, "session_id": "session-b"}
    stdout = f"{json.dumps(initial)}\n{json.dumps(final)}"
    assert not stdout.endswith("\n")
    adapter = _adapter(monkeypatch, AIToolID.CLAUDE, _fake_cli(stdout=stdout))

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSONL)
    )

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("conflicting session provenance" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


def test_claude_text_output_returns_the_json_envelope_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # TEXT runs --output-format json so the envelope's checks still apply.
    request = _request(tmp_path)
    command = _command(AIToolID.CLAUDE, request)
    payload = {**CLAUDE_RESULT, "result": "OK"}
    adapter = _adapter(monkeypatch, AIToolID.CLAUDE, _fake_cli(stdout=json.dumps(payload)))

    result = adapter.run_headless_session(request)

    assert command[command.index("--output-format") + 1] == "json"
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "OK" and not result.final_json_present
    assert result.session_id == CLAUDE_RESULT["session_id"]


def test_claude_text_output_fails_on_a_native_error_that_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(
        monkeypatch, AIToolID.CLAUDE, _fake_cli(stdout=json.dumps(CLAUDE_ERROR_RESULT))
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.final_text is None
    assert any("native error result" in warning for warning in result.warnings)


def test_antigravity_text_output_returns_the_json_envelope_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # TEXT is read from the stream-json result, so the conversation ID and status
    # are checked.
    request = _request(tmp_path)
    payload = {key: value for key, value in AGY_RESULT.items() if key != "json_schema"}
    payload["response"] = "OK\n"
    adapter = _adapter(
        monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=_agy_result(payload))
    )

    result = adapter.run_headless_session(request)

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "OK"
    assert result.conversation_id == AGY_CONVERSATION


@pytest.mark.parametrize(
    ("update", "status"),
    [
        pytest.param({"conversation_id": None}, HeadlessTerminalStatus.INVALID_OUTPUT, id="no-id"),
        pytest.param({"status": "ERROR"}, HeadlessTerminalStatus.FAILED, id="native-failure"),
    ],
)
def test_antigravity_text_output_applies_the_envelope_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    update: dict[str, Any],
    status: HeadlessTerminalStatus,
) -> None:
    payload = {key: value for key, value in AGY_RESULT.items() if key != "json_schema"}
    payload = {key: value for key, value in {**payload, **update}.items() if value is not None}
    adapter = _adapter(
        monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=_agy_result(payload))
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is status
    assert result.final_text is None


def test_session_mcp_servers_are_refused_where_they_would_be_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    servers = {"docs": {"command": "npx", "args": ["-y", "docs-server"]}}

    # A tool with no verified way to take them at all.
    cursor = _adapter(monkeypatch, AIToolID.CURSOR, _fake_cli(stdout="{}"))
    with pytest.raises(HeadlessUnsupportedError, match="MCP servers for one unattended session"):
        cursor.run_headless_session(_request(tmp_path, mcp_servers=servers))

    # Antigravity CLI reads them from a file in its folder: without that file (which a
    # persona folder's provisioning writes) the session would start without them.
    agy = _adapter(monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=_jsonl(AGY_STREAM)))
    with pytest.raises(HeadlessRequestError, match="does not define docs"):
        agy.run_headless_session(_request(tmp_path, mcp_servers=servers))
    (tmp_path / ".agents").mkdir()
    (tmp_path / ".agents" / "mcp_config.json").write_text(json.dumps({"mcpServers": servers}))
    assert agy.run_headless_session(_request(tmp_path, mcp_servers=servers)).status is (
        HeadlessTerminalStatus.SUCCEEDED
    )

    # Codex merges a session server into a configured one of the same name, so a clash
    # is refused; a server under another name is unaffected.
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "config.toml").write_text('[mcp_servers.docs]\ncommand = "old"\nenabled = false\n')
    monkeypatch.setenv("CODEX_HOME", str(home))
    clashing = _adapter(monkeypatch, AIToolID.CODEX, _fake_cli(stdout=_jsonl(CODEX_EVENTS)))
    with pytest.raises(HeadlessRequestError, match="already has an MCP server named docs"):
        clashing.run_headless_session(_request(tmp_path, mcp_servers=servers))
    (home / "config.toml").write_text('[mcp_servers.other]\ncommand = "old"\n')

    # Codex CLI takes them on its command line, which only the unattended mode builds:
    # a brokered session would start without them and still report them as running.
    codex = _adapter(monkeypatch, AIToolID.CODEX, _fake_cli(stdout=_jsonl(CODEX_EVENTS)))
    with pytest.raises(HeadlessUnsupportedError, match="MCP servers for one brokered session"):
        codex.run_headless_session(
            _request(
                tmp_path, mcp_servers=servers, interaction_mode=HeadlessInteractionMode.BROKERED
            ),
            interaction_handler=lambda *_args, **_kwargs: None,  # type: ignore[arg-type]
        )
    assert codex.run_headless_session(_request(tmp_path, mcp_servers=servers)).status is (
        HeadlessTerminalStatus.SUCCEEDED
    )


def test_tool_steps_are_counted_as_each_tool_reports_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def steps(tool: AIToolID, stdout: str) -> int | None:
        adapter = _adapter(monkeypatch, tool, _fake_cli(stdout=stdout))
        result = adapter.run_headless_session(_request(tmp_path))
        assert result.status is HeadlessTerminalStatus.SUCCEEDED
        return result.usage.tool_steps if result.usage is not None else None

    # Claude Code: one model turn per tool round, plus the answer.
    assert steps(AIToolID.CLAUDE, json.dumps({**CLAUDE_RESULT, "num_turns": 1})) == 0
    assert steps(AIToolID.CLAUDE, json.dumps({**CLAUDE_RESULT, "num_turns": 4})) == 3
    without = {key: value for key, value in CLAUDE_RESULT.items() if key != "num_turns"}
    assert steps(AIToolID.CLAUDE, json.dumps(without)) is None

    # Codex CLI: every completed item that is not a message, reasoning or a to-do list.
    assert steps(AIToolID.CODEX, _jsonl(CODEX_EVENTS)) == 0
    command = {
        "id": "item_1",
        "type": "command_execution",
        "command": "cat a",
        "status": "completed",
    }
    started = {"type": "item.started", "item": {**command, "status": "in_progress"}}
    ran = (
        *CODEX_EVENTS[:2],
        {"type": "item.completed", "item": {"id": "item_0", "type": "reasoning", "text": "…"}},
        started,
        {"type": "item.completed", "item": command},
        {"type": "item.completed", "item": {**command, "id": "item_2", "type": "mcp_tool_call"}},
        *CODEX_EVENTS[2:],
    )
    assert steps(AIToolID.CODEX, _jsonl(ran)) == 2

    # Antigravity CLI: each tool step once, however many updates it sent.
    assert steps(AIToolID.ANTIGRAVITY_CLI, _jsonl(AGY_STREAM)) == 0

    def tool(index: int, state: str) -> dict[str, Any]:
        update = {
            "conversation_id": AGY_CONVERSATION,
            "step_index": index,
            "state": state,
            "step_type": "tool",
            "tool_name": "read_file",
        }
        return {"event": "step_update", "step_update": update}

    used = (
        *AGY_STREAM[:-1],
        tool(2, "ACTIVE"),
        tool(2, "DONE"),
        tool(3, "ACTIVE"),
        tool(3, "ERROR"),
        AGY_STREAM[-1],
    )
    assert steps(AIToolID.ANTIGRAVITY_CLI, _jsonl(used)) == 2


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


def test_codex_rejects_item_events_after_its_terminal_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    late_agent_message = {
        "type": "item.completed",
        "item": {"id": "item_late", "type": "agent_message", "text": "late response"},
    }
    adapter = _adapter(
        monkeypatch,
        AIToolID.CODEX,
        _fake_cli(stdout=_jsonl((*CODEX_EVENTS, late_agent_message)), exit_code=0),
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("frame after its terminal turn event" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        pytest.param(
            (
                {"id": "item_0", "type": "agent_message", "text": "I'll read calc.py."},
                {"id": "item_1", "type": "command_execution", "command": "cat calc.py"},
                {"id": "item_2", "type": "reasoning", "text": "add subtracts"},
                {"id": "item_3", "type": "agent_message", "text": "Yes, add subtracts."},
                {"id": "item_4", "type": "agent_message", "text": "Use a + b."},
            ),
            "Yes, add subtracts.\nUse a + b.",
            id="drops-commentary-before-tools",
        ),
        pytest.param(
            (
                {"id": "item_0", "type": "agent_message", "text": "Editing now."},
                {"id": "item_1", "type": "file_change", "changes": []},
            ),
            "Editing now.",
            id="keeps-messages-when-run-ends-on-a-tool",
        ),
    ],
)
def test_codex_final_text_is_the_answer_after_the_last_tool_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    items: tuple[dict[str, Any], ...],
    expected: str,
) -> None:
    thread_started, turn_started, _message, turn_completed = CODEX_EVENTS
    events = (
        thread_started,
        turn_started,
        *({"type": "item.completed", "item": item} for item in items),
        turn_completed,
    )
    adapter = _adapter(monkeypatch, AIToolID.CODEX, _fake_cli(stdout=_jsonl(events), exit_code=0))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == expected


def test_codex_rejects_error_events_after_its_terminal_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(
        monkeypatch,
        AIToolID.CODEX,
        _fake_cli(
            stdout=_jsonl((*CODEX_EVENTS, {"type": "error", "message": "late error"})), exit_code=0
        ),
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("frame after its terminal turn event" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


def test_codex_rejects_thread_events_after_its_terminal_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(
        monkeypatch,
        AIToolID.CODEX,
        _fake_cli(stdout=_jsonl((*CODEX_EVENTS, CODEX_EVENTS[0])), exit_code=0),
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("frame after its terminal turn event" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


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


@pytest.mark.parametrize("invalid_is_error", [None, 0, "false"], ids=["null", "zero", "string"])
def test_cursor_rejects_non_boolean_error_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_is_error: object
) -> None:
    adapter = _adapter(
        monkeypatch,
        AIToolID.CURSOR,
        _fake_cli(stdout=json.dumps({**CURSOR_RESULT, "is_error": invalid_is_error})),
    )

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON)
    )

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("authoritative boolean is_error" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


def test_cursor_rejects_a_missing_error_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = {**CURSOR_RESULT}
    del envelope["is_error"]
    adapter = _adapter(
        monkeypatch,
        AIToolID.CURSOR,
        _fake_cli(stdout=json.dumps(envelope)),
    )

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON)
    )

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("authoritative boolean is_error" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


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


def test_copilot_jsonl_answer_session_and_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(monkeypatch, AIToolID.COPILOT, _fake_cli(stdout=_copilot_events()))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "OK"
    assert result.session_id == COPILOT_SESSION
    assert result.usage is not None
    assert result.usage.premium_requests == 1
    assert [item.model for item in result.usage.model_breakdown] == ["gpt-6-luna"]
    assert result.usage.input_tokens is None  # Copilot reports no token counts
    assert any("--allow-all-tools" in warning for warning in result.warnings)
    # The user.message frame echoes the prompt.
    _assert_prompt_is_private(result)


def test_copilot_counts_the_tool_calls_it_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _copilot_events(answer="I couldn't write the file.", denied=2)
    adapter = _adapter(monkeypatch, AIToolID.COPILOT, _fake_cli(stdout=events))

    result = adapter.run_headless_session(_request(tmp_path, read_only=True))

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    # Tool-request messages have no content: the answer is the final one.
    assert result.final_text == "I couldn't write the file."
    assert result.denials == ("GitHub Copilot CLI denied 2 tool call(s) under its rules",)
    _assert_prompt_is_private(result)


def test_an_empty_marked_final_answer_is_never_replaced_by_commentary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commentary = json.dumps(
        {
            "type": "assistant.message",
            "data": {"model": "gpt-6-luna", "content": "I will inspect the repository."},
        }
    )
    events = _copilot_events(answer="   ")
    turn = '{"type": "assistant.turn_start"'
    events = events.replace(turn, f"{commentary}\n{turn}", 1)
    adapter = _adapter(monkeypatch, AIToolID.COPILOT, _fake_cli(stdout=events))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None


def test_copilot_answer_without_the_final_answer_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _copilot_events().replace('"phase": "final_answer", ', "")
    adapter = _adapter(monkeypatch, AIToolID.COPILOT, _fake_cli(stdout=events))

    assert adapter.run_headless_session(_request(tmp_path)).final_text == "OK"


@pytest.mark.parametrize(
    ("events", "warning"),
    [
        (
            _copilot_events(session="0c1e8b2a-6f4d-4a3b-9c2d-1e5f7a9b3c4d"),
            "reported a different session",
        ),
        (_copilot_events() + _copilot_events(), "emitted events after its result"),
        (_copilot_events().rsplit("\n", 2)[0] + "\n", "ended without its result event"),
        ("not json\n", "malformed JSONL"),
    ],
)
def test_copilot_rejects_an_inconsistent_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: str, warning: str
) -> None:
    adapter = _adapter(monkeypatch, AIToolID.COPILOT, _fake_cli(stdout=events))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert any(warning in item for item in result.warnings)


def test_copilot_native_failure_is_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = _copilot_events(
        answer=None, exit_code=1, premium=0, error='Execution failed: Model "x" is not available.'
    )
    adapter = _adapter(monkeypatch, AIToolID.COPILOT, _fake_cli(stdout=events, exit_code=1))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.native_status == "exit_1"
    assert result.failure_reason is HeadlessFailureReason.MODEL_UNAVAILABLE
    assert all("not available" not in warning for warning in result.warnings)


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


@pytest.mark.parametrize(
    "late_step",
    [OPENCODE_EVENTS[0], OPENCODE_EVENTS[-1]],
    ids=["next-step", "duplicate-stop"],
)
def test_opencode_rejects_step_events_after_terminal_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late_step: dict[str, Any]
) -> None:
    events = (*OPENCODE_EVENTS, late_step)
    adapter = _adapter(
        monkeypatch, AIToolID.OPENCODE, _fake_cli(stdout=_jsonl(events), exit_code=0)
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.exit_code == 0
    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("frame after its terminal stop event" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


def test_opencode_rejects_text_events_after_terminal_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    late_text = {
        "type": "text",
        "sessionID": OPENCODE_SESSION,
        "part": {"id": "prt_late", "type": "text", "text": "late response"},
    }
    adapter = _adapter(
        monkeypatch,
        AIToolID.OPENCODE,
        _fake_cli(stdout=_jsonl((*OPENCODE_EVENTS, late_text)), exit_code=0),
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("frame after its terminal stop event" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


def test_opencode_rejects_unknown_frames_after_terminal_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    late_frame = {"type": "error", "sessionID": OPENCODE_SESSION}
    adapter = _adapter(
        monkeypatch,
        AIToolID.OPENCODE,
        _fake_cli(stdout=_jsonl((*OPENCODE_EVENTS, late_frame)), exit_code=0),
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("frame after its terminal stop event" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


def test_opencode_rejects_prompt_bearing_session_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = json.loads(json.dumps(OPENCODE_EVENTS))
    for event in events:
        event["sessionID"] = PROMPT
    adapter = _adapter(monkeypatch, AIToolID.OPENCODE, _fake_cli(stdout=_jsonl(events)))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("unsafe session provenance" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


def test_opencode_nonterminal_finish_without_stop_is_invalid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete = list(OPENCODE_EVENTS)
    incomplete[-1] = json.loads(json.dumps(incomplete[-1]))
    incomplete[-1]["part"]["reason"] = "tool-calls"
    adapter = _adapter(
        monkeypatch, AIToolID.OPENCODE, _fake_cli(stdout=_jsonl(tuple(incomplete)), exit_code=0)
    )

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.exit_code == 0
    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None and not result.final_json_present
    assert any("without a terminal stop event" in warning for warning in result.warnings)
    _assert_prompt_is_private(result)


def test_antigravity_json_envelope_validates_the_schema_echo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(
        monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=_agy_result(AGY_RESULT))
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
        monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=_agy_result(tampered))
    )

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON, response_schema=SCHEMA)
    )

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert any("echo the requested schema" in warning for warning in result.warnings)


def test_antigravity_rejects_a_schema_echo_with_an_oversized_integer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tampered = json.loads(json.dumps(AGY_RESULT))
    tampered["json_schema"] = f'{{"value": {"9" * 4_301}}}'
    adapter = _adapter(
        monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=_agy_result(tampered))
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
        monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=_agy_result(AGY_WAITING))
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
def test_unattended_sessions_deliver_exactly_the_prompt_on_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: AIToolID,
    native_output: HeadlessNativeOutput,
) -> None:
    """Every native CLI reads its prompt, and only its prompt, from a closed pipe.

    It never inherits the caller's terminal, and never takes the prompt on argv.
    """
    delivered = tmp_path / f"{tool.value}-stdin.txt"
    adapter = _adapter(monkeypatch, tool, _fake_cli(stdout="{}", stdin_record=delivered))

    adapter.run_headless_session(_request(tmp_path, native_output=native_output))

    assert adapter.capabilities().headless.prompt_transport is HeadlessPromptTransport.STDIN
    # agy reads one stream-json message; every other tool reads the raw prompt.
    expected = _stream_input(PROMPT) if tool is AIToolID.ANTIGRAVITY_CLI else PROMPT
    assert delivered.read_text() == expected


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


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX filesystem encoding")
def test_posix_surrogate_escaped_argv_uses_filesystem_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = AbstractAITool.get(AIToolID.CODEX)
    trusted_dir = os.fsdecode(b"/tmp/crossby-trusted-\xff")
    monkeypatch.setattr(
        adapter,
        "_headless_argv_for_validation",
        lambda _request, **_kwargs: ["codex", "--add-dir", trusted_dir],
    )

    adapter._validate_headless_argv(_request(tmp_path))


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

    assert max(len(argument.encode("utf-8")) for argument in argv) < base_mod._MAX_HEADLESS_ARGUMENT
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

    assert max(len(argument.encode("utf-8")) for argument in argv) < base_mod._MAX_HEADLESS_ARGUMENT
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


@pytest.mark.parametrize("model", ["gemini-3.1-pro-medium", "managed-experiment-high"])
def test_antigravity_model_only_suffix_that_would_be_rewritten_fails_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    started = tmp_path / "started.txt"
    adapter = _adapter(
        monkeypatch,
        AIToolID.ANTIGRAVITY_CLI,
        _fake_cli(stdout="{}", stdin_record=started),
    )

    with pytest.raises(HeadlessRequestError, match="model-encoded reasoning effort"):
        adapter.run_headless_session(_request(tmp_path, model=model))

    assert not started.exists()


@pytest.mark.parametrize("model", ["gemini-3.8-flash-low", "gpt-oss-120b-medium"])
def test_antigravity_valid_model_only_suffix_is_preserved(tmp_path: Path, model: str) -> None:
    command = _command(
        AIToolID.ANTIGRAVITY_CLI,
        _request(tmp_path, model=model),
        print_timeout_seconds=30,
    )

    assert command[command.index("--model") + 1] == model


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


@pytest.mark.parametrize("entry_point", ["runtime", "preflight"])
@pytest.mark.parametrize(
    ("tool", "updates", "error_type", "message"),
    [
        (
            AIToolID.CLAUDE,
            {"native_output": HeadlessNativeOutput.TEXT, "response_schema": SCHEMA},
            HeadlessUnsupportedError,
            "structured_output",
        ),
        (
            AIToolID.CURSOR,
            {
                "model": "auto",
                "effort": EffortLevel.HIGH,
                "native_output": HeadlessNativeOutput.JSON,
            },
            HeadlessRequestError,
            "selected model is not known",
        ),
    ],
)
def test_static_adapter_requirements_fail_before_version_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
    tool: AIToolID,
    updates: dict[str, Any],
    error_type: type[Exception],
    message: str,
) -> None:
    adapter = AbstractAITool.get(tool)
    version_probes = 0

    def probe(**_kwargs: Any) -> BinaryVersion:
        nonlocal version_probes
        version_probes += 1
        raise AssertionError("static adapter requirements must fail before version probing")

    monkeypatch.setattr(adapter, "_detect_headless_version", probe)

    with pytest.raises(error_type, match=message):
        request = _request(tmp_path, **updates)
        if entry_point == "runtime":
            adapter.run_headless_session(request)
        else:
            adapter.preflight_headless_session(request)

    assert version_probes == 0


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
                native_output=HeadlessNativeOutput.JSON,
                response_schema={"type": "object", "description": "x" * 33_000},
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
        lambda argv: "x" * base_mod._MAX_HEADLESS_WINDOWS_COMMAND_LINE if len(argv) > 1 else "x",
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
            _request(tmp_path, model="m" * 33_000, native_output=native_output)
        )

    assert version_probes == 0


@pytest.mark.parametrize("platform", ["darwin", "win32"])
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
def test_prompt_size_is_never_bounded_by_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: AIToolID, platform: str
) -> None:
    """A prompt far beyond every argv limit stays off argv and passes validation."""
    prompt = "x" * 1_000_000
    request = _request(tmp_path, prompt=prompt)
    adapter = AbstractAITool.get(tool)
    monkeypatch.setattr(base_mod.sys, "platform", platform)

    argv = adapter._headless_argv_for_validation(request)

    assert argv is not None
    assert all(prompt not in argument for argument in argv)
    adapter._validate_headless_argv(request)  # raises on any OS limit


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


def test_cursor_text_output_reads_the_stream_json_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Cursor's own text format has no terminal status, so TEXT rides the
    # stream-json wire and returns the result envelope's text.
    request = _request(tmp_path, native_output=HeadlessNativeOutput.TEXT)
    command = _command(AIToolID.CURSOR, request)
    adapter = _adapter(monkeypatch, AIToolID.CURSOR, _fake_cli(stdout=_jsonl(CURSOR_STREAM)))

    result = adapter.run_headless_session(request)

    assert command[:4] == ["agent", "--print", "--output-format", "stream-json"]
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "OK" and not result.final_json_present
    assert result.session_id == CURSOR_RESULT["session_id"]
    assert any(event.kind is HeadlessEventKind.PROGRESS for event in result.events)


def _cursor_assistant(text: str) -> dict[str, Any]:
    return {
        "type": "assistant",
        "session_id": CURSOR_RESULT["session_id"],
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


@pytest.mark.parametrize(
    ("result_text", "expected"),
    [
        pytest.param(
            "I'll read calc.py.Yes, add subtracts.",
            "Yes, add subtracts.",
            id="drops-commentary-before-tools",
        ),
        pytest.param(
            "Cursor's own summary.",
            "Cursor's own summary.",
            id="keeps-a-result-that-does-not-end-with-the-answer",
        ),
    ],
)
def test_cursor_text_output_is_the_answer_after_the_last_tool_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result_text: str, expected: str
) -> None:
    session = CURSOR_RESULT["session_id"]
    frames = (
        CURSOR_STREAM[0],
        _cursor_assistant("I'll read calc.py."),
        {"type": "tool_call", "subtype": "started", "session_id": session},
        {"type": "tool_call", "subtype": "completed", "session_id": session},
        _cursor_assistant("Yes, add subtracts."),
        {**CURSOR_RESULT, "result": result_text},
    )
    adapter = _adapter(monkeypatch, AIToolID.CURSOR, _fake_cli(stdout=_jsonl(frames)))

    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.TEXT)
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == expected


def test_cursor_headless_argv_trusts_the_workspace_without_forcing_edits(
    tmp_path: Path,
) -> None:
    command = _command(AIToolID.CURSOR, _request(tmp_path, native_output=HeadlessNativeOutput.JSON))

    assert command[:4] == ["agent", "--print", "--output-format", "json"]
    # --trust answers the workspace-trust gate; per-tool approval stays gated.
    assert "--trust" in command
    assert command[command.index("--sandbox") + 1] == "enabled"
    assert "--force" not in command and "--yolo" not in command
    # The prompt is piped on stdin, never a positional argument.
    assert PROMPT not in command


def test_copilot_headless_argv_removes_the_question_tool(tmp_path: Path) -> None:
    command = _command(AIToolID.COPILOT, _request(tmp_path))

    # The run reads Copilot's JSONL events, not its plain text.
    assert command[command.index("--output-format") + 1] == "json"

    # Prompt mode reads the piped prompt: no --prompt argument at all.
    assert command[0] == "copilot"
    assert "--prompt" not in command and PROMPT not in command
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

    # One stream-json message on stdin; --print= cannot swallow the next flag.
    assert command[:6] == [
        "agy",
        "--print=",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
    ]
    assert PROMPT not in command
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
    payload = {key: value for key, value in AGY_RESULT.items() if key != "json_schema"}
    adapter = _adapter(
        monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=_agy_result(payload))
    )
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
