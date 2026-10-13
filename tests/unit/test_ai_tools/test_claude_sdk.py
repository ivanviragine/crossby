"""Actual pinned SDK control dispatch over deterministic owned child processes."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools.claude import ClaudeAdapter
from crossby.ai_tools.claude_sdk import SDK_VERSION, brokered_profile, sdk_command
from crossby.ai_tools.headless import HeadlessTransportError, HeadlessUnsupportedError
from crossby.models.ai import (
    HeadlessEventKind,
    HeadlessInteractionMode,
    HeadlessNativeOutput,
    HeadlessSessionRequest,
    HeadlessTerminalStatus,
    PlanCommandPolicy,
    PlanInteractionOutcome,
    SessionInteraction,
    SessionInteractionResponse,
)
from crossby.utils.versioning import BinaryVersion

_CHILD = r"""
import json, sys, time, subprocess, os
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
record = Path(sys.argv[2])
mode = config.get('mode', 'normal')
session = config.get('session', 'session-exact')
def emit(data):
    print(json.dumps(data), flush=True)
def save(data):
    with record.open('a') as f:
        f.write(json.dumps(data) + '\n')
save({'type':'fixture_environment', 'entrypoint':os.environ.get('CLAUDE_CODE_ENTRYPOINT'),
      'version':os.environ.get('CLAUDE_AGENT_SDK_VERSION'), 'nested':os.environ.get('CLAUDECODE'),
      'pwd':os.environ.get('PWD')})
background_stage = 0
def result():
    global background_stage
    frame = {'type': 'result', 'subtype': config.get('subtype', 'success'),
             'is_error': config.get('is_error', False), 'duration_ms': 1,
             'duration_api_ms': 1, 'num_turns': 1, 'session_id': session,
             'result': 'Completed', 'usage': {'input_tokens': 5, 'output_tokens': 7},
             'total_cost_usd': 0.25}
    frame.update(config.get('result', {}))
    if mode == 'background_shell':
        emit({'type':'system', 'subtype':'task_started', 'task_id':'shell-exact',
              'task_type':config.get('task_type', 'local_bash'), 'description':'Long lived work',
              'uuid':'shell-start', 'session_id':session})
    if mode == 'background':
        if background_stage == 0:
            emit({'type':'system', 'subtype':'task_started', 'task_id':'bg-exact',
                  'task_type':config.get('task_type', 'local_agent'),
                  'description':'Background', 'uuid':'task-start', 'session_id':session})
            if config.get('with_state', True):
                emit({'type':'system', 'subtype':'session_state_changed',
                      'sdk_host_only':True, 'state':'running', 'session_id':session})
        else:
            frame.update(config.get('background_result', {}))
            frame.update({'origin':{'kind':'task-notification'}, 'result':'Background answer',
                          'usage':{'input_tokens':100, 'output_tokens':200}})
    if mode in {'lifecycle_running', 'lifecycle_running_exit',
                'native_failure_wait', 'native_failure_stubborn'}:
        emit({'type':'system', 'subtype':'session_state_changed',
              'sdk_host_only':True, 'state':'running', 'session_id':session})
    if mode == 'background' and background_stage and config.get('idle_before_result'):
        emit({'type':'system', 'subtype':'session_state_changed', 'sdk_host_only':True,
              'state':'idle', 'session_id':session})
    emit(frame)
    if mode == 'background':
        if background_stage == 0:
            background_stage = 1
            emit({'type':'system', 'subtype':'background_tasks_changed',
                  'tasks':[], 'session_id':session})
            time.sleep(0.03)
            if config.get('completion', 'notification') == 'notification':
                emit({'type':'system', 'subtype':'task_notification', 'task_id':'bg-exact',
                      'status':'completed', 'output_file':'out', 'summary':'Done',
                      'uuid':'task-end', 'session_id':session})
            else:
                emit({'type':'system', 'subtype':'task_updated', 'task_id':'bg-exact',
                      'patch':{'status':'killed'}, 'uuid':'task-end', 'session_id':session})
            emit({'type':'assistant', 'session_id':session,
                  'message':{'model':'claude-opus',
                             'content':[{'type':'text', 'text':'Follow-up'}]}})
            emit({'type':'control_request', 'request_id':'background-wire',
                  'request':{'subtype':'can_use_tool', 'tool_use_id':'background-tool',
                             'tool_name':'Bash', 'input':{'command':'pwd'}}})
        elif config.get('with_state', True):
            emit({'type':'system', 'subtype':'session_state_changed', 'sdk_host_only':True,
                  'state':'idle', 'session_id':session})
    if mode in {'lifecycle', 'lifecycle_running'}:
        time.sleep(config.get('lifecycle_delay', 0))
        emit({'type':'system', 'subtype':'session_state_changed',
              'sdk_host_only':config.get('host_only', True), 'state':config.get('state', 'idle'),
              'session_id':config.get('lifecycle_session', session)})
    if mode == 'trailing':
        emit({'type': 'assistant', 'session_id': session, 'message': {'content': []}})
for line in sys.stdin:
    frame = json.loads(line)
    save(frame)
    if frame['type'] == 'control_request':
        subtype = frame['request']['subtype']
        if subtype == 'initialize' and mode == 'startup_error':
            result()
            break
        if subtype == 'initialize' and mode == 'startup_stall':
            continue
        if mode == 'native_failure_stubborn' and subtype == 'interrupt':
            continue
        emit({'type': 'control_response', 'response': {
              'subtype': 'success', 'request_id': frame['request_id'], 'response': {}}})
    elif frame['type'] == 'user':
        if mode != 'no_session':
            emit({'type': 'system', 'subtype': 'init', 'session_id': session})
        if mode == 'stall':
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
            save({'descendant': child.pid})
            continue
        if mode == 'eof':
            break
        request = config.get('request')
        if request:
            emit({'type': 'control_request', 'request_id': config.get('request_id', 'wire-exact'),
                  'request': {'subtype': 'can_use_tool',
                  'tool_name': config.get('tool', 'AskUserQuestion'),
                  'tool_use_id': config.get('tool_id', 'tool-exact'), 'input': request,
                  'title': 'Display is not executable', 'blocked_path': '/blocked',
                  'agent_id': 'agent-exact', **config.get('native_context', {})}})
            if mode == 'native_cancel':
                time.sleep(config.get('cancel_delay', 0))
                emit({'type': 'control_cancel_request', 'request_id': 'wire-exact'})
                result()
            if mode == 'duplicate':
                emit({'type': 'control_request', 'request_id': 'wire-exact',
                      'request': {'subtype': 'can_use_tool', 'tool_name': 'AskUserQuestion',
                      'tool_use_id': 'tool-exact', 'input': request}})
        else:
            emit({'type': 'stream_event', 'uuid': 'stream-exact', 'session_id': session,
                  'event': {'type': 'content_block_delta'}})
            emit({'type': 'assistant', 'session_id': session,
                  'message': {'model': 'claude-opus', 'content': [
                    {'type': 'text', 'text': 'Private assistant'},
                    {'type': 'tool_use', 'id': 'used', 'name': 'Read',
                     'input': {'file_path':'private'}}]}})
            result()
            if mode == 'lifecycle_running_exit':
                break
    elif frame['type'] == 'control_response':
        if mode == 'reuse':
            emit({'type': 'control_request', 'request_id': 'wire-exact',
                  'request': {'subtype': 'can_use_tool', 'tool_name': 'AskUserQuestion',
                  'tool_use_id': 'tool-exact', 'input': config['request']}})
        else:
            result()
if mode == 'native_failure_stubborn':
    time.sleep(60)
save({'type':'fixture_eof', 'background_stage':background_stage})
sys.exit(config.get('exit', 0))
"""


@pytest.fixture
def sdk_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("claude_agent_sdk")
    assert metadata.version("claude-agent-sdk") == SDK_VERSION
    child = tmp_path / "child.py"
    child.write_text(_CHILD)
    config = tmp_path / "config.json"
    record = tmp_path / "record.jsonl"
    monkeypatch.setattr(
        "crossby.utils.versioning.detect_binary_version_info",
        lambda *_args, **_kwargs: BinaryVersion((2, 1, 263), "2.1.263"),
    )
    monkeypatch.setattr(
        "crossby.ai_tools.claude_sdk.sdk_command",
        lambda _request: [sys.executable, str(child), str(config), str(record)],
    )

    def run(values: dict[str, Any], handler: Any = None, **kwargs: Any) -> Any:
        config.write_text(json.dumps(values))
        request_values = kwargs.pop("request_values", {})
        request = HeadlessSessionRequest(
            prompt=request_values.pop("prompt", "private task"),
            working_dir=tmp_path,
            interaction_mode=HeadlessInteractionMode.BROKERED,
            timeout_seconds=5,
            **request_values,
        )
        return ClaudeAdapter().run_headless_session(
            request,
            interaction_handler=handler
            or (lambda _i: SessionInteractionResponse(outcome=PlanInteractionOutcome.DENIED)),
            **kwargs,
        )

    return run, record


def _questions(*, multiple: bool = False) -> dict[str, Any]:
    return {
        "questions": [
            {
                "question": "Native question?",
                "header": "Header",
                "multiSelect": multiple,
                "options": [
                    {"label": "One, exact", "description": "Description"},
                    {"label": "Two", "description": "Second"},
                ],
            }
        ]
    }


def _records(record: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in record.read_text().splitlines()]


def test_sdk_actual_dispatch_multiselect_and_free_text(sdk_child: Any) -> None:
    run, record = sdk_child
    native = _questions(multiple=True)
    native["questions"].append(
        {**native["questions"][0], "question": "Second?", "multiSelect": False}
    )
    seen = []

    def handler(interaction: SessionInteraction) -> SessionInteractionResponse:
        seen.append(interaction)
        if interaction.question_id == "Second?":
            return SessionInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED, answer="Custom"
            )
        return SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED, option_ids=("One, exact", "Two")
        )

    result = run({"request": native}, handler)
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.session_id == "session-exact"
    assert len(seen) == 2
    assert seen[0].question_id == "Native question?"
    assert seen[0].request_id == "wire-exact"
    assert seen[0].session_id == "session-exact"
    assert seen[0].allow_multiple and seen[0].allow_other
    assert seen[0].options[0].description == "Description"
    reply = next(f["response"] for f in _records(record) if f["type"] == "control_response")
    assert reply["request_id"] == "wire-exact"
    assert reply["response"]["updatedInput"]["answers"] == {
        "Native question?": ["One, exact", "Two"],
        "Second?": "Custom",
    }
    assert reply["response"]["updatedInput"]["questions"] == native["questions"]


@pytest.mark.parametrize(
    "outcome", [PlanInteractionOutcome.APPROVED, PlanInteractionOutcome.DENIED]
)
def test_permission_native_authority(sdk_child: Any, outcome: PlanInteractionOutcome) -> None:
    run, record = sdk_child
    seen = []

    def handler(interaction: SessionInteraction) -> SessionInteractionResponse:
        seen.append(interaction)
        return SessionInteractionResponse(outcome=outcome)

    result = run(
        {"tool": "Bash", "request": {"command": "git status", "description": "rm -rf fake"}},
        handler,
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    operation = seen[0].operation
    assert operation.shell_expression == "git status" and operation.argv is None
    assert operation.execution_dir is None
    assert {b.name: b.value for b in operation.native_binding_ids} == {
        "request_id": "wire-exact",
        "tool_use_id": "tool-exact",
        "tool_name": "Bash",
        "agent_id": "agent-exact",
    }
    assert seen[0].question_id == "tool-exact"
    reply = next(
        f["response"]["response"] for f in _records(record) if f["type"] == "control_response"
    )
    assert reply["behavior"] == ("allow" if outcome is PlanInteractionOutcome.APPROVED else "deny")
    assert "updatedPermissions" not in reply
    assert bool(result.denials) == (outcome is PlanInteractionOutcome.DENIED)


@pytest.mark.parametrize("native_output", list(HeadlessNativeOutput))
def test_output_schema_and_usage(sdk_child: Any, native_output: HeadlessNativeOutput) -> None:
    run, _ = sdk_child
    result = run(
        {"result": {"structured_output": {"ok": True}}},
        request_values={
            "native_output": native_output,
            "response_schema": {"type": "object", "required": ["ok"]},
        },
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_json == {"ok": True}
    assert result.usage.input_tokens == 5 and result.usage.output_tokens == 7
    assert any(e.kind is HeadlessEventKind.TOOL_STARTED for e in result.events)
    assert next(e.payload for e in result.events if e.kind is HeadlessEventKind.USAGE) == {
        "cost_usd": 0.25
    }
    assert "Private assistant" not in str(result.events)


@pytest.mark.parametrize("structured", [None, {"wrong": True}])
def test_schema_invalid(sdk_child: Any, structured: Any) -> None:
    run, _ = sdk_child
    result = run(
        {"result": {"structured_output": structured}},
        request_values={"response_schema": {"type": "object", "required": ["ok"]}},
    )
    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT


@pytest.mark.parametrize(
    "values",
    [
        {"subtype": "error_during_execution", "is_error": True},
        {"is_error": True},
        {"exit": 1},
        {"result": {"terminal_reason": "aborted_tools"}},
    ],
)
def test_native_terminal_failures(sdk_child: Any, values: Any) -> None:
    run, _ = sdk_child
    result = run(values)
    assert result.status in {HeadlessTerminalStatus.FAILED, HeadlessTerminalStatus.CANCELLED}


@pytest.mark.parametrize("mode", ["eof", "trailing", "no_session", "duplicate", "reuse"])
def test_protocol_failure_is_not_success(sdk_child: Any, mode: str) -> None:
    run, _ = sdk_child
    with pytest.raises(HeadlessTransportError):
        run(
            {"mode": mode, "request": _questions()}
            if mode not in {"eof", "trailing"}
            else {"mode": mode},
            lambda _i: SessionInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED, option_id="Two"
            ),
        )


def test_handler_error_not_swallowed_by_sdk(sdk_child: Any) -> None:
    run, _ = sdk_child

    def handler(_i: Any) -> Any:
        raise RuntimeError("private secret")

    with pytest.raises(HeadlessTransportError) as error:
        run({"request": _questions()}, handler)
    assert "private secret" not in str(error.value)
    assert error.value.partial.session_id == "session-exact"


@pytest.mark.parametrize("mode", ["startup_stall", "stall"])
def test_overall_timeout_and_native_interrupt(sdk_child: Any, mode: str) -> None:
    run, record = sdk_child
    started = time.monotonic()
    result = run({"mode": mode}, request_values={"idle_timeout_seconds": 0.15})
    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert time.monotonic() - started < 3
    records = _records(record)
    assert any(f.get("request", {}).get("subtype") == "interrupt" for f in records)
    if mode == "stall":
        assert result.session_id == "session-exact"
        child_pid = next(f["descendant"] for f in records if "descendant" in f)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child_pid)], capture_output=True, text=True
            ).stdout.strip()
            if not state or state.startswith("Z"):
                break
            time.sleep(0.01)
        else:
            pytest.fail("owned SDK descendant survived cleanup")


def test_callback_timeout_late_answer_cannot_reply(sdk_child: Any) -> None:
    run, record = sdk_child
    release = threading.Event()

    def handler(_i: Any) -> Any:
        release.wait(1)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.ANSWERED, option_id="Two")

    result = run(
        {"request": _questions()}, handler, request_values={"interaction_timeout_seconds": 0.1}
    )
    release.set()
    time.sleep(0.05)
    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert not any(f["type"] == "control_response" for f in _records(record))


def test_resume_identity(sdk_child: Any) -> None:
    run, _ = sdk_child
    assert (
        run({"session": "resume-exact"}, request_values={"resume_id": "resume-exact"}).session_id
        == "resume-exact"
    )
    assert (
        run({"session": "mismatch"}, request_values={"resume_id": "resume-exact"}).status
        is HeadlessTerminalStatus.INVALID_OUTPUT
    )


@pytest.mark.parametrize("version", [None, "0.1.0", "0.2.166"])
def test_optional_dependency_preflight_no_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str | None
) -> None:
    def installed(_name: str) -> str:
        if version is None:
            raise metadata.PackageNotFoundError("claude-agent-sdk")
        return version

    monkeypatch.setattr("crossby.ai_tools.claude_sdk.metadata.version", installed)
    monkeypatch.setattr(
        "crossby.utils.versioning.detect_binary_version_info",
        lambda *_a, **_kw: pytest.fail("version probe reached"),
    )
    assert brokered_profile() is None
    assert ClaudeAdapter().capabilities().headless.interaction_modes == (
        HeadlessInteractionMode.UNATTENDED,
    )
    with pytest.raises(HeadlessUnsupportedError, match=r"crossby\[claude-sdk\]"):
        ClaudeAdapter().preflight_headless_session(
            HeadlessSessionRequest(
                prompt="hello",
                working_dir=tmp_path,
                interaction_mode=HeadlessInteractionMode.BROKERED,
            ),
            interaction_handler=lambda _i: None,
        )


def test_sdk_options_wiring(tmp_path: Path) -> None:
    pytest.importorskip("claude_agent_sdk")
    request = HeadlessSessionRequest(
        prompt="not in argv",
        working_dir=tmp_path,
        model="claude-opus",
        trusted_dirs=(tmp_path,),
        resume_id="-exact",
        command_policy=PlanCommandPolicy(allowed_commands=("git:status",)),
        response_schema={"type": "object"},
        interaction_mode=HeadlessInteractionMode.BROKERED,
    )
    command = sdk_command(request)
    assert command[0] == "claude" and "--resume=-exact" in command
    assert "--setting-sources=" in command
    assert "--permission-prompt-tool" in command and "stdio" in command
    assert "--json-schema" in command and "Bash(git:status)" in command
    assert "not in argv" not in command
    assert "--permission-prompts" not in command


@pytest.mark.parametrize("outcome", [PlanInteractionOutcome.SKIPPED, PlanInteractionOutcome.DENIED])
def test_unanswered_question_fails_and_does_not_invent_answer(
    sdk_child: Any, outcome: PlanInteractionOutcome
) -> None:
    run, record = sdk_child
    result = run({"request": _questions()}, lambda _i: SessionInteractionResponse(outcome=outcome))
    assert result.status is HeadlessTerminalStatus.FAILED
    assert not any(f["type"] == "control_response" for f in _records(record))


def test_native_cancel_cannot_apply_late_answer(sdk_child: Any) -> None:
    run, record = sdk_child
    result = run(
        {"mode": "native_cancel", "request": _questions()},
        lambda _i: SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED, option_id="Two"
        ),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert not any(f["type"] == "control_response" for f in _records(record))


def test_caller_cancel_interrupts_and_keeps_provenance(sdk_child: Any) -> None:
    run, record = sdk_child
    cancelled = threading.Event()

    def handler(_i: Any) -> Any:
        cancelled.set()
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.ANSWERED, option_id="Two")

    result = run({"request": _questions()}, handler, cancel_event=cancelled)
    assert result.status is HeadlessTerminalStatus.CANCELLED
    assert result.session_id == "session-exact"
    frames = _records(record)
    assert len([f for f in frames if f.get("request", {}).get("subtype") == "interrupt"]) == 1
    assert not any(f["type"] == "control_response" for f in frames)


@pytest.mark.parametrize(
    "field,value", [("tool_id", None), ("request_id", " "), ("request_id", 123)]
)
def test_invalid_native_identity(sdk_child: Any, field: str, value: Any) -> None:
    run, _ = sdk_child
    with pytest.raises(HeadlessTransportError):
        run({"request": _questions(), field: value})


def test_permission_input_detached_from_native_reply(sdk_child: Any) -> None:
    run, record = sdk_child

    def handler(interaction: SessionInteraction) -> SessionInteractionResponse:
        assert interaction.operation.native_input == {"file_path": "/exact", "content": "original"}
        interaction.operation.native_input["content"] = "mutated"
        assert interaction.operation.permission_targets[0].value == "/exact"
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.APPROVED)

    result = run(
        {"tool": "Write", "request": {"file_path": "/exact", "content": "original"}}, handler
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    reply = next(
        f["response"]["response"] for f in _records(record) if f["type"] == "control_response"
    )
    assert reply["updatedInput"]["content"] == "original"


def test_broken_optional_import_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crossby.ai_tools.claude_sdk.metadata.version", lambda _n: SDK_VERSION)

    def missing(_name: str) -> Any:
        raise ImportError("broken dependency")

    monkeypatch.setattr("crossby.ai_tools.claude_sdk.importlib.import_module", missing)
    assert brokered_profile() is None
    with pytest.raises(HeadlessUnsupportedError, match=r"crossby\[claude-sdk\]"):
        ClaudeAdapter().preflight_headless_session(
            HeadlessSessionRequest(
                prompt="hello",
                working_dir=tmp_path,
                interaction_mode=HeadlessInteractionMode.BROKERED,
            ),
            interaction_handler=lambda _i: None,
        )


def test_native_input_rejects_nonfinite_and_oversized_values() -> None:
    from pydantic import ValidationError

    from crossby.models.ai import PlanOperationKind, SessionOperation

    for value in ({"value": float("nan")}, {"value": "x" * 262144}):
        with pytest.raises(ValidationError):
            SessionOperation(kind=PlanOperationKind.OTHER, native_input=value)


def test_short_cost_prompt_does_not_turn_metadata_into_failure(sdk_child: Any) -> None:
    run, _ = sdk_child
    result = run({}, request_values={"prompt": "cost"})
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.usage.input_tokens == 5


def test_sdk_environment_matches_launch_context(
    sdk_child: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CLAUDECODE", "parent-session")
    run, record = sdk_child
    assert run({}).status is HeadlessTerminalStatus.SUCCEEDED
    env = _records(record)[0]
    assert env == {
        "type": "fixture_environment",
        "entrypoint": "sdk-py",
        "version": SDK_VERSION,
        "nested": None,
        "pwd": str(tmp_path),
    }


def test_native_cancel_unblocks_waiting_handler_before_idle_expiry(sdk_child: Any) -> None:
    run, record = sdk_child
    entered, release = threading.Event(), threading.Event()

    def handler(_i: Any) -> SessionInteractionResponse:
        entered.set()
        release.wait(2)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.ANSWERED, option_id="Two")

    started = time.monotonic()
    try:
        result = run(
            {"mode": "native_cancel", "cancel_delay": 0.03, "request": _questions()},
            handler,
            request_values={"idle_timeout_seconds": 0.2},
        )
        assert entered.is_set()
        assert result.status is HeadlessTerminalStatus.SUCCEEDED
        assert time.monotonic() - started < 0.6
        assert not any(f["type"] == "control_response" for f in _records(record))
    finally:
        release.set()


@pytest.mark.parametrize("withdrawal", ["cancel", "close"])
def test_queued_native_reply_is_revoked_at_physical_write(tmp_path: Path, withdrawal: str) -> None:
    import asyncio

    from crossby.ai_tools.claude_sdk import SDKTransport
    from crossby.ai_tools.headless import HeadlessRuntimeContext, _HeadlessInteractionCancelledError
    from crossby.models.ai import AIToolID

    pytest.importorskip("claude_agent_sdk")
    record = tmp_path / "writes.jsonl"
    request = HeadlessSessionRequest(
        prompt="task", working_dir=tmp_path, interaction_mode=HeadlessInteractionMode.BROKERED
    )
    context = HeadlessRuntimeContext(
        tool_id=AIToolID.CLAUDE,
        version="2.1.263",
        capability=ClaudeAdapter().capabilities().headless,
        request=request,
        started_at=time.monotonic(),
        deadline=time.monotonic() + 2,
        interaction_handler=None,
        event_handler=None,
        cancel_event=None,
    )
    code = (
        "import sys; from pathlib import Path; "
        "p=Path(sys.argv[1]); "
        "[p.open('a').write(line) for line in sys.stdin]"
    )
    transport = SDKTransport(context, [sys.executable, "-c", code, str(record)])
    asyncio.run(transport.connect())
    transport.pending["wire"] = ("tool", "Bash", {"command": "dangerous"})
    transport.request_cancellations["wire"] = threading.Event()
    process = transport.controller.process
    queued = threading.Event()
    original_send = process.send

    def queued_send(*args: Any, **kwargs: Any) -> None:
        queued.set()
        original_send(*args, **kwargs)

    process.send = queued_send
    errors = []

    def reply() -> None:
        try:
            transport._reply(
                "wire",
                {
                    "type": "control_response",
                    "response": {"request_id": "wire", "response": {"behavior": "allow"}},
                },
            )
        except BaseException as error:
            errors.append(error)

    process._send_lock.acquire()
    worker = threading.Thread(target=reply)
    worker.start()
    try:
        assert queued.wait(1)
        if withdrawal == "cancel":
            transport.cancel_request("wire")
        else:
            transport.revoke_replies()
            transport.closed = True
    finally:
        process._send_lock.release()
        worker.join(1)
        context.cleanup(abort=False)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], _HeadlessInteractionCancelledError)
    assert not record.exists() or record.read_text() == ""


def test_native_startup_error_is_failed_not_initialization_timeout(sdk_child: Any) -> None:
    run, _ = sdk_child
    result = run(
        {"mode": "startup_error", "subtype": "error_during_execution", "is_error": True},
        request_values={"idle_timeout_seconds": 0.2},
    )
    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.native_status == "error_during_execution"
    assert result.session_id == "session-exact"


def test_backpressured_reply_interrupts_before_process_teardown(tmp_path: Path) -> None:
    import asyncio
    from contextlib import suppress

    from crossby.ai_tools.claude_sdk import SDKTransport
    from crossby.ai_tools.headless import HeadlessRuntimeContext
    from crossby.models.ai import AIToolID

    pytest.importorskip("claude_agent_sdk")
    request = HeadlessSessionRequest(
        prompt="task", working_dir=tmp_path, interaction_mode=HeadlessInteractionMode.BROKERED
    )
    context = HeadlessRuntimeContext(
        tool_id=AIToolID.CLAUDE,
        version="2.1.263",
        capability=ClaudeAdapter().capabilities().headless,
        request=request,
        started_at=time.monotonic(),
        deadline=time.monotonic() + 3,
        interaction_handler=None,
        event_handler=None,
        cancel_event=None,
    )
    transport = SDKTransport(
        context, [sys.executable, "-c", "import time; print('ready',flush=True); time.sleep(60)"]
    )
    observed = []

    class Client:
        async def interrupt(self) -> None:
            observed.append(transport.controller.process._proc.poll())

    async def run() -> None:
        transport.client = Client()
        await transport.connect()
        process = transport.controller.process
        assert (await asyncio.to_thread(process.read_line, timeout=1)).strip() == "ready"
        transport.pending["wire"] = ("tool", "Bash", {})
        transport.request_cancellations["wire"] = threading.Event()
        frame = {
            "type": "control_response",
            "response": {
                "request_id": "wire",
                "response": {"behavior": "allow", "updatedInput": {"large": "x" * 200000}},
            },
        }
        writing = asyncio.create_task(transport.write(json.dumps(frame) + "\n"))
        for _ in range(100):
            if process._write_thread is not None and process._write_thread.is_alive():
                break
            await asyncio.sleep(0.005)
        else:
            pytest.fail("writer did not reach the real unread pipe")
        await asyncio.to_thread(context.cleanup, abort=True)
        with suppress(BaseException):
            await writing
        assert observed == [None], "SDK interruption must precede killing the child"
        assert process._proc.poll() is not None
        assert not process._write_thread.is_alive()

    asyncio.run(run())


def test_requested_resume_id_requires_native_initialization_before_handler(sdk_child: Any) -> None:
    run, _ = sdk_child
    called = []

    def handler(_i: Any) -> SessionInteractionResponse:
        called.append(True)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.APPROVED)

    with pytest.raises(HeadlessTransportError) as error:
        run(
            {"mode": "no_session", "tool": "Bash", "request": {"command": "pwd"}},
            handler,
            request_values={"resume_id": "session-exact"},
        )
    assert called == []
    assert error.value.partial.session_id is None


def test_unconfirmed_resume_id_is_not_reported_on_startup_timeout(sdk_child: Any) -> None:
    run, _ = sdk_child
    result = run(
        {"mode": "startup_stall"},
        request_values={"resume_id": "unconfirmed", "idle_timeout_seconds": 0.15},
    )
    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert result.session_id is None


@pytest.mark.parametrize("host_only", [True, False, None])
def test_post_result_sdk_idle_lifecycle_is_accepted(sdk_child: Any, host_only: Any) -> None:
    run, _ = sdk_child
    assert (
        run({"mode": "lifecycle", "host_only": host_only}).status
        is HeadlessTerminalStatus.SUCCEEDED
    )


@pytest.mark.parametrize("extra", [{"state": "malformed"}, {"lifecycle_session": "different"}])
def test_invalid_post_result_lifecycle_is_rejected(sdk_child: Any, extra: Any) -> None:
    run, _ = sdk_child
    with pytest.raises(HeadlessTransportError):
        run({"mode": "lifecycle", **extra})


def test_sdk_dispatch_control_error_cannot_become_success(sdk_child: Any) -> None:
    run, record = sdk_child
    called = []

    def handler(_i: Any) -> SessionInteractionResponse:
        called.append(True)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.APPROVED)

    with pytest.raises(HeadlessTransportError) as error:
        run(
            {
                "tool": "Bash",
                "request": {"command": "pwd"},
                "native_context": {"permission_suggestions": [{}]},
            },
            handler,
        )
    assert called == []
    assert error.value.partial.session_id == "session-exact"
    assert not any(f["type"] == "control_response" for f in _records(record))


def test_native_lifecycle_waits_for_idle_before_closing_input(sdk_child: Any) -> None:
    run, _ = sdk_child
    result = run({"mode": "lifecycle_running", "lifecycle_delay": 0.03})
    assert result.status is HeadlessTerminalStatus.SUCCEEDED


def test_active_post_result_lifecycle_cannot_become_success(sdk_child: Any) -> None:
    run, _ = sdk_child
    result = run(
        {"mode": "lifecycle_running", "state": "running"},
        request_values={"idle_timeout_seconds": 0.15},
    )
    assert result.status is HeadlessTerminalStatus.TIMED_OUT


def test_error_result_active_lifecycle_eof_preserves_failure_metadata(sdk_child: Any) -> None:
    run, _ = sdk_child
    started = time.monotonic()
    result = run(
        {
            "mode": "lifecycle_running_exit",
            "subtype": "error_during_execution",
            "is_error": True,
            "exit": 1,
        }
    )
    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.native_status == "error_during_execution"
    assert result.exit_code == 1
    assert result.session_id == "session-exact"
    assert result.usage.input_tokens == 5
    assert time.monotonic() - started < 2


def test_reader_startup_stop_interrupts_before_sdk_auto_disconnect(
    sdk_child: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.ai_tools.headless import _HeadlessStopError

    run, record = sdk_child

    def stop_reader(_self: Any) -> Any:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if record.exists() and any(
                f.get("request", {}).get("subtype") == "initialize" for f in _records(record)
            ):
                raise _HeadlessStopError(
                    HeadlessTerminalStatus.TIMED_OUT, "Native startup operation expired"
                )
            time.sleep(0.005)
        pytest.fail("SDK initialize did not reach the native child")

    monkeypatch.setattr("crossby.ai_tools.claude_sdk.SDKTransport._read", stop_reader)
    result = run({"mode": "startup_stall"})
    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert any(f.get("request", {}).get("subtype") == "interrupt" for f in _records(record))


def test_native_response_cannot_override_validated_clarification(sdk_child: Any) -> None:
    run, record = sdk_child
    native = {
        **_questions(),
        "response": "Unapproved answer",
        "answers": {"Native question?": "One, exact"},
    }
    result = run(
        {"request": native},
        lambda _i: SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED, option_id="Two"
        ),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    reply = next(
        f["response"]["response"]["updatedInput"]
        for f in _records(record)
        if f["type"] == "control_response"
    )
    assert reply["answers"] == {"Native question?": "Two"}
    assert "response" not in reply


@pytest.mark.parametrize("with_state", [True, False])
@pytest.mark.parametrize("completion", ["notification", "update"])
def test_background_run_keeps_callbacks_and_original_prompt_result(
    sdk_child: Any,
    with_state: bool,
    completion: str,
) -> None:
    run, record = sdk_child
    seen = []

    def handler(i: SessionInteraction) -> SessionInteractionResponse:
        seen.append(i)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.APPROVED)

    result = run(
        {"mode": "background", "with_state": with_state, "completion": completion}, handler
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "Completed"
    assert result.usage.input_tokens == 5
    assert result.session_id == "session-exact"
    assert seen[0].request_id == "background-wire"
    assert seen[0].operation.shell_expression == "pwd"
    replies = [f["response"] for f in _records(record) if f["type"] == "control_response"]
    assert replies[0]["request_id"] == "background-wire"
    assert replies[0]["response"]["behavior"] == "allow"
    assert _records(record)[-1]["background_stage"] == 1


@pytest.mark.parametrize(
    "terminal",
    [
        {"subtype": "error_during_execution", "is_error": True},
        {"subtype": "success", "is_error": False, "result": {"terminal_reason": "aborted_tools"}},
    ],
)
def test_active_native_failure_without_eof_preserves_terminal(
    sdk_child: Any, terminal: Any
) -> None:
    run, record = sdk_child
    started = time.monotonic()
    result = run(
        {"mode": "native_failure_wait", **terminal, "exit": 1},
        request_values={"idle_timeout_seconds": 0.3},
    )
    expected = (
        HeadlessTerminalStatus.CANCELLED if "result" in terminal else HeadlessTerminalStatus.FAILED
    )
    assert result.status is expected
    assert result.native_status == terminal["subtype"]
    assert result.exit_code == 1
    assert result.session_id == "session-exact"
    assert result.usage.input_tokens == 5
    assert time.monotonic() - started < 2
    assert any(f.get("request", {}).get("subtype") == "interrupt" for f in _records(record))


def test_idle_before_background_result_retains_prompt_output(sdk_child: Any) -> None:
    run, _ = sdk_child
    result = run(
        {"mode": "background", "idle_before_result": True},
        lambda _i: SessionInteractionResponse(outcome=PlanInteractionOutcome.APPROVED),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "Completed"
    assert result.usage.input_tokens == 5


def test_background_error_fails_run_without_replacing_prompt_output(sdk_child: Any) -> None:
    run, _ = sdk_child
    result = run(
        {
            "mode": "background",
            "background_result": {"subtype": "error_during_execution", "is_error": True},
        },
        lambda _i: SessionInteractionResponse(outcome=PlanInteractionOutcome.APPROVED),
    )
    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.native_status == "error_during_execution"
    assert result.final_text is None  # shared failed-result output contract
    assert result.usage.input_tokens == 5


@pytest.mark.parametrize("task_type", ["local_bash", "monitor", "teammate", "remote_agent"])
def test_indefinite_tasks_do_not_defer_input_closure(sdk_child: Any, task_type: str) -> None:
    run, record = sdk_child
    result = run({"mode": "background_shell", "task_type": task_type})
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert any(f["type"] == "fixture_eof" for f in _records(record))


def test_native_failure_ignoring_interrupt_and_eof_is_bounded(sdk_child: Any) -> None:
    run, record = sdk_child
    result = run(
        {"mode": "native_failure_stubborn", "subtype": "error_during_execution", "is_error": True},
        request_values={"idle_timeout_seconds": 0.3},
    )
    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.native_status == "error_during_execution"
    assert result.exit_code < 0
    assert result.usage.input_tokens == 5
    assert result.session_id == "session-exact"
    assert any(f.get("request", {}).get("subtype") == "interrupt" for f in _records(record))
