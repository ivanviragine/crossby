"""Exercise shared transport bounds with real pipes and blocked socket readers."""

from __future__ import annotations

import io
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools import AbstractAITool, plan_process
from crossby.ai_tools.headless import HeadlessRuntimeContext, _HeadlessStopError
from crossby.ai_tools.opencode_server import BoundedSSEParser, OpenCodeServer
from crossby.ai_tools.session_protocol import ProtocolSessionController
from crossby.models.ai import (
    AIToolID,
    HeadlessEventKind,
    HeadlessInteractionMode,
    HeadlessSessionRequest,
    HeadlessTerminalStatus,
    PlanInteractionOutcome,
    SessionInteractionResponse,
)


def _context(tmp_path: Path) -> HeadlessRuntimeContext:
    now = time.monotonic()
    return HeadlessRuntimeContext(
        tool_id=AIToolID.CODEX,
        version="fixture",
        capability=AbstractAITool.get(AIToolID.CODEX).capabilities().headless,
        request=HeadlessSessionRequest(
            prompt="private task",
            working_dir=tmp_path,
            interaction_mode=HeadlessInteractionMode.BROKERED,
            timeout_seconds=5,
            idle_timeout_seconds=0.5,
        ),
        started_at=now,
        deadline=now + 5,
        interaction_handler=None,
        event_handler=None,
        cancel_event=None,
    )


@pytest.mark.parametrize("headerless", [False, True])
def test_progress_keeps_protocol_reads_and_writes_live_past_initial_idle_budget(
    tmp_path: Path, headerless: bool
) -> None:
    frame = {"method": "progress", **({} if headerless else {"jsonrpc": "2.0"})}
    script = (
        "import json,sys,time\n"
        f"frame={frame!r}\n"
        "for _ in range(6):\n"
        " print(json.dumps(frame),flush=True)\n"
        " time.sleep(0.15)\n"
        "request=json.loads(sys.stdin.readline())\n"
        "frame['id']=request['id']; frame['result']={}\n"
        "print(json.dumps(frame),flush=True)\n"
    )
    context = _context(tmp_path)
    controller = ProtocolSessionController(
        context, [sys.executable, "-c", script], cwd=tmp_path, headerless=headerless
    )
    try:
        for _ in range(6):
            assert controller.read()["method"] == "progress"
            context.mark_progress()
        controller.request(1, "finish", {})
        assert controller.read()["id"] == 1
        assert controller.close_cleanly() == 0
    finally:
        context.cleanup(abort=False)


@pytest.mark.parametrize("operation", ["send", "request", "notify", "respond"])
def test_each_protocol_write_honors_a_shorter_operation_budget(
    tmp_path: Path, operation: str
) -> None:
    rpc = plan_process.JsonRpcProcess(
        [sys.executable, "-c", "import time; print('ready',flush=True); time.sleep(30)"],
        cwd=tmp_path,
        timeout=30,
    )
    assert rpc.read_line(timeout=2).strip() == "ready"
    payload = {"text": "x" * (2 * 1024 * 1024)}
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="writing a JSON-RPC message"):
            if operation == "send":
                rpc.send(payload, timeout=0.05)
            elif operation == "request":
                rpc.request(1, "work", payload, timeout=0.05)
            elif operation == "notify":
                rpc.notify("work", payload, timeout=0.05)
            else:
                rpc.respond(1, payload, timeout=0.05)
        assert time.monotonic() - started < 1
    finally:
        rpc.close()
    assert rpc._write_thread is not None and not rpc._write_thread.is_alive()


@pytest.mark.parametrize("operation", ["read", "write"])
def test_protocol_idle_expiry_is_a_runtime_timeout_not_a_transport_failure(
    tmp_path: Path, operation: str
) -> None:
    context = _context(tmp_path)
    controller = ProtocolSessionController(
        context, [sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path
    )
    try:
        with pytest.raises(_HeadlessStopError) as stopped:
            if operation == "read":
                controller.read()
            else:
                controller.notify("work", {"text": "x" * (2 * 1024 * 1024)})
        assert stopped.value.status is HeadlessTerminalStatus.TIMED_OUT
    finally:
        context.cleanup(abort=True)


def test_windows_protocol_child_is_owned_before_resume_and_reader_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class FakeProcess:
        stdin = io.StringIO()
        stdout = io.StringIO()
        stderr = io.StringIO()
        returncode = 0

        def poll(self) -> int:
            return 0

    process = FakeProcess()
    tree = plan_process._WindowsProcessTree(1, lambda _handle: events.append("kill-tree"))

    def launch(_argv: list[str], **kwargs: Any) -> Any:
        assert kwargs["creationflags"] == getattr(subprocess, "CREATE_SUSPENDED", 4)
        events.append("launch-suspended")
        return process

    def attach(_proc: Any) -> Any:
        events.append("attach-job")
        return tree

    monkeypatch.setattr(plan_process, "_is_windows", lambda: True)
    monkeypatch.setattr(plan_process.subprocess, "Popen", launch)
    monkeypatch.setattr(plan_process, "_create_windows_process_tree", attach)
    monkeypatch.setattr(plan_process, "_resume_windows_process", lambda _p: events.append("resume"))
    rpc = plan_process.JsonRpcProcess(["fixture"], cwd=tmp_path)
    rpc.close()
    assert events == ["launch-suspended", "attach-job", "resume", "kill-tree"]


def test_failed_protocol_ownership_kills_and_reaps_the_spawned_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    children: list[subprocess.Popen[Any]] = []
    popen = subprocess.Popen

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    def fail_ownership(_process: Any) -> None:
        raise OSError("cannot own child")

    monkeypatch.setattr(plan_process.subprocess, "Popen", launch)
    monkeypatch.setattr(plan_process, "own_process_tree", fail_ownership)
    with pytest.raises(OSError, match="cannot own child"):
        plan_process.JsonRpcProcess(
            [sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path
        )
    assert children[0].poll() is not None


@pytest.mark.skipif(os.name != "posix", reason="POSIX orphaned process-group behavior")
def test_protocol_cleanup_kills_descendants_after_parent_exits(tmp_path: Path) -> None:
    script = (
        "import os,subprocess,sys\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
        "print(child.pid,flush=True)\n"
        "os._exit(0)\n"
    )
    context = _context(tmp_path)
    controller = ProtocolSessionController(context, [sys.executable, "-c", script], cwd=tmp_path)
    child_pid = int(controller.process.read_line(timeout=2))
    controller.process._proc.wait(timeout=2)
    try:
        context.cleanup(abort=False)
        limit = time.monotonic() + 2
        while time.monotonic() < limit:
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child_pid)], capture_output=True, text=True
            ).stdout.strip()
            if not state or state.startswith("Z"):
                break
            time.sleep(0.01)
        else:
            pytest.fail("owned descendant survived after its parent exited")
    finally:
        plan_process.kill_process_group(controller.process._proc)
        with suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)


def test_stdio_byte_backpressure_preserves_every_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plan_process, "_QUEUED_OUTPUT_BYTES_LIMIT", 32)
    rpc = plan_process.JsonRpcProcess(
        [sys.executable, "-c", "for i in range(20): print(str(i)*10,flush=True)"], cwd=tmp_path
    )
    try:
        assert [rpc.read_line(timeout=2).strip() for _ in range(20)] == [
            str(i) * 10 for i in range(20)
        ]
    finally:
        rpc.close()


def test_stdio_frame_limit_counts_utf8_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plan_process, "_JSON_RPC_FRAME_LIMIT", 10)
    rpc = plan_process.JsonRpcProcess.__new__(plan_process.JsonRpcProcess)
    rpc._stdout_queue = queue.Queue()
    rpc._closing = threading.Event()
    rpc._discard_stdout = threading.Event()
    rpc._read_stdout(io.StringIO("é" * 6 + "\n"))
    assert isinstance(rpc._stdout_queue.get_nowait(), plan_process.JsonRpcFrameLimitError)


@pytest.mark.parametrize("limit", ["bytes", "lines"])
def test_empty_sse_data_lines_cannot_bypass_aggregate_limits(
    monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    monkeypatch.setattr(
        "crossby.ai_tools.opencode_server."
        + ("_SSE_EVENT_LIMIT" if limit == "bytes" else "_SSE_DATA_LINE_LIMIT"),
        16 if limit == "bytes" else 3,
    )
    parser = BoundedSSEParser()
    with pytest.raises(ValueError, match="event exceeded its size limit"):
        for _ in range(20):
            parser.feed(b"data:\n")


def test_sse_line_limit_is_independent_of_network_chunk_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crossby.ai_tools.opencode_server._SSE_EVENT_LIMIT", 64)
    parser = BoundedSSEParser()
    value = {"text": "x" * 40}
    line = b"data: " + json.dumps(value).encode()
    assert len(line) < 64
    assert parser.feed(line) == []
    assert parser.feed(b"\n\n: heartbeat\n\n") == [value]
    parser.finish()


@pytest.mark.parametrize("prompt", ["a", "tool", "call"])
def test_opencode_tool_events_do_not_reject_short_prompts(tmp_path: Path, prompt: str) -> None:
    from crossby.ai_tools.headless_cli import BoundedProgress

    context = _context(tmp_path)
    context.request = context.request.model_copy(update={"prompt": prompt})
    server = OpenCodeServer.__new__(OpenCodeServer)
    server.runtime = context
    server._events = queue.Queue()
    server._event_error = None
    server._bound_message_ids = {"msg-exact"}
    server._tool_states = {}
    server._events.put(
        {
            "type": "message.part.updated",
            "properties": {
                "part": {
                    "id": "tool-exact",
                    "sessionID": "ses-exact",
                    "type": "tool",
                    "messageID": "msg-exact",
                    "state": {"status": "completed"},
                }
            },
        }
    )
    server.drain_events(context, "ses-exact", "prompt-exact", BoundedProgress(context))
    assert context.events[-1].kind is HeadlessEventKind.TOOL_COMPLETED
    assert context.events[-1].payload is None


def test_sse_cleanup_interrupts_a_blocked_buffered_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader_socket, peer = socket.socketpair()
    reader = reader_socket.makefile("rb")
    entered = threading.Event()

    class FakeConnection:
        sock = reader_socket
        status = 200

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def request(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def getresponse(self) -> Any:
            return self

        def getheader(self, _name: str, _default: str = "") -> str:
            return "text/event-stream"

        def read1(self, amount: int) -> bytes:
            entered.set()
            return reader.read1(amount)

        def close(self) -> None:
            reader_socket.close()

    monkeypatch.setattr(
        "crossby.ai_tools.opencode_server.http.client.HTTPConnection", FakeConnection
    )
    server = OpenCodeServer.__new__(OpenCodeServer)
    server.runtime = None
    server.port = 1234
    server.directory = tmp_path
    server.authorization = "fixture"
    server.deadline = time.monotonic() + 30
    server._event_stop = threading.Event()
    server._event_socket = None
    server._event_connection = None
    server._events = queue.Queue()
    server._event_error = None
    server.open_events()
    try:
        assert entered.wait(timeout=2)
        server._stop_events()
        assert server._event_thread is not None
        server._event_thread.join(timeout=0.5)
        assert not server._event_thread.is_alive()
    finally:
        peer.close()
        reader.close()
        reader_socket.close()


def test_opencode_remaining_budget_tracks_runtime_progress(tmp_path: Path) -> None:
    server = OpenCodeServer.__new__(OpenCodeServer)
    server.runtime = _context(tmp_path)
    server.deadline = server.runtime.deadline
    server.runtime._last_progress -= 0.25
    before = server.remaining()
    server.runtime.mark_progress()
    assert server.remaining() > before + 0.2
    assert server.remaining() <= 0.5


def test_advisory_event_queue_is_bounded_by_bytes_before_its_count_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = {"type": "fixture", "properties": {"text": "é" * 10}}
    size = len(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    monkeypatch.setattr("crossby.ai_tools.opencode_server._SSE_QUEUE_BYTES_LIMIT", size * 2)
    monkeypatch.setattr("crossby.ai_tools.opencode_server.JsonRpcProcess", lambda *_a, **_k: None)
    request = HeadlessSessionRequest(prompt="task", working_dir=tmp_path)
    server = OpenCodeServer(request, time.monotonic() + 5)
    server._events.put_nowait(event)
    server._events.put_nowait(event)
    with pytest.raises(queue.Full):
        server._events.put_nowait(event)
    assert server._events.get_nowait() == event
    server._events.put_nowait(event)


@pytest.mark.parametrize("kind", ["message", "tool"])
def test_opencode_bookkeeping_remains_bounded_after_visible_progress_cap(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from crossby.ai_tools.headless_cli import BoundedProgress

    monkeypatch.setattr("crossby.ai_tools.opencode_server._NATIVE_STATE_LIMIT", 2)
    server = OpenCodeServer.__new__(OpenCodeServer)
    server.runtime = None
    server._events = queue.Queue()
    server._event_error = None
    server._bound_message_ids = {"msg-bound"} if kind == "tool" else set()
    server._tool_states = {}

    class FakeContext:
        def mark_progress(self) -> None:
            pass

    context = FakeContext()
    progress = BoundedProgress(context, limit=0)  # type: ignore[arg-type]
    for index in range(3):
        server._events.put(
            {
                "type": "message.updated" if kind == "message" else "message.part.updated",
                "properties": {
                    "info" if kind == "message" else "part": {
                        "id": f"native-{index}",
                        "sessionID": "ses-exact",
                        "role": "assistant",
                        "parentID": "prompt-exact",
                        "type": "tool",
                        "messageID": "msg-bound",
                        "state": {"status": "completed"},
                    }
                },
            }
        )
    with pytest.raises(ValueError, match="identifier state exceeded"):
        server.drain_events(context, "ses-exact", "prompt-exact", progress)  # type: ignore[arg-type]
    assert len(server._bound_message_ids if kind == "message" else server._tool_states) == 2


def test_opencode_bookkeeping_bounds_individual_identifier_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crossby.ai_tools.opencode_server import _remember_native_id

    monkeypatch.setattr("crossby.ai_tools.opencode_server._NATIVE_ID_BYTES_LIMIT", 3)
    remembered: set[str] = set()
    with pytest.raises(ValueError, match="identifier state exceeded"):
        _remember_native_id(remembered, "éé")
    assert not remembered


def test_opencode_does_not_retain_oversized_tool_statuses(tmp_path: Path) -> None:
    from crossby.ai_tools.headless_cli import BoundedProgress

    context = _context(tmp_path)
    server = OpenCodeServer.__new__(OpenCodeServer)
    server.runtime = context
    server._events = queue.Queue()
    server._event_error = None
    server._bound_message_ids = {"msg-exact"}
    server._tool_states = {}
    server._events.put(
        {
            "type": "message.part.updated",
            "properties": {
                "part": {
                    "id": "tool-exact",
                    "sessionID": "ses-exact",
                    "type": "tool",
                    "messageID": "msg-exact",
                    "state": {"status": "é" * 33},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="tool update omitted its status"):
        server.drain_events(context, "ses-exact", "prompt-exact", BoundedProgress(context))
    assert not server._tool_states


@pytest.mark.parametrize("outcome", ["denied", "skipped", "automatic", "approved"])
@pytest.mark.parametrize("with_kinds", [False, True])
def test_cursor_permission_labels_cannot_change_the_native_decision(
    outcome: str, with_kinds: bool
) -> None:
    from crossby.ai_tools.cursor import _answer_cursor_permission
    from crossby.ai_tools.plan_mode import PlanInteractionRequiredError

    responses: list[Any] = []

    class Rpc:
        def respond(self, request_id: int, result: Any) -> None:
            assert request_id == 90
            responses.append(result)

    message = {
        "id": 90,
        "params": {
            "sessionId": "ses-exact",
            "toolCall": {"toolCallId": "tool-exact"},
            "options": [
                {
                    "optionId": "yes",
                    "name": "Deny safeguards and run",
                    **({"kind": "allow_once"} if with_kinds else {}),
                },
                {
                    "optionId": "no",
                    "name": "Allow task to continue without access",
                    **({"kind": "reject_once"} if with_kinds else {}),
                },
            ],
        },
    }

    def run() -> None:
        _answer_cursor_permission(
            Rpc(),
            message,
            session_id="ses-exact",
            handler=lambda _interaction: SessionInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED
                if outcome == "automatic"
                else PlanInteractionOutcome(outcome)
            ),
            deny_automatically=outcome == "automatic",
            tool_id=AIToolID.CURSOR,
            capability=AbstractAITool.get(AIToolID.CURSOR).capabilities().plan_mode,
        )

    if not with_kinds and outcome == "approved":
        with pytest.raises(PlanInteractionRequiredError):
            run()
        assert responses == []
    else:
        run()
        expected = (
            [
                {
                    "outcome": {
                        "outcome": "selected",
                        "optionId": "yes" if outcome == "approved" else "no",
                    }
                }
            ]
            if with_kinds
            else [{"outcome": {"outcome": "cancelled"}}]
        )
        assert responses == expected
