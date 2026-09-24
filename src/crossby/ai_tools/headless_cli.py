"""Shared managed-CLI transport for unattended headless sessions.

Every terminal adapter drives its native non-interactive CLI through
:func:`run_managed_command`.  The helper owns exactly one child process group,
never inherits the parent's stdin, registers the runtime's single cleanup
sequence before any blocking read, and bounds every wait with
:meth:`HeadlessRuntimeContext.remaining_seconds`.  Adapters keep their native
parsing and contribute only normalized events, provenance, and one terminal
result.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from crossby.ai_tools.headless import (
    HeadlessCleanupContext,
    HeadlessCleanupHooks,
    HeadlessRuntimeContext,
)
from crossby.ai_tools.plan_process import (
    join_threads_until,
    kill_process_group,
    own_process_tree,
    process_tree_popen_kwargs,
)
from crossby.models.ai import (
    HeadlessEventKind,
    HeadlessNativeOutput,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
    TokenUsage,
)

_STDOUT_LIMIT = 8 * 1024 * 1024
_STDERR_LIMIT = 1024 * 1024
_READ_CHUNK = 64 * 1024
_WAIT_POLL_SECONDS = 0.05
_WORKER_JOIN_GRACE_SECONDS = 0.25
_MAX_PROGRESS_EVENTS = 512
_STDERR_TAIL_CHARS = 2000

MISSING: Any = object()
"""Sentinel distinguishing absent structured output from an explicit JSON null."""

StdoutLineHandler = Callable[[Sequence[str]], None]
"""Receives complete native stdout lines while the child is still running."""


@dataclass(frozen=True)
class HeadlessCommandOutput:
    """Bounded captured output and exact status from one owned native CLI run."""

    returncode: int
    stdout: str
    stderr: str
    overflowed: bool = False
    undecodable: bool = False

    @property
    def stderr_tail(self) -> str:
        """Return a bounded stderr tail for private diagnostics, never result warnings."""
        collapsed = " ".join(self.stderr.split())
        return collapsed[-_STDERR_TAIL_CHARS:]


class BoundedProgress:
    """Emit a fixed maximum of normalized progress events for one session.

    Native streaming transports can emit unbounded frames.  Past the cap the
    session still refreshes its idle deadline, but stops appending events so a
    long run cannot exhaust the runtime's fixed event budget.
    """

    def __init__(self, context: HeadlessRuntimeContext, *, limit: int = _MAX_PROGRESS_EVENTS):
        self._context = context
        self._limit = limit
        self._emitted = 0

    def note(self) -> None:
        """Record one fixed progress milestone without exposing native frame data."""
        if self._emitted >= self._limit:
            self._context.mark_progress()
            return
        self._emitted += 1
        self._context.emit(HeadlessEventKind.PROGRESS)


def run_managed_command(
    context: HeadlessRuntimeContext,
    *,
    argv: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    stdin_text: str | None = None,
    on_stdout_lines: StdoutLineHandler | None = None,
) -> HeadlessCommandOutput:
    """Run one owned native CLI process under the managed headless boundary.

    ``stdin_text`` is delivered on a pipe that is closed immediately afterwards.
    A broken or short write fails the transport rather than allowing a native
    response for only part of the requested prompt. When it is ``None`` the
    child receives ``/dev/null``: an unattended session never inherits arbitrary
    parent terminal input, so a native prompt can never block on a caller that
    is not there.

    ``on_stdout_lines`` receives every complete native line while the child is
    still running, from this same thread.  Live delivery is what lets a long
    streaming run refresh its idle deadline and keep safe provenance in the
    partial result when the overall deadline fires.
    """
    context.checkpoint()
    stdin_bytes = stdin_text.encode("utf-8") if stdin_text is not None else None
    proc: subprocess.Popen[bytes] | None = None
    proc_lock = threading.Lock()
    workers: list[threading.Thread] = []
    input_stream: IO[bytes] | None = None
    cleanup_requested = threading.Event()

    def current_process() -> subprocess.Popen[bytes] | None:
        with proc_lock:
            return proc

    def close_input(_cleanup: HeadlessCleanupContext) -> None:
        # This is the first cleanup hook. Remember the state so a child that
        # finishes spawning after cleanup has begun is synchronously torn down.
        cleanup_requested.set()
        if input_stream is not None and not input_stream.closed:
            with suppress(OSError):
                input_stream.close()

    def terminate(_cleanup: HeadlessCleanupContext) -> None:
        child = current_process()
        if child is not None and child.poll() is None:
            with suppress(OSError):
                child.terminate()

    def force_kill(_cleanup: HeadlessCleanupContext) -> None:
        owned_process = current_process()
        if owned_process is not None:
            kill_process_group(owned_process)

    def reap(cleanup: HeadlessCleanupContext) -> None:
        child = current_process()
        remaining = cleanup.remaining_seconds()
        if child is not None and remaining > 0:
            with suppress(subprocess.TimeoutExpired):
                child.wait(timeout=remaining)

    def join_workers(cleanup: HeadlessCleanupContext) -> None:
        join_threads_until(tuple(workers), cleanup.deadline)

    # Popen itself can block. Register hooks first so a deadline or cancellation
    # that fires in that window cannot strand a child once Popen returns.
    context.register_cleanup(
        HeadlessCleanupHooks(
            close_input=close_input,
            terminate=terminate,
            force_kill=force_kill,
            reap=reap,
            join_workers=join_workers,
        )
    )
    try:
        child = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **process_tree_popen_kwargs(),
        )
        with proc_lock:
            proc = child
        own_process_tree(child)
    except (OSError, ValueError):
        owned_process = current_process()
        if owned_process is not None:
            kill_process_group(owned_process)
        raise context.transport_error(
            "The managed headless transport could not start the native CLI."
        ) from None
    assert proc is not None
    if cleanup_requested.is_set():
        kill_process_group(proc)
    # Cleanup may have run while Popen blocked. This checkpoint either lets the
    # normal launch continue or returns the already-published runtime outcome;
    # the process above is always torn down first in the latter case.
    context.checkpoint()
    if (
        proc.stdout is None
        or proc.stderr is None
        or (stdin_bytes is not None and proc.stdin is None)
    ):
        kill_process_group(proc)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_WORKER_JOIN_GRACE_SECONDS)
        raise context.transport_error(
            "The managed headless transport could not create native CLI pipes."
        )

    buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    state_lock = threading.Lock()
    overflowed = False
    stdin_delivery_failed = threading.Event()
    input_stream = proc.stdin

    def read_bounded(stream: IO[bytes], name: str, limit: int) -> None:
        nonlocal overflowed
        # Read whatever is already available instead of a full buffered chunk,
        # so a still-running streaming transport is observable line by line.
        descriptor = stream.fileno()
        while True:
            try:
                chunk = os.read(descriptor, _READ_CHUNK)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            with state_lock:
                remaining = limit - len(buffers[name])
                if remaining > 0:
                    buffers[name].extend(chunk[:remaining])
                if len(chunk) <= remaining:
                    continue
                if overflowed:
                    return
                overflowed = True
            kill_process_group(proc)
            return

    workers.append(
        threading.Thread(
            target=read_bounded,
            args=(proc.stdout, "stdout", _STDOUT_LIMIT),
            daemon=True,
            name="crossby-headless-cli-stdout",
        )
    )
    workers.append(
        threading.Thread(
            target=read_bounded,
            args=(proc.stderr, "stderr", _STDERR_LIMIT),
            daemon=True,
            name="crossby-headless-cli-stderr",
        )
    )

    if stdin_bytes is not None:
        assert input_stream is not None
        prompt_stream = input_stream

        def write_prompt() -> None:
            try:
                written = prompt_stream.write(stdin_bytes)
                if written != len(stdin_bytes):
                    stdin_delivery_failed.set()
                    return
                prompt_stream.flush()
            except (BrokenPipeError, OSError, ValueError):
                stdin_delivery_failed.set()
            finally:
                # Closing is part of delivery: a native CLI that reads its whole
                # prompt from stdin only starts once the stream reaches EOF.
                with suppress(OSError, ValueError):
                    prompt_stream.close()

        workers.append(
            threading.Thread(
                target=write_prompt,
                daemon=True,
                name="crossby-headless-cli-stdin",
            )
        )

    for worker in workers:
        worker.start()
    context.emit(HeadlessEventKind.STARTED)

    consumed = 0

    def drain_stdout_lines() -> None:
        """Hand completed native lines to the adapter from this thread only."""
        nonlocal consumed
        if on_stdout_lines is None:
            return
        with state_lock:
            snapshot = bytes(buffers["stdout"])
        # A newline is always a safe UTF-8 boundary, so a partial frame is
        # never handed over and never decoded twice.
        end = snapshot.rfind(b"\n") + 1
        if end <= consumed:
            return
        chunk = snapshot[consumed:end]
        consumed = end
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            # The final decode reports the undecodable capture authoritatively.
            return
        lines = [line for line in text.splitlines() if line.strip()]
        if lines:
            on_stdout_lines(lines)

    while True:
        remaining = context.remaining_seconds()
        try:
            returncode = proc.wait(timeout=min(_WAIT_POLL_SECONDS, remaining))
            break
        except subprocess.TimeoutExpired:
            drain_stdout_lines()
            continue

    # A direct child can exit while a descendant still holds the capture pipes.
    # Bound that with a fixed grace, then clear the whole owned group.
    if not join_threads_until(tuple(workers), time.monotonic() + _WORKER_JOIN_GRACE_SECONDS):
        kill_process_group(proc)
        join_threads_until(tuple(workers), time.monotonic() + _WORKER_JOIN_GRACE_SECONDS)
    kill_process_group(proc)
    if stdin_delivery_failed.is_set():
        raise context.transport_error(
            "The managed headless transport could not deliver the complete prompt "
            "to the native CLI."
        )
    # A short-lived child can exit before any poll observed its frames.
    drain_stdout_lines()

    decoded: dict[str, str] = {}
    undecodable = False
    for name, buffer in buffers.items():
        try:
            decoded[name] = buffer.decode("utf-8")
        except UnicodeDecodeError:
            undecodable = True
            decoded[name] = buffer.decode("utf-8", errors="replace")
    return HeadlessCommandOutput(
        returncode=returncode,
        stdout=decoded["stdout"],
        stderr=decoded["stderr"],
        overflowed=overflowed,
        undecodable=undecodable,
    )


def frame_streamer(
    context: HeadlessRuntimeContext,
    *,
    kind_of: Callable[[dict[str, Any]], str | None],
    provenance_of: Callable[[dict[str, Any]], dict[str, str | None]],
) -> StdoutLineHandler:
    """Turn live native JSONL frames into provenance and bounded progress.

    Native frames echo prompt and response text, so their data only determines
    whether to emit a content-free progress event. Provenance is recorded as
    soon as the transport emits it, which is what a timed-out session keeps in
    its safe partial result.
    """
    progress = BoundedProgress(context)

    def consume(lines: Sequence[str]) -> None:
        for line in lines:
            frame = parse_json_object(line)
            if frame is None:
                continue
            observed = provenance_of(frame)
            context.validate_provenance(
                session_id=observed.get("session_id"),
                thread_id=observed.get("thread_id"),
                turn_id=observed.get("turn_id"),
                conversation_id=observed.get("conversation_id"),
            )
            context.set_provenance(
                session_id=observed.get("session_id"),
                thread_id=observed.get("thread_id"),
                turn_id=observed.get("turn_id"),
                conversation_id=observed.get("conversation_id"),
            )
            kind = kind_of(frame)
            if kind:
                progress.note()

    return consume


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse one native JSON object envelope, or ``None`` when malformed."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_json_lines(text: str) -> list[dict[str, Any]] | None:
    """Parse native JSONL object frames, or ``None`` when any line is malformed."""
    frames: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict):
            return None
        frames.append(value)
    return frames


def optional_int(value: Any) -> int | None:
    """Return a non-negative native token count, ignoring any other shape."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def usage_from(
    values: Any,
    *,
    total: str | None = None,
    input_tokens: str | None = None,
    output_tokens: str | None = None,
    cached: str | None = None,
    session_id: str | None = None,
) -> TokenUsage | None:
    """Normalize one native usage object, or ``None`` when nothing is usable."""
    if not isinstance(values, Mapping):
        return None
    usage = TokenUsage(
        total_tokens=optional_int(values.get(total)) if total else None,
        input_tokens=optional_int(values.get(input_tokens)) if input_tokens else None,
        output_tokens=optional_int(values.get(output_tokens)) if output_tokens else None,
        cached_tokens=optional_int(values.get(cached)) if cached else None,
        session_id=session_id,
    )
    if (
        usage.total_tokens is None
        and usage.input_tokens is None
        and usage.output_tokens is None
        and usage.cached_tokens is None
    ):
        return None
    return usage


def non_blank_text(value: Any) -> str | None:
    """Return native text only when it is a usable non-blank string."""
    return value if isinstance(value, str) and value.strip() else None


def complete_session(
    context: HeadlessRuntimeContext,
    request: HeadlessSessionRequest,
    *,
    status: HeadlessTerminalStatus,
    exit_code: int,
    response_text: str | None = None,
    native_object: Any = MISSING,
    structured_output: Any = MISSING,
    native_status: str | None = None,
    session_id: str | None = None,
    conversation_id: str | None = None,
    thread_id: str | None = None,
    usage: TokenUsage | None = None,
    warnings: tuple[str, ...] = (),
) -> HeadlessSessionResult:
    """Build the one terminal result in the caller's requested output shape.

    A caller-provided response schema always selects the adapter's native
    structured output, which the runtime then validates.  Otherwise ``TEXT``
    returns the native final response text and ``JSON``/``JSONL`` return the
    native object that carried it, exactly as the CLI emitted it.
    """
    provenance: dict[str, Any] = {
        "exit_code": exit_code,
        "native_status": native_status,
        "session_id": session_id,
        "conversation_id": conversation_id,
        "thread_id": thread_id,
        "usage": usage,
    }
    if status is not HeadlessTerminalStatus.SUCCEEDED:
        return context.complete(status, warnings=warnings, **provenance)
    if request.response_schema is not None:
        if structured_output is MISSING:
            return context.complete(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                warnings=(*warnings, "The native transport omitted schema-constrained output."),
                **provenance,
            )
        return context.complete(final_json=structured_output, warnings=warnings, **provenance)
    if request.native_output is HeadlessNativeOutput.TEXT:
        return context.complete(final_text=response_text, warnings=warnings, **provenance)
    if native_object is MISSING:
        return context.complete(
            HeadlessTerminalStatus.INVALID_OUTPUT,
            warnings=(*warnings, "The native transport omitted its machine-readable result."),
            **provenance,
        )
    return context.complete(final_json=native_object, warnings=warnings, **provenance)


def capture_failure_warnings(output: HeadlessCommandOutput) -> tuple[str, ...]:
    """Describe a native capture failure without exposing raw native frames."""
    warnings: list[str] = []
    if output.overflowed:
        warnings.append("The native CLI exceeded the fixed captured-output limit.")
    if output.undecodable:
        warnings.append("The native CLI emitted output that was not valid UTF-8 text.")
    if output.stderr:
        warnings.append(
            "The native CLI emitted diagnostics that are withheld to protect session content."
        )
    return tuple(warnings)


__all__ = [
    "MISSING",
    "BoundedProgress",
    "HeadlessCommandOutput",
    "StdoutLineHandler",
    "capture_failure_warnings",
    "complete_session",
    "frame_streamer",
    "non_blank_text",
    "optional_int",
    "parse_json_lines",
    "parse_json_object",
    "run_managed_command",
    "usage_from",
]
