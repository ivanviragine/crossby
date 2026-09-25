"""Shared bounded process and wire helpers for managed AI-tool sessions.

The historical module path remains because plan collectors and downstream
tests import it directly.  ``crossby.ai_tools.session_process`` is an alias to
this same module, so both APIs share limits, monkeypatches, and cleanup code.
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, ClassVar

_STDERR_TAIL_LIMIT = 8 * 1024
_STDOUT_QUEUE_LIMIT = 256
_QUEUE_PUT_TIMEOUT_SECONDS = 0.1
_JSON_RPC_FRAME_LIMIT = 1024 * 1024
_JSON_RPC_READ_CHUNK_SIZE = 64 * 1024
_CAPTURED_STDOUT_LIMIT = 8 * 1024 * 1024
_CAPTURED_STDERR_LIMIT = 1024 * 1024
_CAPTURE_CHUNK_SIZE = 64 * 1024
_CAPTURE_CLEANUP_GRACE_SECONDS = 0.2
_PLAN_ARTIFACT_TEXT_LIMIT = 8 * 1024 * 1024

_WINDOWS_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_CLOSE = 0x00002000
_WINDOWS_INVALID_DWORD = 0xFFFFFFFF
_WINDOWS_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_WINDOWS_SNAPSHOT_THREADS = 0x00000004
_WINDOWS_THREAD_SUSPEND_RESUME = 0x0002


class _WindowsJobBasicLimits(ctypes.Structure):
    """Layout of the limits prefix used by Windows extended job information."""

    _fields_ = [
        ("per_process_user_time_limit", ctypes.c_longlong),
        ("per_job_user_time_limit", ctypes.c_longlong),
        ("limit_flags", ctypes.c_uint32),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t),
        ("priority_class", ctypes.c_uint32),
        ("scheduling_class", ctypes.c_uint32),
    ]


class _WindowsIoCounters(ctypes.Structure):
    _fields_ = [
        ("read_operation_count", ctypes.c_ulonglong),
        ("write_operation_count", ctypes.c_ulonglong),
        ("other_operation_count", ctypes.c_ulonglong),
        ("read_transfer_count", ctypes.c_ulonglong),
        ("write_transfer_count", ctypes.c_ulonglong),
        ("other_transfer_count", ctypes.c_ulonglong),
    ]


class _WindowsExtendedJobLimits(ctypes.Structure):
    _fields_ = [
        ("basic_limit_information", _WindowsJobBasicLimits),
        ("io_info", _WindowsIoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ]


class _WindowsThreadEntry(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("usage", ctypes.c_uint32),
        ("thread_id", ctypes.c_uint32),
        ("owner_process_id", ctypes.c_uint32),
        ("base_priority", ctypes.c_long),
        ("delta_priority", ctypes.c_long),
        ("flags", ctypes.c_uint32),
    ]


class _WindowsProcessTree:
    """A Windows Job Object that terminates every owned descendant on close."""

    def __init__(self, handle: int, close_handle: Callable[[int], object]) -> None:
        self._handle = handle
        self._close_handle = close_handle
        self._lock = threading.Lock()
        self._closed = False

    def close(self) -> None:
        """Close once; ``KILL_ON_JOB_CLOSE`` ends the complete process tree."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        with suppress(OSError):
            self._close_handle(self._handle)


@dataclass(frozen=True)
class CapturedProcess:
    """Captured output and exact exit status from a bounded child."""

    returncode: int
    stdout: str
    stderr: str


class CapturedOutputLimitError(subprocess.SubprocessError):
    """A captured child exceeded the configured in-memory output bound."""

    def __init__(self, stream: str, limit: int) -> None:
        super().__init__(f"captured {stream} exceeded the {limit}-byte limit")
        self.stream = stream
        self.limit = limit


class CapturedOutputDecodeError(subprocess.SubprocessError):
    """A captured child emitted bytes invalid for the native UTF-8 contract."""

    def __init__(self, stream: str, encoding: str) -> None:
        super().__init__(f"captured {stream} was not valid {encoding} text")
        self.stream = stream
        self.encoding = encoding


class SessionProcessCancelledError(subprocess.SubprocessError):
    """A caller cancellation event stopped an owned child process group."""


class PlanArtifactSizeError(OSError):
    """A file-backed plan artifact exceeded its in-memory text bound."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"plan artifact exceeded the {limit}-byte limit")
        self.limit = limit


class JsonRpcFrameLimitError(ValueError):
    """A protocol child emitted a frame larger than the configured bound."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"JSON-RPC frame exceeded the {limit}-character limit")
        self.limit = limit


def read_text_bounded(
    path: Path,
    *,
    limit: int | None = None,
    dir_fd: int | None = None,
    expected_stat: os.stat_result | None = None,
) -> str:
    """Read a stable regular UTF-8 artifact with a fixed byte cap.

    A directory descriptor anchors relative paths to the run directory that was
    opened before launch. Identity checks also protect callers on platforms
    without descriptor-relative filesystem operations.
    """
    effective_limit = _PLAN_ARTIFACT_TEXT_LIMIT if limit is None else limit
    if effective_limit <= 0:
        raise ValueError("plan artifact limit must be positive")
    expected = expected_stat or os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
    if not stat.S_ISREG(expected.st_mode):
        raise OSError("plan artifact must be a regular, non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags, dir_fd=dir_fd)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(expected, opened):
            raise OSError("plan artifact was replaced before reading")
        current = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or not os.path.samestat(opened, current):
            raise OSError("plan artifact was replaced before reading")
        content = stream.read(effective_limit + 1)
    if len(content) > effective_limit:
        raise PlanArtifactSizeError(effective_limit)
    return content.decode("utf-8")


def _is_windows() -> bool:
    return os.name == "nt"


def process_tree_popen_kwargs() -> dict[str, Any]:
    """Return launch options that let the managed transport own descendant processes."""
    if _is_windows():
        # Claim the child in a Job Object before it can spawn an unowned descendant.
        return {"creationflags": getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)}
    return {"start_new_session": os.name == "posix"}


def own_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Attach a just-created suspended Windows child to its owned process tree."""
    if not _is_windows():
        return
    tree = _create_windows_process_tree(proc)
    process: Any = proc
    try:
        process._crossby_windows_process_tree = tree
        _resume_windows_process(proc)
    except Exception:
        tree.close()
        process._crossby_windows_process_tree = None
        raise


def _create_windows_process_tree(proc: subprocess.Popen[Any]) -> _WindowsProcessTree:
    """Create a kill-on-close Job Object and assign the suspended child to it."""
    windll = getattr(ctypes, "WinDLL", None)
    if windll is None:
        raise OSError("Windows Job Objects are unavailable on this Python runtime.")
    kernel32 = windll("kernel32", use_last_error=True)
    create_job = kernel32.CreateJobObjectW
    create_job.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
    create_job.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    handle = create_job(None, None)
    if not handle:
        _raise_windows_error("CreateJobObjectW")
    tree = _WindowsProcessTree(int(handle), close_handle)
    try:
        limits = _WindowsExtendedJobLimits()
        limits.basic_limit_information.limit_flags = _WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_CLOSE
        set_information = kernel32.SetInformationJobObject
        set_information.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
        set_information.restype = ctypes.c_int
        if not set_information(
            handle,
            _WINDOWS_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            _raise_windows_error("SetInformationJobObject")
        process_handle = getattr(proc, "_handle", None)
        if not isinstance(process_handle, int) or process_handle == 0:
            raise OSError("Windows subprocess does not expose a process handle for Job ownership.")
        assign_process = kernel32.AssignProcessToJobObject
        assign_process.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        assign_process.restype = ctypes.c_int
        if not assign_process(handle, process_handle):
            _raise_windows_error("AssignProcessToJobObject")
    except Exception:
        tree.close()
        raise
    return tree


def _resume_windows_process(proc: subprocess.Popen[Any]) -> None:
    """Resume a child held at creation until its Job Object owns its descendants."""
    windll = getattr(ctypes, "WinDLL", None)
    if windll is None:
        raise OSError("Windows thread controls are unavailable on this Python runtime.")
    kernel32 = windll("kernel32", use_last_error=True)
    snapshot_threads = kernel32.CreateToolhelp32Snapshot
    snapshot_threads.argtypes = (ctypes.c_uint32, ctypes.c_uint32)
    snapshot_threads.restype = ctypes.c_void_p
    snapshot = snapshot_threads(_WINDOWS_SNAPSHOT_THREADS, 0)
    if not snapshot or snapshot == _WINDOWS_INVALID_HANDLE_VALUE:
        _raise_windows_error("CreateToolhelp32Snapshot")
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    try:
        thread_id = _windows_primary_thread_id(kernel32, snapshot, proc.pid)
        open_thread = kernel32.OpenThread
        open_thread.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        open_thread.restype = ctypes.c_void_p
        thread_handle = open_thread(_WINDOWS_THREAD_SUSPEND_RESUME, False, thread_id)
        if not thread_handle:
            _raise_windows_error("OpenThread")
    finally:
        close_handle(snapshot)
    resume_thread = kernel32.ResumeThread
    resume_thread.argtypes = (ctypes.c_void_p,)
    resume_thread.restype = ctypes.c_uint32
    try:
        if resume_thread(thread_handle) == _WINDOWS_INVALID_DWORD:
            _raise_windows_error("ResumeThread")
    finally:
        close_handle(thread_handle)


def _windows_primary_thread_id(kernel32: Any, snapshot: int, process_id: int) -> int:
    """Find the only user thread a newly ``CREATE_SUSPENDED`` child can have."""
    first_thread = kernel32.Thread32First
    first_thread.argtypes = (ctypes.c_void_p, ctypes.POINTER(_WindowsThreadEntry))
    first_thread.restype = ctypes.c_int
    next_thread = kernel32.Thread32Next
    next_thread.argtypes = (ctypes.c_void_p, ctypes.POINTER(_WindowsThreadEntry))
    next_thread.restype = ctypes.c_int
    entry = _WindowsThreadEntry()
    entry.size = ctypes.sizeof(entry)
    if not first_thread(snapshot, ctypes.byref(entry)):
        _raise_windows_error("Thread32First")
    while True:
        if entry.owner_process_id == process_id:
            return int(entry.thread_id)
        entry.size = ctypes.sizeof(entry)
        if not next_thread(snapshot, ctypes.byref(entry)):
            break
    raise OSError("Could not find the suspended Windows subprocess primary thread.")


def _raise_windows_error(action: str) -> None:
    get_last_error = getattr(ctypes, "get_last_error", None)
    error = int(get_last_error()) if get_last_error is not None else 0
    raise OSError(f"{action} failed with Windows error {error}.")


def _kill_process_group(proc: subprocess.Popen[Any]) -> None:
    """Kill a run-owned process group or tree so descendants cannot outlive it."""
    if _is_windows():
        tree = getattr(proc, "_crossby_windows_process_tree", None)
        if isinstance(tree, _WindowsProcessTree):
            tree.close()
            return
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except OSError:
            pass
    if proc.poll() is None:
        with suppress(OSError):
            proc.kill()


def _join_until(threads: tuple[threading.Thread, ...], deadline: float) -> bool:
    """Join workers against one deadline rather than one timeout per thread."""
    for thread in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        thread.join(timeout=remaining)
    return all(not thread.is_alive() for thread in threads)


def run_captured(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    native_abort: Callable[[], object] | None = None,
) -> CapturedProcess:
    """Run a bounded child with hard output limits and one cleanup path.

    ``deadline`` lets a caller carry an already-running absolute budget across
    preflight, startup, prompt delivery, and collection.  ``timeout`` remains
    for compatibility and can only shorten that deadline.
    """
    computed_deadline = time.monotonic() + timeout
    if deadline is not None:
        computed_deadline = min(computed_deadline, deadline)
    encoding = "utf-8"
    input_bytes = input_text.encode(encoding) if input_text is not None else None
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=os.name == "posix",
    )
    if (
        proc.stdout is None
        or proc.stderr is None
        or (input_bytes is not None and proc.stdin is None)
    ):
        _kill_process_group(proc)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
        raise OSError("failed to create captured subprocess pipes")

    outputs: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    overflow: list[tuple[str, int]] = []
    overflow_lock = threading.Lock()
    abort_lock = threading.Lock()
    abort_called = False

    def abort_once() -> None:
        nonlocal abort_called
        if native_abort is None:
            return
        with abort_lock:
            if abort_called:
                return
            abort_called = True
        abort_finished = threading.Event()

        def invoke_abort() -> None:
            with suppress(BaseException):
                native_abort()
            abort_finished.set()

        thread = threading.Thread(target=invoke_abort, daemon=True)
        thread.start()
        abort_finished.wait(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)

    def read_bounded(stream: IO[bytes], name: str, limit: int) -> None:
        while chunk := stream.read(_CAPTURE_CHUNK_SIZE):
            remaining = limit - len(outputs[name])
            if remaining > 0:
                outputs[name].extend(chunk[:remaining])
            if len(chunk) > remaining:
                with overflow_lock:
                    if not overflow:
                        overflow.append((name, limit))
                        abort_once()
                        _kill_process_group(proc)

    readers = (
        threading.Thread(
            target=read_bounded,
            args=(proc.stdout, "stdout", _CAPTURED_STDOUT_LIMIT),
            daemon=True,
        ),
        threading.Thread(
            target=read_bounded,
            args=(proc.stderr, "stderr", _CAPTURED_STDERR_LIMIT),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()

    writer: threading.Thread | None = None
    if input_bytes is not None:
        input_stream = proc.stdin
        assert input_stream is not None

        def write_input() -> None:
            try:
                input_stream.write(input_bytes)
                input_stream.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                with suppress(OSError):
                    input_stream.close()

        writer = threading.Thread(target=write_input, daemon=True)
        writer.start()

    workers = (*readers, *((writer,) if writer is not None else ()))
    try:
        remaining = computed_deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout)
        if cancel_event is None:
            returncode = proc.wait(timeout=remaining)
        else:
            while True:
                if cancel_event.is_set():
                    raise SessionProcessCancelledError("captured child was cancelled")
                remaining = computed_deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    returncode = proc.wait(timeout=min(_QUEUE_PUT_TIMEOUT_SECONDS, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue

        # A direct child may exit while a descendant retains inherited capture
        # descriptors.  Bound that ambiguity by fixed cleanup grace rather than
        # consuming the rest of a long session deadline; the timeout cleanup
        # below then clears the whole owned group.
        join_deadline = min(
            computed_deadline,
            time.monotonic() + _CAPTURE_CLEANUP_GRACE_SECONDS,
        )
        if not _join_until(workers, join_deadline):
            if overflow:
                raise CapturedOutputLimitError(*overflow[0])
            raise subprocess.TimeoutExpired(command, timeout)
        _kill_process_group(proc)
        if overflow:
            raise CapturedOutputLimitError(*overflow[0])

        decoded: dict[str, str] = {}
        for name, output in outputs.items():
            try:
                decoded[name] = output.decode(encoding)
            except UnicodeDecodeError as exc:
                raise CapturedOutputDecodeError(name, encoding) from exc

        return CapturedProcess(returncode, decoded["stdout"], decoded["stderr"])
    except subprocess.TimeoutExpired:
        abort_once()
        _kill_process_group(proc)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
        _join_until(workers, time.monotonic() + _CAPTURE_CLEANUP_GRACE_SECONDS)
        raise subprocess.TimeoutExpired(command, timeout) from None
    except SessionProcessCancelledError:
        abort_once()
        _kill_process_group(proc)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
        _join_until(workers, time.monotonic() + _CAPTURE_CLEANUP_GRACE_SECONDS)
        raise
    except BaseException:
        abort_once()
        _kill_process_group(proc)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
        _join_until(workers, time.monotonic() + _CAPTURE_CLEANUP_GRACE_SECONDS)
        raise


def run_interactive(command: list[str], *, cwd: Path, timeout: float) -> int:
    """Run a bounded process group attached to the caller's terminal."""
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        start_new_session=os.name == "posix",
    )
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
        raise subprocess.TimeoutExpired(command, timeout) from None
    except BaseException:
        _kill_process_group(proc)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
        raise
    _kill_process_group(proc)
    return returncode


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    """Parse non-blank JSONL lines, rejecting malformed/non-object envelopes."""
    events: list[dict[str, Any]] = []
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSONL at line {number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"malformed JSONL at line {number}: expected an object")
        events.append(value)
    return events


class JsonRpcProcess:
    """Line-delimited JSON-RPC stdio process with bounded reads and cleanup."""

    _include_jsonrpc_version: ClassVar[bool] = True

    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout: float = 600.0,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.command = command
        computed_deadline = time.monotonic() + timeout
        self._deadline = (
            min(computed_deadline, deadline) if deadline is not None else computed_deadline
        )
        self._cancel_event = cancel_event
        self._write_thread: threading.Thread | None = None
        self._proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
            start_new_session=os.name == "posix",
        )
        if self._proc.stdin is None or self._proc.stdout is None or self._proc.stderr is None:
            _kill_process_group(self._proc)
            with suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
            raise OSError("failed to create JSON-RPC stdio pipes")
        self._stdin: IO[str] = self._proc.stdin
        self._stdout_queue: queue.Queue[str | ValueError | None] = queue.Queue(
            maxsize=_STDOUT_QUEUE_LIMIT
        )
        self._stderr_tail = ""
        self._stderr_lock = threading.Lock()
        self._closing = threading.Event()
        self._discard_stdout = threading.Event()
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(self._proc.stdout,),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(self._proc.stderr,),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    @property
    def returncode(self) -> int | None:
        return self._proc.poll()

    @property
    def stderr(self) -> str:
        with self._stderr_lock:
            return self._stderr_tail

    def _read_stdout(self, stream: IO[str]) -> None:
        try:
            while line := stream.readline(_JSON_RPC_FRAME_LIMIT + 1):
                if self._discard_stdout.is_set():
                    continue
                if len(line) > _JSON_RPC_FRAME_LIMIT:
                    proc = getattr(self, "_proc", None)
                    if proc is not None:
                        _kill_process_group(proc)
                    self._queue_stdout(JsonRpcFrameLimitError(_JSON_RPC_FRAME_LIMIT))
                    break
                if not self._queue_stdout(line):
                    break
        except ValueError as exc:
            if not self._closing.is_set():
                self._queue_stdout(exc)
        finally:
            self._queue_stdout(None)

    def _queue_stdout(self, line: str | ValueError | None) -> bool:
        """Queue one stdout record, applying backpressure until read or closed."""
        while not self._closing.is_set():
            if self._discard_stdout.is_set():
                return True
            try:
                self._stdout_queue.put(line, timeout=_QUEUE_PUT_TIMEOUT_SECONDS)
            except queue.Full:
                continue
            return True
        return False

    def discard_stdout(self) -> None:
        """Drain queued stdout and discard subsequent records until shutdown."""
        self._discard_stdout.set()
        while True:
            try:
                self._stdout_queue.get_nowait()
            except queue.Empty:
                return

    def _read_stderr(self, stream: IO[str]) -> None:
        try:
            while chunk := stream.readline(_JSON_RPC_READ_CHUNK_SIZE):
                chunk_tail = chunk[-_STDERR_TAIL_LIMIT:]
                with self._stderr_lock:
                    self._stderr_tail = (self._stderr_tail + chunk_tail)[-_STDERR_TAIL_LIMIT:]
        except ValueError:
            if not self._closing.is_set():
                raise

    def send(self, payload: dict[str, Any]) -> None:
        cancel_event = getattr(self, "_cancel_event", None)
        if cancel_event is not None and cancel_event.is_set():
            raise SessionProcessCancelledError("JSON-RPC child was cancelled")
        if self._proc.poll() is not None:
            raise EOFError(f"JSON-RPC child exited with status {self._proc.returncode}")
        envelope = {"jsonrpc": "2.0", **payload} if self._include_jsonrpc_version else payload
        line = json.dumps(envelope, separators=(",", ":")) + "\n"
        errors: list[OSError | ValueError] = []

        def write() -> None:
            try:
                self._stdin.write(line)
                self._stdin.flush()
            except (OSError, ValueError) as exc:
                errors.append(exc)

        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out writing a JSON-RPC message")
        self._write_thread = threading.Thread(target=write, daemon=True)
        self._write_thread.start()
        while self._write_thread.is_alive():
            cancel_event = getattr(self, "_cancel_event", None)
            if cancel_event is not None and cancel_event.is_set():
                _kill_process_group(self._proc)
                self._write_thread.join(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
                raise SessionProcessCancelledError("JSON-RPC child was cancelled")
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                break
            self._write_thread.join(timeout=min(_QUEUE_PUT_TIMEOUT_SECONDS, remaining))
        if self._write_thread.is_alive():
            _kill_process_group(self._proc)
            self._write_thread.join(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
            raise TimeoutError("timed out writing a JSON-RPC message")
        if errors:
            raise errors[0]

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"method": method}
        if params is not None:
            payload["params"] = params
        self.send(payload)

    def request(self, request_id: int, method: str, params: dict[str, Any]) -> None:
        self.send(
            {
                "id": request_id,
                "method": method,
                "params": params,
            }
        )

    def respond(self, request_id: object, result: Any) -> None:
        self.send({"id": request_id, "result": result})

    def read_line(self, *, timeout: float) -> str:
        """Read a bounded stdout line, also usable for a server's startup banner."""
        read_deadline = min(
            getattr(self, "_deadline", float("inf")),
            time.monotonic() + timeout,
        )
        while True:
            cancel_event = getattr(self, "_cancel_event", None)
            if cancel_event is not None and cancel_event.is_set():
                raise SessionProcessCancelledError("JSON-RPC child was cancelled")
            remaining = read_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for a JSON-RPC message")
            try:
                line = self._stdout_queue.get(timeout=min(_QUEUE_PUT_TIMEOUT_SECONDS, remaining))
            except queue.Empty:
                continue
            break
        if line is None:
            code = self._proc.poll()
            raise EOFError(f"JSON-RPC stream closed (exit status {code})")
        if isinstance(line, ValueError):
            raise line
        return line

    def read(self, *, timeout: float) -> dict[str, Any]:
        line = self.read_line(timeout=timeout)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON-RPC envelope: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError("malformed JSON-RPC envelope: expected an object")
        if self._include_jsonrpc_version:
            if payload.get("jsonrpc") != "2.0":
                raise ValueError("malformed JSON-RPC envelope: expected jsonrpc='2.0'")
        elif "jsonrpc" in payload:
            raise ValueError("malformed headerless JSON-RPC envelope: unexpected jsonrpc field")
        if "id" not in payload and "method" not in payload:
            raise ValueError("malformed JSON-RPC envelope: missing both id and method")
        return payload

    def close(self) -> int:
        """Stop the server and its helpers, then close only drained reader pipes."""
        writer_finished = self._write_thread is None or not self._write_thread.is_alive()
        if writer_finished and not self._stdin.closed:
            with suppress(OSError):
                self._stdin.close()
        if self._proc.poll() is None:
            remaining = self._deadline - time.monotonic()
            if remaining > 0:
                with suppress(subprocess.TimeoutExpired):
                    self._proc.wait(timeout=min(1.0, remaining))
        # The server may already have exited while a helper still owns a pipe.
        _kill_process_group(self._proc)
        if self._proc.poll() is None:
            # Reaping is cleanup, not session work: retain one fixed grace after
            # SIGKILL even when the request deadline has already expired.
            with suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
        if self._write_thread is not None:
            remaining = self._deadline - time.monotonic()
            if remaining > 0:
                self._write_thread.join(timeout=min(_CAPTURE_CLEANUP_GRACE_SECONDS, remaining))
            if not self._write_thread.is_alive() and not self._stdin.closed:
                with suppress(OSError):
                    self._stdin.close()
        self._closing.set()
        readers = (self._stdout_thread, self._stderr_thread)
        _join_until(
            readers,
            min(
                self._deadline,
                time.monotonic() + _CAPTURE_CLEANUP_GRACE_SECONDS,
            ),
        )
        for stream, thread in zip((self._proc.stdout, self._proc.stderr), readers, strict=True):
            # Closing an IO wrapper while its reader holds the lock can block
            # forever (including on non-POSIX hosts without group cleanup).
            if stream is not None and not thread.is_alive() and not stream.closed:
                stream.close()
        returncode = self._proc.poll()
        # Never report success while the child is still observable as alive.
        # The process group has already received SIGKILL, so use that status as
        # the fail-closed result when the OS has not reaped the direct child yet.
        return returncode if returncode is not None else -getattr(signal, "SIGKILL", 9)

    def __enter__(self) -> JsonRpcProcess:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class HeaderlessJsonRpcProcess(JsonRpcProcess):
    """Codex app-server's JSONL dialect, which omits the JSON-RPC version field."""

    _include_jsonrpc_version = False


# Neutral names for new ordinary-session transports.  Plan-prefixed names and
# this module path remain stable for collected-plan consumers.
SessionArtifactSizeError = PlanArtifactSizeError
CapturedSessionProcess = CapturedProcess
# Public spellings of the owned-process primitives shared with the managed
# headless transport, which owns its own cleanup ordering instead of reusing
# ``run_captured``'s single blocking call.
kill_process_group = _kill_process_group
join_threads_until = _join_until


def child_environment(extra: dict[str, str] | None = None) -> dict[str, str] | None:
    """Merge adapter environment additions without mutating the parent."""
    return {**os.environ, **extra} if extra else None
