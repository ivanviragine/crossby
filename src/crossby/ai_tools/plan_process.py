"""Small stdlib process and wire helpers for native plan collectors."""

from __future__ import annotations

import json
import locale
import os
import queue
import signal
import stat
import subprocess
import threading
import time
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
    """A captured child emitted bytes invalid for the selected locale encoding."""

    def __init__(self, stream: str, encoding: str) -> None:
        super().__init__(f"captured {stream} was not valid {encoding} text")
        self.stream = stream
        self.encoding = encoding


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


def _kill_process_group(proc: subprocess.Popen[Any]) -> None:
    """Kill a run-owned process group so descendants cannot outlive the deadline."""
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
) -> CapturedProcess:
    """Run a bounded child with hard stdout/stderr memory limits and no shell."""
    deadline = time.monotonic() + timeout
    encoding = locale.getpreferredencoding(False)
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

    def read_bounded(stream: IO[bytes], name: str, limit: int) -> None:
        while chunk := stream.read(_CAPTURE_CHUNK_SIZE):
            remaining = limit - len(outputs[name])
            if remaining > 0:
                outputs[name].extend(chunk[:remaining])
            if len(chunk) > remaining:
                with overflow_lock:
                    if not overflow:
                        overflow.append((name, limit))
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
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout)
        returncode = proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
        _join_until(workers, time.monotonic() + _CAPTURE_CLEANUP_GRACE_SECONDS)
        raise subprocess.TimeoutExpired(command, timeout) from None

    if not _join_until(workers, deadline):
        _kill_process_group(proc)
        _join_until(workers, time.monotonic() + _CAPTURE_CLEANUP_GRACE_SECONDS)
        if overflow:
            raise CapturedOutputLimitError(*overflow[0])
        raise subprocess.TimeoutExpired(command, timeout) from None
    if overflow:
        raise CapturedOutputLimitError(*overflow[0])

    decoded: dict[str, str] = {}
    for name, output in outputs.items():
        try:
            decoded[name] = output.decode(encoding)
        except UnicodeDecodeError as exc:
            raise CapturedOutputDecodeError(name, encoding) from exc

    return CapturedProcess(returncode, decoded["stdout"], decoded["stderr"])


def run_interactive(command: list[str], *, cwd: Path, timeout: float) -> int:
    """Run a bounded process group attached to the caller's terminal."""
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        start_new_session=os.name == "posix",
    )
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
        raise subprocess.TimeoutExpired(command, timeout) from None


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
    ) -> None:
        self.command = command
        self._deadline = time.monotonic() + timeout
        self._write_thread: threading.Thread | None = None
        self._proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
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
            try:
                self._stdout_queue.put(line, timeout=_QUEUE_PUT_TIMEOUT_SECONDS)
            except queue.Full:
                continue
            return True
        return False

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
        self._write_thread.join(timeout=remaining)
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
        try:
            line = self._stdout_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("timed out waiting for a JSON-RPC message") from exc
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
            with suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=1.0)
        # The server may already have exited while a helper still owns a pipe.
        _kill_process_group(self._proc)
        if self._proc.poll() is None:
            with suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
        if self._write_thread is not None:
            self._write_thread.join(timeout=_CAPTURE_CLEANUP_GRACE_SECONDS)
            if not self._write_thread.is_alive() and not self._stdin.closed:
                with suppress(OSError):
                    self._stdin.close()
        self._closing.set()
        readers = (self._stdout_thread, self._stderr_thread)
        _join_until(readers, time.monotonic() + _CAPTURE_CLEANUP_GRACE_SECONDS)
        for stream, thread in zip((self._proc.stdout, self._proc.stderr), readers, strict=True):
            # Closing an IO wrapper while its reader holds the lock can block
            # forever (including on non-POSIX hosts without group cleanup).
            if stream is not None and not thread.is_alive() and not stream.closed:
                stream.close()
        return self._proc.returncode if self._proc.returncode is not None else 0

    def __enter__(self) -> JsonRpcProcess:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class HeaderlessJsonRpcProcess(JsonRpcProcess):
    """Codex app-server's JSONL dialect, which omits the JSON-RPC version field."""

    _include_jsonrpc_version = False


def child_environment(extra: dict[str, str] | None = None) -> dict[str, str] | None:
    """Merge adapter environment additions without mutating the parent."""
    return {**os.environ, **extra} if extra else None
