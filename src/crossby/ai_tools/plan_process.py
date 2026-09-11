"""Small stdlib process and wire helpers for native plan collectors."""

from __future__ import annotations

import json
import locale
import os
import queue
import subprocess
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, ClassVar

_STDERR_TAIL_LIMIT = 8 * 1024
_STDOUT_QUEUE_LIMIT = 256
_QUEUE_PUT_TIMEOUT_SECONDS = 0.1
_CAPTURED_STDOUT_LIMIT = 8 * 1024 * 1024
_CAPTURED_STDERR_LIMIT = 1024 * 1024
_CAPTURE_CHUNK_SIZE = 64 * 1024


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


def run_captured(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> CapturedProcess:
    """Run a bounded child with hard stdout/stderr memory limits and no shell."""
    encoding = locale.getpreferredencoding(False)
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if (
        proc.stdout is None
        or proc.stderr is None
        or (input_text is not None and proc.stdin is None)
    ):
        proc.kill()
        proc.wait()
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
                        with suppress(OSError):
                            proc.kill()

    readers = (
        threading.Thread(target=read_bounded, args=(proc.stdout, "stdout", _CAPTURED_STDOUT_LIMIT)),
        threading.Thread(target=read_bounded, args=(proc.stderr, "stderr", _CAPTURED_STDERR_LIMIT)),
    )
    for reader in readers:
        reader.start()

    writer: threading.Thread | None = None
    if input_text is not None:
        input_stream = proc.stdin
        assert input_stream is not None

        def write_input() -> None:
            try:
                input_stream.write(input_text.encode(encoding))
                input_stream.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                with suppress(OSError):
                    input_stream.close()

        writer = threading.Thread(target=write_input)
        writer.start()

    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        for reader in readers:
            reader.join()
        if writer is not None:
            writer.join()
        raise subprocess.TimeoutExpired(command, timeout) from None

    for reader in readers:
        reader.join()
    if writer is not None:
        writer.join()
    if overflow:
        raise CapturedOutputLimitError(*overflow[0])

    return CapturedProcess(
        returncode,
        outputs["stdout"].decode(encoding),
        outputs["stderr"].decode(encoding),
    )


def run_interactive(command: list[str], *, cwd: Path, timeout: float) -> int:
    """Run a bounded child attached to the caller's terminal."""
    return subprocess.run(command, cwd=cwd, timeout=timeout, check=False).returncode


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

    def __init__(self, command: list[str], *, cwd: Path) -> None:
        self.command = command
        self._proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self._proc.stdin is None or self._proc.stdout is None or self._proc.stderr is None:
            self._proc.kill()
            raise OSError("failed to create JSON-RPC stdio pipes")
        self._stdin: IO[str] = self._proc.stdin
        self._stdout_queue: queue.Queue[str | None] = queue.Queue(maxsize=_STDOUT_QUEUE_LIMIT)
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
            for line in stream:
                if not self._queue_stdout(line):
                    break
        except ValueError:
            if not self._closing.is_set():
                raise
        finally:
            self._queue_stdout(None)

    def _queue_stdout(self, line: str | None) -> bool:
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
            for line in stream:
                line_tail = line[-_STDERR_TAIL_LIMIT:]
                with self._stderr_lock:
                    self._stderr_tail = (self._stderr_tail + line_tail)[-_STDERR_TAIL_LIMIT:]
        except ValueError:
            if not self._closing.is_set():
                raise

    def send(self, payload: dict[str, Any]) -> None:
        if self._proc.poll() is not None:
            raise EOFError(f"JSON-RPC child exited with status {self._proc.returncode}")
        envelope = {"jsonrpc": "2.0", **payload} if self._include_jsonrpc_version else payload
        self._stdin.write(json.dumps(envelope, separators=(",", ":")) + "\n")
        self._stdin.flush()

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

    def read(self, *, timeout: float) -> dict[str, Any]:
        try:
            line = self._stdout_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("timed out waiting for a JSON-RPC message") from exc
        if line is None:
            code = self._proc.poll()
            raise EOFError(f"JSON-RPC stream closed (exit status {code})")
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
        """Close stdin, then terminate/kill a server that does not stop itself."""
        if not self._stdin.closed:
            with suppress(OSError):
                self._stdin.close()
        if self._proc.poll() is None:
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=1.0)
        self._closing.set()
        readers = (self._stdout_thread, self._stderr_thread)
        for thread in readers:
            thread.join(timeout=0.2)
        for stream in (self._proc.stdout, self._proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        for thread in readers:
            thread.join(timeout=0.2)
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
