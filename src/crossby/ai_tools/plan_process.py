"""Small stdlib process and wire helpers for native plan collectors."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, ClassVar


@dataclass(frozen=True)
class CapturedProcess:
    """Captured output and exact exit status from a bounded child."""

    returncode: int
    stdout: str
    stderr: str


def run_captured(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> CapturedProcess:
    """Run a bounded child with text I/O and no shell interpretation."""
    proc = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    return CapturedProcess(proc.returncode, proc.stdout, proc.stderr)


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
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stderr_lines: list[str] = []
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
        return "".join(self._stderr_lines)

    def _read_stdout(self, stream: IO[str]) -> None:
        try:
            for line in stream:
                self._stdout_queue.put(line)
        except ValueError:
            if not self._closing.is_set():
                raise
        finally:
            self._stdout_queue.put(None)

    def _read_stderr(self, stream: IO[str]) -> None:
        try:
            for line in stream:
                self._stderr_lines.append(line)
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
        readers = (self._stdout_thread, self._stderr_thread)
        for thread in readers:
            thread.join(timeout=0.2)
        self._closing.set()
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
