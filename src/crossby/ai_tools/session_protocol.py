"""Runtime-owned bounded JSON-RPC transport for ordinary brokered sessions."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from crossby.ai_tools.headless import (
    HeadlessCleanupContext,
    HeadlessCleanupHooks,
    HeadlessRuntimeContext,
)
from crossby.ai_tools.plan_process import (
    HeaderlessJsonRpcProcess,
    JsonRpcProcess,
    kill_process_group,
)


class ProtocolSessionController:
    """Own one native process, its outstanding requests, and its cleanup chain.

    A callback result can be sent only while its exact native request remains
    outstanding. Runtime cancellation closes that gate before native abort and
    process cleanup, so a late callback cannot write into a reused session.
    """

    def __init__(
        self,
        context: HeadlessRuntimeContext,
        command: list[str],
        *,
        cwd: Path,
        headerless: bool = False,
        env: dict[str, str] | None = None,
    ) -> None:
        self.context = context
        self._lock = threading.RLock()
        self._pending: set[int | str] = set()
        self._resolved = False
        self._abort_method: str | None = None
        self._abort_params: dict[str, Any] | None = None
        self._abort_notification = False
        process_type = HeaderlessJsonRpcProcess if headerless else JsonRpcProcess
        self.process = process_type(
            command,
            cwd=cwd,
            env=env,
            timeout=context.remaining_seconds(),
            deadline=context.deadline,
        )
        context.register_cleanup(
            HeadlessCleanupHooks(
                native_abort=self._abort,
                close_input=self._close_input,
                terminate=self._terminate,
                force_kill=self._kill,
                reap=self._reap,
                join_workers=self._join,
            )
        )

    @property
    def stderr(self) -> str:
        return self.process.stderr

    def set_abort(self, method: str, params: dict[str, Any], *, notify: bool = False) -> None:
        with self._lock:
            if self._resolved:
                return
            self._abort_method = method
            self._abort_params = params
            self._abort_notification = notify

    def request(self, request_id: int, method: str, params: dict[str, Any]) -> None:
        self.context.checkpoint()
        self.process.request(request_id, method, params)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self.context.checkpoint()
        self.process.notify(method, params)

    def read(self) -> dict[str, Any]:
        self.context.checkpoint()
        frame = self.process.read(timeout=self.context.remaining_seconds())
        self.context.checkpoint()
        return frame

    def track_native_request(self, frame: dict[str, Any]) -> int | str:
        request_id = frame.get("id")
        if isinstance(request_id, bool) or not isinstance(request_id, (int, str)):
            raise ValueError("native request omitted a valid request ID")
        with self._lock:
            if self._resolved or request_id in self._pending:
                raise ValueError("native request reused a pending or resolved ID")
            self._pending.add(request_id)
        return request_id

    def respond_native(self, request_id: int | str, result: Any) -> None:
        with self._lock:
            self.context.checkpoint()
            if self._resolved or request_id not in self._pending:
                raise ValueError("native request is no longer live")
            self.process.respond(request_id, result)
            self._pending.remove(request_id)

    respond = respond_native

    def close_cleanly(self) -> int:
        with self._lock:
            self.context.checkpoint()
            self._resolved = True
            self._pending.clear()
        return self.process.close()

    def _abort(self, _cleanup: HeadlessCleanupContext) -> None:
        with self._lock:
            self._resolved = True
            self._pending.clear()
            method, params = self._abort_method, self._abort_params
            notify = self._abort_notification
        if method is None or params is None or self.process.returncode is not None:
            return
        # Cleanup has a fixed grace independent of the expired work deadline.
        # The runtime bounds this cooperative hook before closing the pipe.
        payload: dict[str, Any] = {"method": method, "params": params}
        if not notify:
            payload["id"] = 999_999
        self.process.send_cleanup(payload)

    def _close_input(self, _cleanup: HeadlessCleanupContext) -> None:
        self.process.close_input()

    def _terminate(self, _cleanup: HeadlessCleanupContext) -> None:
        proc = self.process._proc
        if proc.poll() is not None:
            return
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()

    def _kill(self, _cleanup: HeadlessCleanupContext) -> None:
        if self.process._proc.poll() is None:
            kill_process_group(self.process._proc)

    def _reap(self, cleanup: HeadlessCleanupContext) -> None:
        with suppress(subprocess.TimeoutExpired):
            self.process._proc.wait(timeout=cleanup.remaining_seconds())

    def _join(self, cleanup: HeadlessCleanupContext) -> None:
        self.process.join_workers(timeout=cleanup.remaining_seconds())
