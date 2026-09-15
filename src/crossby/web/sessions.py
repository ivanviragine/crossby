"""Terminal session registry for the local browser UI.

Turns a browser's launch request into argv via the tool adapter, spawns it on a
pseudo-terminal, and tracks the live sessions so later requests can write input,
resize, or close them.

Every request is validated against the adapter's own
:class:`~crossby.models.ai.AIToolCapabilities` before a process is created, so
an unsupported flag is a clean 400 rather than a confusing tool-side error.
"""

from __future__ import annotations

import threading
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from crossby.ai_tools.base import AbstractAITool
from crossby.models.ai import AIToolID, AIToolType, EffortLevel
from crossby.utils.pty_runner import PtySession, WindowSize

logger = structlog.get_logger()

# Sessions a single UI process will hold open at once. The cap exists so a
# runaway page cannot fork unbounded AI tool processes.
MAX_CONCURRENT_SESSIONS = 16


class SessionNotFoundError(KeyError):
    """Raised when a session id does not match a live session."""


class LaunchValidationError(ValueError):
    """Raised when a launch request contradicts the tool's declared capabilities."""


@dataclass(frozen=True)
class LaunchRequest:
    """A browser's request to start one AI tool session."""

    tool: AIToolID
    model: str | None = None
    effort: EffortLevel | None = None
    yolo: bool = False
    initial_message: str | None = None
    size: WindowSize = field(default=WindowSize(cols=80, rows=24))

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LaunchRequest:
        """Build a request from decoded JSON, raising on anything unusable."""
        raw_tool = payload.get("tool")
        if not isinstance(raw_tool, str):
            raise LaunchValidationError("'tool' is required")
        try:
            tool = AIToolID(raw_tool)
        except ValueError as exc:
            raise LaunchValidationError(f"unknown tool: {raw_tool!r}") from exc

        raw_effort = payload.get("effort")
        effort: EffortLevel | None = None
        if isinstance(raw_effort, str) and raw_effort:
            try:
                effort = EffortLevel(raw_effort)
            except ValueError as exc:
                raise LaunchValidationError(f"unknown effort level: {raw_effort!r}") from exc

        model = payload.get("model") or None
        if model is not None and not isinstance(model, str):
            raise LaunchValidationError("'model' must be a string")

        message = payload.get("initial_message") or None
        if message is not None and not isinstance(message, str):
            raise LaunchValidationError("'initial_message' must be a string")

        return cls(
            tool=tool,
            model=model,
            effort=effort,
            yolo=bool(payload.get("yolo", False)),
            initial_message=message,
            size=_window_size(payload),
        )


def _window_size(payload: dict[str, Any], default: WindowSize | None = None) -> WindowSize:
    """Read ``cols``/``rows`` from a payload, falling back to *default*."""
    fallback = default or WindowSize(cols=80, rows=24)
    raw_cols = payload.get("cols", fallback.cols)
    raw_rows = payload.get("rows", fallback.rows)
    if not isinstance(raw_cols, int) or not isinstance(raw_rows, int):
        raise LaunchValidationError("'cols' and 'rows' must be integers")
    try:
        return WindowSize(cols=raw_cols, rows=raw_rows)
    except ValueError as exc:
        raise LaunchValidationError(str(exc)) from exc


def window_size_from_payload(payload: dict[str, Any]) -> WindowSize:
    """Public wrapper used by the resize route."""
    return _window_size(payload)


def embeddable_tools() -> list[AIToolID]:
    """Installed tools that can run inside an embedded terminal.

    GUI tools (VS Code, the Antigravity IDE) are excluded: they override
    ``launch()`` to open a window and have no terminal interface to embed.
    """
    return [
        tool_id
        for tool_id in AbstractAITool.detect_installed()
        if AbstractAITool.get(tool_id).capabilities().tool_type is AIToolType.TERMINAL
    ]


def describe_tools() -> list[dict[str, Any]]:
    """Describe embeddable tools so the page can build its launch form."""
    described: list[dict[str, Any]] = []
    for tool_id in embeddable_tools():
        adapter = AbstractAITool.get(tool_id)
        caps = adapter.capabilities()
        described.append(
            {
                "id": str(tool_id),
                "display_name": caps.display_name,
                "supports_model_flag": caps.supports_model_flag,
                "models": [model.id for model in adapter.get_models()],
                "supports_effort": caps.supports_effort,
                "supported_efforts": [e.value for e in caps.supported_efforts],
                "supports_yolo": caps.supports_yolo,
                "supports_initial_message": caps.supports_initial_message,
            }
        )
    return described


class SessionManager:
    """Owns every live terminal session for one UI server.

    All sessions run in ``project_root``. The browser never supplies a working
    directory: the server is started against one project and stays there, so a
    page cannot walk the filesystem by asking for a different cwd.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._sessions: dict[str, PtySession] = {}
        self._requests: dict[str, LaunchRequest] = {}
        self._lock = threading.Lock()

    def create(self, request: LaunchRequest) -> PtySession:
        """Validate *request*, build its argv, and spawn it on a PTY."""
        adapter = AbstractAITool.get(request.tool)
        caps = adapter.capabilities()

        if caps.tool_type is not AIToolType.TERMINAL:
            raise LaunchValidationError(
                f"{caps.display_name} is a GUI tool and cannot run in an embedded terminal"
            )
        if request.effort is not None and not caps.supports_effort:
            raise LaunchValidationError(f"{caps.display_name} does not support effort levels")
        if request.effort is not None and request.effort not in caps.supported_efforts:
            supported = ", ".join(e.value for e in caps.supported_efforts)
            raise LaunchValidationError(
                f"{caps.display_name} supports these effort levels: {supported}"
            )
        if request.yolo and not caps.supports_yolo:
            raise LaunchValidationError(f"{caps.display_name} does not support YOLO mode")
        if request.initial_message and not caps.supports_initial_message:
            raise LaunchValidationError(f"{caps.display_name} does not accept an initial message")

        with self._lock:
            # Exited sessions linger until explicitly closed; drop them here so a
            # long-lived server does not accumulate dead entries.
            for dead in [sid for sid, s in self._sessions.items() if not s.running]:
                self._sessions.pop(dead, None)
                self._requests.pop(dead, None)
            live = sum(1 for session in self._sessions.values() if session.running)
            if live >= MAX_CONCURRENT_SESSIONS:
                raise LaunchValidationError(
                    f"too many live sessions (limit {MAX_CONCURRENT_SESSIONS}); "
                    "close one before starting another"
                )

        command = adapter.build_launch_command(
            model=request.model,
            initial_message=request.initial_message,
            effort=request.effort,
            yolo=request.yolo,
            working_dir=self.project_root,
        )
        session = PtySession(command, cwd=self.project_root, size=request.size)

        with self._lock:
            self._sessions[session.id] = session
            self._requests[session.id] = request
        logger.info(
            "web.session.created",
            session=session.id,
            tool=str(request.tool),
            model=request.model,
        )
        return session

    def get(self, session_id: str) -> PtySession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def close(self, session_id: str) -> None:
        session = self.get(session_id)
        session.close()
        with self._lock:
            self._sessions.pop(session_id, None)
            self._requests.pop(session_id, None)

    def describe(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        with self._lock:
            request = self._requests.get(session_id)
        return {
            "id": session.id,
            "tool": str(request.tool) if request else None,
            "model": request.model if request else None,
            "command": session.command,
            "cwd": str(session.cwd),
            "pid": session.pid,
            "running": session.running,
            "exit_code": session.exit_code,
            "cols": session.size.cols,
            "rows": session.size.rows,
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        """Describe every live session, tolerating one closed concurrently."""
        with self._lock:
            ids = list(self._sessions)
        described: list[dict[str, Any]] = []
        for session_id in ids:
            with suppress(SessionNotFoundError):
                described.append(self.describe(session_id))
        return described

    def shutdown(self) -> None:
        """Close every live session — called when the server stops."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._requests.clear()
        for session in sessions:
            session.close()
