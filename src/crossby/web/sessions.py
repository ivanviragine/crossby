"""Terminal session registry for the local browser UI.

Turns a browser's launch request into argv via the tool adapter, spawns it on a
pseudo-terminal, and tracks the live sessions so later requests can write input,
resize, or close them.

Every request is validated against the adapter's own
:class:`~crossby.models.ai.AIToolCapabilities` before a process is created, so
an unsupported flag is a clean 400 rather than a confusing tool-side error.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.plan_mode import PlanModeLaunchError
from crossby.config.json_utils import PathContainmentError, assert_within
from crossby.models.ai import AIToolCapabilities, AIToolID, AIToolType, EffortLevel
from crossby.utils.pty_runner import (
    PtySession,
    SubscriberDesyncError,
    Subscription,
    WindowSize,
    drain,
)

logger = structlog.get_logger()

# Sessions a single UI process will hold open at once. The cap exists so a
# runaway page cannot fork unbounded AI tool processes.
MAX_CONCURRENT_SESSIONS = 16

# How long shutdown waits for a create that is still spawning. Bounded so a
# wedged child cannot hold Ctrl-C open indefinitely.
SHUTDOWN_DRAIN_SECONDS = 10.0

# Directories returned by one browse call; a listing is a convenience, not a
# file manager, and an enormous folder should not become an enormous payload.
MAX_BROWSE_ENTRIES = 500


class Autonomy(StrEnum):
    """How much the tool may do without asking.

    Mutually exclusive by construction, mirroring ``crossby launch``: native
    plan mode is exclusive, and the remaining rungs have a fixed precedence
    (yolo > auto > accept-edits > default prompting). Modelling them as one
    choice makes the contradictory combinations unrepresentable rather than
    something to validate after the fact.
    """

    DEFAULT = "default"
    PLAN = "plan"
    ACCEPT_EDITS = "accept-edits"
    AUTO = "auto"
    YOLO = "yolo"


class SessionNotFoundError(KeyError):
    """Raised when a session id does not match a live session."""


class LaunchValidationError(ValueError):
    """Raised when a launch request contradicts the tool's declared capabilities."""


class ManagerClosedError(RuntimeError):
    """Raised when a launch arrives at (or races) :meth:`SessionManager.shutdown`.

    Request threads are daemons, so a shutdown does not wait for one already
    inside ``create()``. Without this the spawned tool would register after the
    shutdown snapshot and keep running unsupervised past "Server stopped".
    """


@dataclass(frozen=True)
class LaunchRequest:
    """A browser's request to start one AI tool session."""

    tool: AIToolID
    model: str | None = None
    effort: EffortLevel | None = None
    autonomy: Autonomy = Autonomy.DEFAULT
    initial_message: str | None = None
    cwd: str | None = None
    network_access: bool = False
    sandbox: bool = True
    size: WindowSize = field(default=WindowSize(cols=80, rows=24))

    @property
    def yolo(self) -> bool:
        return self.autonomy is Autonomy.YOLO

    @property
    def plan_mode(self) -> bool:
        return self.autonomy is Autonomy.PLAN

    @property
    def accept_edits(self) -> bool:
        return self.autonomy is Autonomy.ACCEPT_EDITS

    @property
    def auto(self) -> bool:
        return self.autonomy is Autonomy.AUTO

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

        cwd = payload.get("cwd") or None
        if cwd is not None and not isinstance(cwd, str):
            raise LaunchValidationError("'cwd' must be a string")

        raw_autonomy = payload.get("autonomy") or Autonomy.DEFAULT.value
        try:
            autonomy = Autonomy(raw_autonomy)
        except ValueError as exc:
            raise LaunchValidationError(f"unknown autonomy mode: {raw_autonomy!r}") from exc

        return cls(
            tool=tool,
            model=model,
            effort=effort,
            autonomy=autonomy,
            initial_message=message,
            cwd=cwd,
            network_access=bool(payload.get("network_access", False)),
            sandbox=bool(payload.get("sandbox", True)),
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
                "supports_initial_message": caps.supports_initial_message,
                "autonomy": [mode.value for mode in _supported_autonomy(caps)],
                "supports_network_access": caps.supports_network_access,
                "supports_sandbox_toggle": caps.supports_sandbox_toggle,
            }
        )
    return described


def _supported_autonomy(caps: Any) -> list[Autonomy]:
    """The autonomy rungs one adapter actually implements."""
    modes = [Autonomy.DEFAULT]
    if caps.plan_mode.supported:
        modes.append(Autonomy.PLAN)
    if caps.supports_accept_edits:
        modes.append(Autonomy.ACCEPT_EDITS)
    if caps.supports_auto:
        modes.append(Autonomy.AUTO)
    if caps.supports_yolo:
        modes.append(Autonomy.YOLO)
    return modes


class SessionManager:
    """Owns every live terminal session for one UI server.

    A session runs in any directory at or below one of ``allowed_roots``, which
    only the operator sets (``--path`` plus any ``--allow-dir``). The browser
    chooses *within* that boundary and never outside it: a requested directory is
    resolved and checked against the roots before a process is created, so a page
    cannot walk the filesystem by asking for an arbitrary cwd.
    """

    def __init__(self, project_root: Path, allowed_roots: Sequence[Path] | None = None) -> None:
        self.project_root = project_root.resolve()
        roots = [self.project_root, *(r.resolve() for r in allowed_roots or ())]
        # De-duplicate while keeping the operator's order; the first is the default.
        self.allowed_roots: tuple[Path, ...] = tuple(dict.fromkeys(roots))
        self._sessions: dict[str, PtySession] = {}
        self._requests: dict[str, LaunchRequest] = {}
        self._listeners: list[Callable[[PtySession], None]] = []
        # Slots claimed by an in-flight create that has not registered yet.
        self._reserved = 0
        # Creates that have claimed a slot and not yet finished cleaning up.
        # Separate from _reserved, which stops counting once the session is in
        # the registry: shutdown has to outlast the whole call, including the
        # orphan close that happens after registration is refused.
        self._inflight = 0
        # Set by shutdown(); guards the window between reserving a slot and
        # registering the spawned session.
        self._closed = False
        self._lock = threading.Lock()
        self._settled = threading.Condition(self._lock)

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
        if request.autonomy not in _supported_autonomy(caps):
            supported = ", ".join(m.value for m in _supported_autonomy(caps))
            raise LaunchValidationError(
                f"{caps.display_name} does not support {request.autonomy.value!r} "
                f"autonomy; supported: {supported}"
            )
        if request.network_access and not caps.supports_network_access:
            raise LaunchValidationError(f"{caps.display_name} has no sandbox network opt-in")
        if not request.sandbox and not caps.supports_sandbox_toggle:
            raise LaunchValidationError(
                f"{caps.display_name} does not support disabling its sandbox"
            )
        if request.initial_message and not caps.supports_initial_message:
            raise LaunchValidationError(f"{caps.display_name} does not accept an initial message")

        # Resolve the working directory before anything is claimed. It rejects a
        # directory that vanished between selection and launch, and doing that
        # after the reservation leaked a slot on every rejection — sixteen bad
        # requests and the server refused all launches until restarted.
        workdir = self.resolve_workdir(request.cwd)

        # Reap exited sessions and reserve a slot in one critical section. The
        # check and the registration used to be separate, so two concurrent
        # creates could both pass the limit before either registered.
        with self._lock:
            if self._closed:
                raise ManagerClosedError("server is shutting down")
            dead = [
                self._sessions.pop(sid) for sid, s in list(self._sessions.items()) if not s.running
            ]
            for entry in dead:
                self._requests.pop(entry.id, None)
            live = sum(1 for session in self._sessions.values() if session.running)
            if live + self._reserved >= MAX_CONCURRENT_SESSIONS:
                raise LaunchValidationError(
                    f"too many live sessions (limit {MAX_CONCURRENT_SESSIONS}); "
                    "close one before starting another"
                )
            self._reserved += 1
            self._inflight += 1

        try:
            return self._spawn(request, workdir, adapter, caps, dead)
        finally:
            with self._settled:
                self._inflight -= 1
                self._settled.notify_all()

    def _spawn(
        self,
        request: LaunchRequest,
        workdir: Path,
        adapter: AbstractAITool,
        caps: AIToolCapabilities,
        dead: list[PtySession],
    ) -> PtySession:
        """Build the argv and start the child. Runs with a slot already claimed."""
        # Closing releases a master fd and joins a reader thread, so it must not
        # happen under the lock — and dropping the reference without closing
        # leaked both.
        for entry in dead:
            entry.close()

        command_kwargs: dict[str, Any] = {
            "model": request.model,
            "initial_message": request.initial_message,
            "effort": request.effort,
            "yolo": request.yolo,
            "plan_mode": request.plan_mode,
            "accept_edits": request.accept_edits,
            "auto": request.auto,
            "network_access": request.network_access,
            "working_dir": workdir,
        }
        # Only adapters declaring the toggle accept the keyword; passing it
        # unconditionally would break the others' builders.
        if caps.supports_sandbox_toggle:
            command_kwargs["sandbox"] = request.sandbox
        try:
            try:
                command = adapter.build_launch_command(**command_kwargs)
            except PlanModeLaunchError as exc:
                # The adapter's own pre-launch gate; surface it as a bad request
                # rather than a server error.
                raise LaunchValidationError(str(exc)) from exc
            session = PtySession(command, cwd=workdir, size=request.size)
        except BaseException:
            with self._lock:
                self._reserved -= 1
            raise

        # shutdown() may have snapshotted the registry while this session was
        # being spawned. Registering now would hide a live tool from the only
        # code that closes sessions, so the loser of that race cleans up its own
        # child instead of orphaning it.
        with self._lock:
            self._reserved -= 1
            if self._closed:
                orphan: PtySession | None = session
            else:
                orphan = None
                self._sessions[session.id] = session
                self._requests[session.id] = request
            listeners = list(self._listeners)
        if orphan is not None:
            orphan.close()
            raise ManagerClosedError("server is shutting down")
        logger.info(
            "web.session.created",
            session=session.id,
            tool=str(request.tool),
            model=request.model,
            cwd=str(workdir),
        )
        for listener in listeners:
            listener(session)
        return session

    def resolve_workdir(self, requested: str | None) -> Path:
        """Resolve a requested working directory, or the default when absent.

        Containment is checked against the resolved path, so a symlink pointing
        out of an allowed root is refused rather than followed.
        """
        if requested is None:
            return self.project_root

        candidate = Path(requested).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise LaunchValidationError(f"no such directory: {requested}") from exc
        if not resolved.is_dir():
            raise LaunchValidationError(f"not a directory: {requested}")

        for root in self.allowed_roots:
            try:
                assert_within(root, resolved)
            except PathContainmentError:
                continue
            return resolved

        allowed = ", ".join(str(root) for root in self.allowed_roots)
        raise LaunchValidationError(
            f"{resolved} is outside the directories this server may use ({allowed}). "
            "Restart crossby ui with --path or --allow-dir to widen it."
        )

    def browse(self, requested: str | None) -> dict[str, Any]:
        """List the sub-directories of *requested* for the folder picker.

        Only directories inside an allowed root are listed, and a parent is
        offered only while it also stays inside one — so the picker cannot be
        used to enumerate the filesystem.
        """
        path = self.resolve_workdir(requested)

        parent: str | None = None
        if path.parent != path:
            try:
                parent = str(self.resolve_workdir(str(path.parent)))
            except LaunchValidationError:
                parent = None

        children: list[str] = []
        try:
            for child in sorted(path.iterdir(), key=lambda c: c.name.lower()):
                if len(children) >= MAX_BROWSE_ENTRIES:
                    break
                # Hidden directories are overwhelmingly tooling (.git, .venv),
                # not somewhere anyone launches an agent.
                if child.name.startswith(".") or not child.is_dir():
                    continue
                children.append(child.name)
        except (PermissionError, OSError) as exc:
            raise LaunchValidationError(f"cannot read {path}: {exc}") from exc

        return {
            "path": str(path),
            "parent": parent,
            "children": children,
            "roots": [str(root) for root in self.allowed_roots],
            "default": str(self.project_root),
        }

    def add_listener(self, callback: Callable[[PtySession], None]) -> None:
        """Notify *callback* of each session created from now on."""
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[PtySession], None]) -> None:
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def live_sessions(self) -> list[PtySession]:
        with self._lock:
            return list(self._sessions.values())

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
            "autonomy": request.autonomy.value if request else None,
            "command": session.command,
            "cwd": str(session.cwd),
            "pid": session.pid,
            "running": session.running,
            "exit_code": session.exit_code,
            "stopped": session.stopped,
            "exit_signal": session.exit_signal,
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
        """Close every live session — called when the server stops.

        Marking the manager closed inside the same critical section as the
        snapshot is what makes this total: a create that has not yet registered
        sees the flag and closes its own session, and one that already
        registered is in the snapshot.

        Then it waits for those creates to finish. Request threads are daemons,
        so returning early let the interpreter exit while a handler was still
        inside ``PtySession(...)`` — the thread died before it could close the
        child it had just spawned, and the tool outlived the server with
        nothing left that could stop it.
        """
        with self._settled:
            self._closed = True
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._requests.clear()
            if self._inflight:
                logger.info("web.shutdown.draining", inflight=self._inflight)
                if not self._settled.wait_for(
                    lambda: self._inflight == 0, timeout=SHUTDOWN_DRAIN_SECONDS
                ):
                    # Bounded so a wedged spawn cannot hold Ctrl-C forever.
                    logger.warning("web.shutdown.drain_timeout", inflight=self._inflight)
        for session in sessions:
            session.close()


@dataclass(frozen=True)
class StreamEvent:
    """One tagged event on the multiplexed stream."""

    kind: str  # "output" | "exit"
    session_id: str
    chunk: bytes = b""
    exit_code: int | None = None
    stopped: bool = False
    exit_signal: str | None = None


class MultiplexedStream:
    """Every session's output on one iterator, tagged by session id.

    A browser holds at most ~6 HTTP/1.1 connections per origin, and a
    per-session stream would spend one apiece — the sixth open terminal
    exhausts the pool and stalls *every* other request, keystrokes included.
    Folding all sessions into a single stream removes that ceiling: tab count
    stops being a transport concern.

    One pump thread per session feeds a shared queue. Sessions created while the
    stream is open are picked up through :meth:`SessionManager.add_listener`.
    Closing the stream detaches every subscription, which also wakes pumps
    parked on a quiet session.
    """

    def __init__(self, manager: SessionManager, *, capacity: int = 4096) -> None:
        self._manager = manager
        self._queue: queue.Queue[StreamEvent | None] = queue.Queue(maxsize=capacity)
        self._subscriptions: dict[str, tuple[PtySession, Subscription]] = {}
        self._started: set[str] = set()
        self._pumps: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._desynced = threading.Event()

    def __enter__(self) -> MultiplexedStream:
        self._manager.add_listener(self._on_session_created)
        for session in self._manager.live_sessions():
            self._start_pump(session)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._manager.remove_listener(self._on_session_created)
        self._closed.set()
        with self._lock:
            pending = list(self._subscriptions.values())
            self._subscriptions.clear()
        for session, subscription in pending:
            session.detach(subscription)
        # Evict if necessary: losing this sentinel parks __iter__ in get()
        # forever and the SSE handler thread never returns.
        for _ in range(self._queue.maxsize + 8):
            try:
                self._queue.put_nowait(None)
                return
            except queue.Full:
                with suppress(queue.Empty):
                    self._queue.get_nowait()
        logger.error("web.stream.sentinel_undeliverable")

    def __iter__(self) -> Iterator[StreamEvent]:
        while True:
            event = self._queue.get()
            if event is None:
                return
            yield event

    @property
    def desynced(self) -> bool:
        """Whether the consumer fell behind and the stream was cut."""
        return self._desynced.is_set()

    def _on_session_created(self, session: PtySession) -> None:
        if not self._closed.is_set():
            self._start_pump(session)

    def _start_pump(self, session: PtySession) -> None:
        # Claim the session before the thread exists. `_pump` only registers its
        # subscription after `attach()` returns, so checking `_subscriptions`
        # here let `__enter__`'s snapshot and the creation listener each start a
        # pump for the same session — and its output arrived twice.
        with self._lock:
            if session.id in self._started or self._closed.is_set():
                return
            self._started.add(session.id)
        thread = threading.Thread(
            target=self._pump, args=(session,), name=f"mux-{session.id[:8]}", daemon=True
        )
        self._pumps.append(thread)
        thread.start()

    def _pump(self, session: PtySession) -> None:
        """Forward one session's output into the shared queue, tagged."""
        backlog, subscription = session.attach()
        if subscription is not None:
            with self._lock:
                if self._closed.is_set():
                    session.detach(subscription)
                    return
                self._subscriptions[session.id] = (session, subscription)
        if backlog:
            self._emit(StreamEvent("output", session.id, chunk=backlog))
        try:
            if subscription is not None:
                for chunk in drain(subscription):
                    self._emit(StreamEvent("output", session.id, chunk=chunk))
        except SubscriberDesyncError:
            self._cut()
            return
        finally:
            if subscription is not None:
                with self._lock:
                    self._subscriptions.pop(session.id, None)
                session.detach(subscription)
        # drain() also returns on detach, which is not an exit — only report one
        # when the child has actually finished.
        if not session.running:
            self._emit(
                StreamEvent(
                    "exit",
                    session.id,
                    exit_code=session.exit_code,
                    stopped=session.stopped,
                    exit_signal=session.exit_signal,
                )
            )

    def _emit(self, event: StreamEvent) -> None:
        if self._closed.is_set():
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Same reasoning as a single session: never drop terminal bytes.
            self._cut()

    def _cut(self) -> None:
        if self._desynced.is_set():
            return
        self._desynced.set()
        logger.warning("web.stream.multiplexed_desync")
        self.close()
