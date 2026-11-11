"""What ``crossby panel`` shows on stderr while a panel runs.

Every message is printed as it is committed. A heading names its turn, its
sender and its recipients, and a footer gives its time and any denials. On a
terminal, one live line also shows what is running now: the tools being
checked, the lead writing, or each addressed member writing, answered, failed
or queued. Off a terminal the same moments are printed as plain lines, so a
log reads in order.

A human seat holds the terminal while it is asked. The live line stops and
anything else to print waits until the reply is in, so nothing is drawn over
what the person types.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from crossby.cli.converse import _duration, _escape, _stderr
from crossby.ui.console import console

if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.live import Live

    from crossby.services.ask import CliBackend
    from crossby.services.journal import ConversationEvent
    from crossby.services.panel import Member, PanelSpec

# The version inside a tool's own --version text, e.g. "2.1.317 (Claude Code)",
# "codex-cli 0.166.0", "2026.11.10-731fb78" or "GitHub Copilot CLI 1.0.109.".
_VERSION = re.compile(r"\d+(?:\.\d+)+(?:[-+][0-9A-Za-z](?:[0-9A-Za-z.-]*[0-9A-Za-z])?)?")

_HUMAN_STATUS = {
    "timed_out": "no reply in time",
    "cancelled": "the reply was cancelled",
    "empty": "the reply was empty",
}


def version_number(text: str | None) -> str | None:
    """The version in a tool's ``--version`` output, without the words around it."""
    if not text:
        return None
    match = _VERSION.search(text)
    return match.group(0) if match else text.strip()


def backend_label(backend: CliBackend) -> str:
    """``codex``, or ``codex (gpt-5.5, high effort)`` when a model or effort is set."""
    settings = [backend.model] if backend.model else []
    if backend.effort is not None:
        settings.append(f"{backend.effort.value} effort")
    return backend.tool.value + (f" ({', '.join(settings)})" if settings else "")


def member_label(member: Member) -> str:
    """``codex``, ``cursor, persona folder``, ``Rui, codex`` or ``Dana, a person``."""
    from crossby.services.ask import CliBackend
    from crossby.services.workspaces import OwnedWorkspace

    parts = [member.name] if member.name else []
    if isinstance(member.backend, CliBackend):
        parts.append(backend_label(member.backend))
        if isinstance(member.workspace, OwnedWorkspace):
            parts.append("persona folder")
    else:
        parts.append("a person")
    return ", ".join(parts)


def show_opening(spec: PanelSpec, *, transcript: str, events: str | None) -> None:
    """Who leads which members, how they answer, the limits, and where the files are."""
    from crossby.services.ask import CliBackend

    backend = spec.lead.backend
    lead = f"{backend_label(backend)} leads" if isinstance(backend, CliBackend) else "You lead"
    count = len(spec.members)
    _stderr(
        f"  [step]▸[/] {_escape(lead)} a {spec.visibility} panel of {count} "
        f"member{'' if count == 1 else 's'}:"
    )
    width = max(len(member.id) for member in spec.members) + 1
    for member in spec.members:
        _stderr(f"      [bold]@{member.id:<{width}}[/] {_escape(member_label(member))}")
    if count == 1:
        how = "the member answers"
    elif spec.concurrency >= count:
        how = "members answer in parallel"
    elif spec.concurrency == 1:
        how = "members answer one at a time"
    else:
        how = f"members answer in parallel, {spec.concurrency} at a time"
    if any(not member.human for member in spec.members):
        how += "; agents are read-only"
    _stderr(f"[dim]      {how}[/]")
    allowed = spec.max_rounds
    limits = (
        f"up to {allowed} message{'' if allowed == 1 else 's'} to the members, "
        f"{_duration(spec.timeout_seconds)} in all, {_duration(spec.turn_timeout_seconds)} per turn"
    )
    if spec.lead.human or any(member.human for member in spec.members):
        limits += f", {_duration(spec.human_timeout_seconds)} per human reply"
    _stderr(f"[dim]      limits: {limits}[/]")
    _stderr(f"[dim]      transcript: {_escape(transcript)}[/]")
    if events is not None:
        _stderr(f"[dim]      events:     {_escape(events)}[/]")


@dataclass(frozen=True)
class _Seat:
    """One addressed member during a dispatch."""

    id: str
    state: str  # queued, writing, waiting, answered or failed
    since: float | None = None
    seconds: float | None = None


@dataclass(frozen=True)
class _Activity:
    """What the live line shows; replaced whole, never changed in place."""

    kind: str  # checking, lead or members
    since: float
    label: str = ""
    seats: tuple[_Seat, ...] = ()


def activity_text(activity: _Activity, now: float) -> str:
    """The live line's text, e.g. ``@ops writing 42s · @design ✓ 14s · @support queued``."""
    if activity.kind == "checking":
        return f"checking {activity.label}… {_duration(now - activity.since)}"
    if activity.kind == "lead":
        return f"{activity.label} is writing… {_duration(now - activity.since)}"
    parts = []
    for seat in activity.seats:
        if seat.state == "writing" and seat.since is not None:
            parts.append(f"@{seat.id} writing {_duration(now - seat.since)}")
        elif seat.state == "waiting" and seat.since is not None:
            parts.append(f"@{seat.id} waiting for a reply {_duration(now - seat.since)}")
        elif seat.state == "answered":
            parts.append(f"@{seat.id} ✓ {_duration(seat.seconds or 0.0)}")
        elif seat.state == "failed":
            parts.append(f"@{seat.id} ✗")
        else:
            parts.append(f"@{seat.id} queued")
    return " · ".join(parts)


class _Line:
    """The live line: a spinner and the current activity, redrawn by rich."""

    def __init__(self, view: PanelView) -> None:
        from rich.spinner import Spinner

        self._view = view
        self._spinner = Spinner("dots", style="step")

    def __rich__(self) -> RenderableType:
        from rich.text import Text

        # Read without a lock: rich calls this from its refresh thread, which
        # stopping the live display joins while the view's lock is held.
        activity = self._view.activity
        text = activity_text(activity, time.monotonic()) if activity is not None else ""
        self._spinner.update(text=Text(f" {text}", style="dim"))
        return self._spinner


class PanelView:
    """An event sink that shows a panel run on stderr. Safe from any thread.

    ``live`` draws the live line (on a terminal only). ``typed_here`` means a
    person types human seats' replies at this terminal and can see them, so
    their committed messages are not printed a second time.
    """

    def __init__(self, spec: PanelSpec, *, live: bool, typed_here: bool) -> None:
        self._spec = spec
        self._live_enabled = live
        self._typed_here = typed_here
        self._lock = threading.RLock()
        self._held = 0
        self._deferred: list[Callable[[], None]] = []
        self._live: Live | None = None
        self.activity: _Activity | None = None
        self._versions: dict[str, str | None] = {}
        self._versions_shown = False
        self._announced = threading.Condition()
        self._requests: set[str] = set()
        self._humans = {member.id for member in spec.members if member.human}
        if spec.lead.human:
            self._humans.add("lead")
        self._names = {member.id: member.name for member in spec.members}

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Show that the tools are being checked, until the first turn starts."""
        tools: list[str] = []
        for backend in [self._spec.lead.backend, *(m.backend for m in self._spec.members)]:
            name = getattr(getattr(backend, "tool", None), "value", None)
            if name is not None and name not in tools:
                tools.append(name)
        with self._lock:
            if tools:
                self._set(_Activity("checking", time.monotonic(), label=", ".join(tools)))

    def close(self) -> None:
        """Stop the live line and print anything still waiting."""
        with self._lock:
            self._held = 0
            self.activity = None
            self._stop_live()
            deferred, self._deferred = self._deferred, []
        for show in deferred:
            show()

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Keep the terminal still while a person types: no live line, no other output."""
        with self._lock:
            self._held += 1
            self._stop_live()
        try:
            yield
        finally:
            with self._lock:
                self._held -= 1
                if self._held == 0:
                    deferred, self._deferred = self._deferred, []
                    for show in deferred:
                        show()
                    self._start_live()

    def wait_announced(self, request_id: str, timeout: float) -> bool:
        """Whether this view has shown everything up to ``request_id``'s input request.

        Events reach the view on the journal's dispatcher thread, a moment after
        they happen. A human seat waits for this before prompting, so the
        messages it answers are already on screen above the prompt.
        """
        deadline = time.monotonic() + timeout
        with self._announced:
            while request_id not in self._requests:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._announced.wait(left)
            return True

    # -- the sink ----------------------------------------------------------

    def __call__(self, event: ConversationEvent) -> None:
        from crossby.services.journal import EventType

        data = event.data
        handlers: dict[EventType, Callable[[dict[str, Any]], None]] = {
            EventType.PARTICIPANT_READY: self._ready,
            EventType.TURN_STARTED: self._turn_started,
            EventType.TURN_FINISHED: self._turn_finished,
            EventType.INPUT_REQUESTED: self._input_requested,
            EventType.INPUT_RESOLVED: self._input_resolved,
            EventType.MESSAGE_COMMITTED: self._committed,
            EventType.CONVERSATION_FINISHED: self._finished,
        }
        handler = handlers.get(event.type)
        if handler is not None:
            with self._lock:
                handler(data)

    def _ready(self, data: dict[str, Any]) -> None:
        self._versions.setdefault(str(data["tool"]), version_number(data.get("version")))

    def _turn_started(self, data: dict[str, Any]) -> None:
        self._show_versions()
        participant, now = data["participant"], time.monotonic()
        if participant == "lead":
            turn, tool = data["turn"], data["tool"]
            self._set(_Activity("lead", now, label=f"[{turn}] lead ({tool})"))
            if not self._live_enabled:
                line = f"[step.count]\\[{turn}][/] lead ({_escape(str(tool))}) is writing…"
                self._print(lambda: _stderr(line))
            return
        self._seat(participant, state="writing", since=now)

    def _turn_finished(self, data: dict[str, Any]) -> None:
        participant = data["participant"]
        if participant == "lead":
            return
        error = data.get("error")
        seconds = float(data.get("duration_seconds") or 0.0)
        self._seat(participant, state="failed" if error else "answered", seconds=seconds)
        if error:
            self._print(lambda: _stderr(f"  [warning]![/] @{participant}: {_escape(str(error))}"))
        elif not self._live_enabled:
            self._print(
                lambda: _stderr(f"[dim]      @{participant} answered · {_duration(seconds)}[/]")
            )

    def _input_requested(self, data: dict[str, Any]) -> None:
        self._show_versions()
        participant = data["participant"]
        if data.get("kind") == "lead":
            self._set(None)
        else:
            self._seat(participant, state="waiting", since=time.monotonic())
        with self._announced:
            self._requests.add(str(data["request_id"]))
            self._announced.notify_all()

    def _input_resolved(self, data: dict[str, Any]) -> None:
        participant, status = data["participant"], data.get("status")
        if participant == "lead":
            return
        seat = self._find(participant)
        since = seat.since if seat is not None else None
        seconds = time.monotonic() - since if since is not None else 0.0
        if status == "answered":
            self._seat(participant, state="answered", seconds=seconds)
            return
        self._seat(participant, state="failed")
        reason = _HUMAN_STATUS.get(str(status), str(status))
        self._print(lambda: _stderr(f"  [warning]![/] @{participant}: {_escape(reason)}"))

    def _committed(self, data: dict[str, Any]) -> None:
        sender = data["participant"]
        recipients = [str(item) for item in data.get("recipients", [])]
        if sender == "lead":
            target = "final answer" if data.get("done") else " ".join(f"@{r}" for r in recipients)
            who = "you" if self._spec.lead.human else self._tool("lead")
            title = f"lead ({who}) → {target}"
        else:
            name, tool = self._names.get(sender), self._tool(sender)
            who = ", ".join(part for part in (name, tool) if part and part != "a person") or tool
            title = f"@{sender} ({who}) → lead"
        typed = sender in self._humans and self._typed_here
        text = str(data.get("text") or "") or "(empty message)"
        footer = [_duration(float(data.get("duration_seconds") or 0.0))]
        footer += [f"denied: {denial}" for denial in data.get("denials", [])]
        if typed:
            footer.insert(0, "typed above")
        heading = f"[step.count]\\[{data['turn']}][/] {_escape(title)}"

        def show() -> None:
            _stderr(heading)
            if not typed:
                sys.stderr.write(f"\n{text}\n\n")
                sys.stderr.flush()
            _stderr(f"[dim]      {_escape(' · '.join(footer))}[/]")

        self._print(show)
        if sender == "lead" and not data.get("done"):
            seats = tuple(_Seat(member_id, "queued") for member_id in recipients)
            self._set(_Activity("members", time.monotonic(), seats=seats))
            if not self._live_enabled:
                asked = ", ".join(f"@{member_id}" for member_id in recipients)
                self._print(lambda: _stderr(f"[dim]      asking {asked}…[/]"))
        elif sender == "lead":
            self._set(None)

    def _finished(self, _data: dict[str, Any]) -> None:
        self._set(None)

    # -- helpers -------------------------------------------------------------

    def _tool(self, participant: str) -> str:
        if participant == "lead":
            backend: Any = self._spec.lead.backend
        else:
            backend = next(m.backend for m in self._spec.members if m.id == participant)
        tool = getattr(getattr(backend, "tool", None), "value", None)
        return str(tool) if tool is not None else "a person"

    def _show_versions(self) -> None:
        if self._versions_shown or not self._versions:
            return
        self._versions_shown = True
        tools = " · ".join(
            name + (f" {version}" if version else "") for name, version in self._versions.items()
        )
        self._print(lambda: _stderr(f"[dim]      tools: {_escape(tools)}[/]"))

    def _find(self, member_id: str) -> _Seat | None:
        activity = self.activity
        if activity is None or activity.kind != "members":
            return None
        return next((seat for seat in activity.seats if seat.id == member_id), None)

    def _seat(self, member_id: str, **changes: Any) -> None:
        activity = self.activity
        if activity is None or activity.kind != "members":
            return
        seats = tuple(
            replace(seat, **changes) if seat.id == member_id else seat for seat in activity.seats
        )
        self._set(replace(activity, seats=seats))

    def _set(self, activity: _Activity | None) -> None:
        self.activity = activity
        if activity is None:
            self._stop_live()
        else:
            self._start_live()

    def _print(self, show: Callable[[], None]) -> None:
        if self._held:
            self._deferred.append(show)
            return
        running = self._live is not None
        self._stop_live()
        show()
        if running or self.activity is not None:
            self._start_live()

    def _start_live(self) -> None:
        if not self._live_enabled or self._held or self._live is not None:
            return
        if self.activity is None:
            return
        from rich.live import Live

        self._live = Live(_Line(self), console=console.err, transient=True, refresh_per_second=4)
        self._live.start()

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
