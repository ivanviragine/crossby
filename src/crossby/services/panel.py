"""Panels: one lead consults several read-only members.

A panel generalizes a two-party consultation. The lead holds the goal and
addresses members by id on the last line of each message::

    NEXT: ASK @cto @founder     (those members)
    NEXT: ASK @all              (every member)
    NEXT: DONE                  (the reply is the final answer)

Addressed members answer in parallel (at most ``concurrency`` at once), each
from its own folder and resumed session, read-only; their replies reach the
lead in roster order, each under its ``@id`` label. A member that fails is a
factual note in that message; a dispatch in which every member fails stops
the panel. ``max_rounds`` counts the lead's dispatches, plus one final turn.

In a ``private`` panel a member receives only what the lead addresses to it.
In a ``shared`` panel a member also receives, on its next turn, the other
members' replies it has not seen, under their labels: no extra turns run.

crossby adds only protocol text: the lead's opening notice (the goal, the
roster and the limit; replaceable through ``protocol_notice``, whose grammar
never changes), the ``@id`` labels, failure notes and the final-turn nudge.
Everything is journaled (see :mod:`crossby.services.journal`).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from crossby.models.ai import (
    HeadlessEvent,
    HeadlessInteractionMode,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
    RestrictionChannel,
    TokenUsage,
)
from crossby.services.ask import CliBackend, ProjectWorkspace, Workspace, render_envelope
from crossby.services.converse import (
    EXIT_CODES,
    MIN_TURN_SECONDS,
    PREFLIGHT_RESUME_ID,
    ConverseError,
    HeadlessTurnRunner,
    StopReason,
    TurnRunner,
    check_depth,
    child_environment,
)
from crossby.services.workspaces import OwnedWorkspace

if TYPE_CHECKING:
    from crossby.services.journal import Journal

PANEL_NOTICE_VERSION = 1
MAX_MEMBERS = 16
_MEMBER_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
# "all" addresses every member and "lead" names the lead in labels, events
# and sessions, so neither can be a member's id.
_RESERVED_IDS = frozenset({"all", "lead"})
_NEXT_LINE = re.compile(
    r"^[\s*_`>#-]*next\s*:\s*[*_`]*\s*(ask|done)\b(?P<args>[^\n]*?)[\s*_`.!]*$",
    re.IGNORECASE,
)

PANEL_NOTICE = (
    "Goal: {goal}\n\n"
    "You lead a panel of {count}: {roster}. Each reply you write goes to the members you "
    "name on its last line, and their answers come back to you together, each under its "
    "@id. End every reply with `NEXT: ASK @id` (one or more ids, or `@all`) to send it, or "
    "with `NEXT: DONE` when the reply is your final answer. You can send at most "
    "{max_rounds} messages."
)
FINAL_NUDGE = (
    "You have used all {max_rounds} messages. Write your final answer now and end with the "
    "line `NEXT: DONE`."
)


class PanelError(ConverseError):
    """A panel request is invalid before any agent starts."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Member(_Frozen):
    """One read-only member: an id the lead addresses, an agent and its folder."""

    id: str
    name: str | None = None
    backend: CliBackend
    workspace: Workspace

    @field_validator("id")
    @classmethod
    def _id_is_addressable(cls, value: str) -> str:
        if not _MEMBER_ID.fullmatch(value) or value in _RESERVED_IDS:
            raise ValueError(
                f"member id {value!r} must be lowercase letters, digits and hyphens "
                "(not 'all' or 'lead')"
            )
        return value

    @model_validator(mode="after")
    def _owned_folder_matches(self) -> Member:
        if isinstance(self.workspace, OwnedWorkspace) and self.workspace.tool is not (
            self.backend.tool
        ):
            raise ValueError(f"member {self.id}'s persona folder is for {self.workspace.tool}")
        return self

    @property
    def label(self) -> str:
        return f"@{self.id}"


class Lead(_Frozen):
    """The agent that holds the goal; it runs in ``folder`` with its normal policy."""

    backend: CliBackend
    folder: Path


class PanelSpec(_Frozen):
    goal: str
    lead: Lead
    members: tuple[Member, ...]
    visibility: Literal["private", "shared"] = "private"
    max_rounds: int = Field(default=5, ge=1)
    concurrency: int = Field(default=4, ge=1, le=MAX_MEMBERS)
    timeout_seconds: float = Field(default=1800.0, gt=0)
    turn_timeout_seconds: float = Field(default=600.0, gt=0)
    # Channels every member must have verified blocked (see crossby.api).
    require_blocked: frozenset[RestrictionChannel] = frozenset()
    allow_project_hooks: bool = False
    # Replaces the wording of the lead's opening notice; it is formatted with
    # {goal}, {roster}, {count} and {max_rounds}. The grammar never changes.
    protocol_notice: str | None = None

    @field_validator("goal")
    @classmethod
    def _goal_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the goal is empty")
        return value

    @model_validator(mode="after")
    def _roster_is_valid(self) -> PanelSpec:
        if not 1 <= len(self.members) <= MAX_MEMBERS:
            raise ValueError(f"a panel needs 1 to {MAX_MEMBERS} members")
        ids = [member.id for member in self.members]
        if len(set(ids)) != len(ids):
            raise ValueError("member ids must be unique")
        return self


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------


class ProtocolError(ValueError):
    """The lead's last line does not address the panel validly."""


@dataclass(frozen=True)
class Dispatch:
    done: bool
    recipients: tuple[str, ...] = ()


def parse_dispatch(text: str, member_ids: Sequence[str]) -> tuple[str, Dispatch]:
    """Split a lead reply into ``(body, dispatch)``, or raise :class:`ProtocolError`.

    Only the last non-blank line is read. ``NEXT: ASK`` with no ids addresses
    the only member of a one-member panel; ``@all`` cannot be combined with
    ids; ``NEXT: DONE`` takes none.
    """
    lines = text.rstrip().splitlines()
    index = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()), None)
    if index is None:
        raise ProtocolError("the message is empty")
    match = _NEXT_LINE.match(lines[index])
    body = "\n".join(lines[:index]).strip()
    if match is None:
        if len(member_ids) == 1:
            # A reply without the marker is sent on, as in a consultation.
            return text.strip(), Dispatch(done=False, recipients=(member_ids[0],))
        raise ProtocolError("the last line is not `NEXT: ASK @id …` or `NEXT: DONE`")
    action = match.group(1).lower()
    names = [token for token in re.split(r"[\s,]+", match.group("args").strip()) if token]
    if action == "done":
        if names:
            raise ProtocolError("`NEXT: DONE` takes no members")
        return body, Dispatch(done=True)
    if not names:
        if len(member_ids) == 1:
            return body, Dispatch(done=False, recipients=(member_ids[0],))
        raise ProtocolError("`NEXT: ASK` must name members (`@id` or `@all`)")
    if any(not name.startswith("@") for name in names):
        raise ProtocolError("members are named as `@id`")
    ids = [name[1:].lower() for name in names]
    if "all" in ids:
        if len(ids) > 1:
            raise ProtocolError("`@all` cannot be combined with member ids")
        return body, Dispatch(done=False, recipients=tuple(member_ids))
    unknown = [name for name in ids if name not in member_ids]
    if unknown:
        raise ProtocolError(f"no member is named @{unknown[0]}")
    if len(set(ids)) != len(ids):
        raise ProtocolError("a member is named twice")
    # Roster order, whatever order the lead named them in.
    return body, Dispatch(done=False, recipients=tuple(m for m in member_ids if m in ids))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelTurn:
    """One committed message: from the lead, or one member's reply."""

    number: int
    sender: str  # "lead" or a member id
    text: str
    recipients: tuple[str, ...] = ()
    done: bool = False
    duration_seconds: float = 0.0
    denials: tuple[str, ...] = ()
    usage: TokenUsage | None = None


@dataclass
class PanelResult:
    spec: PanelSpec
    started_at: datetime
    finished_at: datetime
    stop_reason: StopReason
    turns: list[PanelTurn] = field(default_factory=list)
    error: str | None = None
    # Member failures noted to the lead: (dispatch number, member id, reason).
    member_failures: list[tuple[int, str, str]] = field(default_factory=list)
    sessions: dict[str, str | None] = field(default_factory=dict)
    answered_at_limit: bool = False
    # Messages the lead actually sent to members.
    dispatches: int = 0

    @property
    def answered(self) -> bool:
        return self.stop_reason is StopReason.ANSWERED

    @property
    def final_answer(self) -> str | None:
        last = self.turns[-1] if self.turns else None
        return last.text if last is not None and last.done else None

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.stop_reason]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class _StopError(Exception):
    def __init__(self, reason: StopReason, error: str) -> None:
        super().__init__(error)
        self.reason = reason
        self.error = error


@dataclass
class _Seat:
    member: Member
    session_id: str | None = None
    # Other members' replies a shared member has not seen yet.
    pending: list[tuple[str, str]] = field(default_factory=list)


def render_notice(spec: PanelSpec) -> str:
    roster = ", ".join(
        f"@{member.id}" + (f" ({member.name})" if member.name else "") for member in spec.members
    )
    template = spec.protocol_notice or PANEL_NOTICE
    return template.format(
        goal=spec.goal, roster=roster, count=len(spec.members), max_rounds=spec.max_rounds
    ).strip()


def combine_replies(
    replies: Sequence[tuple[str, str | None, str | None]],
) -> str:
    """The lead's next message: each reply under its label, failures as notes."""
    parts: list[str] = []
    for member_id, text, failure in replies:
        if text is not None:
            parts.append(f"@{member_id}:\n{text}")
        else:
            parts.append(f"@{member_id} (no answer: {failure})")
    return "\n\n".join(parts)


def member_message(seat: _Seat, body: str, *, shared: bool) -> str:
    """What a member receives: the lead's message, plus missed replies if shared."""
    if not shared or not seat.pending:
        return body
    seen = "\n\n".join(f"@{member_id}:\n{text}" for member_id, text in seat.pending)
    return f"{seen}\n\n@lead:\n{body}"


def run_panel(
    spec: PanelSpec,
    *,
    runner: TurnRunner | None = None,
    journal: Journal | None = None,
    clock: Callable[[], float] = monotonic,
) -> PanelResult:
    """Run a panel until the lead gives its final answer, or a limit stops it."""
    from crossby.ai_tools.headless import HeadlessSessionError
    from crossby.services.journal import EventType

    check_depth()
    active: TurnRunner = runner or HeadlessTurnRunner()
    started_at = datetime.now(tz=UTC)
    deadline = clock() + spec.timeout_seconds
    member_ids = [member.id for member in spec.members]
    seats = {member.id: _Seat(member) for member in spec.members}
    lead_session: list[str | None] = [None]
    turns: list[PanelTurn] = []
    failures: list[tuple[int, str, str]] = []
    cancel = threading.Event()
    shared = spec.visibility == "shared"
    answered_at_limit = [False]
    sent = [0]
    next_turn = [1]

    def claim() -> int:
        """The next turn number: every lead turn and member attempt gets one."""
        with lock:
            number = next_turn[0]
            next_turn[0] += 1
            return number

    lock = threading.Lock()

    def emit(type_: EventType, **data: Any) -> None:
        if journal is not None:
            journal.emit(type_, **data)

    def sessions() -> dict[str, str | None]:
        return {"lead": lead_session[0], **{key: seat.session_id for key, seat in seats.items()}}

    def finish(reason: StopReason, error: str | None = None) -> PanelResult:
        result = PanelResult(
            spec=spec,
            started_at=started_at,
            finished_at=datetime.now(tz=UTC),
            stop_reason=reason,
            turns=turns,
            error=error,
            member_failures=failures,
            sessions=sessions(),
            answered_at_limit=answered_at_limit[0],
            dispatches=sent[0],
        )
        emit(
            EventType.CONVERSATION_FINISHED,
            stop_reason=reason.value,
            error=error,
            answered_at_limit=answered_at_limit[0],
            rounds=result.dispatches,
            sessions=result.sessions,
            finished_at=result.finished_at.isoformat(),
        )
        return result

    def remaining() -> float:
        return deadline - clock()

    def request(
        prompt: str,
        backend: CliBackend,
        folder: Path,
        *,
        resume_id: str | None,
        read_only: bool,
        minimum_version: str | None = None,
    ) -> HeadlessSessionRequest:
        left = remaining()
        if left < MIN_TURN_SECONDS:
            raise _StopError(
                StopReason.TIMEOUT, f"panel time limit of {spec.timeout_seconds:g}s reached"
            )
        return HeadlessSessionRequest(
            prompt=prompt,
            working_dir=folder,
            model=backend.model,
            effort=backend.effort,
            interaction_mode=HeadlessInteractionMode.UNATTENDED,
            timeout_seconds=min(spec.turn_timeout_seconds, left),
            resume_id=resume_id,
            read_only=read_only,
            require_blocked=spec.require_blocked if read_only else frozenset(),
            allow_project_hooks=spec.allow_project_hooks,
            minimum_version=minimum_version,
            environment=child_environment(),
        )

    def member_folder(member: Member) -> tuple[Path, str | None]:
        workspace = member.workspace
        if isinstance(workspace, OwnedWorkspace):
            return workspace.path, workspace.layout.verified_version
        return workspace.path, None

    def progress(participant: str, number: int) -> Callable[[HeadlessEvent], None]:
        def handle(event: HeadlessEvent) -> None:
            kind = event.kind.value
            if kind in {"tool_started", "tool_completed", "assistant_message"}:
                emit(
                    EventType.TURN_PROGRESS,
                    turn=number,
                    participant=participant,
                    kind=kind,
                    elapsed_seconds=round(event.elapsed_seconds, 3),
                )

        return handle

    def run_one(
        participant: str, number: int, built: HeadlessSessionRequest, tool: Any
    ) -> tuple[HeadlessSessionResult | None, str | None, float]:
        """Run one session; ``(result, failure, seconds)``. Never raises a session error."""
        emit(
            EventType.TURN_STARTED,
            turn=number,
            participant=participant,
            tool=tool.value,
            resumed=built.resume_id is not None,
            session_id=built.resume_id,
        )
        begun = clock()
        failure: str | None
        try:
            result = active.run(tool, built, cancel, on_event=progress(participant, number))
        except HeadlessSessionError as exc:
            failure = str(exc)
            emit(
                EventType.TURN_FINISHED,
                turn=number,
                participant=participant,
                status="failed",
                error=failure,
                duration_seconds=round(clock() - begun, 3),
            )
            return None, failure, clock() - begun
        failure = None
        text = (result.final_text or "").strip()
        if result.status is not HeadlessTerminalStatus.SUCCEEDED:
            from crossby.services.ask import describe_session_status

            failure = f"the session ended with status {describe_session_status(result)}"
        elif not text:
            failure = "the session returned no text"
        session = result.session_id or result.thread_id or result.conversation_id
        emit(
            EventType.TURN_FINISHED,
            turn=number,
            participant=participant,
            status=result.status.value,
            session_id=session,
            duration_seconds=round(clock() - begun, 3),
            denials=list(result.denials),
            failure_reason=result.failure_reason.value if result.failure_reason else None,
            error=failure,
        )
        return result, failure, clock() - begun

    def commit(turn: PanelTurn) -> None:
        turns.append(turn)
        emit(
            EventType.MESSAGE_COMMITTED,
            turn=turn.number,
            participant=turn.sender,
            text=turn.text,
            recipients=list(turn.recipients),
            done=turn.done,
            duration_seconds=round(turn.duration_seconds, 3),
            denials=list(turn.denials),
        )

    def lead_turn(message: str, number: int) -> tuple[str, float, HeadlessSessionResult]:
        built = request(
            message,
            spec.lead.backend,
            spec.lead.folder,
            resume_id=lead_session[0],
            read_only=False,
        )
        result, failure, seconds = run_one("lead", number, built, spec.lead.backend.tool)
        if result is None or failure is not None:
            if remaining() <= 0:
                raise _StopError(
                    StopReason.TIMEOUT, f"panel time limit reached during turn {number}"
                )
            raise _StopError(StopReason.FAILED, f"lead turn {number}: {failure}")
        if lead_session[0] is None:
            handle = result.session_id or result.thread_id or result.conversation_id
            if handle is None:
                raise _StopError(
                    StopReason.FAILED,
                    f"lead turn {number}: {spec.lead.backend.tool.value} reported no session",
                )
            lead_session[0] = handle
        return (result.final_text or "").strip(), seconds, result

    def member_turn(
        seat: _Seat, body: str, number: int
    ) -> tuple[str, str | None, str | None, float, HeadlessSessionResult | None]:
        member = seat.member
        folder, minimum = member_folder(member)
        message = member_message(seat, body, shared=shared)
        if seat.session_id is None and isinstance(member.workspace, ProjectWorkspace):
            message = render_envelope(message, member.workspace.instructions)
        built = request(
            message,
            member.backend,
            folder,
            resume_id=seat.session_id,
            read_only=True,
            minimum_version=minimum,
        )
        result, failure, seconds = run_one(member.id, number, built, member.backend.tool)
        text = (result.final_text or "").strip() if result is not None else None
        if failure is None and result is not None and seat.session_id is None:
            handle = result.session_id or result.thread_id or result.conversation_id
            if handle is None:
                failure = f"{member.backend.tool.value} reported no session to resume"
            else:
                seat.session_id = handle
        if failure is None:
            with lock:
                seat.pending.clear()
        return member.id, (text if failure is None else None), failure, seconds, result

    def preflight() -> None:
        from crossby.services.workspaces import WorkspaceDriftError, verify

        for member in spec.members:
            if isinstance(member.workspace, OwnedWorkspace):
                try:
                    verify(member.workspace)
                except WorkspaceDriftError as exc:
                    raise _StopError(
                        StopReason.FAILED,
                        f"@{member.id}'s persona folder changed since it was provisioned: "
                        f"{', '.join(exc.changed[:5])}; provision it again",
                    ) from None
        checks: list[tuple[str, CliBackend, Path, bool, str | None]] = [
            ("lead", spec.lead.backend, spec.lead.folder, False, None)
        ]
        for member in spec.members:
            folder, minimum = member_folder(member)
            checks.append((member.id, member.backend, folder, True, minimum))
        for participant, backend, folder, read_only, minimum in checks:
            probe = request(
                "preflight",
                backend,
                folder,
                resume_id=PREFLIGHT_RESUME_ID,
                read_only=read_only,
                minimum_version=minimum,
            )
            try:
                report = active.preflight(backend.tool, probe)
            except HeadlessSessionError as exc:
                raise _StopError(
                    StopReason.FAILED, f"{participant} cannot run headless: {exc}"
                ) from None
            emit(
                EventType.PARTICIPANT_READY,
                participant=participant,
                tool=backend.tool.value,
                model=backend.model,
                folder=str(folder),
                read_only=read_only,
                version=report.detected_version if report is not None else None,
                restrictions=(
                    {item.channel.value: item.state.value for item in report.restrictions}
                    if report is not None
                    else {}
                ),
            )

    def fan_out(body: str, recipients: tuple[str, ...], dispatch: int) -> str:
        # Each member's attempt gets its own turn number, kept when its reply
        # is committed; a failed attempt's number is never reused.
        numbers = {member_id: claim() for member_id in recipients}
        outcomes: dict[str, tuple[str | None, str | None, float, HeadlessSessionResult | None]] = {}
        with ThreadPoolExecutor(max_workers=min(spec.concurrency, len(recipients))) as pool:
            futures: dict[Future[Any], str] = {
                pool.submit(member_turn, seats[member_id], body, numbers[member_id]): member_id
                for member_id in recipients
            }
            pending_futures = set(futures)
            try:
                while pending_futures:
                    done, pending_futures = wait(
                        pending_futures, timeout=0.1, return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        member_id, text, failure, seconds, result = future.result()
                        outcomes[member_id] = (text, failure, seconds, result)
            except KeyboardInterrupt:
                cancel.set()
                raise
        replies: list[tuple[str, str | None, str | None]] = []
        for member_id in recipients:  # roster order
            text, failure, seconds, result = outcomes[member_id]
            if text is not None:
                commit(
                    PanelTurn(
                        number=numbers[member_id],
                        sender=member_id,
                        text=text,
                        recipients=("lead",),
                        duration_seconds=seconds,
                        denials=tuple(result.denials) if result is not None else (),
                        usage=result.usage if result is not None else None,
                    )
                )
                if shared:
                    with lock:
                        for other in seats.values():
                            if other.member.id != member_id:
                                other.pending.append((member_id, text))
            else:
                failures.append((dispatch, member_id, failure or "no answer"))
            replies.append((member_id, text, failure))
        if all(text is None for _member, text, _failure in replies):
            raise _StopError(
                StopReason.FAILED,
                f"every member addressed in message {dispatch} failed: "
                + "; ".join(f"@{member} {failure}" for member, _text, failure in replies),
            )
        return combine_replies(replies)

    emit(
        EventType.CONVERSATION_STARTED,
        goal=spec.goal,
        kind="panel",
        visibility=spec.visibility,
        notice_version=PANEL_NOTICE_VERSION,
        participants=[
            {"participant": "lead", "tool": spec.lead.backend.tool.value, "read_only": False},
            *[
                {
                    "participant": member.id,
                    "name": member.name,
                    "tool": member.backend.tool.value,
                    "read_only": True,
                }
                for member in spec.members
            ],
        ],
        max_rounds=spec.max_rounds,
        started_at=started_at.isoformat(),
    )

    try:
        preflight()
        message = render_notice(spec)
        dispatches = 0
        while True:
            final = dispatches >= spec.max_rounds
            number = claim()
            text, seconds, result = lead_turn(message, number)
            try:
                body, dispatch = parse_dispatch(text, member_ids)
            except ProtocolError as exc:
                raise _StopError(
                    StopReason.FAILED, f"lead turn {number}: {exc}; nothing was sent"
                ) from None
            if not body:
                raise _StopError(StopReason.FAILED, f"lead turn {number} sent only the marker")
            commit(
                PanelTurn(
                    number=number,
                    sender="lead",
                    text=body,
                    recipients=dispatch.recipients,
                    done=dispatch.done,
                    duration_seconds=seconds,
                    denials=tuple(result.denials),
                    usage=result.usage,
                )
            )
            if dispatch.done:
                answered_at_limit[0] = final
                return finish(StopReason.ANSWERED)
            if final:
                return finish(StopReason.ROUND_LIMIT)
            dispatches += 1
            sent[0] = dispatches
            message = fan_out(body, dispatch.recipients, dispatches)
            if dispatches == spec.max_rounds:
                message = f"{message}\n\n{FINAL_NUDGE.format(max_rounds=spec.max_rounds)}"
    except _StopError as stop:
        return finish(stop.reason, stop.error)
    except KeyboardInterrupt:
        cancel.set()
        return finish(StopReason.INTERRUPTED, f"interrupted after {len(turns)} messages")


def render_panel_transcript(result: PanelResult) -> str:
    """The panel as markdown: roster, goal, then every committed message."""
    from crossby.services.converse import _fenced

    spec = result.spec
    lines = [
        "# Panel",
        "",
        f"- Lead: {spec.lead.backend.tool.value}, in `{spec.lead.folder}`",
    ]
    for member in spec.members:
        folder = member.workspace.path
        kind = "persona folder" if isinstance(member.workspace, OwnedWorkspace) else "folder"
        name = f" ({member.name})" if member.name else ""
        lines.append(
            f"- @{member.id}{name}: {member.backend.tool.value}, read-only, {kind} `{folder}`"
        )
    lines += [
        f"- Visibility: {spec.visibility}",
        f"- Started: {result.started_at.isoformat(timespec='seconds')}",
        f"- Finished: {result.finished_at.isoformat(timespec='seconds')}",
        f"- Messages allowed: {spec.max_rounds}",
        f"- Outcome: {result.stop_reason.value}" + (f": {result.error}" if result.error else ""),
        "",
        "## Goal (sent only to the lead)",
        "",
        _fenced(spec.goal),
        "",
    ]
    for turn in result.turns:
        if turn.sender == "lead":
            target = "final answer" if turn.done else " ".join(f"@{r}" for r in turn.recipients)
            heading = f"## Turn {turn.number} · lead ({spec.lead.backend.tool.value}) → {target}"
        else:
            heading = f"## Turn {turn.number} · @{turn.sender} → lead"
        lines += ["---", "", heading, "", turn.text or "(empty message)", ""]
    if result.member_failures:
        lines += ["---", "", "## Members that did not answer", ""]
        lines += [
            f"- message {dispatch}: @{member} — {reason}"
            for dispatch, member, reason in result.member_failures
        ]
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "FINAL_NUDGE",
    "MAX_MEMBERS",
    "PANEL_NOTICE",
    "PANEL_NOTICE_VERSION",
    "Dispatch",
    "Lead",
    "Member",
    "PanelError",
    "PanelResult",
    "PanelSpec",
    "PanelTurn",
    "ProtocolError",
    "combine_replies",
    "member_message",
    "parse_dispatch",
    "render_notice",
    "render_panel_transcript",
    "run_panel",
]
