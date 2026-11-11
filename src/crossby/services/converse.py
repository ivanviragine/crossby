"""Goal-driven consultations between two headless AI coding agents.

``crossby converse ASKER RESPONDER --goal ...`` lets one agent consult another.
The asker holds the goal: it decides what to ask, judges each answer, asks
follow-up questions, and decides when it has enough. The responder never sees
the goal; it only answers the asker's messages. ``crossby ask`` (see
:mod:`crossby.services.ask`) is the single-question primitive.

Each side keeps its own native session: every turn after its first resumes
that session and carries only the other agent's latest message, whole. Nothing
is replayed, budgeted or cut, so the only limit on a conversation is each
model's own context, which its tool manages. A CLI build whose unattended
resume was not verified is refused at preflight, with the build to upgrade to,
and a turn that reports no session to resume stops the conversation.

The responder never changes files: its turns (and ``ask`` and the summary) run
with the adapter's verified read-only profile, and a tool without one is
refused at preflight. The asker is the user's own agent and keeps its folder's
normal unattended policy.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Protocol

from crossby.models.ai import (
    AIToolID,
    EffortLevel,
    HeadlessEvent,
    HeadlessEventKind,
    HeadlessInteractionMode,
    HeadlessSessionPreflight,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
    TokenUsage,
)

if TYPE_CHECKING:
    from crossby.handoff.models import ConversationTranscript
    from crossby.services.journal import ConversationEvent, Journal

DEPTH_ENV = "CROSSBY_CONVERSE_DEPTH"
# Agents started by ``ask``/``converse`` may not start another one: the depth
# counter is inherited by child processes, so this stops agents from consulting
# each other in an unbounded chain.
MAX_DEPTH = 1

DEFAULT_MAX_ROUNDS = 5
DEFAULT_TIMEOUT_SECONDS = 1800.0
DEFAULT_TURN_TIMEOUT_SECONDS = 600.0
DEFAULT_SUMMARY_TIMEOUT_SECONDS = 300.0
PREFLIGHT_TIMEOUT_SECONDS = 20.0
# A turn needs a realistic minimum to start a native CLI and answer; with less
# than this left on the overall deadline the conversation stops as timed out.
MIN_TURN_SECONDS = 5.0
# Preflight validates each side as the resumed session every turn after its
# first will be: the adapter then refuses a build whose resume is unverified.
# Only the ID's shape is checked (some tools require a UUID); nothing is opened.
PREFLIGHT_RESUME_ID = "00000000-0000-4000-8000-000000000000"

ASKER_PROMPT_FILE = "converse_asker.md"
SUMMARY_PROMPT_FILE = "converse_summary.md"

FINAL_NUDGE = (
    "You have used all {max_rounds} messages. Write your final answer now and end "
    "with the line `NEXT: DONE`."
)


class ConverseError(RuntimeError):
    """A consultation request is invalid before any agent is started."""


class ConverseDepthError(ConverseError):
    """``ask``/``converse`` was invoked from inside another consultation."""


class ConverseSummaryError(RuntimeError):
    """The summary session failed; the transcript is unaffected."""


class Role(StrEnum):
    ASKER = "asker"
    RESPONDER = "responder"


class Next(StrEnum):
    """What the asker's reply is: a message to send, or its final answer."""

    ASK = "ASK"
    DONE = "DONE"
    MISSING = "MISSING"


class StopReason(StrEnum):
    """Why a consultation ended."""

    ANSWERED = "answered"
    ROUND_LIMIT = "round_limit"
    TIMEOUT = "timeout"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


EXIT_CODES: dict[StopReason, int] = {
    StopReason.ANSWERED: 0,
    StopReason.FAILED: 1,
    StopReason.ROUND_LIMIT: 2,
    StopReason.TIMEOUT: 2,
    StopReason.INTERRUPTED: 130,
}


@dataclass(frozen=True)
class Participant:
    """One side of a consultation."""

    tool: AIToolID
    role: Role
    model: str | None = None
    effort: EffortLevel | None = None

    @property
    def label(self) -> str:
        return f"{self.role.value} ({self.tool.value})"

    @property
    def settings(self) -> str:
        """``claude-opus-5-5, high effort``: the model and effort asked for, if any."""
        parts = [self.model] if self.model else []
        if self.effort is not None:
            parts.append(f"{self.effort.value} effort")
        return ", ".join(parts)


@dataclass(frozen=True)
class ConverseTurn:
    """One relayed message."""

    number: int
    role: Role
    tool: AIToolID
    text: str
    next: Next | None
    duration_seconds: float
    # What the tool reported denying during this turn (bounded, count-only
    # diagnostics from the adapter, never the denied input).
    denials: tuple[str, ...] = ()
    # Tokens as the tool reported them for this turn, when it did.
    usage: TokenUsage | None = None

    @property
    def is_final_answer(self) -> bool:
        return self.role is Role.ASKER and self.next is Next.DONE


@dataclass
class ConversationResult:
    """The full outcome of a consultation, including partial progress."""

    goal: str
    asker: Participant
    responder: Participant
    max_rounds: int
    started_at: datetime
    finished_at: datetime
    stop_reason: StopReason
    turns: list[ConverseTurn] = field(default_factory=list)
    error: str | None = None
    answered_at_limit: bool = False
    # Each side's native session (``None`` until its first turn reported one).
    sessions: dict[Role, str | None] = field(default_factory=dict)
    # The folder each side ran in.
    folders: dict[Role, Path] = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.stop_reason is StopReason.ANSWERED

    @property
    def final_answer(self) -> str | None:
        last = self.turns[-1] if self.turns else None
        return last.text if last is not None and last.is_final_answer else None

    @property
    def rounds(self) -> int:
        """Answers the responder gave."""
        return sum(1 for turn in self.turns if turn.role is Role.RESPONDER)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.stop_reason]


class TurnRunner(Protocol):
    """Runs managed headless sessions; injectable so tests need no real CLIs."""

    def preflight(
        self, tool: AIToolID, request: HeadlessSessionRequest
    ) -> HeadlessSessionPreflight | None: ...

    def run(
        self,
        tool: AIToolID,
        request: HeadlessSessionRequest,
        cancel_event: threading.Event,
        on_event: Callable[[HeadlessEvent], None] | None = None,
    ) -> HeadlessSessionResult: ...


class HeadlessTurnRunner:
    """Default runner backed by each adapter's managed headless session."""

    def preflight(
        self, tool: AIToolID, request: HeadlessSessionRequest
    ) -> HeadlessSessionPreflight | None:
        from crossby.ai_tools import preflight_headless_session

        return preflight_headless_session(tool, request, timeout_seconds=PREFLIGHT_TIMEOUT_SECONDS)

    def run(
        self,
        tool: AIToolID,
        request: HeadlessSessionRequest,
        cancel_event: threading.Event,
        on_event: Callable[[HeadlessEvent], None] | None = None,
    ) -> HeadlessSessionResult:
        """Run one session in a worker so Ctrl-C cancels it cleanly."""
        from crossby.services.ask import run_cancellable

        return run_cancellable(tool, request, cancel_event, event_handler=on_event)


# ---------------------------------------------------------------------------
# Recursion guard
# ---------------------------------------------------------------------------


def current_depth() -> int:
    raw = os.environ.get(DEPTH_ENV, "").strip()
    try:
        return max(int(raw), 0) if raw else 0
    except ValueError:
        return 0


def check_depth() -> None:
    """Refuse to start agents from inside an agent that crossby started."""
    if current_depth() >= MAX_DEPTH:
        raise ConverseDepthError(
            "crossby ask/converse was called from an agent that crossby itself started "
            f"({DEPTH_ENV}={current_depth()}). Nested consultations are disabled to "
            "prevent agents from calling each other in a loop."
        )


def child_environment() -> dict[str, str]:
    """The variables that mark an agent crossby starts as one level deeper.

    They travel in each session request, so only that agent's process gets
    them: the caller's own environment never changes, and concurrent
    consultations in one process cannot see or undo each other's.
    """
    return {DEPTH_ENV: str(current_depth() + 1)}


# ---------------------------------------------------------------------------
# Capabilities, prompts and parsing
# ---------------------------------------------------------------------------


def make_participants(
    asker: AIToolID,
    responder: AIToolID,
    *,
    asker_model: str | None = None,
    responder_model: str | None = None,
    asker_effort: EffortLevel | None = None,
    responder_effort: EffortLevel | None = None,
) -> tuple[Participant, Participant]:
    return (
        Participant(tool=asker, role=Role.ASKER, model=asker_model, effort=asker_effort),
        Participant(
            tool=responder, role=Role.RESPONDER, model=responder_model, effort=responder_effort
        ),
    )


_NEXT_LINE = re.compile(
    r"^[\s*_`>#-]*next\s*:\s*[*_`]*\s*(ask|done)\b[\s*_`.!]*$",
    re.IGNORECASE,
)


def parse_next(text: str) -> tuple[str, Next]:
    """Split an asker reply into ``(body, next)``.

    The marker must be the last non-empty line; markdown decoration such as
    ``**NEXT: DONE**`` is tolerated. A reply without one is ``MISSING`` and is
    sent on as a message, never taken as a final answer.
    """
    lines = text.rstrip().splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip():
            match = _NEXT_LINE.match(lines[index])
            if match is None:
                return text.strip(), Next.MISSING
            return "\n".join(lines[:index]).strip(), Next(match.group(1).upper())
    return "", Next.MISSING


def _load(filename: str) -> str:
    from crossby.handoff.prompts import load_bundled_prompt

    return load_bundled_prompt(filename)


def asker_opening(*, goal: str, responder: Participant, max_rounds: int) -> str:
    return (
        _load(ASKER_PROMPT_FILE)
        .format(goal=goal, responder=responder.tool.value, max_rounds=max_rounds)
        .strip()
    )


def reply_for_asker(reply: str, *, final: bool, max_rounds: int) -> str:
    """What the asker receives: the responder's answer as is.

    Only after the last allowed message is one line added, asking for the
    final answer.
    """
    if not final:
        return reply
    return f"{reply}\n\n{FINAL_NUDGE.format(max_rounds=max_rounds)}"


def _request(
    participant: Participant | None,
    *,
    prompt: str,
    working_dir: Path,
    timeout_seconds: float,
    model: str | None = None,
    effort: EffortLevel | None = None,
    resume_id: str | None = None,
    read_only: bool = False,
    allow_project_hooks: bool = False,
) -> HeadlessSessionRequest:
    return HeadlessSessionRequest(
        prompt=prompt,
        working_dir=working_dir,
        model=participant.model if participant is not None else model,
        effort=participant.effort if participant is not None else effort,
        interaction_mode=HeadlessInteractionMode.UNATTENDED,
        timeout_seconds=timeout_seconds,
        resume_id=resume_id,
        read_only=read_only,
        allow_project_hooks=allow_project_hooks,
        environment=child_environment(),
    )


def _read_only(participant: Participant) -> bool:
    """A responder never changes files: its sessions run read-only.

    The asker is the user's own agent in the user's project, so it keeps the
    folder's normal unattended policy.
    """
    return participant.role is Role.RESPONDER


def _status(result: HeadlessSessionResult) -> str:
    """``failed``, plus the recognized reason and the tool's own status.

    For example ``failed: not signed in (native: api_error)``; see
    :func:`crossby.services.ask.describe_session_status`.
    """
    from crossby.services.ask import describe_session_status

    return describe_session_status(result)


def _session_handle(result: HeadlessSessionResult) -> str | None:
    return result.session_id or result.thread_id or result.conversation_id


# ---------------------------------------------------------------------------
# converse
# ---------------------------------------------------------------------------


class _StopError(Exception):
    def __init__(self, reason: StopReason, error: str | None = None) -> None:
        super().__init__(error)
        self.reason = reason
        self.error = error


@dataclass
class _Side:
    participant: Participant
    session_id: str | None = None


@dataclass(frozen=True)
class _Reply:
    """What one turn produced, before the asker's marker is parsed."""

    text: str
    duration: float
    denials: tuple[str, ...]
    usage: TokenUsage | None = None


# Runtime event kinds forwarded as ``turn.progress``; each names only the kind
# of step a tool took, never its content.
_PROGRESS_KINDS = {
    HeadlessEventKind.TOOL_STARTED: "tool_started",
    HeadlessEventKind.TOOL_COMPLETED: "tool_completed",
    HeadlessEventKind.ASSISTANT_MESSAGE: "assistant_message",
    HeadlessEventKind.PROGRESS: "progress",
}
# At most one generic "progress" event per second per turn: streaming tools
# emit one per native frame.
_PROGRESS_INTERVAL_SECONDS = 1.0


def _usage_data(usage: TokenUsage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    data = usage.model_dump(
        include={
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "total_tokens",
            "premium_requests",
            "reported_cost_usd",
        },
        exclude_none=True,
    )
    return data or None


def run_conversation(
    goal: str,
    participants: tuple[Participant, Participant],
    *,
    working_dir: Path,
    responder_dir: Path | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    turn_timeout_seconds: float = DEFAULT_TURN_TIMEOUT_SECONDS,
    runner: TurnRunner | None = None,
    on_turn_start: Callable[[int, Participant], None] | None = None,
    on_turn: Callable[[ConverseTurn], None] | None = None,
    clock: Callable[[], float] = monotonic,
    allow_project_hooks: bool = False,
    journal: Journal | None = None,
    resume: ConversationResult | None = None,
) -> ConversationResult:
    """Let the asker consult the responder until it gives a final answer.

    The asker runs in ``working_dir`` and the responder in ``responder_dir``
    (default: the same folder), each with that folder's own files and setup.

    The asker holds the goal and ends every reply with ``NEXT: ASK`` (send it
    to the responder) or ``NEXT: DONE`` (its final answer). After the last
    allowed round the asker is told to answer now; a limit, a failure or Ctrl-C
    always returns the turns completed so far.

    ``journal`` receives every event as it happens (see
    :mod:`crossby.services.journal`). ``resume`` continues a consultation that
    stopped without an answer: both sides resume their own sessions, the next
    side gets the message it was owed, and up to ``max_rounds`` more answers
    are allowed.
    """
    from crossby.ai_tools.headless import HeadlessSessionError
    from crossby.services.journal import EventType

    check_depth()
    if not goal.strip():
        raise ConverseError("The goal is empty.")
    if max_rounds < 1:
        raise ConverseError("--max-rounds must be at least 1.")
    if timeout_seconds <= 0 or turn_timeout_seconds <= 0:
        raise ConverseError("Timeouts must be positive.")
    asker, responder = participants
    if asker.role is not Role.ASKER or responder.role is not Role.RESPONDER:
        raise ConverseError("Participants must be (asker, responder).")
    if resume is not None and (
        resume.answered or (resume.turns and resume.turns[-1].is_final_answer)
    ):
        raise ConverseError("That consultation already has its final answer.")

    active_runner: TurnRunner = runner or HeadlessTurnRunner()
    folders = {Role.ASKER: working_dir, Role.RESPONDER: responder_dir or working_dir}
    started_at = resume.started_at if resume is not None else datetime.now(tz=UTC)
    deadline = clock() + timeout_seconds
    turns: list[ConverseTurn] = list(resume.turns) if resume is not None else []
    sides = {Role.ASKER: _Side(asker), Role.RESPONDER: _Side(responder)}
    if resume is not None:
        for role, side in sides.items():
            side.session_id = resume.sessions.get(role)
    answers = sum(1 for turn in turns if turn.role is Role.RESPONDER)
    # Answers allowed in all: the ones already given plus this run's.
    cap = answers + max_rounds
    opening = asker_opening(goal=goal, responder=responder, max_rounds=max_rounds)
    in_progress: int | None = None
    answered_at_limit = False

    def emit(type_: EventType, **data: Any) -> None:
        if journal is not None:
            journal.emit(type_, **data)

    def side_data(participant: Participant) -> dict[str, Any]:
        return {
            "role": participant.role.value,
            "tool": participant.tool.value,
            "model": participant.model,
            "effort": participant.effort.value if participant.effort is not None else None,
            "folder": str(folders[participant.role]),
            "read_only": _read_only(participant),
        }

    if resume is None:
        emit(
            EventType.CONVERSATION_STARTED,
            goal=goal,
            participants=[side_data(asker), side_data(responder)],
            max_rounds=cap,
            timeout_seconds=timeout_seconds,
            turn_timeout_seconds=turn_timeout_seconds,
            started_at=started_at.isoformat(),
        )
    else:
        emit(
            EventType.CONVERSATION_CONTINUED,
            previous_stop_reason=resume.stop_reason.value,
            max_rounds=cap,
            added_rounds=max_rounds,
            timeout_seconds=timeout_seconds,
            turn_timeout_seconds=turn_timeout_seconds,
        )

    def finish(reason: StopReason, error: str | None = None) -> ConversationResult:
        result = ConversationResult(
            goal=goal,
            asker=asker,
            responder=responder,
            max_rounds=cap,
            started_at=started_at,
            finished_at=datetime.now(tz=UTC),
            stop_reason=reason,
            turns=turns,
            error=error,
            answered_at_limit=answered_at_limit,
            sessions={role: side.session_id for role, side in sides.items()},
            folders=dict(folders),
        )
        emit(
            EventType.CONVERSATION_FINISHED,
            stop_reason=reason.value,
            error=error,
            answered_at_limit=answered_at_limit,
            rounds=result.rounds,
            sessions={role.value: session for role, session in result.sessions.items()},
            finished_at=result.finished_at.isoformat(),
        )
        return result

    def out_of_time(when: str) -> _StopError:
        return _StopError(
            StopReason.TIMEOUT, f"conversation time limit of {timeout_seconds:g}s reached {when}"
        )

    def progress_handler(number: int) -> Callable[[HeadlessEvent], None]:
        last_progress = [float("-inf")]

        def handle(event: HeadlessEvent) -> None:
            kind = _PROGRESS_KINDS.get(event.kind)
            if kind is None:
                return
            if event.kind is HeadlessEventKind.PROGRESS:
                if event.elapsed_seconds - last_progress[0] < _PROGRESS_INTERVAL_SECONDS:
                    return
                last_progress[0] = event.elapsed_seconds
            emit(
                EventType.TURN_PROGRESS,
                turn=number,
                kind=kind,
                elapsed_seconds=round(event.elapsed_seconds, 3),
            )

        return handle

    def turn_finished(
        number: int,
        role: Role,
        result: HeadlessSessionResult | None,
        *,
        status: str,
        duration: float,
        error: str | None = None,
    ) -> None:
        emit(
            EventType.TURN_FINISHED,
            turn=number,
            role=role.value,
            tool=sides[role].participant.tool.value,
            status=status,
            duration_seconds=round(duration, 3),
            session_id=(_session_handle(result) if result is not None else None)
            or sides[role].session_id,
            usage=_usage_data(result.usage) if result is not None else None,
            denials=list(result.denials) if result is not None else [],
            failure_reason=(
                result.failure_reason.value
                if result is not None and result.failure_reason is not None
                else None
            ),
            error=error,
        )

    def take_turn(role: Role, message: str) -> _Reply:
        """Run one turn for ``role`` and return its reply.

        The prompt is the message itself, whole: the session holds everything
        said before. Only the asker's very first turn sends its opening.
        """
        nonlocal in_progress
        side = sides[role]
        number = len(turns) + 1
        remaining = deadline - clock()
        if remaining < MIN_TURN_SECONDS:
            raise out_of_time(f"before turn {number}")
        bounded_by_deadline = remaining <= turn_timeout_seconds
        request = _request(
            side.participant,
            prompt=message if turns else opening,
            working_dir=folders[role],
            timeout_seconds=min(turn_timeout_seconds, remaining),
            resume_id=side.session_id,
            read_only=_read_only(side.participant),
            allow_project_hooks=allow_project_hooks,
        )
        in_progress = number
        emit(
            EventType.TURN_STARTED,
            turn=number,
            role=role.value,
            tool=side.participant.tool.value,
            resumed=side.session_id is not None,
            session_id=side.session_id,
        )
        if on_turn_start is not None:
            on_turn_start(number, side.participant)
        turn_started = clock()
        label = side.participant.label

        def stop(reason: StopReason, error: str, result: HeadlessSessionResult | None) -> None:
            status = result.status.value if result is not None else "failed"
            turn_finished(
                number, role, result, status=status, duration=clock() - turn_started, error=error
            )
            raise _StopError(reason, error)

        try:
            result = active_runner.run(
                side.participant.tool, request, cancel_event, on_event=progress_handler(number)
            )
        except HeadlessSessionError as exc:
            # The runtime re-probes the version before it starts and raises
            # (rather than returning TIMED_OUT) when the deadline runs out there.
            if clock() >= deadline:
                timeout = out_of_time(f"during turn {number}")
                stop(timeout.reason, timeout.error or "", None)
            stop(StopReason.FAILED, f"{label} turn {number}: {exc}", None)
            raise AssertionError("unreachable") from None  # pragma: no cover
        if result.status is HeadlessTerminalStatus.TIMED_OUT:
            if bounded_by_deadline:
                timeout = out_of_time(f"during turn {number}")
                stop(timeout.reason, timeout.error or "", result)
            stop(
                StopReason.FAILED,
                f"{label} turn {number} timed out after {turn_timeout_seconds:g}s",
                result,
            )
        # A tool that denied an action may end its turn without an answer
        # (Antigravity CLI does); name the denials so the failure explains itself.
        denied = f" ({'; '.join(result.denials)})" if result.denials else ""
        if result.status is not HeadlessTerminalStatus.SUCCEEDED:
            stop(
                StopReason.FAILED,
                f"{label} turn {number} ended with status {_status(result)}{denied}",
                result,
            )
        text = (result.final_text or "").strip()
        if not text:
            stop(StopReason.FAILED, f"{label} turn {number} returned no text{denied}", result)
        if side.session_id is None:
            handle = _session_handle(result)
            if handle is None:
                # Its next turn could only start over without the conversation.
                stop(
                    StopReason.FAILED,
                    f"{label} turn {number}: {side.participant.tool.value} reported no "
                    "session to resume, so the conversation cannot continue",
                    result,
                )
            side.session_id = handle
        duration = clock() - turn_started
        turn_finished(number, role, result, status=result.status.value, duration=duration)
        return _Reply(text, duration, tuple(result.denials), result.usage)

    def record(role: Role, reply: _Reply, body: str, next_: Next | None) -> ConverseTurn:
        nonlocal in_progress
        turn = ConverseTurn(
            number=len(turns) + 1,
            role=role,
            tool=sides[role].participant.tool,
            text=body,
            next=next_,
            duration_seconds=reply.duration,
            denials=reply.denials,
            usage=reply.usage,
        )
        turns.append(turn)
        in_progress = None
        emit(
            EventType.MESSAGE_COMMITTED,
            turn=turn.number,
            role=role.value,
            tool=turn.tool.value,
            text=body,
            next=next_.value if next_ is not None else None,
            duration_seconds=round(reply.duration, 3),
            denials=list(reply.denials),
            usage=_usage_data(reply.usage),
        )
        if on_turn is not None:
            on_turn(turn)
        return turn

    def preflight() -> None:
        for participant in dict.fromkeys(participants):
            remaining = deadline - clock()
            if remaining < MIN_TURN_SECONDS:
                raise out_of_time("during preflight")
            probe = _request(
                participant,
                prompt="preflight",
                working_dir=folders[participant.role],
                timeout_seconds=min(turn_timeout_seconds, remaining),
                resume_id=PREFLIGHT_RESUME_ID,
                read_only=_read_only(participant),
                allow_project_hooks=allow_project_hooks,
            )
            try:
                report = active_runner.preflight(participant.tool, probe)
            except HeadlessSessionError as exc:
                if clock() >= deadline:
                    raise out_of_time("during preflight") from None
                raise _StopError(
                    StopReason.FAILED, f"{participant.label} cannot run headless: {exc}"
                ) from None
            emit(
                EventType.PARTICIPANT_READY,
                **side_data(participant),
                version=report.detected_version if report is not None else None,
                restrictions=(
                    {item.channel.value: item.state.value for item in report.restrictions}
                    if report is not None
                    else {}
                ),
            )

    def relay() -> ConversationResult:
        nonlocal answered_at_limit, answers
        preflight()
        # Whose turn it is, and the message it is owed.
        if turns and turns[-1].role is Role.RESPONDER:
            next_role = Role.ASKER
            message = reply_for_asker(turns[-1].text, final=answers >= cap, max_rounds=cap)
        elif turns:
            next_role, message = Role.RESPONDER, turns[-1].text
        else:
            next_role, message = Role.ASKER, ""  # the first turn sends the opening
        while True:
            if next_role is Role.ASKER:
                final = answers >= cap
                asked = take_turn(Role.ASKER, message)
                body, next_ = parse_next(asked.text)
                if not body:
                    # A bare marker is neither an answer nor a message to send.
                    what = "final answer" if next_ is Next.DONE else "message"
                    raise _StopError(
                        StopReason.FAILED,
                        f"{asker.label} turn {len(turns) + 1} sent only the NEXT marker, "
                        f"with no {what}",
                    )
                record(Role.ASKER, asked, body, next_)
                if next_ is Next.DONE:
                    answered_at_limit = final
                    return finish(StopReason.ANSWERED)
                if final:
                    return finish(StopReason.ROUND_LIMIT)
                next_role, message = Role.RESPONDER, body
            else:
                answer = take_turn(Role.RESPONDER, message)
                record(Role.RESPONDER, answer, answer.text, None)
                answers += 1
                message = reply_for_asker(answer.text, final=answers == cap, max_rounds=cap)
                next_role = Role.ASKER

    cancel_event = threading.Event()
    # Ctrl-C can land anywhere: in a session, a preflight, or a caller's
    # callback while it prints a reply. Every case keeps the completed turns.
    try:
        return relay()
    except _StopError as stop:
        return finish(stop.reason, stop.error)
    except KeyboardInterrupt:
        where = f"during turn {in_progress}" if in_progress else f"after {_turns(len(turns))}"
        if in_progress is not None:
            emit(
                EventType.TURN_FINISHED,
                turn=in_progress,
                role=(
                    sides[Role.ASKER] if in_progress % 2 else sides[Role.RESPONDER]
                ).participant.role.value,
                status="cancelled",
                error="interrupted",
            )
        return finish(StopReason.INTERRUPTED, f"interrupted {where}")


def result_from_journal(events: Sequence[ConversationEvent]) -> ConversationResult:
    """Rebuild a consultation's result from its journal, as of its last event.

    A run that never recorded its end is reported as interrupted.
    """
    from crossby.services.journal import EventType

    started = next((e for e in events if e.type is EventType.CONVERSATION_STARTED), None)
    if started is None:
        raise ConverseError("The journal has no conversation.started event.")
    data = started.data
    sides: dict[Role, Participant] = {}
    folders: dict[Role, Path] = {}
    for item in data.get("participants", []):
        role = Role(item["role"])
        effort = item.get("effort")
        sides[role] = Participant(
            tool=AIToolID(item["tool"]),
            role=role,
            model=item.get("model"),
            effort=EffortLevel(effort) if effort else None,
        )
        folders[role] = Path(item["folder"])
    if set(sides) != {Role.ASKER, Role.RESPONDER}:
        raise ConverseError("The journal does not name both sides of the consultation.")
    max_rounds = int(data.get("max_rounds", DEFAULT_MAX_ROUNDS))
    turns: list[ConverseTurn] = []
    sessions: dict[Role, str | None] = {Role.ASKER: None, Role.RESPONDER: None}
    stop_reason = StopReason.INTERRUPTED
    error: str | None = "the run ended without recording its outcome"
    answered_at_limit = False
    finished_at = events[-1].ts if events else started.ts

    def moment(value: object, fallback: datetime) -> datetime:
        return datetime.fromisoformat(value) if isinstance(value, str) else fallback

    for event in events:
        if event.type is EventType.CONVERSATION_CONTINUED:
            max_rounds = int(event.data.get("max_rounds", max_rounds))
            stop_reason = StopReason.INTERRUPTED
            error = "the run ended without recording its outcome"
        elif event.type is EventType.MESSAGE_COMMITTED:
            role = Role(event.data["role"])
            usage = event.data.get("usage")
            turns.append(
                ConverseTurn(
                    number=int(event.data["turn"]),
                    role=role,
                    tool=AIToolID(event.data["tool"]),
                    text=str(event.data["text"]),
                    next=Next(event.data["next"]) if event.data.get("next") else None,
                    duration_seconds=float(event.data.get("duration_seconds", 0.0)),
                    denials=tuple(event.data.get("denials", ())),
                    usage=TokenUsage.model_validate(usage) if usage else None,
                )
            )
        elif event.type is EventType.TURN_FINISHED:
            session = event.data.get("session_id")
            if session:
                sessions[Role(event.data["role"])] = session
        elif event.type is EventType.CONVERSATION_FINISHED:
            stop_reason = StopReason(event.data["stop_reason"])
            error = event.data.get("error")
            answered_at_limit = bool(event.data.get("answered_at_limit"))
            finished_at = moment(event.data.get("finished_at"), event.ts)
            for role_name, session in (event.data.get("sessions") or {}).items():
                if session:
                    sessions[Role(role_name)] = session
    if turns and turns[-1].is_final_answer and stop_reason is not StopReason.ANSWERED:
        # The final answer was committed but the run ended (a crash or Ctrl-C)
        # before recording its outcome: it is answered, never continuable.
        stop_reason, error = StopReason.ANSWERED, None
    return ConversationResult(
        goal=str(data["goal"]),
        asker=sides[Role.ASKER],
        responder=sides[Role.RESPONDER],
        max_rounds=max_rounds,
        started_at=moment(data.get("started_at"), started.ts),
        finished_at=finished_at,
        stop_reason=stop_reason,
        turns=turns,
        error=error,
        answered_at_limit=answered_at_limit,
        sessions=sessions,
        folders=folders,
    )


# ---------------------------------------------------------------------------
# Transcript, summary, and files
# ---------------------------------------------------------------------------


def _turns(count: int) -> str:
    return f"{count} turn" if count == 1 else f"{count} turns"


def _questions(count: int) -> str:
    return f"{count} question" if count == 1 else f"{count} questions"


def _asked(result: ConversationResult) -> int:
    """Questions the asker sent; its final answer is not one."""
    return sum(1 for turn in result.turns if turn.role is Role.ASKER and not turn.is_final_answer)


def _where(result: ConversationResult) -> str:
    """Where a consultation that did not answer stood when it stopped."""
    asked = _asked(result)
    if asked > result.rounds:
        return f"while the responder was answering question {asked}"
    if asked == 0:
        return "before the asker's first question"
    return f"after {_questions(asked)}, while the asker was writing"


def describe_stop(result: ConversationResult) -> str:
    """How the consultation ended, in one sentence, for people and ``--json``'s ``outcome``."""
    if result.stop_reason is StopReason.ANSWERED:
        asked = _asked(result)
        after = f"after {_questions(asked)}" if asked else "without asking a question"
        when = " at the question limit" if result.answered_at_limit else ""
        return f"answered: the asker gave its final answer {after}{when}"
    if result.stop_reason is StopReason.ROUND_LIMIT:
        return (
            f"question limit reached ({_questions(result.max_rounds)}) "
            "and the asker gave no final answer"
        )
    if result.stop_reason is StopReason.INTERRUPTED:
        # The error only restates where it stopped.
        return f"interrupted by the user {_where(result)}"
    detail = f": {result.error}" if result.error else ""
    if result.stop_reason is StopReason.TIMEOUT:
        return f"stopped on the time limit {_where(result)}{detail}"
    return f"a session failed {_where(result)}{detail}"


def render_transcript(result: ConversationResult) -> str:
    """Render a finished consultation as markdown."""
    return _render_transcript(
        goal=result.goal,
        asker=result.asker,
        responder=result.responder,
        max_rounds=result.max_rounds,
        started_at=result.started_at,
        finished_at=result.finished_at,
        turns=result.turns,
        outcome=describe_stop(result),
        folders=result.folders,
        sessions=result.sessions,
    )


def render_partial_transcript(
    *,
    goal: str,
    asker: Participant,
    responder: Participant,
    max_rounds: int,
    started_at: datetime,
    turns: Sequence[ConverseTurn],
    folders: Mapping[Role, Path] | None = None,
) -> str:
    """Render a consultation that is still running, so completed turns survive a crash."""
    return _render_transcript(
        goal=goal,
        asker=asker,
        responder=responder,
        max_rounds=max_rounds,
        started_at=started_at,
        finished_at=None,
        turns=turns,
        outcome=f"in progress: {_turns(len(turns))} completed",
        folders=folders or {},
        sessions={},
    )


def _fenced(text: str) -> str:
    """``text`` verbatim in a code fence no line of it can close."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _side_line(
    label: str, participant: Participant, folder: Path | None, *, read_only: bool
) -> str:
    settings = participant.settings
    facts = [participant.tool.value + (f" ({settings})" if settings else "")]
    if read_only:
        facts.append("read-only")
    if folder is not None:
        facts.append(f"in `{folder}`")
    return f"- {label}: {', '.join(facts)}"


def _render_transcript(
    *,
    goal: str,
    asker: Participant,
    responder: Participant,
    max_rounds: int,
    started_at: datetime,
    finished_at: datetime | None,
    turns: Sequence[ConverseTurn],
    outcome: str,
    folders: Mapping[Role, Path],
    sessions: Mapping[Role, str | None],
) -> str:
    """The transcript: run facts, the goal, then every message as written.

    The goal sits in a code fence and each turn under a numbered "Turn"
    heading after a rule, so headings inside the goal or a message never read
    as the transcript's own structure.
    """
    lines = [
        "# Consultation",
        "",
        _side_line("Asker", asker, folders.get(Role.ASKER), read_only=False),
        _side_line("Responder", responder, folders.get(Role.RESPONDER), read_only=True),
        f"- Started: {started_at.isoformat(timespec='seconds')}",
    ]
    if finished_at is not None:
        lines.append(f"- Finished: {finished_at.isoformat(timespec='seconds')}")
    lines += [f"- Rounds allowed: {max_rounds}", f"- Outcome: {outcome}"]
    known = [
        f"{side.tool.value} `{sessions[side.role]}`"
        for side in (asker, responder)
        if sessions.get(side.role)
    ]
    if known:
        lines.append(f"- Sessions: {', '.join(known)}")
    lines += ["", "## Goal (sent only to the asker)", "", _fenced(goal), ""]
    for turn in turns:
        if turn.is_final_answer:
            heading = f"## Turn {turn.number} · final answer ({turn.tool.value})"
        elif turn.role is Role.ASKER:
            heading = f"## Turn {turn.number} · asker ({turn.tool.value}) → responder"
        else:
            heading = f"## Turn {turn.number} · responder ({turn.tool.value}) → asker"
        body = turn.text or "(empty message)"
        lines += ["---", "", heading, "", body, "", f"*{turn_footer(turn)}*", ""]
    return "\n".join(lines).rstrip() + "\n"


def turn_footer(turn: ConverseTurn) -> str:
    """The one-line run facts shown under a turn: marker, time, denials."""
    parts = [f"NEXT: {turn.next.value}"] if turn.next is not None else []
    parts.append(f"{turn.duration_seconds:.0f}s")
    parts += [f"denied: {denial}" for denial in turn.denials]
    return " · ".join(parts)


def to_handoff_transcript(result: ConversationResult, path: Path) -> ConversationTranscript:
    """Express the consultation in the handoff model so the summarizer can read it."""
    from crossby.handoff.models import ConversationTranscript, ConversationTurn, SessionRef

    def label(turn: ConverseTurn) -> str:
        if turn.is_final_answer:
            return f"asker ({turn.tool.value}), final answer"
        return f"{turn.role.value} ({turn.tool.value})"

    return ConversationTranscript(
        session_ref=SessionRef(
            tool_id=result.asker.tool,
            session_id=path.parent.name,
            path=path,
            started_at=result.started_at,
        ),
        turns=[
            ConversationTurn(
                role="user" if turn.role is Role.ASKER else "assistant",
                content=f"{label(turn)}, turn {turn.number}:\n{turn.text}",
            )
            for turn in result.turns
        ],
    )


def build_summary_prompt(result: ConversationResult, *, transcript_path: Path) -> str:
    """Assemble the summary prompt: the instructions, then the whole transcript.

    Nothing is dropped or clipped; a transcript too large for the summarizer's
    model fails in that tool, which reports it.
    """
    from crossby.handoff.summarizer import render_turn

    header = (
        _load(SUMMARY_PROMPT_FILE)
        .format(
            asker=result.asker.tool.value,
            responder=result.responder.tool.value,
            goal=result.goal,
            stop_reason=describe_stop(result),
        )
        .strip()
    )
    lines = [header, "", "--- Transcript ---\n"]
    for turn in to_handoff_transcript(result, transcript_path).turns:
        lines += [render_turn(turn), ""]
    return "\n".join(lines)


def summarize_conversation(
    result: ConversationResult,
    *,
    transcript_path: Path,
    tool: AIToolID,
    working_dir: Path,
    model: str | None = None,
    effort: EffortLevel | None = None,
    timeout_seconds: float = DEFAULT_SUMMARY_TIMEOUT_SECONDS,
    runner: TurnRunner | None = None,
    allow_project_hooks: bool = False,
) -> str:
    """Summarize a finished consultation in one managed read-only session.

    The summary runs through the same boundary as the turns, so it gets the
    runtime's deadline, cancellation, process-group cleanup and each tool's
    read-only profile, plus the recursion guard. Failures raise
    :class:`ConverseSummaryError`; the transcript is already on disk.
    """
    from crossby.ai_tools.headless import HeadlessSessionError

    prompt = build_summary_prompt(result, transcript_path=transcript_path)
    request = _request(
        None,
        prompt=prompt,
        working_dir=working_dir,
        timeout_seconds=timeout_seconds,
        model=model,
        effort=effort,
        read_only=True,
        allow_project_hooks=allow_project_hooks,
    )
    runner = runner or HeadlessTurnRunner()
    try:
        session = runner.run(tool, request, threading.Event())
    except HeadlessSessionError as exc:
        raise ConverseSummaryError(str(exc)) from None
    if session.status is not HeadlessTerminalStatus.SUCCEEDED:
        raise ConverseSummaryError(
            f"{tool.value} summary session ended with status {_status(session)}"
        )
    text = (session.final_text or "").strip()
    if not text:
        raise ConverseSummaryError(f"{tool.value} returned an empty summary")
    return text


def render_summary(result: ConversationResult, body: str) -> str:
    return (
        "# Consultation summary\n\n"
        f"**Goal:** {result.goal}\n\n"
        f"- Asker: {result.asker.tool.value}\n"
        f"- Responder: {result.responder.tool.value}\n"
        f"- Outcome: {describe_stop(result)}\n\n"
        f"{body.strip()}\n"
    )


def conversation_dir(project_root: Path, started_at: datetime) -> Path:
    stamp = started_at.strftime("%Y%m%dT%H%M%S%f")
    return project_root / ".crossby" / "conversations" / stamp
