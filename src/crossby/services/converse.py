"""Goal-driven consultations between two headless AI coding agents.

``crossby converse ASKER RESPONDER --goal ...`` lets one agent consult another.
The asker holds the goal: it decides what to ask, judges each answer, asks
follow-up questions, and decides when it has enough. The responder never sees
the goal; it only answers the asker's messages. ``crossby ask`` is the
single-question primitive underneath.

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
import queue
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Protocol

from crossby.models.ai import (
    AIToolID,
    HeadlessInteractionMode,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
)

if TYPE_CHECKING:
    from crossby.handoff.models import ConversationTranscript

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

    @property
    def label(self) -> str:
        return f"{self.role.value} ({self.tool.value})"


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


@dataclass(frozen=True)
class AskResult:
    """The outcome of a single ``ask`` turn."""

    tool: AIToolID
    status: HeadlessTerminalStatus | None
    text: str
    duration_seconds: float
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is HeadlessTerminalStatus.SUCCEEDED and self.error is None


class TurnRunner(Protocol):
    """Runs managed headless sessions; injectable so tests need no real CLIs."""

    def preflight(self, tool: AIToolID, request: HeadlessSessionRequest) -> None: ...

    def run(
        self,
        tool: AIToolID,
        request: HeadlessSessionRequest,
        cancel_event: threading.Event,
    ) -> HeadlessSessionResult: ...


class HeadlessTurnRunner:
    """Default runner backed by each adapter's managed headless session."""

    def preflight(self, tool: AIToolID, request: HeadlessSessionRequest) -> None:
        from crossby.ai_tools import preflight_headless_session

        preflight_headless_session(tool, request, timeout_seconds=PREFLIGHT_TIMEOUT_SECONDS)

    def run(
        self,
        tool: AIToolID,
        request: HeadlessSessionRequest,
        cancel_event: threading.Event,
    ) -> HeadlessSessionResult:
        """Run one session in a worker so Ctrl-C cancels it cleanly.

        ``KeyboardInterrupt`` only reaches the main thread. Waiting here and
        setting ``cancel_event`` on interrupt lets the managed runtime tear down
        the native process group before the interrupt propagates.
        """
        from crossby.ai_tools.base import AbstractAITool

        adapter = AbstractAITool.get(tool)
        box: queue.Queue[HeadlessSessionResult | BaseException] = queue.Queue(1)

        def work() -> None:
            try:
                box.put(adapter.run_headless_session(request, cancel_event=cancel_event))
            except BaseException as exc:
                box.put(exc)

        worker = threading.Thread(target=work, daemon=True, name=f"crossby-converse-{tool}")
        worker.start()
        try:
            while worker.is_alive():
                worker.join(0.1)
        except KeyboardInterrupt:
            cancel_event.set()
            worker.join(30)
            raise
        value = box.get_nowait()
        if isinstance(value, BaseException):
            raise value
        return value


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


@contextmanager
def child_depth() -> Iterator[None]:
    """Mark processes started inside this block as one level deeper."""
    previous = os.environ.get(DEPTH_ENV)
    os.environ[DEPTH_ENV] = str(current_depth() + 1)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(DEPTH_ENV, None)
        else:
            os.environ[DEPTH_ENV] = previous


# ---------------------------------------------------------------------------
# Capabilities, prompts and parsing
# ---------------------------------------------------------------------------


def make_participants(
    asker: AIToolID,
    responder: AIToolID,
    *,
    asker_model: str | None = None,
    responder_model: str | None = None,
) -> tuple[Participant, Participant]:
    return (
        Participant(tool=asker, role=Role.ASKER, model=asker_model),
        Participant(tool=responder, role=Role.RESPONDER, model=responder_model),
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
    resume_id: str | None = None,
    read_only: bool = False,
) -> HeadlessSessionRequest:
    return HeadlessSessionRequest(
        prompt=prompt,
        working_dir=working_dir,
        model=participant.model if participant is not None else model,
        interaction_mode=HeadlessInteractionMode.UNATTENDED,
        timeout_seconds=timeout_seconds,
        resume_id=resume_id,
        read_only=read_only,
    )


def _read_only(participant: Participant) -> bool:
    """A responder never changes files: its sessions run read-only.

    The asker is the user's own agent in the user's project, so it keeps the
    folder's normal unattended policy.
    """
    return participant.role is Role.RESPONDER


def _session_handle(result: HeadlessSessionResult) -> str | None:
    return result.session_id or result.thread_id or result.conversation_id


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def ask_agent(
    tool: AIToolID,
    prompt: str,
    *,
    working_dir: Path,
    model: str | None = None,
    timeout_seconds: float = DEFAULT_TURN_TIMEOUT_SECONDS,
    runner: TurnRunner | None = None,
    clock: Callable[[], float] = monotonic,
) -> AskResult:
    """Ask one agent one question in a read-only unattended headless session.

    The asked agent is a responder: it answers from its folder and never
    changes it. A tool without a verified read-only profile is refused.
    """
    from crossby.ai_tools.headless import HeadlessSessionError

    check_depth()
    if not prompt.strip():
        raise ConverseError("The question is empty.")
    runner = runner or HeadlessTurnRunner()
    request = _request(
        None,
        prompt=prompt.strip(),
        working_dir=working_dir,
        timeout_seconds=timeout_seconds,
        model=model,
        read_only=True,
    )
    started = clock()
    try:
        with child_depth():
            result = runner.run(tool, request, threading.Event())
    except HeadlessSessionError as exc:
        return AskResult(tool, None, "", clock() - started, error=str(exc))
    text = (result.final_text or "").strip()
    error = None
    if result.status is not HeadlessTerminalStatus.SUCCEEDED:
        error = f"{tool.value} session ended with status {result.status.value}"
    elif not text:
        error = f"{tool.value} returned no text"
    return AskResult(tool, result.status, text, clock() - started, error=error)


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
) -> ConversationResult:
    """Let the asker consult the responder until it gives a final answer.

    The asker runs in ``working_dir`` and the responder in ``responder_dir``
    (default: the same folder), each with that folder's own files and setup.

    The asker holds the goal and ends every reply with ``NEXT: ASK`` (send it
    to the responder) or ``NEXT: DONE`` (its final answer). After the last
    allowed round the asker is told to answer now; a limit, a failure or Ctrl-C
    always returns the turns completed so far.
    """
    from crossby.ai_tools.headless import HeadlessSessionError

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

    active_runner: TurnRunner = runner or HeadlessTurnRunner()
    folders = {Role.ASKER: working_dir, Role.RESPONDER: responder_dir or working_dir}
    started_at = datetime.now(tz=UTC)
    deadline = clock() + timeout_seconds
    turns: list[ConverseTurn] = []
    sides = {Role.ASKER: _Side(asker), Role.RESPONDER: _Side(responder)}
    opening = asker_opening(goal=goal, responder=responder, max_rounds=max_rounds)
    in_progress: int | None = None
    answered_at_limit = False

    def finish(reason: StopReason, error: str | None = None) -> ConversationResult:
        return ConversationResult(
            goal=goal,
            asker=asker,
            responder=responder,
            max_rounds=max_rounds,
            started_at=started_at,
            finished_at=datetime.now(tz=UTC),
            stop_reason=reason,
            turns=turns,
            error=error,
            answered_at_limit=answered_at_limit,
            sessions={role: side.session_id for role, side in sides.items()},
        )

    def out_of_time(when: str) -> _StopError:
        return _StopError(
            StopReason.TIMEOUT, f"conversation time limit of {timeout_seconds:g}s reached {when}"
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
        )
        in_progress = number
        if on_turn_start is not None:
            on_turn_start(number, side.participant)
        turn_started = clock()
        label = side.participant.label
        try:
            result = active_runner.run(side.participant.tool, request, cancel_event)
        except HeadlessSessionError as exc:
            # The runtime re-probes the version before it starts and raises
            # (rather than returning TIMED_OUT) when the deadline runs out there.
            if clock() >= deadline:
                raise out_of_time(f"during turn {number}") from None
            raise _StopError(StopReason.FAILED, f"{label} turn {number}: {exc}") from None
        if result.status is HeadlessTerminalStatus.TIMED_OUT:
            if bounded_by_deadline:
                raise out_of_time(f"during turn {number}")
            raise _StopError(
                StopReason.FAILED,
                f"{label} turn {number} timed out after {turn_timeout_seconds:g}s",
            )
        # A tool that denied an action may end its turn without an answer
        # (Antigravity CLI does); name the denials so the failure explains itself.
        denied = f" ({'; '.join(result.denials)})" if result.denials else ""
        if result.status is not HeadlessTerminalStatus.SUCCEEDED:
            raise _StopError(
                StopReason.FAILED,
                f"{label} turn {number} ended with status {result.status.value}{denied}",
            )
        text = (result.final_text or "").strip()
        if not text:
            raise _StopError(StopReason.FAILED, f"{label} turn {number} returned no text{denied}")
        if side.session_id is None:
            handle = _session_handle(result)
            if handle is None:
                # Its next turn could only start over without the conversation.
                raise _StopError(
                    StopReason.FAILED,
                    f"{label} turn {number}: {side.participant.tool.value} reported no "
                    "session to resume, so the conversation cannot continue",
                )
            side.session_id = handle
        return _Reply(text, clock() - turn_started, tuple(result.denials))

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
        )
        turns.append(turn)
        in_progress = None
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
            )
            try:
                active_runner.preflight(participant.tool, probe)
            except HeadlessSessionError as exc:
                if clock() >= deadline:
                    raise out_of_time("during preflight") from None
                raise _StopError(
                    StopReason.FAILED, f"{participant.label} cannot run headless: {exc}"
                ) from None

    def relay() -> ConversationResult:
        nonlocal answered_at_limit
        preflight()
        message_for_asker = ""  # the asker's first turn uses its opening
        for round_number in range(1, max_rounds + 2):
            final = round_number > max_rounds
            asked = take_turn(Role.ASKER, message_for_asker)
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
            answer = take_turn(Role.RESPONDER, body)
            record(Role.RESPONDER, answer, answer.text, None)
            message_for_asker = reply_for_asker(
                answer.text, final=round_number == max_rounds, max_rounds=max_rounds
            )
        return finish(StopReason.ROUND_LIMIT)  # pragma: no cover - loop always returns

    cancel_event = threading.Event()
    # Ctrl-C can land anywhere: in a session, a preflight, or a caller's
    # callback while it prints a reply. Every case keeps the completed turns.
    try:
        with child_depth():
            return relay()
    except _StopError as stop:
        return finish(stop.reason, stop.error)
    except KeyboardInterrupt:
        where = f"during turn {in_progress}" if in_progress else f"after {_turns(len(turns))}"
        return finish(StopReason.INTERRUPTED, f"interrupted {where}")


# ---------------------------------------------------------------------------
# Transcript, summary, and files
# ---------------------------------------------------------------------------


def _turns(count: int) -> str:
    return f"{count} turn" if count == 1 else f"{count} turns"


def _rounds(count: int) -> str:
    return f"{count} round" if count == 1 else f"{count} rounds"


def describe_stop(result: ConversationResult) -> str:
    rounds = _rounds(result.rounds)
    if result.stop_reason is StopReason.ANSWERED:
        when = " at the round limit" if result.answered_at_limit else ""
        return f"answered: the asker gave its final answer after {rounds}{when}"
    if result.stop_reason is StopReason.ROUND_LIMIT:
        return (
            f"round limit reached ({_rounds(result.max_rounds)}) and the asker gave no final answer"
        )
    detail = f": {result.error}" if result.error else ""
    if result.stop_reason is StopReason.TIMEOUT:
        return f"stopped on the time limit after {rounds}{detail}"
    if result.stop_reason is StopReason.INTERRUPTED:
        return f"interrupted by the user after {rounds}{detail}"
    return f"a session failed after {rounds}{detail}"


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
    )


def render_partial_transcript(
    *,
    goal: str,
    asker: Participant,
    responder: Participant,
    max_rounds: int,
    started_at: datetime,
    turns: Sequence[ConverseTurn],
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
    )


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
) -> str:
    lines = [
        "# Consultation",
        "",
        f"**Goal (sent only to the asker):** {goal}",
        "",
        f"- Asker: {asker.tool.value}",
        f"- Responder: {responder.tool.value}",
        f"- Started: {started_at.isoformat(timespec='seconds')}",
    ]
    if finished_at is not None:
        lines.append(f"- Finished: {finished_at.isoformat(timespec='seconds')}")
    lines += [f"- Rounds allowed: {max_rounds}", f"- Outcome: {outcome}", ""]
    for turn in turns:
        if turn.is_final_answer:
            heading = f"## {turn.number}. Final answer ({turn.tool.value})"
        elif turn.role is Role.ASKER:
            heading = f"## {turn.number}. asker ({turn.tool.value}) → responder"
        else:
            heading = f"## {turn.number}. responder ({turn.tool.value}) → asker"
        lines += [heading, "", turn.text or "(empty message)", "", f"*{turn_footer(turn)}*", ""]
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
    timeout_seconds: float = DEFAULT_SUMMARY_TIMEOUT_SECONDS,
    runner: TurnRunner | None = None,
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
        read_only=True,
    )
    runner = runner or HeadlessTurnRunner()
    try:
        with child_depth():
            session = runner.run(tool, request, threading.Event())
    except HeadlessSessionError as exc:
        raise ConverseSummaryError(str(exc)) from None
    if session.status is not HeadlessTerminalStatus.SUCCEEDED:
        raise ConverseSummaryError(
            f"{tool.value} summary session ended with status {session.status.value}"
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
