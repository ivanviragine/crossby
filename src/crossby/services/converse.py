"""Goal-driven consultations between two headless AI coding agents.

``crossby converse ASKER RESPONDER --goal ...`` lets one agent consult another.
The asker holds the goal: it decides what to ask, judges each answer, asks
follow-up questions, and decides when it has enough. The responder never sees
the goal; it only answers the asker's messages. ``crossby ask`` is the
single-question primitive underneath.

Each side keeps its own native session. Where the adapter supports unattended
resume (verified for Claude Code), every turn after the first resumes that
side's session and carries only the other agent's latest message. A tool
without verified resume falls back to replay: each turn is a fresh session
whose prompt replays that side's earlier exchanges, bounded by a character
budget and the transport's size limit. The transcript records which mode each
side used.
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
DEFAULT_HISTORY_CHARS = 80_000
MIN_HISTORY_CHARS = 1000
DEFAULT_SUMMARY_TIMEOUT_SECONDS = 300.0
DEFAULT_SUMMARY_TOKEN_BUDGET = 32_000
PREFLIGHT_TIMEOUT_SECONDS = 20.0
# A turn needs a realistic minimum to start a native CLI and answer; with less
# than this left on the overall deadline the conversation stops as timed out.
MIN_TURN_SECONDS = 5.0

ASKER_PROMPT_FILE = "converse_asker.md"
SUMMARY_PROMPT_FILE = "converse_summary.md"
# Character ceiling for the assembled summary prompt, before the summarizer
# transport's own size check.
SUMMARY_PROMPT_CHARS = 100_000

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


class SessionMode(StrEnum):
    """How a side keeps its context between turns."""

    RESUMED = "resumed"
    REPLAYED = "replayed"


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
    mode: SessionMode

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
    modes: dict[Role, SessionMode] = field(default_factory=dict)

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


PromptFits = Callable[[AIToolID, str], bool]
SupportsResume = Callable[[AIToolID], bool]


def native_prompt_fits(tool: AIToolID, prompt: str) -> bool:
    """Whether ``tool``'s unattended transport can receive ``prompt``."""
    from crossby.ai_tools.base import AbstractAITool

    return AbstractAITool.get(tool).headless_prompt_fits(prompt)


def native_supports_resume(tool: AIToolID) -> bool:
    """Whether ``tool`` declares verified unattended session resume."""
    from crossby.ai_tools.base import AbstractAITool

    headless = AbstractAITool.get(tool).capabilities().headless
    try:
        return headless.profile_for(HeadlessInteractionMode.UNATTENDED).supports_resume
    except ValueError:
        return False


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


def _clip(text: str, limit: int) -> str:
    """Keep the head and tail of an oversized message."""
    if len(text) <= limit:
        return text
    marker = "\n[… {n} characters omitted …]\n"
    keep = max(limit - len(marker.format(n=len(text))), 2)
    head = keep // 2
    tail = keep - head
    omitted = len(text) - head - tail
    return text[:head] + marker.format(n=omitted) + text[-tail:]


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


def _render_exchanges(blocks: Sequence[str], *, budget_chars: int, heading: str) -> tuple[str, int]:
    """Keep the newest blocks that fit ``budget_chars``; return text and omitted count."""
    kept: list[str] = []
    used = 0
    for block in reversed(blocks):
        if kept and used + len(block) > budget_chars:
            break
        if not kept:
            block = _clip(block, budget_chars)
        kept.append(block)
        used += len(block) + 2
    kept.reverse()
    omitted = len(blocks) - len(kept)
    parts = [heading, ""]
    if omitted:
        parts += [f"({omitted} earlier message(s) omitted to fit the size limit.)", ""]
    parts.append("\n\n".join(kept))
    return "\n".join(parts), omitted


def replay_prompt(
    role: Role,
    *,
    opening: str,
    turns: Sequence[ConverseTurn],
    latest: str,
    responder: Participant,
    budget_chars: int,
) -> str:
    """Rebuild a fresh session's context for a side whose tool cannot resume.

    The prompt is the earlier exchanges from that side's point of view, then
    the message it is answering now (after the asker's goal, for the asker).
    """
    asker_side = role is Role.ASKER
    if not turns:
        return f"{opening}\n\n{latest}" if asker_side else latest
    blocks: list[str] = []
    for turn in turns:
        if turn.role is Role.ASKER:
            who = "You" if asker_side else "Question"
        else:
            who = responder.tool.value if asker_side else "Your answer"
        blocks.append(f"### {who}\n{turn.text}")
    history, _omitted = _render_exchanges(
        blocks, budget_chars=budget_chars, heading="Earlier in this conversation:"
    )
    if asker_side:
        return f"{opening}\n\n{history}\n\n### {responder.tool.value}\n{latest}"
    return f"{history}\n\n### Question\n{latest}"


def _request(
    participant: Participant | None,
    *,
    prompt: str,
    working_dir: Path,
    timeout_seconds: float,
    model: str | None = None,
    resume_id: str | None = None,
) -> HeadlessSessionRequest:
    return HeadlessSessionRequest(
        prompt=prompt,
        working_dir=working_dir,
        model=participant.model if participant is not None else model,
        interaction_mode=HeadlessInteractionMode.UNATTENDED,
        timeout_seconds=timeout_seconds,
        resume_id=resume_id,
    )


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
    """Ask one agent one question in an unattended headless session."""
    from crossby.ai_tools.headless import HeadlessSessionError

    if not prompt.strip():
        raise ConverseError("The question is empty.")
    runner = runner or HeadlessTurnRunner()
    request = _request(
        None,
        prompt=prompt.strip(),
        working_dir=working_dir,
        timeout_seconds=timeout_seconds,
        model=model,
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
    mode: SessionMode
    session_id: str | None = None
    opening: str = ""


def run_conversation(
    goal: str,
    participants: tuple[Participant, Participant],
    *,
    working_dir: Path,
    responder_dir: Path | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    turn_timeout_seconds: float = DEFAULT_TURN_TIMEOUT_SECONDS,
    history_chars: int = DEFAULT_HISTORY_CHARS,
    runner: TurnRunner | None = None,
    on_turn_start: Callable[[int, Participant], None] | None = None,
    on_turn: Callable[[ConverseTurn], None] | None = None,
    clock: Callable[[], float] = monotonic,
    prompt_fits: PromptFits | None = None,
    supports_resume: SupportsResume | None = None,
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

    if not goal.strip():
        raise ConverseError("The goal is empty.")
    if max_rounds < 1:
        raise ConverseError("--max-rounds must be at least 1.")
    if timeout_seconds <= 0 or turn_timeout_seconds <= 0:
        raise ConverseError("Timeouts must be positive.")
    if history_chars < MIN_HISTORY_CHARS:
        raise ConverseError(f"The history budget must be at least {MIN_HISTORY_CHARS} characters.")
    asker, responder = participants
    if asker.role is not Role.ASKER or responder.role is not Role.RESPONDER:
        raise ConverseError("Participants must be (asker, responder).")

    active_runner: TurnRunner = runner or HeadlessTurnRunner()
    folders = {Role.ASKER: working_dir, Role.RESPONDER: responder_dir or working_dir}
    fits: PromptFits = prompt_fits or native_prompt_fits
    resumes: SupportsResume = supports_resume or native_supports_resume
    started_at = datetime.now(tz=UTC)
    deadline = clock() + timeout_seconds
    turns: list[ConverseTurn] = []
    sides = {
        Role.ASKER: _Side(
            asker,
            SessionMode.RESUMED if resumes(asker.tool) else SessionMode.REPLAYED,
            opening=asker_opening(goal=goal, responder=responder, max_rounds=max_rounds),
        ),
        Role.RESPONDER: _Side(
            responder,
            SessionMode.RESUMED if resumes(responder.tool) else SessionMode.REPLAYED,
        ),
    }
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
            modes={role: side.mode for role, side in sides.items()},
        )

    def out_of_time(when: str) -> _StopError:
        return _StopError(
            StopReason.TIMEOUT, f"conversation time limit of {timeout_seconds:g}s reached {when}"
        )

    def build_prompt(side: _Side, message: str, budget: int) -> str:
        """The prompt for this side's next turn, given the message it answers.

        The message itself is never shortened; ``budget`` bounds only the
        replayed history of a side whose tool cannot resume.
        """
        if side.session_id is not None:  # resumed: only the new message
            return message
        if not turns:  # the very first turn: the asker's opening
            return side.opening
        if side.mode is SessionMode.RESUMED:  # first turn of a resumable responder
            return message
        # Replay: the latest turn is the message being answered, sent separately.
        return replay_prompt(
            side.participant.role,
            opening=side.opening,
            turns=turns[:-1],
            latest=message,
            responder=responder,
            budget_chars=budget,
        )

    def take_turn(role: Role, message: str) -> tuple[str, float]:
        """Run one turn for ``role``; return its reply text and duration."""
        nonlocal in_progress
        side = sides[role]
        number = len(turns) + 1
        remaining = deadline - clock()
        if remaining < MIN_TURN_SECONDS:
            raise out_of_time(f"before turn {number}")
        bounded_by_deadline = remaining <= turn_timeout_seconds
        # Only replayed history shrinks to fit the transport; a message that
        # still does not fit stops the consultation instead of being cut.
        budget = history_chars
        while True:
            prompt = build_prompt(side, message, budget)
            if fits(side.participant.tool, prompt):
                break
            if budget <= MIN_HISTORY_CHARS:
                raise _StopError(
                    StopReason.FAILED,
                    f"{side.participant.label} turn {number}: the message "
                    f"({len(message):,} characters) is too large for "
                    f"{side.participant.tool.value}'s command-line prompt limit",
                )
            budget = max(budget // 2, MIN_HISTORY_CHARS)
        request = _request(
            side.participant,
            prompt=prompt,
            working_dir=folders[role],
            timeout_seconds=min(turn_timeout_seconds, remaining),
            resume_id=side.session_id,
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
        if result.status is not HeadlessTerminalStatus.SUCCEEDED:
            raise _StopError(
                StopReason.FAILED, f"{label} turn {number} ended with status {result.status.value}"
            )
        text = (result.final_text or "").strip()
        if not text:
            raise _StopError(StopReason.FAILED, f"{label} turn {number} returned no text")
        if side.mode is SessionMode.RESUMED:
            handle = _session_handle(result)
            if side.session_id is None and handle is None:
                # The tool resumes in principle but gave no handle back; keep
                # going by replaying, and say so in the transcript.
                side.mode = SessionMode.REPLAYED
            elif side.session_id is None:
                side.session_id = handle
        return text, clock() - turn_started

    def record(role: Role, text: str, next_: Next | None, duration: float) -> ConverseTurn:
        nonlocal in_progress
        turn = ConverseTurn(
            number=len(turns) + 1,
            role=role,
            tool=sides[role].participant.tool,
            text=text,
            next=next_,
            duration_seconds=duration,
            mode=sides[role].mode,
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
            raw, duration = take_turn(Role.ASKER, message_for_asker)
            body, next_ = parse_next(raw)
            if not body:
                # A bare marker is neither an answer nor a message to send.
                what = "final answer" if next_ is Next.DONE else "message"
                raise _StopError(
                    StopReason.FAILED,
                    f"{asker.label} turn {len(turns) + 1} sent only the NEXT marker, "
                    f"with no {what}",
                )
            record(Role.ASKER, body, next_, duration)
            if next_ is Next.DONE:
                answered_at_limit = final
                return finish(StopReason.ANSWERED)
            if final:
                return finish(StopReason.ROUND_LIMIT)
            answer, duration = take_turn(Role.RESPONDER, body)
            record(Role.RESPONDER, answer, None, duration)
            message_for_asker = reply_for_asker(
                answer, final=round_number == max_rounds, max_rounds=max_rounds
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
        marker = f"NEXT: {turn.next.value} · " if turn.next is not None else ""
        lines += [
            heading,
            "",
            turn.text or "(empty message)",
            "",
            f"*{marker}{turn.duration_seconds:.0f}s · session {turn.mode.value}*",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


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


def build_summary_prompt(
    result: ConversationResult,
    *,
    transcript_path: Path,
    token_budget: int = DEFAULT_SUMMARY_TOKEN_BUDGET,
    max_chars: int = SUMMARY_PROMPT_CHARS,
    fits: Callable[[str], bool] | None = None,
    on_truncate: Callable[[int, int], None] | None = None,
) -> str:
    """Assemble the summary prompt with the handoff formatting and truncation.

    The handoff token budget keeps the most recent turns. Further old turns
    are then dropped (and a single oversized turn clipped) until the prompt is
    within ``max_chars`` and ``fits`` accepts it; ``fits`` is the summarizer
    transport's own size check, so byte-limited argv tools always receive it.
    """
    from crossby.handoff.summarizer import render_turn
    from crossby.handoff.truncate import truncate_transcript

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
    transcript = to_handoff_transcript(result, transcript_path)
    prepared = truncate_transcript(transcript, token_budget)
    turns = list(prepared.turns)
    truncated = prepared.truncated

    def assemble() -> str:
        lines = [header, ""]
        if truncated:
            lines.append(
                "Note: earlier turns were dropped to fit the size limit. Rely on the "
                "last turns for where the consultation ended.\n"
            )
        lines.append("--- Transcript ---\n")
        for turn in turns:
            lines += [render_turn(turn), ""]
        return "\n".join(lines)

    def too_big(text: str) -> bool:
        return len(text) > max_chars or (fits is not None and not fits(text))

    prompt = assemble()
    while too_big(prompt) and len(turns) > 1:
        turns.pop(0)
        truncated = True
        prompt = assemble()
    if too_big(prompt) and turns:
        only = turns[0]
        room = min(len(only.content), max_chars)
        while too_big(prompt) and room > MIN_HISTORY_CHARS:
            room = max(room // 2, MIN_HISTORY_CHARS)
            turns = [only.model_copy(update={"content": _clip(only.content, room)})]
            truncated = True
            prompt = assemble()
    if truncated and on_truncate is not None:
        on_truncate(len(transcript.turns), len(turns))
    return prompt


def summarize_conversation(
    result: ConversationResult,
    *,
    transcript_path: Path,
    tool: AIToolID,
    working_dir: Path,
    model: str | None = None,
    timeout_seconds: float = DEFAULT_SUMMARY_TIMEOUT_SECONDS,
    runner: TurnRunner | None = None,
    on_truncate: Callable[[int, int], None] | None = None,
    prompt_fits: PromptFits = native_prompt_fits,
) -> str:
    """Summarize a finished consultation in one managed unattended session.

    The summary runs through the same boundary as the turns, so it gets the
    runtime's deadline, cancellation, process-group cleanup and each tool's
    unattended permission policy, plus the recursion guard. Failures raise
    :class:`ConverseSummaryError`; the transcript is already on disk.
    """
    from crossby.ai_tools.headless import HeadlessSessionError

    prompt = build_summary_prompt(
        result,
        transcript_path=transcript_path,
        fits=lambda text: prompt_fits(tool, text),
        on_truncate=on_truncate,
    )
    request = _request(
        None, prompt=prompt, working_dir=working_dir, timeout_seconds=timeout_seconds, model=model
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
