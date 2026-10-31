"""Goal-driven conversations between two headless AI coding agents.

``crossby converse`` relays messages between two managed headless sessions
until both agents say the goal is met, a turn or time limit is reached, or a
session fails. ``crossby ask`` is the single-turn primitive underneath.

Unattended managed sessions cannot resume a native session on any adapter
today, so every turn is a fresh unattended session whose prompt replays the
goal and the conversation so far (bounded by a character budget). The relayed
transcript is therefore the single source of truth, which keeps the loop
uniform across every headless-capable tool and stateless between runs.
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
    from crossby.ai_tools.base import AbstractAITool
    from crossby.handoff.models import ConversationTranscript

DEPTH_ENV = "CROSSBY_CONVERSE_DEPTH"
# Agents started by ``ask``/``converse`` may not start another one: the depth
# counter is inherited by child processes, so this stops agents from consulting
# each other in an unbounded chain.
MAX_DEPTH = 1

DEFAULT_MAX_TURNS = 6
DEFAULT_TIMEOUT_SECONDS = 1800.0
DEFAULT_TURN_TIMEOUT_SECONDS = 600.0
DEFAULT_HISTORY_CHARS = 80_000
DEFAULT_SUMMARY_TIMEOUT_SECONDS = 300
PREFLIGHT_TIMEOUT_SECONDS = 20.0
# A turn needs a realistic minimum to start a native CLI and answer; with less
# than this left on the overall deadline the conversation stops as timed out.
MIN_TURN_SECONDS = 5.0

TURN_PROMPT_FILE = "converse_turn.md"
SUMMARY_PROMPT_FILE = "converse_summary.md"


class ConverseError(RuntimeError):
    """A conversation request is invalid before any agent is started."""


class ConverseDepthError(ConverseError):
    """``ask``/``converse`` was invoked from inside another conversation."""


class AgentStatus(StrEnum):
    """The status line an agent ends its reply with."""

    CONTINUE = "CONTINUE"
    DONE = "DONE"
    MISSING = "MISSING"


class StopReason(StrEnum):
    """Why a conversation ended."""

    AGREED = "agreed"
    TURN_LIMIT = "turn_limit"
    TIMEOUT = "timeout"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


EXIT_CODES: dict[StopReason, int] = {
    StopReason.AGREED: 0,
    StopReason.FAILED: 1,
    StopReason.TURN_LIMIT: 2,
    StopReason.TIMEOUT: 2,
    StopReason.INTERRUPTED: 130,
}


@dataclass(frozen=True)
class Participant:
    """One side of a conversation."""

    tool: AIToolID
    label: str
    model: str | None = None


@dataclass(frozen=True)
class ConverseTurn:
    """One relayed reply."""

    number: int
    speaker: str
    tool: AIToolID
    text: str
    status: AgentStatus
    duration_seconds: float


@dataclass
class ConversationResult:
    """The full outcome of a conversation, including partial progress."""

    goal: str
    participants: tuple[Participant, Participant]
    max_turns: int
    started_at: datetime
    finished_at: datetime
    stop_reason: StopReason
    turns: list[ConverseTurn] = field(default_factory=list)
    error: str | None = None

    @property
    def agreed(self) -> bool:
        return self.stop_reason is StopReason.AGREED

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
# Participants and prompts
# ---------------------------------------------------------------------------


def make_participants(
    tool_a: AIToolID,
    tool_b: AIToolID,
    *,
    model_a: str | None = None,
    model_b: str | None = None,
) -> tuple[Participant, Participant]:
    """Build both sides, disambiguating labels when the same tool talks to itself."""
    if tool_a == tool_b:
        labels = (f"{tool_a.value}-1", f"{tool_b.value}-2")
    else:
        labels = (tool_a.value, tool_b.value)
    return (
        Participant(tool=tool_a, label=labels[0], model=model_a),
        Participant(tool=tool_b, label=labels[1], model=model_b),
    )


_STATUS_LINE = re.compile(
    r"^[\s*_`>#-]*status\s*:\s*[*_`]*\s*(continue|done)\b[\s*_`.!]*$",
    re.IGNORECASE,
)


def parse_status(text: str) -> tuple[str, AgentStatus]:
    """Split an agent reply into ``(body, status)``.

    The status must be the last non-empty line. Markdown decoration such as
    ``**STATUS: DONE**`` or a code span is tolerated. A reply without one is
    treated as wanting to continue (``MISSING``), never as agreement.
    """
    lines = text.rstrip().splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip():
            match = _STATUS_LINE.match(lines[index])
            if match is None:
                return text.strip(), AgentStatus.MISSING
            body = "\n".join(lines[:index]).strip()
            return body, AgentStatus(match.group(1).upper())
    return "", AgentStatus.MISSING


def _clip(text: str, limit: int) -> str:
    """Keep the head and tail of an oversized reply."""
    if len(text) <= limit:
        return text
    marker = "\n[… {n} characters omitted …]\n"
    keep = max(limit - len(marker.format(n=len(text))), 2)
    head = keep // 2
    tail = keep - head
    omitted = len(text) - head - tail
    return text[:head] + marker.format(n=omitted) + text[-tail:]


def render_history(turns: Sequence[ConverseTurn], *, peer: str, budget_chars: int) -> str:
    """Render the conversation so far, keeping the most recent turns that fit."""
    if not turns:
        return (
            f"You speak first. Open the conversation with {peer}: give your initial "
            "assessment of the goal and what you checked."
        )
    blocks = [
        f"### {turn.speaker} (turn {turn.number}, STATUS: {turn.status.value})\n"
        f"{turn.text or '(empty reply)'}"
        for turn in turns
    ]
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
    parts = ["Conversation so far:", ""]
    if omitted:
        parts += [f"({omitted} earlier turn(s) omitted to fit the size limit.)", ""]
    parts.append("\n\n".join(kept))
    parts += ["", f"It is your turn. Reply to {peer}."]
    return "\n".join(parts)


def load_turn_template() -> str:
    from crossby.handoff.prompts import load_bundled_prompt

    return load_bundled_prompt(TURN_PROMPT_FILE)


def build_turn_prompt(
    *,
    template: str,
    goal: str,
    speaker: Participant,
    peer: Participant,
    turn: int,
    max_turns: int,
    history: Sequence[ConverseTurn],
    history_chars: int,
) -> str:
    rendered_history = render_history(history, peer=peer.label, budget_chars=history_chars)
    return template.format(
        speaker=speaker.label,
        peer=peer.label,
        goal=goal,
        turn=turn,
        max_turns=max_turns,
        history=rendered_history,
    ).strip()


def _request(
    participant: Participant | None,
    *,
    prompt: str,
    working_dir: Path,
    timeout_seconds: float,
    model: str | None = None,
) -> HeadlessSessionRequest:
    return HeadlessSessionRequest(
        prompt=prompt,
        working_dir=working_dir,
        model=participant.model if participant is not None else model,
        interaction_mode=HeadlessInteractionMode.UNATTENDED,
        timeout_seconds=timeout_seconds,
    )


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
        None, prompt=prompt, working_dir=working_dir, timeout_seconds=timeout_seconds, model=model
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


def run_conversation(
    goal: str,
    participants: tuple[Participant, Participant],
    *,
    working_dir: Path,
    max_turns: int = DEFAULT_MAX_TURNS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    turn_timeout_seconds: float = DEFAULT_TURN_TIMEOUT_SECONDS,
    history_chars: int = DEFAULT_HISTORY_CHARS,
    runner: TurnRunner | None = None,
    template: str | None = None,
    on_turn_start: Callable[[int, Participant], None] | None = None,
    on_turn: Callable[[ConverseTurn], None] | None = None,
    clock: Callable[[], float] = monotonic,
) -> ConversationResult:
    """Relay between two agents until they agree or a limit stops them.

    Agreement means two consecutive replies (one from each agent) ending in
    ``STATUS: DONE``. Turn and time limits always stop the loop, and every
    outcome, including failure and Ctrl-C, returns the turns completed so far.
    """
    from crossby.ai_tools.headless import HeadlessSessionError

    if not goal.strip():
        raise ConverseError("The goal is empty.")
    if max_turns < 2:
        raise ConverseError("--max-turns must be at least 2 so both agents can reply.")
    if timeout_seconds <= 0 or turn_timeout_seconds <= 0:
        raise ConverseError("Timeouts must be positive.")
    if history_chars < 1000:
        raise ConverseError("The history budget must be at least 1000 characters.")

    runner = runner or HeadlessTurnRunner()
    template = template if template is not None else load_turn_template()
    started_at = datetime.now(tz=UTC)
    deadline = clock() + timeout_seconds
    turns: list[ConverseTurn] = []

    def finish(reason: StopReason, error: str | None = None) -> ConversationResult:
        return ConversationResult(
            goal=goal,
            participants=participants,
            max_turns=max_turns,
            started_at=started_at,
            finished_at=datetime.now(tz=UTC),
            stop_reason=reason,
            turns=turns,
            error=error,
        )

    def out_of_time(when: str) -> ConversationResult:
        return finish(
            StopReason.TIMEOUT,
            f"conversation time limit of {timeout_seconds:g}s reached {when}",
        )

    # Fail fast, before the first agent spends time, if either side cannot run.
    # Preflights spend the same overall budget as the turns.
    for participant in dict.fromkeys(participants):
        remaining = deadline - clock()
        if remaining < MIN_TURN_SECONDS:
            return out_of_time("during preflight")
        probe = _request(
            participant,
            prompt="preflight",
            working_dir=working_dir,
            timeout_seconds=min(turn_timeout_seconds, remaining),
        )
        try:
            runner.preflight(participant.tool, probe)
        except HeadlessSessionError as exc:
            if clock() >= deadline:
                return out_of_time("during preflight")
            return finish(StopReason.FAILED, f"{participant.label} cannot run headless: {exc}")

    cancel_event = threading.Event()
    with child_depth():
        for number in range(1, max_turns + 1):
            speaker = participants[(number - 1) % 2]
            peer = participants[number % 2]
            remaining = deadline - clock()
            if remaining < MIN_TURN_SECONDS:
                return out_of_time(f"before turn {number}")
            bounded_by_deadline = remaining <= turn_timeout_seconds
            prompt = build_turn_prompt(
                template=template,
                goal=goal,
                speaker=speaker,
                peer=peer,
                turn=number,
                max_turns=max_turns,
                history=turns,
                history_chars=history_chars,
            )
            request = _request(
                speaker,
                prompt=prompt,
                working_dir=working_dir,
                timeout_seconds=min(turn_timeout_seconds, remaining),
            )
            if on_turn_start is not None:
                on_turn_start(number, speaker)
            turn_started = clock()
            try:
                result = runner.run(speaker.tool, request, cancel_event)
            except KeyboardInterrupt:
                return finish(StopReason.INTERRUPTED, f"interrupted during turn {number}")
            except HeadlessSessionError as exc:
                return finish(StopReason.FAILED, f"{speaker.label} turn {number}: {exc}")
            duration = clock() - turn_started

            if result.status is HeadlessTerminalStatus.TIMED_OUT:
                if bounded_by_deadline:
                    return out_of_time(f"during turn {number}")
                return finish(
                    StopReason.FAILED,
                    f"{speaker.label} turn {number} timed out after {turn_timeout_seconds:g}s",
                )
            if result.status is not HeadlessTerminalStatus.SUCCEEDED:
                return finish(
                    StopReason.FAILED,
                    f"{speaker.label} turn {number} ended with status {result.status.value}",
                )
            text = (result.final_text or "").strip()
            if not text:
                return finish(StopReason.FAILED, f"{speaker.label} turn {number} returned no text")

            body, status = parse_status(text)
            turn = ConverseTurn(
                number=number,
                speaker=speaker.label,
                tool=speaker.tool,
                text=body,
                status=status,
                duration_seconds=duration,
            )
            turns.append(turn)
            if on_turn is not None:
                on_turn(turn)
            if (
                status is AgentStatus.DONE
                and len(turns) >= 2
                and turns[-2].status is AgentStatus.DONE
            ):
                return finish(StopReason.AGREED)
    return finish(StopReason.TURN_LIMIT)


# ---------------------------------------------------------------------------
# Transcript, summary, and files
# ---------------------------------------------------------------------------


def _turns(count: int) -> str:
    return f"{count} turn" if count == 1 else f"{count} turns"


def describe_stop(result: ConversationResult) -> str:
    done = _turns(len(result.turns))
    if result.stop_reason is StopReason.AGREED:
        return f"agreed: both agents marked the goal as met after {done}"
    if result.stop_reason is StopReason.TURN_LIMIT:
        return f"turn limit reached ({_turns(result.max_turns)}) without agreement"
    detail = f": {result.error}" if result.error else ""
    if result.stop_reason is StopReason.TIMEOUT:
        return f"stopped on the time limit after {done} without agreement{detail}"
    if result.stop_reason is StopReason.INTERRUPTED:
        return f"interrupted by the user after {done}{detail}"
    return f"a session failed after {done}{detail}"


def render_transcript(result: ConversationResult) -> str:
    """Render the whole conversation as markdown."""
    a, b = result.participants
    lines = [
        "# Conversation",
        "",
        f"**Goal:** {result.goal}",
        "",
        f"- Participants: {a.label}, {b.label}",
        f"- Started: {result.started_at.isoformat(timespec='seconds')}",
        f"- Finished: {result.finished_at.isoformat(timespec='seconds')}",
        f"- Turns: {len(result.turns)} of at most {result.max_turns}",
        f"- Outcome: {describe_stop(result)}",
        "",
    ]
    for turn in result.turns:
        lines += [
            f"## Turn {turn.number}: {turn.speaker}",
            "",
            turn.text or "(empty reply)",
            "",
            f"*STATUS: {turn.status.value} · {turn.duration_seconds:.0f}s*",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def to_handoff_transcript(result: ConversationResult, path: Path) -> ConversationTranscript:
    """Express the conversation in the handoff model so the summarizer can read it."""
    from crossby.handoff.models import ConversationTranscript, ConversationTurn, SessionRef

    return ConversationTranscript(
        session_ref=SessionRef(
            tool_id=result.participants[0].tool,
            session_id=path.parent.name,
            path=path,
            started_at=result.started_at,
        ),
        turns=[
            ConversationTurn(
                role="assistant",
                content=f"{turn.speaker} (turn {turn.number}, STATUS: {turn.status.value}):\n"
                f"{turn.text}",
            )
            for turn in result.turns
        ],
    )


def summarize_conversation(
    result: ConversationResult,
    *,
    transcript_path: Path,
    summarizer_tool: AbstractAITool,
    working_dir: Path,
    timeout_seconds: int = DEFAULT_SUMMARY_TIMEOUT_SECONDS,
    on_truncate: Callable[[int, int], None] | None = None,
) -> str:
    """Summarize a finished conversation with a neutral summarizer pass.

    Raises the handoff summarizer's errors unchanged so the caller can report
    them; the transcript is already on disk when this runs.
    """
    from crossby.handoff.prompts import load_bundled_prompt
    from crossby.handoff.summarizer import HandoffSummarizer

    a, b = result.participants
    template = load_bundled_prompt(SUMMARY_PROMPT_FILE).format(
        participants=f"{a.label} and {b.label}",
        goal=result.goal,
        stop_reason=describe_stop(result),
    )
    summarizer = HandoffSummarizer(
        summarizer_tool,
        prompt_template=template.strip(),
        timeout_seconds=timeout_seconds,
        working_dir=working_dir,
    )
    # The summarizer is an agent too: it must not be able to start another
    # consultation, so it runs under the same recursion guard as the turns.
    with child_depth():
        raw = summarizer.summarize_raw(
            to_handoff_transcript(result, transcript_path),
            source_tool=a.tool,
            target_tool=b.tool,
            prompt_source="converse-summary",
            on_truncate=on_truncate,
        )
    return raw.body


def render_summary(result: ConversationResult, body: str) -> str:
    a, b = result.participants
    return (
        "# Conversation summary\n\n"
        f"**Goal:** {result.goal}\n\n"
        f"- Participants: {a.label}, {b.label}\n"
        f"- Outcome: {describe_stop(result)}\n\n"
        f"{body.strip()}\n"
    )


def conversation_dir(project_root: Path, started_at: datetime) -> Path:
    stamp = started_at.strftime("%Y%m%dT%H%M%S%f")
    return project_root / ".crossby" / "conversations" / stamp
