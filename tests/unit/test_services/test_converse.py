"""Tests for the ask/converse engine, driven by a fake headless runner."""

from __future__ import annotations

import os
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.headless import HeadlessUnsupportedError
from crossby.models.ai import (
    AIToolID,
    HeadlessEvent,
    HeadlessEventKind,
    HeadlessInteractionMode,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
)
from crossby.services import converse
from crossby.services.converse import (
    DEPTH_ENV,
    ConversationResult,
    ConverseDepthError,
    ConverseError,
    ConverseTurn,
    Next,
    Role,
    StopReason,
)

Reply = str | HeadlessTerminalStatus | BaseException
GOAL = "Check whether the PFO diagnosis is sound"


def _result(
    tool: AIToolID,
    status: HeadlessTerminalStatus,
    text: str | None,
    session_id: str | None = None,
    denials: tuple[str, ...] = (),
) -> HeadlessSessionResult:
    return HeadlessSessionResult(
        tool=tool,
        version="1.0.0",
        status=status,
        events=(
            HeadlessEvent(sequence=1, kind=HeadlessEventKind.TERMINAL, terminal_status=status),
        ),
        final_text=text,
        session_id=session_id,
        duration_seconds=1.0,
        denials=denials,
    )


def _unsupported(tool: AIToolID, message: str) -> HeadlessUnsupportedError:
    return HeadlessUnsupportedError(
        message, tool_id=tool, capability=AbstractAITool.get(tool).capabilities().headless
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeRunner:
    """Replays scripted replies; new sessions get a fresh ID, resumed ones keep it."""

    def __init__(
        self,
        replies: list[Reply],
        *,
        clock: FakeClock | None = None,
        seconds_per_turn: float = 1.0,
        seconds_per_preflight: float = 0.0,
        preflight_error: BaseException | None = None,
        issue_session_ids: bool = True,
        denials: dict[int, tuple[str, ...]] | None = None,
    ) -> None:
        self.replies = list(replies)
        self.clock = clock
        self.seconds_per_turn = seconds_per_turn
        self.seconds_per_preflight = seconds_per_preflight
        self.preflight_error = preflight_error
        self.issue_session_ids = issue_session_ids
        # Denials the tool reports, by 1-based call number.
        self.denials = denials or {}
        self.preflights: list[AIToolID] = []
        self.preflight_requests: list[HeadlessSessionRequest] = []
        self.preflight_timeouts: list[float] = []
        self.calls: list[tuple[AIToolID, HeadlessSessionRequest]] = []
        self.depths: list[str | None] = []
        # Runtime event kinds every run reports to its event handler.
        self.progress: list[HeadlessEventKind] = []
        self.parent_depths: list[str | None] = []
        self.sessions_created = 0

    def preflight(self, tool: AIToolID, request: HeadlessSessionRequest) -> None:
        self.preflights.append(tool)
        self.preflight_requests.append(request)
        self.preflight_timeouts.append(request.timeout_seconds)
        if self.clock is not None:
            self.clock.now += self.seconds_per_preflight
        if self.preflight_error is not None:
            raise self.preflight_error

    def run(
        self,
        tool: AIToolID,
        request: HeadlessSessionRequest,
        cancel_event: threading.Event,
        on_event: Any = None,
    ) -> HeadlessSessionResult:
        self.calls.append((tool, request))
        if on_event is not None:
            for sequence, kind in enumerate(self.progress, start=1):
                on_event(
                    HeadlessEvent(sequence=sequence, kind=kind, elapsed_seconds=0.5 * sequence)
                )
        self.depths.append(request.environment.get(DEPTH_ENV))
        self.parent_depths.append(os.environ.get(DEPTH_ENV))
        if self.clock is not None:
            self.clock.now += self.seconds_per_turn
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        session = request.resume_id
        if session is None and self.issue_session_ids:
            self.sessions_created += 1
            session = f"session-{self.sessions_created}"
        denials = self.denials.get(len(self.calls), ())
        if isinstance(item, HeadlessTerminalStatus):
            return _result(tool, item, None, denials=denials)
        return _result(tool, HeadlessTerminalStatus.SUCCEEDED, item, session, denials=denials)


PARTICIPANTS = converse.make_participants(AIToolID.CLAUDE, AIToolID.CODEX)
ASKER, RESPONDER = PARTICIPANTS


def _run(runner: FakeRunner, tmp_path: Path, **kwargs: object) -> ConversationResult:
    return converse.run_conversation(
        GOAL,
        PARTICIPANTS,
        working_dir=tmp_path,
        runner=runner,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture(autouse=True)
def _no_inherited_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEPTH_ENV, raising=False)


# ---------------------------------------------------------------------------
# parse_next
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What is a PFO?\nNEXT: ASK", Next.ASK),
        ("Verdict: sound.\n\nNEXT: DONE\n\n", Next.DONE),
        ("ok\n**NEXT: DONE**", Next.DONE),
        ("ok\n`next: ask`", Next.ASK),
        ("ok\nNext: Done.", Next.DONE),
        ("NEXT: DONE is what I'd say later\nbut not yet", Next.MISSING),
        ("no marker here", Next.MISSING),
        ("ok\nNEXT: MAYBE", Next.MISSING),
        ("ok\nSTATUS: DONE", Next.MISSING),
    ],
)
def test_parse_next(text: str, expected: Next) -> None:
    assert converse.parse_next(text)[1] is expected


def test_parse_next_strips_the_marker_line() -> None:
    body, next_ = converse.parse_next("Line one.\nLine two.\n\nNEXT: DONE")
    assert body == "Line one.\nLine two."
    assert next_ is Next.DONE


# ---------------------------------------------------------------------------
# The consultation loop (resumed sessions)
# ---------------------------------------------------------------------------

ANSWERING = [
    "What is a patent foramen ovale?\nNEXT: ASK",
    "A flap-like opening between the atria.",
    "How can it cause a paradoxical embolism?\nNEXT: ASK",
    "A venous clot crosses right-to-left and bypasses the lungs.",
    "Verdict: the responder understands the mechanism.\nNEXT: DONE",
]


def test_asker_consults_until_it_gives_a_final_answer(tmp_path: Path) -> None:
    runner = FakeRunner(ANSWERING)
    result = _run(runner, tmp_path)

    assert result.stop_reason is StopReason.ANSWERED
    assert result.answered
    assert result.exit_code == 0
    assert result.final_answer == "Verdict: the responder understands the mechanism."
    assert result.rounds == 2
    assert not result.answered_at_limit
    assert [t.role for t in result.turns] == [
        Role.ASKER,
        Role.RESPONDER,
        Role.ASKER,
        Role.RESPONDER,
        Role.ASKER,
    ]
    assert [t.next for t in result.turns] == [Next.ASK, None, Next.ASK, None, Next.DONE]
    assert [tool for tool, _req in runner.calls] == [
        AIToolID.CLAUDE,
        AIToolID.CODEX,
        AIToolID.CLAUDE,
        AIToolID.CODEX,
        AIToolID.CLAUDE,
    ]
    assert result.sessions == {Role.ASKER: "session-1", Role.RESPONDER: "session-2"}


def test_only_the_asker_sees_the_goal(tmp_path: Path) -> None:
    runner = FakeRunner(ANSWERING)
    _run(runner, tmp_path)
    asker_prompts = [req.prompt for tool, req in runner.calls if tool is AIToolID.CLAUDE]
    responder_prompts = [req.prompt for tool, req in runner.calls if tool is AIToolID.CODEX]
    assert GOAL in asker_prompts[0]
    assert all(GOAL not in prompt for prompt in responder_prompts)


def test_messages_are_relayed_as_written(tmp_path: Path) -> None:
    runner = FakeRunner(ANSWERING)
    _run(runner, tmp_path, max_rounds=4)
    requests = [req for _tool, req in runner.calls]

    # The asker gets its goal and the bare mechanics of the exchange.
    assert requests[0].prompt == converse.asker_opening(
        goal=GOAL, responder=RESPONDER, max_rounds=4
    )
    assert requests[0].prompt.startswith(f"Goal: {GOAL}\n")
    assert "NEXT: ASK" in requests[0].prompt and "NEXT: DONE" in requests[0].prompt
    assert "at most 4 messages" in requests[0].prompt
    assert requests[0].resume_id is None

    # Everything else travels exactly as the other agent wrote it.
    assert requests[1].prompt == "What is a patent foramen ovale?"
    assert requests[1].resume_id is None
    assert requests[2].prompt == "A flap-like opening between the atria."
    assert requests[2].resume_id == "session-1"
    assert requests[3].prompt == "How can it cause a paradoxical embolism?"
    assert requests[3].resume_id == "session-2"
    assert requests[4].prompt == "A venous clot crosses right-to-left and bypasses the lungs."
    assert requests[4].resume_id == "session-1"


def test_asker_opening_has_no_advice() -> None:
    opening = converse.asker_opening(goal=GOAL, responder=RESPONDER, max_rounds=5)
    for advice in ("judge", "verify", "modify", "files", "follow-up", "challenge"):
        assert advice not in opening.lower()


def test_responder_runs_in_its_own_folder(tmp_path: Path) -> None:
    elsewhere = tmp_path / "responder-project"
    elsewhere.mkdir()
    runner = FakeRunner(ANSWERING)
    _run(runner, tmp_path, responder_dir=elsewhere)
    folders = {(tool, req.working_dir) for tool, req in runner.calls}
    assert folders == {(AIToolID.CLAUDE, tmp_path), (AIToolID.CODEX, elsewhere)}


def test_same_tool_on_both_sides_keeps_separate_sessions(tmp_path: Path) -> None:
    runner = FakeRunner(ANSWERING)
    participants = converse.make_participants(AIToolID.CLAUDE, AIToolID.CLAUDE)
    converse.run_conversation(GOAL, participants, working_dir=tmp_path, runner=runner)
    resumes = [req.resume_id for _tool, req in runner.calls]
    assert resumes == [None, None, "session-1", "session-2", "session-1"]


def test_participant_models_reach_their_requests(tmp_path: Path) -> None:
    participants = converse.make_participants(
        AIToolID.CLAUDE, AIToolID.CODEX, asker_model="opus-y", responder_model="gpt-x"
    )
    runner = FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"])
    converse.run_conversation(GOAL, participants, working_dir=tmp_path, runner=runner)
    assert [req.model for _tool, req in runner.calls] == ["opus-y", "gpt-x", "opus-y"]


def test_every_request_is_unattended_in_the_project(tmp_path: Path) -> None:
    runner = FakeRunner(ANSWERING)
    _run(runner, tmp_path)
    for _tool, request in runner.calls:
        assert request.interaction_mode is HeadlessInteractionMode.UNATTENDED
        assert request.working_dir == tmp_path


def test_the_responder_runs_read_only_and_the_asker_does_not(tmp_path: Path) -> None:
    runner = FakeRunner(ANSWERING)
    _run(runner, tmp_path)
    assert [request.read_only for _tool, request in runner.calls] == [
        False,
        True,
        False,
        True,
        False,
    ]
    # Preflight checks the same posture, so a tool without a verified read-only
    # profile is refused before any agent starts.
    assert [request.read_only for request in runner.preflight_requests] == [False, True]


def test_denials_are_recorded_on_their_turn_and_shown(tmp_path: Path) -> None:
    denial = "Claude Code reported 1 permission denials under --permission-prompts none"
    runner = FakeRunner(ANSWERING, denials={2: (denial,)})
    result = _run(runner, tmp_path)
    assert [turn.denials for turn in result.turns] == [(), (denial,), (), (), ()]
    transcript = converse.render_transcript(result)
    assert f"s · denied: {denial}*" in transcript


def test_an_empty_reply_after_denials_names_them(tmp_path: Path) -> None:
    denial = "Antigravity CLI denied 1 action(s) that needed a permission"
    # agy ends a turn with no answer when it denies an action; the runtime
    # reports a successful exit without final output as invalid output.
    runner = FakeRunner(
        ["q\nNEXT: ASK", HeadlessTerminalStatus.INVALID_OUTPUT], denials={2: (denial,)}
    )
    result = _run(runner, tmp_path)
    assert result.stop_reason is StopReason.FAILED
    assert result.error == (f"responder (codex) turn 2 ended with status invalid_output ({denial})")


def test_answering_on_the_first_turn_is_allowed(tmp_path: Path) -> None:
    result = _run(FakeRunner(["I already know: sound.\nNEXT: DONE"]), tmp_path)
    assert result.answered
    assert result.rounds == 0
    assert result.final_answer == "I already know: sound."


def test_missing_marker_is_sent_on_as_a_message(tmp_path: Path) -> None:
    runner = FakeRunner(["What is a PFO?", "An opening.", "Fine.\nNEXT: DONE"])
    result = _run(runner, tmp_path)
    assert result.turns[0].next is Next.MISSING
    assert runner.calls[1][1].prompt == "What is a PFO?"
    assert result.answered


@pytest.mark.parametrize(
    ("reply", "what"),
    [
        ("NEXT: DONE", "final answer"),
        ("**NEXT: ASK**", "message"),
        ("\n NEXT: done \n", "final answer"),
    ],
)
def test_marker_only_reply_is_a_failure_not_an_answer(
    tmp_path: Path, reply: str, what: str
) -> None:
    runner = FakeRunner(["q\nNEXT: ASK", "a", reply])
    result = _run(runner, tmp_path)
    assert result.stop_reason is StopReason.FAILED
    assert result.exit_code == 1
    assert result.final_answer is None
    assert result.error == f"asker (claude) turn 3 sent only the NEXT marker, with no {what}"
    assert len(result.turns) == 2
    assert len(runner.calls) == 3  # nothing empty was sent to the responder


# ---------------------------------------------------------------------------
# Round limit
# ---------------------------------------------------------------------------


def test_round_limit_asks_for_a_final_answer(tmp_path: Path) -> None:
    runner = FakeRunner(["q1\nNEXT: ASK", "a1", "q2\nNEXT: ASK", "a2", "Final: sound.\nNEXT: DONE"])
    result = _run(runner, tmp_path, max_rounds=2)

    assert result.answered
    assert result.answered_at_limit
    assert result.final_answer == "Final: sound."
    last_asker_prompt = runner.calls[4][1].prompt
    assert last_asker_prompt.startswith("a2\n\n")
    assert "You have used all 2 messages" in last_asker_prompt
    assert "Write your final answer now" in last_asker_prompt
    # The earlier reply carried no nudge.
    assert runner.calls[2][1].prompt == "a1"


def test_asker_that_never_answers_stops_at_the_round_limit(tmp_path: Path) -> None:
    runner = FakeRunner(["q1\nNEXT: ASK", "a1", "q2\nNEXT: ASK", "unused"])
    result = _run(runner, tmp_path, max_rounds=1)
    assert result.stop_reason is StopReason.ROUND_LIMIT
    assert result.exit_code == 2
    assert result.final_answer is None
    assert len(runner.calls) == 3


# ---------------------------------------------------------------------------
# Resume only: nothing is replayed
# ---------------------------------------------------------------------------


def test_every_turn_after_a_sides_first_resumes_and_sends_only_the_new_message(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(ANSWERING)
    _run(runner, tmp_path)
    requests = [req for _tool, req in runner.calls]
    # Only the asker's first turn carries the goal; nothing earlier is repeated.
    assert GOAL in requests[0].prompt
    assert all(GOAL not in req.prompt for req in requests[1:])
    assert [req.resume_id for req in requests] == [
        None,
        None,
        "session-1",
        "session-2",
        "session-1",
    ]
    assert requests[3].prompt == "How can it cause a paradoxical embolism?"


def test_preflight_validates_each_side_as_a_resumed_session(tmp_path: Path) -> None:
    """The adapter then refuses a build whose resume is unverified, before any turn."""
    runner = FakeRunner(ANSWERING)
    _run(runner, tmp_path)
    assert [req.resume_id for req in runner.preflight_requests] == [
        converse.PREFLIGHT_RESUME_ID,
        converse.PREFLIGHT_RESUME_ID,
    ]


def test_a_build_that_cannot_resume_is_refused_before_any_turn(tmp_path: Path) -> None:
    message = (
        "Resuming an unattended Codex CLI session was verified on 0.166.0; the installed "
        "build is 0.160.0. Upgrade Codex CLI to resume sessions."
    )
    runner = FakeRunner(["unused"], preflight_error=_unsupported(AIToolID.CODEX, message))
    result = _run(runner, tmp_path)
    assert result.stop_reason is StopReason.FAILED
    assert "Upgrade Codex CLI to resume sessions" in (result.error or "")
    assert runner.calls == []


def test_a_turn_that_reports_no_session_stops_the_conversation(tmp_path: Path) -> None:
    runner = FakeRunner(ANSWERING, issue_session_ids=False)
    result = _run(runner, tmp_path)
    assert result.stop_reason is StopReason.FAILED
    assert result.error == (
        "asker (claude) turn 1: claude reported no session to resume, so the "
        "conversation cannot continue"
    )
    assert len(runner.calls) == 1
    assert result.sessions == {Role.ASKER: None, Role.RESPONDER: None}


def test_messages_of_any_size_are_sent_whole(tmp_path: Path) -> None:
    question = "Q" * 2_000_000
    answer = "A" * 2_000_000
    runner = FakeRunner([f"{question}\nNEXT: ASK", answer, "done\nNEXT: DONE"])
    result = _run(runner, tmp_path)
    assert result.answered
    assert runner.calls[1][1].prompt == question
    assert runner.calls[2][1].prompt == answer


# ---------------------------------------------------------------------------
# Failures, limits and interrupts
# ---------------------------------------------------------------------------


def test_a_failed_turn_names_the_tools_own_status(tmp_path: Path) -> None:
    class NativeFailure(FakeRunner):
        def run(
            self,
            tool: AIToolID,
            request: HeadlessSessionRequest,
            cancel_event: threading.Event,
            on_event: Any = None,
        ) -> HeadlessSessionResult:
            result = super().run(tool, request, cancel_event)
            if tool is AIToolID.CODEX:
                return result.model_copy(
                    update={"status": HeadlessTerminalStatus.FAILED, "native_status": "turn.failed"}
                )
            return result

    result = _run(NativeFailure(["q\nNEXT: ASK", "a"]), tmp_path)
    assert result.error == (
        "responder (codex) turn 2 ended with status failed (native: turn.failed)"
    )


def test_preflight_failure_stops_before_any_turn(tmp_path: Path) -> None:
    runner = FakeRunner(
        ["unused"], preflight_error=_unsupported(AIToolID.CODEX, "codex is too old")
    )
    result = _run(runner, tmp_path)
    assert result.stop_reason is StopReason.FAILED
    assert result.exit_code == 1
    assert "cannot run headless: codex is too old" in (result.error or "")
    assert runner.calls == []


def test_session_error_mid_consultation_keeps_completed_turns(tmp_path: Path) -> None:
    runner = FakeRunner(["q\nNEXT: ASK", _unsupported(AIToolID.CODEX, "boom")])
    result = _run(runner, tmp_path)
    assert result.stop_reason is StopReason.FAILED
    assert len(result.turns) == 1
    assert result.error == "responder (codex) turn 2: boom"


@pytest.mark.parametrize(
    "status", [HeadlessTerminalStatus.FAILED, HeadlessTerminalStatus.INVALID_OUTPUT]
)
def test_failed_session_status_stops(tmp_path: Path, status: HeadlessTerminalStatus) -> None:
    result = _run(FakeRunner([status]), tmp_path)
    assert result.stop_reason is StopReason.FAILED
    assert result.error == f"asker (claude) turn 1 ended with status {status.value}"


def test_turn_timeout_inside_the_overall_budget_is_a_failure(tmp_path: Path) -> None:
    runner = FakeRunner([HeadlessTerminalStatus.TIMED_OUT])
    result = _run(runner, tmp_path, timeout_seconds=1000, turn_timeout_seconds=10)
    assert result.stop_reason is StopReason.FAILED
    assert result.error == "asker (claude) turn 1 timed out after 10s"


def test_overall_deadline_bounds_each_turn_and_stops_as_timeout(tmp_path: Path) -> None:
    clock = FakeClock()
    runner = FakeRunner(["q\nNEXT: ASK", "a", "unused"], clock=clock, seconds_per_turn=48)
    result = _run(runner, tmp_path, timeout_seconds=100, turn_timeout_seconds=600, clock=clock)
    assert result.stop_reason is StopReason.TIMEOUT
    assert result.exit_code == 2
    assert len(result.turns) == 2
    assert [req.timeout_seconds for _tool, req in runner.calls] == [100, 52]


def test_deadline_hit_during_a_turn_is_a_timeout(tmp_path: Path) -> None:
    runner = FakeRunner([HeadlessTerminalStatus.TIMED_OUT])
    result = _run(runner, tmp_path, timeout_seconds=50, turn_timeout_seconds=600)
    assert result.stop_reason is StopReason.TIMEOUT
    assert "during turn 1" in (result.error or "")


def test_startup_error_after_the_deadline_is_a_timeout(tmp_path: Path) -> None:
    clock = FakeClock()
    runner = FakeRunner(
        ["q\nNEXT: ASK", _unsupported(AIToolID.CODEX, "version probe ran out of time")],
        clock=clock,
        seconds_per_turn=30,
    )
    result = _run(runner, tmp_path, timeout_seconds=50, clock=clock)
    assert result.stop_reason is StopReason.TIMEOUT
    assert result.error == "conversation time limit of 50s reached during turn 2"


def test_preflights_spend_the_overall_budget(tmp_path: Path) -> None:
    clock = FakeClock()
    runner = FakeRunner(["done\nNEXT: DONE"], clock=clock, seconds_per_preflight=8)
    result = _run(runner, tmp_path, timeout_seconds=30, turn_timeout_seconds=600, clock=clock)
    assert result.answered
    assert runner.preflight_timeouts == [30, 22]
    assert [req.timeout_seconds for _tool, req in runner.calls] == [14]


def test_slow_preflight_that_exhausts_the_budget_is_a_timeout(tmp_path: Path) -> None:
    clock = FakeClock()
    runner = FakeRunner(["unused"], clock=clock, seconds_per_preflight=18)
    result = _run(runner, tmp_path, timeout_seconds=20, clock=clock)
    assert result.stop_reason is StopReason.TIMEOUT
    assert result.error == "conversation time limit of 20s reached during preflight"
    assert runner.calls == []


def test_keyboard_interrupt_in_a_session_keeps_completed_turns(tmp_path: Path) -> None:
    result = _run(FakeRunner(["q\nNEXT: ASK", KeyboardInterrupt()]), tmp_path)
    assert result.stop_reason is StopReason.INTERRUPTED
    assert result.exit_code == 130
    assert len(result.turns) == 1
    assert result.error == "interrupted during turn 2"


def test_interrupt_while_the_caller_prints_a_reply(tmp_path: Path) -> None:
    def on_turn(turn: ConverseTurn) -> None:
        if turn.number == 2:
            raise KeyboardInterrupt

    result = _run(FakeRunner(["q\nNEXT: ASK", "a", "unused"]), tmp_path, on_turn=on_turn)
    assert result.stop_reason is StopReason.INTERRUPTED
    assert [t.text for t in result.turns] == ["q", "a"]
    assert result.error == "interrupted after 2 turns"
    assert DEPTH_ENV not in os.environ


def test_interrupt_during_preflight(tmp_path: Path) -> None:
    result = _run(FakeRunner([], preflight_error=KeyboardInterrupt()), tmp_path)
    assert result.stop_reason is StopReason.INTERRUPTED
    assert result.turns == []


def test_turn_callbacks_fire_in_order(tmp_path: Path) -> None:
    events: list[str] = []
    _run(
        FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"]),
        tmp_path,
        on_turn_start=lambda n, p: events.append(f"start {n} {p.role.value}"),
        on_turn=lambda t: events.append(f"done {t.number} {t.role.value}"),
    )
    assert events == [
        "start 1 asker",
        "done 1 asker",
        "start 2 responder",
        "done 2 responder",
        "start 3 asker",
        "done 3 asker",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_rounds": 0}, "at least 1"),
        ({"timeout_seconds": 0}, "positive"),
    ],
)
def test_invalid_settings_are_rejected(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ConverseError, match=message):
        _run(FakeRunner([]), tmp_path, **kwargs)


def test_blank_goal_and_swapped_roles_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConverseError, match="goal is empty"):
        converse.run_conversation("  ", PARTICIPANTS, working_dir=tmp_path, runner=FakeRunner([]))
    with pytest.raises(ConverseError, match="asker, responder"):
        converse.run_conversation(
            GOAL, (RESPONDER, ASKER), working_dir=tmp_path, runner=FakeRunner([])
        )


# ---------------------------------------------------------------------------
# Message size: nothing is budgeted or cut
# ---------------------------------------------------------------------------


def test_a_long_message_with_a_requirement_in_the_middle_arrives_whole(tmp_path: Path) -> None:
    # The reviewer's case: a 90,062-character question with a requirement in
    # the middle used to be cut to the 80,000-character history budget.
    question = "A" * 45_000 + " REQUIREMENT-IN-THE-MIDDLE " + "B" * 45_035
    answer = "C" * 45_000 + " DETAIL-IN-THE-MIDDLE " + "D" * 45_000
    assert len(question) == 90_062
    runner = FakeRunner([question + "\nNEXT: ASK", answer, "done\nNEXT: DONE"])
    result = _run(runner, tmp_path)
    assert result.answered
    assert runner.calls[1][1].prompt == question
    assert runner.calls[2][1].prompt == answer
    assert all("characters omitted" not in req.prompt for _tool, req in runner.calls)


# ---------------------------------------------------------------------------
# Recursion guard
# ---------------------------------------------------------------------------


def test_agents_run_one_level_deeper_without_changing_the_parent(tmp_path: Path) -> None:
    runner = FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"])
    _run(runner, tmp_path)
    # The depth travels in each session request, never in this process's
    # environment, so concurrent consultations cannot see or undo each other's.
    assert runner.depths == ["1", "1", "1"]
    assert runner.parent_depths == [None, None, None]
    assert DEPTH_ENV not in os.environ


def test_the_depth_counts_up_from_the_callers_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(converse, "MAX_DEPTH", 3)
    monkeypatch.setenv(DEPTH_ENV, "2")
    runner = FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"])
    _run(runner, tmp_path)
    assert runner.depths == ["3", "3", "3"]
    assert os.environ[DEPTH_ENV] == "2"


def test_nested_consultation_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    converse.check_depth()
    monkeypatch.setenv(DEPTH_ENV, "1")
    with pytest.raises(ConverseDepthError, match="Nested consultations are disabled"):
        converse.check_depth()


def test_unparseable_depth_is_treated_as_top_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEPTH_ENV, "nope")
    converse.check_depth()


def test_library_calls_enforce_the_recursion_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The guard lives in the engine, not only in the CLI, so a product that
    # imports the library is covered too (``ask`` is covered in test_ask.py).
    monkeypatch.setenv(DEPTH_ENV, "1")
    runner = FakeRunner(["unused"])
    with pytest.raises(ConverseDepthError):
        _run(runner, tmp_path)
    assert runner.calls == []


# ---------------------------------------------------------------------------
# Transcript and summary
# ---------------------------------------------------------------------------


def _turn(number: int, role: Role, text: str, next_: Next | None) -> ConverseTurn:
    tool = AIToolID.CLAUDE if role is Role.ASKER else AIToolID.CODEX
    return ConverseTurn(number, role, tool, text, next_, 1.0)


def _finished(reason: StopReason, *, error: str | None = None) -> ConversationResult:
    return ConversationResult(
        goal=GOAL,
        asker=ASKER,
        responder=RESPONDER,
        max_rounds=5,
        started_at=datetime(2026, 10, 31, 12, 0, tzinfo=UTC),
        finished_at=datetime(2026, 10, 31, 12, 5, tzinfo=UTC),
        stop_reason=reason,
        turns=[
            _turn(1, Role.ASKER, "What is a PFO?", Next.ASK),
            _turn(2, Role.RESPONDER, "An opening between the atria.", None),
            _turn(3, Role.ASKER, "It knows the anatomy.", Next.DONE),
        ],
        error=error,
        sessions={Role.ASKER: "session-1", Role.RESPONDER: "session-2"},
    )


def test_render_transcript() -> None:
    text = converse.render_transcript(_finished(StopReason.ANSWERED))
    assert f"## Goal (sent only to the asker)\n\n```text\n{GOAL}\n```" in text
    assert "- Asker: claude" in text
    assert "- Responder: codex" in text
    assert "- Outcome: answered: the asker gave its final answer after 1 question" in text
    assert "## Turn 1 · asker (claude) → responder\n\nWhat is a PFO?" in text
    assert "## Turn 2 · responder (codex) → asker\n\nAn opening between the atria." in text
    assert "## Turn 3 · final answer (claude)\n\nIt knows the anatomy." in text
    assert "*NEXT: ASK · 1s*" in text


def test_transcript_keeps_headings_in_the_goal_and_messages_out_of_its_structure(
    tmp_path: Path,
) -> None:
    result = _finished(StopReason.ANSWERED)
    result.goal = "# Brief\n\n## 1. Context\n\n```python\nprint('hi')\n```"
    result.turns[1] = _turn(2, Role.RESPONDER, "## 1. Decisions\n\nUse X.", None)
    result.asker = converse.Participant(AIToolID.CLAUDE, Role.ASKER, model="opus-y")
    result.folders = {Role.ASKER: tmp_path / "a", Role.RESPONDER: tmp_path / "r"}
    text = converse.render_transcript(result)

    # The goal sits in a fence longer than any backtick run inside it.
    assert f"````text\n{result.goal}\n````" in text
    # Each turn starts after a rule, under its own "Turn" heading.
    assert "---\n\n## Turn 2 · responder (codex) → asker\n\n## 1. Decisions" in text
    assert f"- Asker: claude (opus-y), in `{tmp_path / 'a'}`" in text
    assert f"- Responder: codex, read-only, in `{tmp_path / 'r'}`" in text
    assert "- Sessions: claude `session-1`, codex `session-2`" in text


def test_partial_transcript_has_no_sessions_yet() -> None:
    text = converse.render_partial_transcript(
        goal=GOAL,
        asker=ASKER,
        responder=RESPONDER,
        max_rounds=5,
        started_at=datetime(2026, 10, 31, 12, 0, tzinfo=UTC),
        turns=[],
    )
    assert "- Sessions:" not in text
    assert "- Responder: codex, read-only" in text


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (
            StopReason.ROUND_LIMIT,
            "question limit reached (5 questions) and the asker gave no final answer",
        ),
        (
            StopReason.TIMEOUT,
            "stopped on the time limit after 1 question, while the asker was writing: late",
        ),
        (
            StopReason.FAILED,
            "a session failed after 1 question, while the asker was writing: late",
        ),
        # An interruption's error only restates where it stopped.
        (
            StopReason.INTERRUPTED,
            "interrupted by the user after 1 question, while the asker was writing",
        ),
    ],
)
def test_describe_stop(reason: StopReason, expected: str) -> None:
    assert converse.describe_stop(_finished(reason, error="late")) == expected


def test_answered_at_the_limit_is_described() -> None:
    result = _finished(StopReason.ANSWERED)
    result.answered_at_limit = True
    assert converse.describe_stop(result).endswith("after 1 question at the question limit")


def test_a_stop_says_where_the_consultation_stood() -> None:
    waiting = _finished(StopReason.TIMEOUT, error="late")
    waiting.turns = waiting.turns[:1]  # the asker's question, still unanswered
    assert converse.describe_stop(waiting) == (
        "stopped on the time limit while the responder was answering question 1: late"
    )
    waiting.turns = []
    assert converse.describe_stop(waiting) == (
        "stopped on the time limit before the asker's first question: late"
    )


def test_partial_transcript_marks_the_consultation_in_progress() -> None:
    text = converse.render_partial_transcript(
        goal=GOAL,
        asker=ASKER,
        responder=RESPONDER,
        max_rounds=5,
        started_at=datetime(2026, 10, 31, 12, 0, tzinfo=UTC),
        turns=[_turn(1, Role.ASKER, "What is a PFO?", Next.ASK)],
    )
    assert "- Outcome: in progress: 1 turn completed" in text
    assert "- Finished:" not in text


def test_summary_runs_as_a_managed_unattended_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_plain_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the summary must not bypass the managed runtime")

    monkeypatch.setattr(subprocess, "run", no_plain_subprocess)
    runner = FakeRunner(["## Outcome\nSound."])
    body = converse.summarize_conversation(
        _finished(StopReason.ANSWERED),
        transcript_path=tmp_path / "run" / "transcript.md",
        tool=AIToolID.CLAUDE,
        working_dir=tmp_path,
        timeout_seconds=42,
        runner=runner,
    )
    assert body == "## Outcome\nSound."
    ((_tool, request),) = runner.calls
    assert request.interaction_mode is HeadlessInteractionMode.UNATTENDED
    assert request.read_only
    assert request.timeout_seconds == 42
    assert runner.depths == ["1"]


def test_summary_prompt_carries_roles_goal_and_transcript(tmp_path: Path) -> None:
    prompt = converse.build_summary_prompt(
        _finished(StopReason.ANSWERED), transcript_path=tmp_path / "t.md"
    )
    assert "The asker (claude) held the goal" in prompt
    assert "the responder (codex) never saw the goal" in prompt
    assert f"Goal: {GOAL}" in prompt
    assert "Do not read, modify, create, or delete files" in prompt
    assert "not independent verification" in prompt
    assert "## Points of Disagreement" in prompt
    assert "asker (claude), turn 1:\nWhat is a PFO?" in prompt
    assert "asker (claude), final answer, turn 3:\nIt knows the anatomy." in prompt


def test_summary_prompt_carries_the_whole_transcript_uncut(tmp_path: Path) -> None:
    result = _finished(StopReason.ROUND_LIMIT)
    texts = [f"turn-{n} " + "x" * 400_000 + f" end-of-turn-{n}" for n in range(1, 4)]
    result.turns = [
        _turn(n, Role.ASKER if n % 2 else Role.RESPONDER, text, None)
        for n, text in enumerate(texts, start=1)
    ]
    runner = FakeRunner(["## Outcome\nok"])
    converse.summarize_conversation(
        result,
        transcript_path=tmp_path / "t.md",
        tool=AIToolID.CURSOR,
        working_dir=tmp_path,
        runner=runner,
    )
    prompt = runner.calls[0][1].prompt
    assert all(text in prompt for text in texts)
    assert "dropped" not in prompt and "omitted" not in prompt


@pytest.mark.parametrize("reply", [HeadlessTerminalStatus.TIMED_OUT, HeadlessTerminalStatus.FAILED])
def test_summary_failures_raise_summary_errors(
    tmp_path: Path, reply: HeadlessTerminalStatus
) -> None:
    with pytest.raises(converse.ConverseSummaryError, match="summary session ended"):
        converse.summarize_conversation(
            _finished(StopReason.ANSWERED),
            transcript_path=tmp_path / "t.md",
            tool=AIToolID.CLAUDE,
            working_dir=tmp_path,
            runner=FakeRunner([reply]),
        )


def test_render_summary_and_conversation_dir(tmp_path: Path) -> None:
    result = _finished(StopReason.ANSWERED)
    text = converse.render_summary(result, "## Outcome\nDone.")
    assert text.startswith(f"# Consultation summary\n\n**Goal:** {GOAL}")
    assert "- Asker: claude" in text
    assert text.endswith("## Outcome\nDone.\n")
    assert converse.conversation_dir(tmp_path, result.started_at) == (
        tmp_path / ".crossby" / "conversations" / "20261031T120000000000"
    )
