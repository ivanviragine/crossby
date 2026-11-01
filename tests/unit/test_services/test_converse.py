"""Tests for the ask/converse engine, driven by a fake headless runner."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

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
    SessionMode,
    StopReason,
)

Reply = str | HeadlessTerminalStatus | BaseException
GOAL = "Check whether the PFO diagnosis is sound"


def _result(
    tool: AIToolID,
    status: HeadlessTerminalStatus,
    text: str | None,
    session_id: str | None = None,
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
    ) -> None:
        self.replies = list(replies)
        self.clock = clock
        self.seconds_per_turn = seconds_per_turn
        self.seconds_per_preflight = seconds_per_preflight
        self.preflight_error = preflight_error
        self.issue_session_ids = issue_session_ids
        self.preflights: list[AIToolID] = []
        self.preflight_timeouts: list[float] = []
        self.calls: list[tuple[AIToolID, HeadlessSessionRequest]] = []
        self.depths: list[str | None] = []
        self.sessions_created = 0

    def preflight(self, tool: AIToolID, request: HeadlessSessionRequest) -> None:
        self.preflights.append(tool)
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
    ) -> HeadlessSessionResult:
        self.calls.append((tool, request))
        self.depths.append(os.environ.get(DEPTH_ENV))
        if self.clock is not None:
            self.clock.now += self.seconds_per_turn
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        session = request.resume_id
        if session is None and self.issue_session_ids:
            self.sessions_created += 1
            session = f"session-{self.sessions_created}"
        if isinstance(item, HeadlessTerminalStatus):
            return _result(tool, item, None)
        return _result(tool, HeadlessTerminalStatus.SUCCEEDED, item, session)


PARTICIPANTS = converse.make_participants(AIToolID.CLAUDE, AIToolID.CODEX)
ASKER, RESPONDER = PARTICIPANTS


def _resumes(_tool: AIToolID) -> bool:
    return True


def _replays(_tool: AIToolID) -> bool:
    return False


def _run(runner: FakeRunner, tmp_path: Path, **kwargs: object) -> ConversationResult:
    kwargs.setdefault("supports_resume", _resumes)
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
    # The fake runner reports version 1.0.0; the per-version resume check has
    # its own tests, so engine tests treat every build as resumable.
    monkeypatch.setattr(converse, "native_resume_supported_on", lambda _tool, _version: True)


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
    assert result.modes == {Role.ASKER: SessionMode.RESUMED, Role.RESPONDER: SessionMode.RESUMED}


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
    converse.run_conversation(
        GOAL, participants, working_dir=tmp_path, runner=runner, supports_resume=_resumes
    )
    resumes = [req.resume_id for _tool, req in runner.calls]
    assert resumes == [None, None, "session-1", "session-2", "session-1"]


def test_participant_models_reach_their_requests(tmp_path: Path) -> None:
    participants = converse.make_participants(
        AIToolID.CLAUDE, AIToolID.CODEX, asker_model="opus-y", responder_model="gpt-x"
    )
    runner = FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"])
    converse.run_conversation(
        GOAL, participants, working_dir=tmp_path, runner=runner, supports_resume=_resumes
    )
    assert [req.model for _tool, req in runner.calls] == ["opus-y", "gpt-x", "opus-y"]


def test_every_request_is_unattended_in_the_project(tmp_path: Path) -> None:
    runner = FakeRunner(ANSWERING)
    _run(runner, tmp_path)
    for _tool, request in runner.calls:
        assert request.interaction_mode is HeadlessInteractionMode.UNATTENDED
        assert request.working_dir == tmp_path


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
# Replay fallback for tools without verified resume
# ---------------------------------------------------------------------------


def test_replay_rebuilds_each_sides_context(tmp_path: Path) -> None:
    runner = FakeRunner(ANSWERING)
    result = _run(runner, tmp_path, supports_resume=_replays)
    requests = [req for _tool, req in runner.calls]

    assert all(req.resume_id is None for req in requests)
    assert result.modes == {
        Role.ASKER: SessionMode.REPLAYED,
        Role.RESPONDER: SessionMode.REPLAYED,
    }
    # The asker's replay: its opening, earlier exchanges, then the latest reply once.
    asker_turn_2 = requests[2].prompt
    assert f"Goal: {GOAL}" in asker_turn_2
    assert asker_turn_2.endswith("### codex\nA flap-like opening between the atria.")
    assert asker_turn_2.count("A flap-like opening") == 1
    assert "### You\nWhat is a patent foramen ovale?" in asker_turn_2
    # The responder's replay: its earlier Q&A, then the new question.
    responder_turn_2 = requests[3].prompt
    assert GOAL not in responder_turn_2
    assert "### Question\nWhat is a patent foramen ovale?" in responder_turn_2
    assert "### Your answer\nA flap-like opening between the atria." in responder_turn_2
    assert responder_turn_2.startswith("Earlier in this conversation:")
    assert responder_turn_2.endswith("### Question\nHow can it cause a paradoxical embolism?")
    assert responder_turn_2.count("How can it cause") == 1


def test_resume_without_a_returned_handle_falls_back_to_replay(tmp_path: Path) -> None:
    runner = FakeRunner(ANSWERING, issue_session_ids=False)
    result = _run(runner, tmp_path)
    assert all(req.resume_id is None for _tool, req in runner.calls)
    assert result.modes[Role.ASKER] is SessionMode.REPLAYED
    assert f"Goal: {GOAL}" in runner.calls[2][1].prompt


def test_an_unverified_cli_build_falls_back_to_replay(tmp_path: Path) -> None:
    seen: list[tuple[AIToolID, str]] = []

    def resumes_on(tool: AIToolID, version: str) -> bool:
        seen.append((tool, version))
        return False

    runner = FakeRunner(ANSWERING)
    result = _run(runner, tmp_path, resume_supported_on=resumes_on)
    assert all(req.resume_id is None for _tool, req in runner.calls)
    assert result.modes == {
        Role.ASKER: SessionMode.REPLAYED,
        Role.RESPONDER: SessionMode.REPLAYED,
    }
    assert seen == [(AIToolID.CLAUDE, "1.0.0"), (AIToolID.CODEX, "1.0.0")]
    assert result.answered


def test_replay_keeps_the_newest_exchanges_within_budget(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            "q1 " + "A" * 900 + "\nNEXT: ASK",
            "a1 " + "B" * 900,
            "q2\nNEXT: ASK",
            "a2 " + "C" * 900,
            "q3\nNEXT: ASK",
            "a3",
            "done\nNEXT: DONE",
        ]
    )
    _run(runner, tmp_path, supports_resume=_replays, history_chars=1500)
    asker_turn_3 = runner.calls[4][1].prompt
    assert "A" * 900 not in asker_turn_3
    assert "earlier message(s) omitted" in asker_turn_3
    assert "C" * 900 in asker_turn_3


# ---------------------------------------------------------------------------
# Failures, limits and interrupts
# ---------------------------------------------------------------------------


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
        ({"history_chars": 10}, "1000"),
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
# Prompt sizing for argument-transport tools
# ---------------------------------------------------------------------------

CJK = "中"  # three UTF-8 bytes per character


def test_long_messages_reach_a_stdin_tool_whole(tmp_path: Path) -> None:
    # The reviewer's case: a 90,062-character question with a requirement in
    # the middle used to be cut to the 80,000-character history budget.
    question = "A" * 45_000 + " REQUIREMENT-IN-THE-MIDDLE " + "B" * 45_035
    answer = "C" * 45_000 + " DETAIL-IN-THE-MIDDLE " + "D" * 45_000
    assert len(question) == 90_062
    runner = FakeRunner([question + "\nNEXT: ASK", answer, "done\nNEXT: DONE"])
    participants = converse.make_participants(AIToolID.CLAUDE, AIToolID.CLAUDE)
    result = converse.run_conversation(
        GOAL, participants, working_dir=tmp_path, runner=runner, supports_resume=_resumes
    )
    assert result.answered
    assert runner.calls[1][1].prompt == question
    assert runner.calls[2][1].prompt == answer
    assert all("characters omitted" not in req.prompt for _tool, req in runner.calls)


def test_long_messages_survive_replay_and_only_history_shrinks(tmp_path: Path) -> None:
    long_question = "q2 " + "Q" * 90_000
    runner = FakeRunner(
        [
            "q1 " + "H" * 50_000 + "\nNEXT: ASK",
            "a1",
            long_question + "\nNEXT: ASK",
            "a2",
            "x\nNEXT: DONE",
        ]
    )
    _run(runner, tmp_path, supports_resume=_replays)
    responder_turn_2 = runner.calls[3][1].prompt
    assert responder_turn_2.endswith("### Question\n" + long_question)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX byte limit")
def test_history_is_dropped_entirely_before_a_message_is_rejected(tmp_path: Path) -> None:
    # Codex's case: a 119,500-character question fits Copilot's 120,000-byte
    # limit on its own, but not with even the 1,000-character history floor.
    question = "Q" * 119_500
    participants = converse.make_participants(AIToolID.CLAUDE, AIToolID.COPILOT)
    runner = FakeRunner(
        ["q1\nNEXT: ASK", "A" * 3_000, question + "\nNEXT: ASK", "a2", "done\nNEXT: DONE"]
    )
    result = converse.run_conversation(
        GOAL, participants, working_dir=tmp_path, runner=runner, supports_resume=_replays
    )
    assert result.answered, result.error
    copilot_turn_2 = runner.calls[3][1].prompt
    assert copilot_turn_2.endswith("### Question\n" + question)
    assert "2 earlier message(s) omitted" in copilot_turn_2
    assert "A" * 100 not in copilot_turn_2
    assert len(os.fsencode(copilot_turn_2)) <= 120_000


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX byte limit")
def test_message_too_large_for_an_argv_tool_fails_explicitly(tmp_path: Path) -> None:
    participants = converse.make_participants(AIToolID.CLAUDE, AIToolID.CURSOR)
    runner = FakeRunner([CJK * 50_000 + "\nNEXT: ASK", "unused"])
    result = converse.run_conversation(
        GOAL, participants, working_dir=tmp_path, runner=runner, supports_resume=_resumes
    )
    assert result.stop_reason is StopReason.FAILED
    assert result.error == (
        "responder (cursor) turn 2: the message (50,000 characters) is too large "
        "for cursor's command-line prompt limit"
    )
    assert len(runner.calls) == 1  # nothing was sent to cursor
    assert result.turns[0].text == CJK * 50_000


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX byte limit")
def test_replayed_history_shrinks_for_an_argv_tool(tmp_path: Path) -> None:
    participants = converse.make_participants(AIToolID.CLAUDE, AIToolID.COPILOT)
    runner = FakeRunner(
        [
            "q1\nNEXT: ASK",
            CJK * 30_000,
            "q2 " + CJK * 30_000 + "\nNEXT: ASK",
            "a2",
            "x\nNEXT: DONE",
        ]
    )
    converse.run_conversation(
        GOAL, participants, working_dir=tmp_path, runner=runner, supports_resume=_replays
    )
    copilot_turn_2 = runner.calls[3][1].prompt
    assert len(os.fsencode(copilot_turn_2)) <= 120_000


@pytest.fixture
def simulated_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measure argument prompts the way the runtime does on Windows."""
    from crossby.ai_tools import base

    monkeypatch.setattr(base.sys, "platform", "win32")
    monkeypatch.setattr(base, "_MAX_HEADLESS_ARGUMENT_PROMPT", 30_000)


def _windows_units(prompt: str) -> int:
    return len(subprocess.list2cmdline([prompt]).encode("utf-16-le")) // 2


EMOJI = "\U0001f600"  # two UTF-16 units


@pytest.mark.usefixtures("simulated_windows")
@pytest.mark.parametrize(
    ("prompt", "fits"),
    [
        ("x" * 30_000, True),
        ("x" * 30_001, False),
        ('"' * 15_000, True),  # each quote is escaped as \" (two units)
        ('"' * 15_001, False),
        (EMOJI * 15_000, True),
        (EMOJI * 15_001, False),
    ],
)
def test_headless_prompt_fits_uses_windows_quoting_and_utf16(prompt: str, fits: bool) -> None:
    assert AbstractAITool.get(AIToolID.CURSOR).headless_prompt_fits(prompt) is fits
    assert AbstractAITool.get(AIToolID.CLAUDE).headless_prompt_fits(prompt)


@pytest.mark.usefixtures("simulated_windows")
def test_windows_limit_fails_explicitly_instead_of_cutting(tmp_path: Path) -> None:
    participants = converse.make_participants(AIToolID.CLAUDE, AIToolID.COPILOT)
    runner = FakeRunner([EMOJI * 20_000 + "\nNEXT: ASK", "unused"])
    result = converse.run_conversation(
        GOAL, participants, working_dir=tmp_path, runner=runner, supports_resume=_resumes
    )
    assert result.stop_reason is StopReason.FAILED
    assert "too large for copilot's command-line prompt limit" in (result.error or "")
    assert len(runner.calls) == 1


@pytest.mark.usefixtures("simulated_windows")
def test_messages_within_the_windows_limit_go_whole(tmp_path: Path) -> None:
    participants = converse.make_participants(AIToolID.CLAUDE, AIToolID.COPILOT)
    question = EMOJI * 10_000
    runner = FakeRunner([question + "\nNEXT: ASK", "a", "done\nNEXT: DONE"])
    converse.run_conversation(
        GOAL, participants, working_dir=tmp_path, runner=runner, supports_resume=_resumes
    )
    assert runner.calls[1][1].prompt == question
    assert _windows_units(question) <= 30_000


def test_headless_prompt_fits_measures_encoded_argument_bytes() -> None:
    cursor = AbstractAITool.get(AIToolID.CURSOR)
    claude = AbstractAITool.get(AIToolID.CLAUDE)
    assert cursor.headless_prompt_fits("short")
    assert claude.headless_prompt_fits(CJK * 200_000)  # stdin: no argument limit
    if not sys.platform.startswith("win"):
        assert cursor.headless_prompt_fits(CJK * 40_000)
        assert not cursor.headless_prompt_fits(CJK * 40_001)


def test_native_supports_resume_matches_the_adapters() -> None:
    assert converse.native_supports_resume(AIToolID.CLAUDE)
    assert not converse.native_supports_resume(AIToolID.CURSOR)


# ---------------------------------------------------------------------------
# Recursion guard and ask
# ---------------------------------------------------------------------------


def test_agents_run_one_level_deeper_and_depth_is_restored(tmp_path: Path) -> None:
    runner = FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"])
    _run(runner, tmp_path)
    assert runner.depths == ["1", "1", "1"]
    assert DEPTH_ENV not in os.environ


def test_nested_consultation_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    converse.check_depth()
    monkeypatch.setenv(DEPTH_ENV, "1")
    with pytest.raises(ConverseDepthError, match="Nested consultations are disabled"):
        converse.check_depth()


def test_unparseable_depth_is_treated_as_top_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEPTH_ENV, "nope")
    converse.check_depth()


def test_ask_sends_the_question_as_written(tmp_path: Path) -> None:
    runner = FakeRunner(["  The answer.  "])
    result = converse.ask_agent(
        AIToolID.CODEX, "Question?", working_dir=tmp_path, model="m", runner=runner
    )
    assert result.succeeded
    assert result.text == "The answer."
    ((_tool, request),) = runner.calls
    assert request.prompt == "Question?"
    assert request.model == "m"
    assert runner.depths == ["1"]


def test_ask_reports_failures(tmp_path: Path) -> None:
    failed = converse.ask_agent(
        AIToolID.CODEX,
        "q",
        working_dir=tmp_path,
        runner=FakeRunner([HeadlessTerminalStatus.FAILED]),
    )
    assert not failed.succeeded
    assert failed.error == "codex session ended with status failed"
    errored = converse.ask_agent(
        AIToolID.CODEX,
        "q",
        working_dir=tmp_path,
        runner=FakeRunner([_unsupported(AIToolID.CODEX, "not installed")]),
    )
    assert errored.status is None
    assert errored.error == "not installed"


# ---------------------------------------------------------------------------
# Transcript and summary
# ---------------------------------------------------------------------------


def _turn(number: int, role: Role, text: str, next_: Next | None) -> ConverseTurn:
    tool = AIToolID.CLAUDE if role is Role.ASKER else AIToolID.CODEX
    return ConverseTurn(number, role, tool, text, next_, 1.0, SessionMode.RESUMED)


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
        modes={Role.ASKER: SessionMode.RESUMED, Role.RESPONDER: SessionMode.REPLAYED},
    )


def test_render_transcript() -> None:
    text = converse.render_transcript(_finished(StopReason.ANSWERED))
    assert f"**Goal (sent only to the asker):** {GOAL}" in text
    assert "- Asker: claude" in text
    assert "- Responder: codex" in text
    assert "- Outcome: answered: the asker gave its final answer after 1 round" in text
    assert "## 1. asker (claude) → responder\n\nWhat is a PFO?" in text
    assert "## 2. responder (codex) → asker\n\nAn opening between the atria." in text
    assert "## 3. Final answer (claude)\n\nIt knows the anatomy." in text
    assert "*NEXT: ASK · 1s · session resumed*" in text


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (
            StopReason.ROUND_LIMIT,
            "round limit reached (5 rounds) and the asker gave no final answer",
        ),
        (StopReason.TIMEOUT, "stopped on the time limit after 1 round: late"),
        (StopReason.FAILED, "a session failed after 1 round: late"),
        (StopReason.INTERRUPTED, "interrupted by the user after 1 round: late"),
    ],
)
def test_describe_stop(reason: StopReason, expected: str) -> None:
    assert converse.describe_stop(_finished(reason, error="late")) == expected


def test_answered_at_the_limit_is_described() -> None:
    result = _finished(StopReason.ANSWERED)
    result.answered_at_limit = True
    assert converse.describe_stop(result).endswith("after 1 round at the round limit")


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


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX byte limit")
def test_summary_prompt_fits_an_argv_summarizer(tmp_path: Path) -> None:
    result = _finished(StopReason.ROUND_LIMIT)
    result.turns = [
        _turn(n, Role.ASKER if n % 2 else Role.RESPONDER, CJK * 25_000, None) for n in range(1, 4)
    ]
    runner = FakeRunner(["## Outcome\nok"])
    converse.summarize_conversation(
        result,
        transcript_path=tmp_path / "t.md",
        tool=AIToolID.CURSOR,
        working_dir=tmp_path,
        runner=runner,
    )
    assert len(os.fsencode(runner.calls[0][1].prompt)) <= 120_000


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
