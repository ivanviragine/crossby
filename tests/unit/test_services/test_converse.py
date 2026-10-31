"""Tests for the ask/converse engine, driven by a fake headless runner."""

from __future__ import annotations

import os
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
    AgentStatus,
    ConversationResult,
    ConverseDepthError,
    ConverseError,
    ConverseTurn,
    StopReason,
)

Reply = str | HeadlessTerminalStatus | BaseException


def _result(
    tool: AIToolID, status: HeadlessTerminalStatus, text: str | None
) -> HeadlessSessionResult:
    return HeadlessSessionResult(
        tool=tool,
        version="1.0.0",
        status=status,
        events=(
            HeadlessEvent(sequence=1, kind=HeadlessEventKind.TERMINAL, terminal_status=status),
        ),
        final_text=text,
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
    def __init__(
        self,
        replies: list[Reply],
        *,
        clock: FakeClock | None = None,
        seconds_per_turn: float = 1.0,
        seconds_per_preflight: float = 0.0,
        preflight_error: BaseException | None = None,
    ) -> None:
        self.replies = list(replies)
        self.clock = clock
        self.seconds_per_turn = seconds_per_turn
        self.preflight_error = preflight_error
        self.seconds_per_preflight = seconds_per_preflight
        self.preflights: list[AIToolID] = []
        self.preflight_timeouts: list[float] = []
        self.calls: list[tuple[AIToolID, HeadlessSessionRequest]] = []
        self.depths: list[str | None] = []

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
        if isinstance(item, HeadlessTerminalStatus):
            return _result(tool, item, None)
        return _result(tool, HeadlessTerminalStatus.SUCCEEDED, item)


PARTICIPANTS = converse.make_participants(AIToolID.CODEX, AIToolID.CLAUDE)


def _run(runner: FakeRunner, tmp_path: Path, **kwargs: object) -> ConversationResult:
    return converse.run_conversation(
        "Agree on a fix",
        PARTICIPANTS,
        working_dir=tmp_path,
        runner=runner,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture(autouse=True)
def _no_inherited_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEPTH_ENV, raising=False)


# ---------------------------------------------------------------------------
# parse_status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("Looks right.\nSTATUS: DONE", AgentStatus.DONE),
        ("Not yet.\n\nSTATUS: CONTINUE\n\n", AgentStatus.CONTINUE),
        ("ok\n**STATUS: DONE**", AgentStatus.DONE),
        ("ok\n`STATUS: done`", AgentStatus.DONE),
        ("ok\nStatus: Continue.", AgentStatus.CONTINUE),
        ("STATUS: DONE is what I'd say later\nbut more to check", AgentStatus.MISSING),
        ("no status line here", AgentStatus.MISSING),
        ("ok\nSTATUS: MAYBE", AgentStatus.MISSING),
    ],
)
def test_parse_status(text: str, status: AgentStatus) -> None:
    assert converse.parse_status(text)[1] is status


def test_parse_status_strips_the_status_line_from_the_body() -> None:
    body, status = converse.parse_status("Line one.\nLine two.\n\nSTATUS: DONE")
    assert body == "Line one.\nLine two."
    assert status is AgentStatus.DONE


# ---------------------------------------------------------------------------
# Prompts and history
# ---------------------------------------------------------------------------


def _turn(number: int, speaker: str, text: str, status: AgentStatus) -> ConverseTurn:
    tool = AIToolID.CODEX if speaker == "codex" else AIToolID.CLAUDE
    return ConverseTurn(number, speaker, tool, text, status, 1.0)


def test_first_turn_prompt_carries_goal_roles_and_status_rule() -> None:
    a, b = PARTICIPANTS
    template = converse.load_turn_template()
    prompt = converse.build_turn_prompt(
        template=template,
        goal="Find the flaky test",
        speaker=a,
        peer=b,
        turn=1,
        max_turns=6,
        history=[],
        history_chars=10_000,
    )
    assert "You are codex" in prompt
    assert "with claude" in prompt
    assert "Goal: Find the flaky test" in prompt
    assert "turn 1 of at most 6" in prompt
    assert "STATUS: DONE" in prompt
    assert "You speak first" in prompt


def test_goal_with_braces_is_not_reinterpreted() -> None:
    a, b = PARTICIPANTS
    prompt = converse.build_turn_prompt(
        template=converse.load_turn_template(),
        goal="Fix {placeholder} handling in f'{x}'",
        speaker=a,
        peer=b,
        turn=1,
        max_turns=2,
        history=[],
        history_chars=10_000,
    )
    assert "Fix {placeholder} handling in f'{x}'" in prompt


def test_history_keeps_most_recent_turns_within_budget() -> None:
    turns = [
        _turn(1, "codex", "A" * 900, AgentStatus.CONTINUE),
        _turn(2, "claude", "B" * 900, AgentStatus.CONTINUE),
        _turn(3, "codex", "C" * 900, AgentStatus.DONE),
    ]
    rendered = converse.render_history(turns, peer="claude", budget_chars=2000)
    assert "A" * 900 not in rendered
    assert "B" * 900 in rendered
    assert "C" * 900 in rendered
    assert "1 earlier turn(s) omitted" in rendered
    assert "### codex (turn 3, STATUS: DONE)" in rendered
    assert rendered.rstrip().endswith("It is your turn. Reply to claude.")


def test_history_clips_a_single_oversized_reply() -> None:
    turns = [_turn(1, "codex", "head" + "x" * 5000 + "tail", AgentStatus.CONTINUE)]
    rendered = converse.render_history(turns, peer="claude", budget_chars=1000)
    assert "characters omitted" in rendered
    assert "head" in rendered
    assert "tail" in rendered
    assert len(rendered) < 1300


def test_same_tool_on_both_sides_gets_distinct_labels() -> None:
    a, b = converse.make_participants(AIToolID.CLAUDE, AIToolID.CLAUDE, model_b="other")
    assert (a.label, b.label) == ("claude-1", "claude-2")
    assert b.model == "other"


# ---------------------------------------------------------------------------
# run_conversation
# ---------------------------------------------------------------------------


def test_agreement_stops_after_consecutive_done(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            "Cause is X.\nSTATUS: CONTINUE",
            "Agree it is X; fix with Y.\nSTATUS: DONE",
            "Y works.\nSTATUS: DONE",
            "should never be asked",
        ]
    )
    result = _run(runner, tmp_path)

    assert result.stop_reason is StopReason.AGREED
    assert result.agreed
    assert result.exit_code == 0
    assert [t.speaker for t in result.turns] == ["codex", "claude", "codex"]
    assert [t.status for t in result.turns] == [
        AgentStatus.CONTINUE,
        AgentStatus.DONE,
        AgentStatus.DONE,
    ]
    assert result.turns[1].text == "Agree it is X; fix with Y."
    assert len(runner.calls) == 3


def test_turns_alternate_and_replay_the_conversation(tmp_path: Path) -> None:
    runner = FakeRunner(["first take\nSTATUS: CONTINUE", "reply\nSTATUS: CONTINUE"])
    _run(runner, tmp_path, max_turns=2)

    (tool_1, req_1), (tool_2, req_2) = runner.calls
    assert (tool_1, tool_2) == (AIToolID.CODEX, AIToolID.CLAUDE)
    assert "You speak first" in req_1.prompt
    assert "### codex (turn 1, STATUS: CONTINUE)\nfirst take" in req_2.prompt
    assert "first take" not in req_1.prompt
    for request in (req_1, req_2):
        assert request.interaction_mode is HeadlessInteractionMode.UNATTENDED
        assert request.resume_id is None
        assert request.working_dir == tmp_path


def test_turn_limit_without_agreement(tmp_path: Path) -> None:
    runner = FakeRunner([f"take {n}\nSTATUS: CONTINUE" for n in range(4)])
    result = _run(runner, tmp_path, max_turns=4)
    assert result.stop_reason is StopReason.TURN_LIMIT
    assert result.exit_code == 2
    assert len(result.turns) == 4


def test_a_single_done_is_not_agreement(tmp_path: Path) -> None:
    runner = FakeRunner(["done\nSTATUS: DONE", "no\nSTATUS: CONTINUE", "hmm\nSTATUS: DONE"])
    result = _run(runner, tmp_path, max_turns=3)
    assert result.stop_reason is StopReason.TURN_LIMIT


def test_missing_status_line_counts_as_continue(tmp_path: Path) -> None:
    runner = FakeRunner(["STATUS: DONE", "I forgot the status line", "STATUS: DONE"])
    result = _run(runner, tmp_path, max_turns=3)
    assert result.turns[1].status is AgentStatus.MISSING
    assert result.stop_reason is StopReason.TURN_LIMIT


def test_participant_models_reach_their_requests(tmp_path: Path) -> None:
    participants = converse.make_participants(
        AIToolID.CODEX, AIToolID.CLAUDE, model_a="gpt-x", model_b="opus-y"
    )
    runner = FakeRunner(["a\nSTATUS: CONTINUE", "b\nSTATUS: CONTINUE"])
    converse.run_conversation(
        "goal", participants, working_dir=tmp_path, runner=runner, max_turns=2
    )
    assert [req.model for _tool, req in runner.calls] == ["gpt-x", "opus-y"]


def test_preflight_failure_stops_before_any_turn(tmp_path: Path) -> None:
    runner = FakeRunner(
        ["unused"], preflight_error=_unsupported(AIToolID.CODEX, "codex is too old")
    )
    result = _run(runner, tmp_path)
    assert result.stop_reason is StopReason.FAILED
    assert result.exit_code == 1
    assert "codex cannot run headless: codex is too old" in (result.error or "")
    assert runner.calls == []


def test_preflights_spend_the_overall_budget(tmp_path: Path) -> None:
    clock = FakeClock()
    runner = FakeRunner(
        ["a\nSTATUS: DONE", "b\nSTATUS: DONE"], clock=clock, seconds_per_preflight=8
    )
    result = _run(runner, tmp_path, timeout_seconds=30, turn_timeout_seconds=600, clock=clock)
    assert result.agreed
    assert runner.preflight_timeouts == [30, 22]
    assert [req.timeout_seconds for _tool, req in runner.calls] == [14, 13]


def test_slow_preflight_that_exhausts_the_budget_is_a_timeout(tmp_path: Path) -> None:
    clock = FakeClock()
    runner = FakeRunner(["unused"], clock=clock, seconds_per_preflight=18)
    result = _run(runner, tmp_path, timeout_seconds=20, clock=clock)
    assert result.stop_reason is StopReason.TIMEOUT
    assert result.error == "conversation time limit of 20s reached during preflight"
    assert runner.preflights == [AIToolID.CODEX]
    assert runner.calls == []


def test_preflight_error_after_the_deadline_is_a_timeout(tmp_path: Path) -> None:
    clock = FakeClock()
    runner = FakeRunner(
        [],
        clock=clock,
        seconds_per_preflight=25,
        preflight_error=_unsupported(AIToolID.CODEX, "version probe timed out"),
    )
    result = _run(runner, tmp_path, timeout_seconds=20, clock=clock)
    assert result.stop_reason is StopReason.TIMEOUT
    assert "during preflight" in (result.error or "")


def test_session_error_mid_conversation_keeps_completed_turns(tmp_path: Path) -> None:
    runner = FakeRunner(["a\nSTATUS: CONTINUE", _unsupported(AIToolID.CLAUDE, "boom")])
    result = _run(runner, tmp_path)
    assert result.stop_reason is StopReason.FAILED
    assert len(result.turns) == 1
    assert result.error == "claude turn 2: boom"


@pytest.mark.parametrize(
    "status", [HeadlessTerminalStatus.FAILED, HeadlessTerminalStatus.INVALID_OUTPUT]
)
def test_failed_session_status_stops_the_conversation(
    tmp_path: Path, status: HeadlessTerminalStatus
) -> None:
    runner = FakeRunner([status])
    result = _run(runner, tmp_path)
    assert result.stop_reason is StopReason.FAILED
    assert result.error == f"codex turn 1 ended with status {status.value}"


def test_reply_without_text_is_a_failure(tmp_path: Path) -> None:
    # A successful session always carries final output; JSON without text is
    # the only way to get no text, and the relay needs text.
    runner = FakeRunner([])
    json_only = HeadlessSessionResult(
        tool=AIToolID.CODEX,
        version="1.0.0",
        status=HeadlessTerminalStatus.SUCCEEDED,
        events=(
            HeadlessEvent(
                sequence=1,
                kind=HeadlessEventKind.TERMINAL,
                terminal_status=HeadlessTerminalStatus.SUCCEEDED,
            ),
        ),
        final_json={"answer": 1},
        duration_seconds=1.0,
    )
    runner.run = lambda tool, request, cancel_event: json_only  # type: ignore[method-assign]
    result = _run(runner, tmp_path)
    assert result.stop_reason is StopReason.FAILED
    assert result.error == "codex turn 1 returned no text"


def test_turn_timeout_inside_the_overall_budget_is_a_failure(tmp_path: Path) -> None:
    runner = FakeRunner([HeadlessTerminalStatus.TIMED_OUT])
    result = _run(runner, tmp_path, timeout_seconds=1000, turn_timeout_seconds=10)
    assert result.stop_reason is StopReason.FAILED
    assert result.error == "codex turn 1 timed out after 10s"


def test_overall_deadline_bounds_each_turn_and_stops_as_timeout(tmp_path: Path) -> None:
    clock = FakeClock()
    runner = FakeRunner(
        ["a\nSTATUS: CONTINUE", "b\nSTATUS: CONTINUE", "unused"],
        clock=clock,
        seconds_per_turn=48,
    )
    result = _run(runner, tmp_path, timeout_seconds=100, turn_timeout_seconds=600, clock=clock)
    assert result.stop_reason is StopReason.TIMEOUT
    assert result.exit_code == 2
    assert len(result.turns) == 2
    timeouts = [req.timeout_seconds for _tool, req in runner.calls]
    assert timeouts == [100, 52]


def test_deadline_hit_during_a_turn_is_a_timeout(tmp_path: Path) -> None:
    runner = FakeRunner([HeadlessTerminalStatus.TIMED_OUT])
    result = _run(runner, tmp_path, timeout_seconds=50, turn_timeout_seconds=600)
    assert result.stop_reason is StopReason.TIMEOUT
    assert "during turn 1" in (result.error or "")


def test_startup_error_after_the_deadline_is_a_timeout(tmp_path: Path) -> None:
    clock = FakeClock()
    runner = FakeRunner(
        ["a\nSTATUS: CONTINUE", _unsupported(AIToolID.CLAUDE, "version probe ran out of time")],
        clock=clock,
        seconds_per_turn=30,
    )
    result = _run(runner, tmp_path, timeout_seconds=50, clock=clock)
    assert result.stop_reason is StopReason.TIMEOUT
    assert result.exit_code == 2
    assert result.error == "conversation time limit of 50s reached during turn 2"
    assert len(result.turns) == 1


def test_keyboard_interrupt_returns_partial_result(tmp_path: Path) -> None:
    runner = FakeRunner(["a\nSTATUS: CONTINUE", KeyboardInterrupt()])
    result = _run(runner, tmp_path)
    assert result.stop_reason is StopReason.INTERRUPTED
    assert result.exit_code == 130
    assert len(result.turns) == 1


def test_interrupt_while_the_caller_prints_a_reply_keeps_completed_turns(
    tmp_path: Path,
) -> None:
    def on_turn(turn: ConverseTurn) -> None:
        if turn.number == 2:
            raise KeyboardInterrupt

    runner = FakeRunner(["a\nSTATUS: CONTINUE", "b\nSTATUS: CONTINUE", "unused"])
    result = _run(runner, tmp_path, on_turn=on_turn)
    assert result.stop_reason is StopReason.INTERRUPTED
    assert [t.text for t in result.turns] == ["a", "b"]
    assert result.error == "interrupted after 2 turns"
    assert DEPTH_ENV not in os.environ


def test_interrupt_in_turn_start_callback_is_reported_for_that_turn(tmp_path: Path) -> None:
    def on_turn_start(number: int, _speaker: object) -> None:
        if number == 2:
            raise KeyboardInterrupt

    result = _run(FakeRunner(["a\nSTATUS: CONTINUE"]), tmp_path, on_turn_start=on_turn_start)
    assert result.stop_reason is StopReason.INTERRUPTED
    assert result.error == "interrupted during turn 2"
    assert len(result.turns) == 1


def test_interrupt_during_preflight_is_an_interruption(tmp_path: Path) -> None:
    result = _run(FakeRunner([], preflight_error=KeyboardInterrupt()), tmp_path)
    assert result.stop_reason is StopReason.INTERRUPTED
    assert result.turns == []


def test_turn_callbacks_fire_in_order(tmp_path: Path) -> None:
    events: list[str] = []
    runner = FakeRunner(["a\nSTATUS: DONE", "b\nSTATUS: DONE"])
    _run(
        runner,
        tmp_path,
        on_turn_start=lambda n, p: events.append(f"start {n} {p.label}"),
        on_turn=lambda t: events.append(f"done {t.number} {t.status.value}"),
    )
    assert events == ["start 1 codex", "done 1 DONE", "start 2 claude", "done 2 DONE"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_turns": 1}, "at least 2"),
        ({"timeout_seconds": 0}, "positive"),
        ({"history_chars": 10}, "1000"),
    ],
)
def test_invalid_settings_are_rejected(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ConverseError, match=message):
        _run(FakeRunner([]), tmp_path, **kwargs)


def test_blank_goal_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConverseError, match="goal is empty"):
        converse.run_conversation("  ", PARTICIPANTS, working_dir=tmp_path, runner=FakeRunner([]))


# ---------------------------------------------------------------------------
# Recursion guard
# ---------------------------------------------------------------------------


def test_agents_run_one_level_deeper_and_depth_is_restored(tmp_path: Path) -> None:
    runner = FakeRunner(["a\nSTATUS: DONE", "b\nSTATUS: DONE"])
    _run(runner, tmp_path)
    assert runner.depths == ["1", "1"]
    assert DEPTH_ENV not in os.environ


def test_nested_consultation_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    converse.check_depth()
    monkeypatch.setenv(DEPTH_ENV, "1")
    with pytest.raises(ConverseDepthError, match="Nested consultations are disabled"):
        converse.check_depth()


def test_unparseable_depth_is_treated_as_top_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEPTH_ENV, "nope")
    converse.check_depth()


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def test_ask_returns_the_answer(tmp_path: Path) -> None:
    runner = FakeRunner(["  The answer.  "])
    result = converse.ask_agent(
        AIToolID.CODEX, "Question?", working_dir=tmp_path, model="m", runner=runner
    )
    assert result.succeeded
    assert result.text == "The answer."
    ((_tool, request),) = runner.calls
    assert request.prompt.endswith("Question:\n\nQuestion?")
    assert "Do not modify, create, or delete files." in request.prompt
    assert request.model == "m"
    assert request.interaction_mode is HeadlessInteractionMode.UNATTENDED
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


def _finished(reason: StopReason, *, error: str | None = None) -> ConversationResult:
    return ConversationResult(
        goal="Agree on a fix",
        participants=PARTICIPANTS,
        max_turns=6,
        started_at=datetime(2026, 10, 31, 12, 0, tzinfo=UTC),
        finished_at=datetime(2026, 10, 31, 12, 5, tzinfo=UTC),
        stop_reason=reason,
        turns=[
            _turn(1, "codex", "Cause is X.", AgentStatus.CONTINUE),
            _turn(2, "claude", "Agreed.", AgentStatus.DONE),
        ],
        error=error,
    )


def test_render_transcript() -> None:
    text = converse.render_transcript(_finished(StopReason.AGREED))
    assert "**Goal:** Agree on a fix" in text
    assert "- Participants: codex, claude" in text
    assert "- Outcome: agreed: both agents declared the goal met after 2 turns" in text
    assert "## Turn 1: codex\n\nCause is X." in text
    assert "*STATUS: DONE · 1s*" in text


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (StopReason.TURN_LIMIT, "turn limit reached (6 turns) without agreement"),
        (StopReason.TIMEOUT, "stopped on the time limit after 2 turns without agreement: late"),
        (StopReason.FAILED, "a session failed after 2 turns: late"),
        (StopReason.INTERRUPTED, "interrupted by the user after 2 turns: late"),
    ],
)
def test_describe_stop(reason: StopReason, expected: str) -> None:
    assert converse.describe_stop(_finished(reason, error="late")) == expected


def test_summary_runs_as_a_managed_unattended_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    def no_plain_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the summary must not bypass the managed runtime")

    monkeypatch.setattr(subprocess, "run", no_plain_subprocess)
    runner = FakeRunner(["## Outcome\nThey agreed.\n"])
    body = converse.summarize_conversation(
        _finished(StopReason.TURN_LIMIT),
        transcript_path=tmp_path / "run" / "transcript.md",
        tool=AIToolID.CLAUDE,
        working_dir=tmp_path,
        model="haiku",
        timeout_seconds=42,
        runner=runner,
    )

    assert body == "## Outcome\nThey agreed."
    ((tool, request),) = runner.calls
    assert tool is AIToolID.CLAUDE
    assert request.interaction_mode is HeadlessInteractionMode.UNATTENDED
    assert request.timeout_seconds == 42
    assert request.model == "haiku"
    assert request.working_dir == tmp_path
    assert runner.depths == ["1"]
    assert DEPTH_ENV not in os.environ


def test_summary_prompt_carries_goal_stop_reason_rules_and_transcript(tmp_path: Path) -> None:
    prompt = converse.build_summary_prompt(
        _finished(StopReason.TURN_LIMIT), transcript_path=tmp_path / "t.md"
    )
    assert "Goal: Agree on a fix" in prompt
    assert "Stop reason: turn limit reached (6 turns) without agreement" in prompt
    assert "codex and claude" in prompt
    assert "## Open Disagreements" in prompt
    assert "Do not read, modify, create, or delete files" in prompt
    assert "It is not independent verification." in prompt
    assert "--- Transcript ---" in prompt
    assert "codex (turn 1, STATUS: CONTINUE):\nCause is X." in prompt
    assert "claude (turn 2, STATUS: DONE):\nAgreed." in prompt
    assert "earlier turns were dropped" not in prompt


def test_summary_prompt_fits_the_character_ceiling(tmp_path: Path) -> None:
    result = _finished(StopReason.TURN_LIMIT)
    result.turns = [
        _turn(n, "codex" if n % 2 else "claude", str(n) * 30_000, AgentStatus.CONTINUE)
        for n in range(1, 7)
    ]
    truncations: list[tuple[int, int]] = []
    prompt = converse.build_summary_prompt(
        result,
        transcript_path=tmp_path / "t.md",
        max_chars=70_000,
        on_truncate=lambda total, kept: truncations.append((total, kept)),
    )
    assert len(prompt) <= 70_000
    assert "earlier turns were dropped" in prompt
    assert "6" * 30_000 in prompt
    assert "1" * 30_000 not in prompt
    assert truncations == [(6, 2)]


def test_summary_prompt_clips_a_single_oversized_turn(tmp_path: Path) -> None:
    result = _finished(StopReason.TURN_LIMIT)
    result.turns = [_turn(1, "codex", "x" * 150_000, AgentStatus.CONTINUE)]
    prompt = converse.build_summary_prompt(
        result, transcript_path=tmp_path / "t.md", max_chars=50_000
    )
    assert len(prompt) <= 50_000
    assert "characters omitted" in prompt


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (HeadlessTerminalStatus.TIMED_OUT, "claude summary session ended with status timed_out"),
        (HeadlessTerminalStatus.FAILED, "claude summary session ended with status failed"),
    ],
)
def test_summary_failures_raise_summary_errors(
    tmp_path: Path, reply: HeadlessTerminalStatus, message: str
) -> None:
    with pytest.raises(converse.ConverseSummaryError, match=message):
        converse.summarize_conversation(
            _finished(StopReason.AGREED),
            transcript_path=tmp_path / "t.md",
            tool=AIToolID.CLAUDE,
            working_dir=tmp_path,
            runner=FakeRunner([reply]),
        )


def test_summary_session_errors_raise_summary_errors(tmp_path: Path) -> None:
    with pytest.raises(converse.ConverseSummaryError, match="argv too long"):
        converse.summarize_conversation(
            _finished(StopReason.AGREED),
            transcript_path=tmp_path / "t.md",
            tool=AIToolID.CURSOR,
            working_dir=tmp_path,
            runner=FakeRunner([_unsupported(AIToolID.CURSOR, "argv too long")]),
        )


CJK = "\u4e2d"  # three UTF-8 bytes per character


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX byte limit")
def test_summary_prompt_fits_an_argv_tools_byte_limit(tmp_path: Path) -> None:
    # Codex's scenario: three 25,000-character CJK replies fit the character
    # ceiling (75,000 chars) but not Cursor's 120,000-byte argument limit.
    result = _finished(StopReason.TURN_LIMIT)
    result.turns = [
        _turn(n, "codex" if n % 2 else "claude", CJK * 25_000, AgentStatus.CONTINUE)
        for n in range(1, 4)
    ]
    runner = FakeRunner(["## Outcome\nok"])
    converse.summarize_conversation(
        result,
        transcript_path=tmp_path / "t.md",
        tool=AIToolID.CURSOR,
        working_dir=tmp_path,
        runner=runner,
    )
    ((_tool, request),) = runner.calls
    assert len(request.prompt) <= converse.SUMMARY_PROMPT_CHARS
    assert len(os.fsencode(request.prompt)) <= 120_000
    assert AbstractAITool.get(AIToolID.CURSOR).headless_prompt_fits(request.prompt)


def test_summary_prompt_for_a_stdin_tool_keeps_the_character_ceiling(tmp_path: Path) -> None:
    result = _finished(StopReason.TURN_LIMIT)
    result.turns = [
        _turn(n, "codex" if n % 2 else "claude", CJK * 25_000, AgentStatus.CONTINUE)
        for n in range(1, 4)
    ]
    runner = FakeRunner(["## Outcome\nok"])
    converse.summarize_conversation(
        result,
        transcript_path=tmp_path / "t.md",
        tool=AIToolID.CLAUDE,
        working_dir=tmp_path,
        runner=runner,
    )
    ((_tool, request),) = runner.calls
    # Claude reads the prompt from stdin, so all three turns still fit.
    assert request.prompt.count(CJK * 25_000) == 3


def test_summary_clips_one_turn_until_the_transport_accepts_it(tmp_path: Path) -> None:
    result = _finished(StopReason.TURN_LIMIT)
    result.turns = [_turn(1, "codex", "x" * 90_000, AgentStatus.CONTINUE)]
    prompt = converse.build_summary_prompt(
        result, transcript_path=tmp_path / "t.md", fits=lambda text: len(text) <= 20_000
    )
    assert len(prompt) <= 20_000
    assert "characters omitted" in prompt


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX byte limit")
def test_turn_prompts_shrink_history_for_an_argv_tool(tmp_path: Path) -> None:
    participants = converse.make_participants(AIToolID.CODEX, AIToolID.CURSOR)
    runner = FakeRunner([CJK * 50_000 + "\nSTATUS: CONTINUE", CJK * 100 + "\nSTATUS: CONTINUE"])
    converse.run_conversation(
        "goal", participants, working_dir=tmp_path, runner=runner, max_turns=2
    )
    (_codex, _first), (tool, second) = runner.calls
    assert tool is AIToolID.CURSOR
    assert len(os.fsencode(second.prompt)) <= 120_000
    assert "characters omitted" in second.prompt


def test_turn_prompt_history_is_left_alone_when_it_fits(tmp_path: Path) -> None:
    checked: list[tuple[AIToolID, int]] = []

    def fits(tool: AIToolID, prompt: str) -> bool:
        checked.append((tool, len(prompt)))
        return True

    runner = FakeRunner(["a" * 5000 + "\nSTATUS: CONTINUE", "b\nSTATUS: CONTINUE"])
    converse.run_conversation(
        "goal", PARTICIPANTS, working_dir=tmp_path, runner=runner, max_turns=2, prompt_fits=fits
    )
    assert [tool for tool, _size in checked] == [AIToolID.CODEX, AIToolID.CLAUDE]
    assert "a" * 5000 in runner.calls[1][1].prompt


def test_headless_prompt_fits_measures_encoded_argument_bytes() -> None:
    cursor = AbstractAITool.get(AIToolID.CURSOR)
    claude = AbstractAITool.get(AIToolID.CLAUDE)
    assert cursor.headless_prompt_fits("short")
    assert claude.headless_prompt_fits(CJK * 200_000)  # stdin: no argument limit
    if not sys.platform.startswith("win"):
        assert cursor.headless_prompt_fits(CJK * 40_000)
        assert not cursor.headless_prompt_fits(CJK * 40_001)


def test_partial_transcript_marks_the_conversation_in_progress() -> None:
    text = converse.render_partial_transcript(
        goal="Agree on a fix",
        participants=PARTICIPANTS,
        max_turns=6,
        started_at=datetime(2026, 10, 31, 12, 0, tzinfo=UTC),
        turns=[_turn(1, "codex", "Cause is X.", AgentStatus.CONTINUE)],
    )
    assert "- Outcome: in progress: 1 turn completed" in text
    assert "- Finished:" not in text
    assert "## Turn 1: codex\n\nCause is X." in text


def test_render_summary_and_conversation_dir(tmp_path: Path) -> None:
    result = _finished(StopReason.AGREED)
    text = converse.render_summary(result, "## Outcome\nDone.")
    assert text.startswith("# Conversation summary\n\n**Goal:** Agree on a fix")
    assert "- Outcome: agreed" in text
    assert text.endswith("## Outcome\nDone.\n")
    assert converse.conversation_dir(tmp_path, result.started_at) == (
        tmp_path / ".crossby" / "conversations" / "20261031T120000000000"
    )
