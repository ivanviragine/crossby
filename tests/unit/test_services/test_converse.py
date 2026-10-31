"""Tests for the ask/converse engine, driven by a fake headless runner."""

from __future__ import annotations

import os
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
        preflight_error: BaseException | None = None,
    ) -> None:
        self.replies = list(replies)
        self.clock = clock
        self.seconds_per_turn = seconds_per_turn
        self.preflight_error = preflight_error
        self.preflights: list[AIToolID] = []
        self.calls: list[tuple[AIToolID, HeadlessSessionRequest]] = []
        self.depths: list[str | None] = []

    def preflight(self, tool: AIToolID, request: HeadlessSessionRequest) -> None:
        self.preflights.append(tool)
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


def test_keyboard_interrupt_returns_partial_result(tmp_path: Path) -> None:
    runner = FakeRunner(["a\nSTATUS: CONTINUE", KeyboardInterrupt()])
    result = _run(runner, tmp_path)
    assert result.stop_reason is StopReason.INTERRUPTED
    assert result.exit_code == 130
    assert len(result.turns) == 1


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
    assert request.prompt == "Question?"
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
    assert "- Outcome: agreed: both agents marked the goal as met after 2 turns" in text
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


def test_summary_uses_the_handoff_summarizer_with_goal_and_stop_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.handoff import summarizer as summarizer_mod
    from crossby.handoff.models import RawHandoff

    captured: dict[str, object] = {}

    def fake_summarize_raw(self, transcript, **kwargs):  # type: ignore[no-untyped-def]
        captured["template"] = self.prompt_template
        captured["working_dir"] = self.working_dir
        captured["turns"] = [turn.content for turn in transcript.turns]
        captured["kwargs"] = kwargs
        return RawHandoff(
            source_tool=kwargs["source_tool"],
            target_tool=kwargs["target_tool"],
            session_ref=transcript.session_ref,
            body="## Outcome\nThey agreed.\n",
            prompt_source=kwargs["prompt_source"],
        )

    monkeypatch.setattr(summarizer_mod.HandoffSummarizer, "summarize_raw", fake_summarize_raw)
    result = _finished(StopReason.TURN_LIMIT)
    body = converse.summarize_conversation(
        result,
        transcript_path=tmp_path / "run" / "transcript.md",
        summarizer_tool=AbstractAITool.get(AIToolID.CLAUDE),
        working_dir=tmp_path,
    )

    assert body == "## Outcome\nThey agreed.\n"
    template = str(captured["template"])
    assert "Goal: Agree on a fix" in template
    assert "Stop reason: turn limit reached (6 turns) without agreement" in template
    assert "codex and claude" in template
    assert "## Open Disagreements" in template
    assert captured["working_dir"] == tmp_path
    assert captured["turns"] == [
        "codex (turn 1, STATUS: CONTINUE):\nCause is X.",
        "claude (turn 2, STATUS: DONE):\nAgreed.",
    ]


def test_render_summary_and_conversation_dir(tmp_path: Path) -> None:
    result = _finished(StopReason.AGREED)
    text = converse.render_summary(result, "## Outcome\nDone.")
    assert text.startswith("# Conversation summary\n\n**Goal:** Agree on a fix")
    assert "- Outcome: agreed" in text
    assert text.endswith("## Outcome\nDone.\n")
    assert converse.conversation_dir(tmp_path, result.started_at) == (
        tmp_path / ".crossby" / "conversations" / "20261031T120000000000"
    )
