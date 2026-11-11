"""The event journal, the views built from it, and continuing a consultation."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from crossby.models.ai import AIToolID, HeadlessEventKind, HeadlessTerminalStatus
from crossby.services import converse
from crossby.services.converse import (
    DEPTH_ENV,
    ConverseError,
    Role,
    StopReason,
    render_transcript,
    result_from_journal,
)
from crossby.services.journal import (
    JOURNAL_FILE,
    ConversationEvent,
    EventType,
    Journal,
    JournalError,
    finished,
    read_journal,
)
from tests.unit.test_services.test_converse import GOAL, PARTICIPANTS, FakeRunner

T = EventType


@pytest.fixture(autouse=True)
def _top_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEPTH_ENV, raising=False)


def _run(runner: FakeRunner, tmp_path: Path, journal: Journal, **kwargs: Any) -> Any:
    return converse.run_conversation(
        GOAL, PARTICIPANTS, working_dir=tmp_path, runner=runner, journal=journal, **kwargs
    )


def _journal(tmp_path: Path, **kwargs: Any) -> Journal:
    return Journal("run-1", path=tmp_path / "out" / JOURNAL_FILE, **kwargs)


def test_a_consultation_is_one_ordered_stream(tmp_path: Path) -> None:
    runner = FakeRunner(["Is it a PFO?\nNEXT: ASK", "Yes, a flap.", "It is.\nNEXT: DONE"])
    journal = _journal(tmp_path)
    result = _run(runner, tmp_path, journal)
    journal.close()

    types = [event.type for event in journal.events]
    assert types == [
        T.CONVERSATION_STARTED,
        T.PARTICIPANT_READY,
        T.PARTICIPANT_READY,
        *[T.TURN_STARTED, T.TURN_FINISHED, T.MESSAGE_COMMITTED] * 3,
        T.CONVERSATION_FINISHED,
    ]
    assert [event.seq for event in journal.events] == list(range(1, len(types) + 1))
    assert {event.run_id for event in journal.events} == {"run-1"}
    started = journal.events[0].data
    assert started["goal"] == GOAL
    assert [side["role"] for side in started["participants"]] == ["asker", "responder"]
    assert started["participants"][1]["read_only"] is True
    committed = [event.data for event in journal.events if event.type is T.MESSAGE_COMMITTED]
    assert [item["text"] for item in committed] == ["Is it a PFO?", "Yes, a flap.", "It is."]
    assert [item["next"] for item in committed] == ["ASK", None, "DONE"]
    assert journal.events[-1].data["stop_reason"] == "answered"
    assert journal.events[-1].data["sessions"] == {"asker": "session-1", "responder": "session-2"}
    assert result.answered
    assert finished(journal.events)


def test_the_file_is_the_journal_and_the_transcript_is_a_view_of_it(tmp_path: Path) -> None:
    runner = FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"])
    journal = _journal(tmp_path)
    result = _run(runner, tmp_path, journal)
    journal.close()

    events = read_journal(tmp_path / "out" / JOURNAL_FILE)
    assert events == journal.events
    assert render_transcript(result_from_journal(events)) == render_transcript(result)


def test_progress_is_forwarded_without_content_and_throttled(tmp_path: Path) -> None:
    runner = FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"])
    runner.progress = [
        HeadlessEventKind.TOOL_STARTED,  # 0.5s
        HeadlessEventKind.PROGRESS,  # 1.0s, kept
        HeadlessEventKind.PROGRESS,  # 1.5s, under a second since the last: dropped
        HeadlessEventKind.TOOL_COMPLETED,  # 2.0s
        HeadlessEventKind.USAGE,  # not forwarded
    ]
    journal = _journal(tmp_path)
    _run(runner, tmp_path, journal)
    journal.close()

    first_turn = [
        e.data for e in journal.events if e.type is T.TURN_PROGRESS and e.data["turn"] == 1
    ]
    assert [item["kind"] for item in first_turn] == ["tool_started", "progress", "tool_completed"]
    assert all(set(item) == {"turn", "kind", "elapsed_seconds"} for item in first_turn)


def test_a_failed_turn_is_recorded_before_the_end(tmp_path: Path) -> None:
    runner = FakeRunner(["q\nNEXT: ASK", HeadlessTerminalStatus.FAILED])
    journal = _journal(tmp_path)
    result = _run(runner, tmp_path, journal)
    journal.close()

    assert result.stop_reason is StopReason.FAILED
    failed = [e.data for e in journal.events if e.type is T.TURN_FINISHED][-1]
    assert (failed["turn"], failed["role"], failed["status"]) == (2, "responder", "failed")
    assert failed["error"] == result.error
    assert journal.events[-1].type is T.CONVERSATION_FINISHED
    assert result_from_journal(journal.events).stop_reason is StopReason.FAILED


def test_a_failing_or_slow_sink_never_stops_the_consultation(tmp_path: Path) -> None:
    seen: list[int] = []

    def slow(event: ConversationEvent) -> None:
        time.sleep(0.01)
        seen.append(event.seq)

    def broken(event: ConversationEvent) -> None:
        raise RuntimeError("UI went away")

    runner = FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"])
    journal = _journal(tmp_path, sinks=[broken, slow])
    result = _run(runner, tmp_path, journal)
    journal.close()

    assert result.answered
    assert seen == [event.seq for event in journal.events]
    assert journal.sink_errors and "UI went away" in journal.sink_errors[0]


def test_a_sink_runs_off_the_session_thread(tmp_path: Path) -> None:
    threads: set[str] = set()
    journal = _journal(tmp_path, sinks=[lambda _e: threads.add(threading.current_thread().name)])
    _run(FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"]), tmp_path, journal)
    journal.close()
    assert threads == {"crossby-journal"}


def test_reading_tolerates_only_an_unfinished_last_line(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _run(FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"]), tmp_path, journal)
    journal.close()
    path = tmp_path / "out" / JOURNAL_FILE
    lines = path.read_text().splitlines()

    path.write_text("\n".join(lines) + '\n{"schema_version": 1, "run_id": "run-1", "se')
    assert len(read_journal(path)) == len(lines)

    path.write_text("\n".join([lines[0], "not json", *lines[1:]]) + "\n")
    with pytest.raises(JournalError, match="line 2"):
        read_journal(path)

    path.write_text("\n".join([lines[0], *lines[2:]]) + "\n")
    with pytest.raises(JournalError, match="skips"):
        read_journal(path)


# ---------------------------------------------------------------------------
# Continuing
# ---------------------------------------------------------------------------


def test_continuing_after_the_round_limit_resumes_both_sessions(tmp_path: Path) -> None:
    first = FakeRunner(["q1\nNEXT: ASK", "a1", "q2 anyway\nNEXT: ASK"])
    journal = _journal(tmp_path)
    stopped = _run(first, tmp_path, journal, max_rounds=1)
    journal.close()
    assert stopped.stop_reason is StopReason.ROUND_LIMIT

    events = read_journal(tmp_path / "out" / JOURNAL_FILE)
    prior = result_from_journal(events)
    assert prior.stop_reason is StopReason.ROUND_LIMIT
    assert prior.sessions == {Role.ASKER: "session-1", Role.RESPONDER: "session-2"}

    second = FakeRunner(["a2", "verdict\nNEXT: DONE"])
    resumed = Journal("run-1", path=tmp_path / "out" / JOURNAL_FILE, start_seq=events[-1].seq + 1)
    result = _run(second, tmp_path, resumed, max_rounds=2, resume=prior)
    resumed.close()

    assert result.answered
    assert [turn.text for turn in result.turns] == ["q1", "a1", "q2 anyway", "a2", "verdict"]
    # The responder owed an answer to the asker's last message; both sides resume.
    (responder_tool, responder_request), (_asker_tool, asker_request) = second.calls
    assert responder_tool is AIToolID.CODEX
    assert responder_request.prompt == "q2 anyway"
    assert responder_request.resume_id == "session-2"
    assert asker_request.prompt == "a2"
    assert asker_request.resume_id == "session-1"
    # Every preflight probe still checks the resumed sessions' builds.
    assert len(second.preflights) == 2
    everything = read_journal(tmp_path / "out" / JOURNAL_FILE)
    assert [event.seq for event in everything] == list(range(1, len(everything) + 1))
    assert T.CONVERSATION_CONTINUED in {event.type for event in everything}
    rebuilt = result_from_journal(everything)
    assert rebuilt.answered and rebuilt.max_rounds == 3
    assert render_transcript(rebuilt) == render_transcript(result)


def test_continuing_after_an_interrupted_answer_resends_the_owed_message(tmp_path: Path) -> None:
    first = FakeRunner(["q1\nNEXT: ASK", KeyboardInterrupt()])
    journal = _journal(tmp_path)
    stopped = _run(first, tmp_path, journal)
    journal.close()
    assert stopped.stop_reason is StopReason.INTERRUPTED

    prior = result_from_journal(read_journal(tmp_path / "out" / JOURNAL_FILE))
    second = FakeRunner(["a1", "ok\nNEXT: DONE"])
    result = _run(second, tmp_path, _journal(tmp_path), resume=prior)

    assert result.answered
    assert second.calls[0][1].prompt == "q1"
    # The responder never reported a session, so it starts one.
    assert second.calls[0][1].resume_id is None


def test_an_answered_consultation_cannot_be_continued(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _run(FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"]), tmp_path, journal)
    journal.close()
    prior = result_from_journal(journal.events)
    with pytest.raises(ConverseError, match="already has its final answer"):
        _run(FakeRunner([]), tmp_path, _journal(tmp_path), resume=prior)


def test_a_run_that_never_recorded_its_end_reads_as_interrupted(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _run(FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"]), tmp_path, journal)
    journal.close()
    events = [event for event in journal.events if event.type is not T.CONVERSATION_FINISHED]
    events = [event for event in events if event.data.get("next") != "DONE"]
    prior = result_from_journal(events)
    assert prior.stop_reason is StopReason.INTERRUPTED
    assert not finished(events)
    assert json.loads(events[0].model_dump_json())["type"] == "conversation.started"


@pytest.mark.parametrize("complete", [False, True])
def test_appending_repairs_an_unfinished_last_line(tmp_path: Path, complete: bool) -> None:
    journal = _journal(tmp_path)
    _run(FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"]), tmp_path, journal)
    journal.close()
    path = tmp_path / "out" / JOURNAL_FILE
    lines = path.read_text().splitlines()
    if complete:
        # The last event is whole; only its newline is missing.
        path.write_text("\n".join(lines))
        kept = len(lines)
    else:
        path.write_text("\n".join(lines[:-1]) + '\n{"schema_version": 1, "run_id": "run-1", "se')
        kept = len(lines) - 1
    resumed = Journal("run-1", path=path, start_seq=kept + 1)
    resumed.emit(T.CONVERSATION_CONTINUED, max_rounds=9)
    resumed.close()
    events = read_journal(path)
    assert len(events) == kept + 1
    assert events[-1].type is T.CONVERSATION_CONTINUED
