"""CLI tests for ``crossby ask`` and ``crossby converse``.

The headless runner and the summarizer are replaced with fakes, so no native
CLI or model is ever started. Every consultation here resumes both sessions
unless a test says otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID, HeadlessTerminalStatus
from crossby.services import converse
from crossby.services.converse import DEPTH_ENV
from tests.unit.test_services.test_converse import FakeRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_inherited_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEPTH_ENV, raising=False)


def _use_runner(monkeypatch: pytest.MonkeyPatch, fake: FakeRunner) -> FakeRunner:
    monkeypatch.setattr(converse, "HeadlessTurnRunner", lambda: fake)
    return fake


def _fake_summary(
    monkeypatch: pytest.MonkeyPatch,
    body: str = "## Outcome\nThe responder knows the mechanism.",
    error: BaseException | None = None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake(result: Any, **kwargs: Any) -> str:
        calls.append({"result": result, **kwargs})
        if error is not None:
            raise error
        return body

    monkeypatch.setattr(converse, "summarize_conversation", fake)
    return calls


CONSULTING = [
    "What is a patent foramen ovale?\nNEXT: ASK",
    "A flap-like opening between the atria.",
    "Verdict: it knows the anatomy.\nNEXT: DONE",
]


def _invoke(tmp_path: Path, *args: str, input: str | None = None) -> Any:
    return runner.invoke(app, [*args, "--path", str(tmp_path)], input=input)


# ---------------------------------------------------------------------------
# converse
# ---------------------------------------------------------------------------


def test_converse_prints_the_final_answer_and_writes_the_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(CONSULTING))
    summaries = _fake_summary(monkeypatch)

    result = _invoke(tmp_path, "converse", "claude", "codex", "--goal", "Check the PFO answer")

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "Verdict: it knows the anatomy."
    assert "asker (claude) is writing" in result.stderr
    assert "responder (codex) is writing" in result.stderr
    assert "A flap-like opening between the atria." in result.stderr
    assert "NEXT: ASK" in result.stderr
    assert "answered: the asker gave its final answer after 1 round" in result.stderr
    assert summaries == []  # the summary is opt-in

    (run_dir,) = (tmp_path / ".crossby" / "conversations").iterdir()
    transcript = (run_dir / "transcript.md").read_text(encoding="utf-8")
    assert "## Goal (sent only to the asker)\n\n```text\nCheck the PFO answer\n```" in transcript
    assert "## Turn 3 · final answer (claude)\n\nVerdict: it knows the anatomy." in transcript
    assert not (run_dir / "summary.md").exists()
    assert [tool for tool, _req in fake.calls] == [
        AIToolID.CLAUDE,
        AIToolID.CODEX,
        AIToolID.CLAUDE,
    ]
    assert fake.calls[2][1].resume_id == "session-1"


def test_converse_opens_with_who_consults_whom_and_where_the_transcript_goes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_runner(monkeypatch, FakeRunner(CONSULTING))
    result = _invoke(tmp_path, "converse", "claude", "codex", "--goal", "g", "--max-rounds", "3")
    assert result.exit_code == 0, result.output
    head = result.stderr.splitlines()[:3]
    assert "claude asks codex (read-only, in the asker's folder)" in head[0]
    assert "limits: up to 3 messages, 30m in all" in head[1]
    assert "transcript: " in head[2] and head[2].endswith("transcript.md")


def test_quiet_converse_prints_no_opening(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_runner(monkeypatch, FakeRunner(CONSULTING))
    result = _invoke(tmp_path, "converse", "claude", "codex", "--goal", "g", "-q")
    assert result.exit_code == 0, result.output
    assert "asks codex" not in result.stderr
    assert result.stdout.strip() == "Verdict: it knows the anatomy."


def test_the_final_answer_is_not_repeated_where_it_was_just_shown(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from crossby.cli import converse as cli

    finished = converse.ConversationResult(
        goal="g",
        asker=converse.Participant(AIToolID.CLAUDE, converse.Role.ASKER),
        responder=converse.Participant(AIToolID.CODEX, converse.Role.RESPONDER),
        max_rounds=5,
        started_at=converse.datetime(2026, 11, 10, tzinfo=converse.UTC),
        finished_at=converse.datetime(2026, 11, 10, tzinfo=converse.UTC),
        stop_reason=converse.StopReason.ANSWERED,
        turns=[
            converse.ConverseTurn(
                1, converse.Role.ASKER, AIToolID.CLAUDE, "Final.", converse.Next.DONE, 1.0
            )
        ],
    )
    common = {
        "transcript_path": tmp_path / "t.md",
        "summary_path": None,
        "summary_body": None,
        "summary_error": None,
        "as_json": False,
    }
    cli._report(finished, answer_shown=True, **common)
    assert capsys.readouterr().out == ""
    cli._report(finished, answer_shown=False, **common)
    assert capsys.readouterr().out == "Final.\n"


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0s"), (42.9, "42s"), (60, "1m"), (185, "3m 05s"), (1800, "30m"), (3720, "1h 02m")],
)
def test_durations_read_naturally(seconds: float, text: str) -> None:
    from crossby.cli import converse as cli

    assert cli._duration(seconds) == text


def test_the_live_wait_line_is_a_no_op_off_a_terminal(capsys: pytest.CaptureFixture[str]) -> None:
    from crossby.cli import converse as cli

    with cli._waiting("Asking codex…"):
        pass
    wait = cli._TurnWait()
    wait.stop()  # stopping before or twice is safe
    wait.stop()
    assert capsys.readouterr().err == ""


def test_converse_summary_is_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_runner(monkeypatch, FakeRunner(CONSULTING))
    summaries = _fake_summary(monkeypatch)

    result = _invoke(tmp_path, "converse", "claude", "codex", "-g", "x", "-q", "--summary")

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == (
        "Verdict: it knows the anatomy.\n\n## Outcome\nThe responder knows the mechanism."
    )
    (call,) = summaries
    assert call["tool"] is AIToolID.CLAUDE
    (run_dir,) = (tmp_path / ".crossby" / "conversations").iterdir()
    assert (run_dir / "summary.md").read_text(encoding="utf-8").startswith("# Consultation summary")


def test_converse_round_limit_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_runner(monkeypatch, FakeRunner(["q1\nNEXT: ASK", "a1", "q2\nNEXT: ASK"]))
    result = _invoke(tmp_path, "converse", "claude", "codex", "-g", "x", "--max-rounds", "1")
    assert result.exit_code == 2
    assert result.stdout.strip() == ""
    assert "round limit reached (1 round) and the asker gave no final answer" in result.stderr


def test_converse_marker_only_answer_is_not_reported_as_answered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_runner(monkeypatch, FakeRunner(["q\nNEXT: ASK", "a", "NEXT: DONE"]))
    result = _invoke(tmp_path, "converse", "claude", "codex", "-g", "x", "-q")
    assert result.exit_code == 1
    assert result.stdout.strip() == ""
    assert "sent only the NEXT marker, with no final answer" in result.stderr


def test_converse_json_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_runner(monkeypatch, FakeRunner(CONSULTING))
    out_dir = tmp_path / "out"

    result = _invoke(
        tmp_path,
        "converse",
        "claude",
        "codex",
        "-g",
        "x",
        "--json",
        "--quiet",
        "--responder-model",
        "gpt-x",
        "--output-dir",
        str(out_dir),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "answered"
    assert payload["answered"] is True
    assert payload["final_answer"] == "Verdict: it knows the anatomy."
    assert payload["rounds"] == 1
    assert payload["turns"] == 3
    assert payload["asker"] == {
        "tool": "claude",
        "model": None,
        "session_id": "session-1",
        "denials": [],
    }
    assert payload["responder"] == {
        "tool": "codex",
        "model": "gpt-x",
        "session_id": "session-2",
        "denials": [],
    }
    assert payload["transcript_path"] == str(out_dir.resolve() / "transcript.md")
    assert payload["summary_path"] is None
    assert "is writing" not in result.stderr


def test_converse_summary_failure_is_a_warning_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_runner(monkeypatch, FakeRunner(CONSULTING))
    _fake_summary(monkeypatch, error=converse.ConverseSummaryError("tool exited 1"))

    result = _invoke(
        tmp_path, "converse", "claude", "codex", "-g", "x", "-q", "--json", "--summary"
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["summary"] is None
    assert payload["summary_error"] == "tool exited 1"
    assert "Summary failed: tool exited 1" in result.stderr


def test_converse_failure_exits_1_and_keeps_the_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_runner(monkeypatch, FakeRunner(["q\nNEXT: ASK", HeadlessTerminalStatus.FAILED]))
    result = _invoke(tmp_path, "converse", "claude", "codex", "-g", "x", "-q")
    assert result.exit_code == 1
    assert "responder (codex) turn 2 ended with status failed" in result.stderr
    (run_dir,) = (tmp_path / ".crossby" / "conversations").iterdir()
    transcript = (run_dir / "transcript.md").read_text(encoding="utf-8")
    assert "## Turn 1 · asker (claude) → responder" in transcript


def test_converse_persists_each_turn_before_the_next_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    seen: list[str] = []

    class Peeking(FakeRunner):
        def run(self, tool, request, cancel_event):  # type: ignore[no-untyped-def]
            transcript = out_dir / "transcript.md"
            seen.append(transcript.read_text(encoding="utf-8") if transcript.exists() else "")
            return super().run(tool, request, cancel_event)

    _use_runner(monkeypatch, Peeking(CONSULTING))
    result = _invoke(
        tmp_path, "converse", "claude", "codex", "-g", "x", "-q", "--output-dir", str(out_dir)
    )

    assert result.exit_code == 0, result.output
    assert seen[0] == ""
    assert "## Turn 1 · asker (claude) → responder\n\nWhat is a patent foramen ovale?" in seen[1]
    assert "- Outcome: in progress: 1 turn completed" in seen[1]
    assert "## Turn 2 · responder (codex) → asker" in seen[2]
    final = (out_dir / "transcript.md").read_text(encoding="utf-8")
    assert "- Outcome: answered" in final


def test_failed_transcript_update_keeps_the_previous_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from crossby.cli import converse as cli
    from crossby.config import json_utils

    path = tmp_path / "run" / "transcript.md"
    cli._write(path, "turn 1 saved\n")

    def failing_replace(_src: object, _dst: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(json_utils.os, "replace", failing_replace)
    with pytest.raises(OSError, match="No space left"):
        cli._write(path, "turn 1 saved\nturn 2 half-writ")

    assert path.read_text(encoding="utf-8") == "turn 1 saved\n"
    assert [p.name for p in path.parent.iterdir()] == ["transcript.md"]


def test_write_that_fails_part_way_keeps_the_previous_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os

    from crossby.cli import converse as cli
    from crossby.config import json_utils

    path = tmp_path / "transcript.md"
    cli._write(path, "turn 1 saved\n")
    real_fdopen = os.fdopen

    class HalfWritten:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> HalfWritten:
            return self

        def __exit__(self, *exc: object) -> None:
            self.handle.close()

        def write(self, text: str) -> int:
            self.handle.write(text[: len(text) // 2])
            raise OSError(5, "Input/output error")

    monkeypatch.setattr(
        json_utils.os, "fdopen", lambda fd, *a, **k: HalfWritten(real_fdopen(fd, *a, **k))
    )
    with pytest.raises(OSError, match="Input/output error"):
        cli._write(path, "turn 1 saved\nturn 2 saved\n")

    assert path.read_text(encoding="utf-8") == "turn 1 saved\n"
    assert [p.name for p in tmp_path.iterdir()] == ["transcript.md"]


def test_transcript_write_replaces_unencodable_characters(tmp_path: Path) -> None:
    # On POSIX, invalid bytes in argv (e.g. --goal) arrive as lone surrogates.
    from crossby.cli import converse as cli

    path = tmp_path / "transcript.md"
    cli._write(path, "**Goal:** fix caf\udce9 bug\n")
    assert path.read_text(encoding="utf-8") == "**Goal:** fix caf? bug\n"


def test_failed_per_turn_save_warns_and_the_consultation_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from crossby.cli import converse as cli

    _use_runner(monkeypatch, FakeRunner(CONSULTING))
    real_write = cli._write
    calls = {"n": 0}

    def flaky_write(path: Path, text: str) -> None:
        calls["n"] += 1
        if calls["n"] == 2:  # the save after turn 2
            raise OSError(28, "No space left on device")
        real_write(path, text)

    monkeypatch.setattr(cli, "_write", flaky_write)
    result = _invoke(tmp_path, "converse", "claude", "codex", "-g", "x", "-q")

    assert result.exit_code == 0, result.output
    assert "Could not save the transcript after turn 2" in result.stderr
    (run_dir,) = (tmp_path / ".crossby" / "conversations").iterdir()
    final = (run_dir / "transcript.md").read_text(encoding="utf-8")
    assert "## Turn 3 · final answer (claude)" in final


def test_converse_ctrl_c_while_printing_keeps_the_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from crossby.cli import converse as cli

    _use_runner(monkeypatch, FakeRunner(CONSULTING))

    def interrupted_print(turn: Any, **_kwargs: Any) -> None:
        if turn.number == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_show_turn", interrupted_print)
    result = _invoke(tmp_path, "converse", "claude", "codex", "-g", "x")

    assert result.exit_code == 130
    (run_dir,) = (tmp_path / ".crossby" / "conversations").iterdir()
    transcript = (run_dir / "transcript.md").read_text(encoding="utf-8")
    assert "## Turn 2 · responder (codex) → asker\n\nA flap-like opening" in transcript
    assert "- Outcome: interrupted by the user after 1 round" in transcript


def test_converse_ctrl_c_during_summary_exits_130_with_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_runner(monkeypatch, FakeRunner(CONSULTING))
    _fake_summary(monkeypatch, error=KeyboardInterrupt())

    result = _invoke(
        tmp_path, "converse", "claude", "codex", "-g", "x", "-q", "--json", "--summary"
    )

    assert result.exit_code == 130
    payload = json.loads(result.stdout)
    assert payload["status"] == "answered"
    assert payload["summary_error"] == "interrupted"
    (run_dir,) = (tmp_path / ".crossby" / "conversations").iterdir()
    assert (run_dir / "transcript.md").exists()
    assert not (run_dir / "summary.md").exists()


def test_converse_runs_the_responder_in_its_own_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(CONSULTING))
    elsewhere = tmp_path / "other-project"
    elsewhere.mkdir()
    result = _invoke(
        tmp_path,
        "converse",
        "claude",
        "codex",
        "-g",
        "x",
        "-q",
        "--responder-path",
        str(elsewhere),
    )
    assert result.exit_code == 0, result.output
    assert [req.working_dir for _tool, req in fake.calls] == [
        tmp_path.resolve(),
        elsewhere.resolve(),
        tmp_path.resolve(),
    ]
    # The transcript stays with the asker's project.
    assert (tmp_path / ".crossby" / "conversations").is_dir()
    assert not (elsewhere / ".crossby").exists()


def test_converse_rejects_a_missing_responder_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(CONSULTING))
    result = _invoke(
        tmp_path,
        "converse",
        "claude",
        "codex",
        "-g",
        "x",
        "--responder-path",
        str(tmp_path / "nope"),
    )
    assert result.exit_code == 1
    assert "--responder-path is not a directory" in result.stderr
    assert fake.calls == []


def test_converse_reads_goal_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(CONSULTING))
    goal_file = tmp_path / "goal.md"
    goal_file.write_text("Decide whether the PFO diagnosis is sound\n", encoding="utf-8")

    result = _invoke(tmp_path, "converse", "claude", "codex", "--goal-file", str(goal_file), "-q")

    assert result.exit_code == 0, result.output
    assert "Goal: Decide whether the PFO diagnosis is sound" in fake.calls[0][1].prompt
    assert "Decide whether" not in fake.calls[1][1].prompt


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["converse", "claude", "codex"], "A goal is required"),
        (["converse", "claude", "codex", "-g", "   "], "The goal is empty"),
        (["converse", "nope", "codex", "-g", "x"], "Unknown tool 'nope' for ASKER"),
        (
            ["converse", "claude", "vscode", "-g", "x"],
            "vscode can't run headless sessions, so it can't be the RESPONDER",
        ),
        (["converse", "claude", "codex", "-g", "x", "--max-rounds", "0"], "at least 1"),
        (
            ["converse", "claude", "codex", "-g", "x", "--summary", "--summarizer-tool", "nope"],
            "Unknown tool 'nope' for --summarizer-tool",
        ),
    ],
)
def test_converse_rejects_bad_input_without_starting_agents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: list[str], message: str
) -> None:
    fake = _use_runner(monkeypatch, FakeRunner([]))
    result = _invoke(tmp_path, *args)
    assert result.exit_code == 1
    assert message in result.stderr
    assert fake.calls == []


def test_converse_refuses_to_nest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(CONSULTING))
    monkeypatch.setenv(DEPTH_ENV, "1")
    result = _invoke(tmp_path, "converse", "claude", "codex", "-g", "x")
    assert result.exit_code == 1
    assert "Nested consultations are disabled" in result.stderr
    assert fake.calls == []


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def test_ask_prints_answer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(["It is a race."]))
    result = _invoke(tmp_path, "ask", "codex", "Is there a race?")
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "It is a race."
    assert fake.calls[0][1].prompt == "Is there a race?"


def test_ask_reads_stdin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(["Looks fine."]))
    result = _invoke(tmp_path, "ask", "claude", "-", input="diff --git a b\n")
    assert result.exit_code == 0, result.output
    assert fake.calls[0][1].prompt == "diff --git a b"


def test_ask_json_and_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_runner(monkeypatch, FakeRunner([HeadlessTerminalStatus.TIMED_OUT]))
    result = _invoke(tmp_path, "ask", "codex", "q", "--json")
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "timed_out"
    assert payload["error"] == "codex session ended with status timed_out"


def test_ask_requires_a_question(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_runner(monkeypatch, FakeRunner([]))
    result = _invoke(tmp_path, "ask", "codex")
    assert result.exit_code == 1
    assert "A question is required" in result.stderr


def test_ask_refuses_to_nest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(["x"]))
    monkeypatch.setenv(DEPTH_ENV, "1")
    result = _invoke(tmp_path, "ask", "codex", "q")
    assert result.exit_code == 1
    assert fake.calls == []
