"""CLI tests for ``crossby ask`` and ``crossby converse``.

The headless runner and the summarizer are replaced with fakes, so no native
CLI or model is ever started.
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
    body: str = "## Outcome\nThey agreed on Y.",
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


AGREEING = [
    "Cause is X.\nSTATUS: CONTINUE",
    "Agreed, fix with Y.\nSTATUS: DONE",
    "Yes.\nSTATUS: DONE",
]


# ---------------------------------------------------------------------------
# converse
# ---------------------------------------------------------------------------


def test_converse_agrees_writes_files_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(AGREEING))
    summaries = _fake_summary(monkeypatch)

    result = runner.invoke(
        app, ["converse", "codex", "claude", "--goal", "Fix it", "--path", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "## Outcome\nThey agreed on Y."
    assert "codex is replying" in result.stderr
    assert "Agreed, fix with Y." in result.stderr
    assert "agreed: both agents declared the goal met after 3 turns" in result.stderr

    (run_dir,) = (tmp_path / ".crossby" / "conversations").iterdir()
    transcript = (run_dir / "transcript.md").read_text(encoding="utf-8")
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "**Goal:** Fix it" in transcript
    assert "## Turn 2: claude\n\nAgreed, fix with Y." in transcript
    assert summary.startswith("# Conversation summary")
    assert "They agreed on Y." in summary

    assert [tool for tool, _req in fake.calls] == [
        AIToolID.CODEX,
        AIToolID.CLAUDE,
        AIToolID.CODEX,
    ]
    (call,) = summaries
    assert call["tool"] is AIToolID.CODEX
    assert call["transcript_path"] == run_dir / "transcript.md"
    assert call["working_dir"] == tmp_path.resolve()


def test_converse_turn_limit_exits_2_and_still_summarizes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_runner(monkeypatch, FakeRunner(["a\nSTATUS: CONTINUE", "b\nSTATUS: CONTINUE"]))
    summaries = _fake_summary(monkeypatch, body="## Outcome\nNo agreement.")

    result = runner.invoke(
        app,
        ["converse", "codex", "claude", "-g", "x", "--max-turns", "2", "--path", str(tmp_path)],
    )

    assert result.exit_code == 2
    assert "No agreement." in result.stdout
    assert "turn limit reached (2 turns) without agreement" in result.stderr
    assert len(summaries) == 1


def test_converse_json_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_runner(monkeypatch, FakeRunner(AGREEING))
    _fake_summary(monkeypatch)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "converse",
            "codex",
            "claude",
            "-g",
            "x",
            "--json",
            "--quiet",
            "--model-b",
            "opus-y",
            "--output-dir",
            str(out_dir),
            "--path",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "agreed"
    assert payload["agreed"] is True
    assert payload["turns"] == 3
    assert payload["participants"] == [
        {"label": "codex", "tool": "codex", "model": None},
        {"label": "claude", "tool": "claude", "model": "opus-y"},
    ]
    assert payload["transcript_path"] == str(out_dir.resolve() / "transcript.md")
    assert payload["summary_path"] == str(out_dir.resolve() / "summary.md")
    assert payload["summary"].startswith("## Outcome")
    assert payload["summary_error"] is None
    assert "is replying" not in result.stderr


def test_converse_no_summary_prints_last_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_runner(monkeypatch, FakeRunner(AGREEING))
    summaries = _fake_summary(monkeypatch)

    result = runner.invoke(
        app,
        ["converse", "codex", "claude", "-g", "x", "-q", "--no-summary", "--path", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "Yes."
    assert summaries == []
    (run_dir,) = (tmp_path / ".crossby" / "conversations").iterdir()
    assert not (run_dir / "summary.md").exists()


def test_converse_summary_failure_is_a_warning_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_runner(monkeypatch, FakeRunner(AGREEING))
    _fake_summary(monkeypatch, error=converse.ConverseSummaryError("tool exited 1"))

    result = runner.invoke(
        app, ["converse", "codex", "claude", "-g", "x", "-q", "--json", "--path", str(tmp_path)]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["summary"] is None
    assert payload["summary_path"] is None
    assert payload["summary_error"] == "tool exited 1"
    assert "Summary failed: tool exited 1" in result.stderr


def test_converse_failure_exits_1_and_keeps_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_runner(monkeypatch, FakeRunner(["a\nSTATUS: CONTINUE", HeadlessTerminalStatus.FAILED]))
    summaries = _fake_summary(monkeypatch)

    result = runner.invoke(
        app, ["converse", "codex", "claude", "-g", "x", "-q", "--path", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "claude turn 2 ended with status failed" in result.stderr
    (run_dir,) = (tmp_path / ".crossby" / "conversations").iterdir()
    assert "## Turn 1: codex" in (run_dir / "transcript.md").read_text(encoding="utf-8")
    assert len(summaries) == 1


def test_converse_interrupt_skips_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_runner(monkeypatch, FakeRunner(["a\nSTATUS: CONTINUE", KeyboardInterrupt()]))
    summaries = _fake_summary(monkeypatch)

    result = runner.invoke(
        app, ["converse", "codex", "claude", "-g", "x", "-q", "--path", str(tmp_path)]
    )

    assert result.exit_code == 130
    assert summaries == []


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

    _use_runner(monkeypatch, Peeking(AGREEING))
    _fake_summary(monkeypatch)
    result = runner.invoke(
        app,
        [
            "converse",
            "codex",
            "claude",
            "-g",
            "x",
            "-q",
            "--output-dir",
            str(out_dir),
            "--path",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen[0] == ""
    assert "## Turn 1: codex\n\nCause is X." in seen[1]
    assert "- Outcome: in progress: 1 turn completed" in seen[1]
    assert "## Turn 2: claude" in seen[2]
    final = (out_dir / "transcript.md").read_text(encoding="utf-8")
    assert "- Outcome: agreed" in final


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


def test_failed_per_turn_save_warns_and_the_conversation_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from crossby.cli import converse as cli

    _use_runner(monkeypatch, FakeRunner(AGREEING))
    _fake_summary(monkeypatch)
    real_write = cli._write
    calls = {"n": 0}

    def flaky_write(path: Path, text: str) -> None:
        calls["n"] += 1
        if calls["n"] == 2:  # the save after turn 2
            raise OSError(28, "No space left on device")
        real_write(path, text)

    monkeypatch.setattr(cli, "_write", flaky_write)
    result = runner.invoke(
        app, ["converse", "codex", "claude", "-g", "x", "-q", "--path", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "Could not save the transcript after turn 2" in result.stderr
    (run_dir,) = (tmp_path / ".crossby" / "conversations").iterdir()
    final = (run_dir / "transcript.md").read_text(encoding="utf-8")
    assert "## Turn 3: codex" in final
    assert "- Outcome: agreed" in final


def test_converse_ctrl_c_while_printing_keeps_the_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from crossby.cli import converse as cli

    _use_runner(monkeypatch, FakeRunner(AGREEING))
    summaries = _fake_summary(monkeypatch)

    def interrupted_print(turn: Any) -> None:
        if turn.number == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_show_turn", interrupted_print)
    result = runner.invoke(app, ["converse", "codex", "claude", "-g", "x", "--path", str(tmp_path)])

    assert result.exit_code == 130
    assert summaries == []
    (run_dir,) = (tmp_path / ".crossby" / "conversations").iterdir()
    transcript = (run_dir / "transcript.md").read_text(encoding="utf-8")
    assert "## Turn 2: claude\n\nAgreed, fix with Y." in transcript
    assert "- Outcome: interrupted by the user after 2 turns" in transcript


def test_converse_ctrl_c_during_summary_exits_130_with_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_runner(monkeypatch, FakeRunner(AGREEING))
    _fake_summary(monkeypatch, error=KeyboardInterrupt())  # type: ignore[arg-type]

    result = runner.invoke(
        app, ["converse", "codex", "claude", "-g", "x", "-q", "--json", "--path", str(tmp_path)]
    )

    assert result.exit_code == 130
    payload = json.loads(result.stdout)
    assert payload["status"] == "agreed"
    assert payload["summary_error"] == "interrupted"
    (run_dir,) = (tmp_path / ".crossby" / "conversations").iterdir()
    assert (run_dir / "transcript.md").exists()
    assert not (run_dir / "summary.md").exists()


def test_converse_summarizer_tool_must_run_headless(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(AGREEING))
    result = runner.invoke(
        app,
        [
            "converse",
            "codex",
            "claude",
            "-g",
            "x",
            "--summarizer-tool",
            "vscode",
            "--path",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "vscode cannot run a managed headless session" in result.stderr
    assert fake.calls == []


def test_converse_reads_goal_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(AGREEING))
    _fake_summary(monkeypatch)
    goal_file = tmp_path / "goal.md"
    goal_file.write_text("Decide between A and B\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "converse",
            "codex",
            "claude",
            "--goal-file",
            str(goal_file),
            "-q",
            "--path",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Goal: Decide between A and B" in fake.calls[0][1].prompt


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["converse", "codex", "claude"], "A goal is required"),
        (["converse", "codex", "claude", "-g", "   "], "The goal is empty"),
        (["converse", "nope", "claude", "-g", "x"], "Unknown tool for TOOL_A"),
        (["converse", "codex", "vscode", "-g", "x"], "vscode cannot run a managed headless"),
        (["converse", "codex", "claude", "-g", "x", "--max-turns", "1"], "at least 2"),
        (
            ["converse", "codex", "claude", "-g", "x", "--summarizer-tool", "nope"],
            "Unknown tool for --summarizer-tool",
        ),
    ],
)
def test_converse_rejects_bad_input_without_starting_agents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: list[str], message: str
) -> None:
    fake = _use_runner(monkeypatch, FakeRunner([]))
    result = runner.invoke(app, [*args, "--path", str(tmp_path)])
    assert result.exit_code == 1
    assert message in result.stderr
    assert fake.calls == []


def test_converse_refuses_to_nest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(AGREEING))
    monkeypatch.setenv(DEPTH_ENV, "1")
    result = runner.invoke(app, ["converse", "codex", "claude", "-g", "x", "--path", str(tmp_path)])
    assert result.exit_code == 1
    assert "Nested consultations are disabled" in result.stderr
    assert fake.calls == []


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def test_ask_prints_answer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(["It is a race."]))
    result = runner.invoke(app, ["ask", "codex", "Is there a race?", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "It is a race."
    assert fake.calls[0][1].prompt.endswith("Question:\n\nIs there a race?")


def test_ask_reads_stdin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(["Looks fine."]))
    result = runner.invoke(
        app, ["ask", "claude", "-", "--path", str(tmp_path)], input="diff --git a b\n"
    )
    assert result.exit_code == 0, result.output
    assert fake.calls[0][1].prompt.endswith("Question:\n\ndiff --git a b")


def test_ask_json_and_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_runner(monkeypatch, FakeRunner([HeadlessTerminalStatus.TIMED_OUT]))
    result = runner.invoke(app, ["ask", "codex", "q", "--json", "--path", str(tmp_path)])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "timed_out"
    assert payload["error"] == "codex session ended with status timed_out"


def test_ask_requires_a_question(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_runner(monkeypatch, FakeRunner([]))
    result = runner.invoke(app, ["ask", "codex", "--path", str(tmp_path)])
    assert result.exit_code == 1
    assert "A question is required" in result.stderr


def test_ask_refuses_to_nest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _use_runner(monkeypatch, FakeRunner(["x"]))
    monkeypatch.setenv(DEPTH_ENV, "1")
    result = runner.invoke(app, ["ask", "codex", "q", "--path", str(tmp_path)])
    assert result.exit_code == 1
    assert fake.calls == []
