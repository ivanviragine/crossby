"""crossby ask / crossby converse — headless agent-to-agent consultation.

Both commands are non-interactive: they never prompt, write progress to
stderr, and reserve stdout for the result so they can be piped or scripted.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from crossby.ui.console import console

if TYPE_CHECKING:
    from crossby.models.ai import AIToolID
    from crossby.services.converse import ConversationResult, ConverseTurn, Participant


def ask(
    tool: str = typer.Argument(..., help="Tool to ask (e.g. codex, claude, cursor)."),
    question: str | None = typer.Argument(
        None, help="The question. Use '-' to read it from stdin."
    ),
    question_file: Path | None = typer.Option(
        None, "--file", help="Read the question from this file instead."
    ),
    model: str | None = typer.Option(None, "--model", help="Model for the asked tool."),
    timeout: float = typer.Option(
        600.0, "--timeout", help="Seconds before the session is stopped."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON result object."),
    path: Path = typer.Option(Path("."), "--path", help="Project root directory."),
) -> None:
    """Ask another AI coding agent one question, headless, and print its answer.

    The agent runs unattended in the project: unexpected questions fail the
    session and unapproved permissions are denied. The question is framed
    with the same rules as a conversation turn: read and check, but do not
    modify files. Each call is a fresh session, so include the context the
    agent needs in the question.

    Examples::

        crossby ask codex "Is there a race in src/cache.py invalidate()?"
        git diff | crossby ask claude -
    """
    from crossby.services.converse import ConverseError, ask_agent, check_depth

    project_root = _project_root(path)
    tool_id = _headless_tool(tool, "TOOL")
    prompt = _read_text_input(
        question, question_file, what="question", hint="an argument, '-' for stdin, or --file"
    )
    try:
        check_depth()
        result = ask_agent(
            tool_id, prompt, working_dir=project_root, model=model, timeout_seconds=timeout
        )
    except ConverseError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from None

    if as_json:
        _print_json(
            {
                "tool": tool_id.value,
                "status": result.status.value if result.status else "failed",
                "answer": result.text,
                "error": result.error,
                "duration_seconds": round(result.duration_seconds, 1),
            }
        )
    elif result.text:
        print(result.text)
    if not result.succeeded:
        if not as_json:
            console.error(result.error or f"{tool_id.value} did not answer.")
        raise typer.Exit(1)


def converse(
    tool_a: str = typer.Argument(..., help="Agent that speaks first (e.g. codex)."),
    tool_b: str = typer.Argument(..., help="Agent that replies (e.g. claude)."),
    goal: str | None = typer.Option(
        None, "--goal", "-g", help="What the two agents should achieve together."
    ),
    goal_file: Path | None = typer.Option(
        None, "--goal-file", help="Read the goal from this file instead."
    ),
    max_turns: int = typer.Option(6, "--max-turns", help="Maximum replies across both agents."),
    timeout: float = typer.Option(
        1800.0, "--timeout", help="Seconds for the whole conversation (summary excluded)."
    ),
    turn_timeout: float = typer.Option(600.0, "--turn-timeout", help="Seconds per reply."),
    model_a: str | None = typer.Option(None, "--model-a", help="Model for the first agent."),
    model_b: str | None = typer.Option(None, "--model-b", help="Model for the second agent."),
    history_chars: int = typer.Option(
        80_000,
        "--history-chars",
        help="Character budget for the conversation replayed into each turn.",
    ),
    summarizer_tool: str | None = typer.Option(
        None,
        "--summarizer-tool",
        help="Tool that writes the final summary (defaults to the first agent).",
    ),
    no_summary: bool = typer.Option(False, "--no-summary", help="Skip the final summary."),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory for transcript.md and summary.md "
        "(default: .crossby/conversations/<timestamp>/).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON result object."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Do not print turns as they arrive."),
    path: Path = typer.Option(Path("."), "--path", help="Project root directory."),
) -> None:
    """Have two AI coding agents work toward a goal in a headless back-and-forth.

    Each agent ends every reply with STATUS: CONTINUE or STATUS: DONE. The
    conversation stops when both agree in consecutive replies, or on the turn
    or time limit. A neutral summary of the whole conversation is then written
    next to the transcript and printed to stdout; progress goes to stderr.

    Exit codes: 0 agreed, 2 stopped by the turn or time limit, 1 failed,
    130 interrupted.

    Examples::

        crossby converse codex claude --goal "Find why test_sync_hooks is flaky and agree on a fix"
        crossby converse claude cursor --goal-file goal.md --max-turns 4 --json
    """
    from crossby.services import converse as engine

    project_root = _project_root(path)
    participant_ids = (_headless_tool(tool_a, "TOOL_A"), _headless_tool(tool_b, "TOOL_B"))
    goal_text = _read_text_input(goal, goal_file, what="goal", hint="--goal or --goal-file")
    summarizer_id: AIToolID | None = None
    if not no_summary:
        summarizer_id = (
            _headless_tool(summarizer_tool, "--summarizer-tool")
            if summarizer_tool is not None
            else participant_ids[0]
        )
    participants = engine.make_participants(*participant_ids, model_a=model_a, model_b=model_b)

    started_at = datetime.now(tz=UTC)
    if output_dir is not None:
        out_dir = output_dir.resolve()
    else:
        out_dir = engine.conversation_dir(project_root, started_at)
    transcript_path = out_dir / "transcript.md"
    completed: list[ConverseTurn] = []

    def on_turn_start(number: int, speaker: Participant) -> None:
        if not quiet:
            _stderr(f"[step.count]\\[{number}/{max_turns}][/] {speaker.label} is replying…")

    def on_turn(turn: ConverseTurn) -> None:
        # Persist before printing, so a crash or Ctrl-C keeps every completed turn.
        completed.append(turn)
        _write(
            transcript_path,
            engine.render_partial_transcript(
                goal=goal_text,
                participants=participants,
                max_turns=max_turns,
                started_at=started_at,
                turns=completed,
            ),
        )
        if not quiet:
            _show_turn(turn)

    try:
        engine.check_depth()
        result = engine.run_conversation(
            goal_text,
            participants,
            working_dir=project_root,
            max_turns=max_turns,
            timeout_seconds=timeout,
            turn_timeout_seconds=turn_timeout,
            history_chars=history_chars,
            on_turn_start=on_turn_start,
            on_turn=on_turn,
        )
    except engine.ConverseError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from None

    _write(transcript_path, engine.render_transcript(result))

    summary_path: Path | None = None
    summary_body: str | None = None
    summary_error: str | None = None
    exit_code = result.exit_code
    if summarizer_id is not None and result.turns and not _interrupted(result):
        summary_body, summary_error, interrupted = _summarize(
            result, transcript_path, summarizer_id, project_root, quiet=quiet
        )
        if interrupted:
            exit_code = engine.EXIT_CODES[engine.StopReason.INTERRUPTED]
        if summary_body is not None:
            summary_path = out_dir / "summary.md"
            _write(summary_path, engine.render_summary(result, summary_body))

    _report(
        result,
        transcript_path=transcript_path,
        summary_path=summary_path,
        summary_body=summary_body,
        summary_error=summary_error,
        as_json=as_json,
    )
    if exit_code:
        raise typer.Exit(exit_code)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_root(path: Path) -> Path:
    root = path.resolve()
    if not root.is_dir():
        console.error(f"--path is not a directory: {root}")
        raise typer.Exit(1)
    return root


def _headless_tool(value: str, name: str) -> AIToolID:
    from crossby.ai_tools.base import AbstractAITool
    from crossby.models.ai import AIToolID

    usable = sorted(
        tid.value
        for tid in AbstractAITool.available_tools()
        if AbstractAITool.get(tid).capabilities().headless.managed_supported
    )
    try:
        tool_id = AIToolID(value)
    except ValueError:
        console.error(f"Unknown tool for {name}: {value!r}. Valid: {', '.join(usable)}")
        raise typer.Exit(1) from None
    if tool_id.value not in usable:
        console.error(
            f"{tool_id.value} cannot run a managed headless session. Choose one of: "
            f"{', '.join(usable)}"
        )
        raise typer.Exit(1)
    return tool_id


def _read_text_input(
    value: str | None,
    file: Path | None,
    *,
    what: str,
    hint: str,
) -> str:
    if value is not None and file is not None:
        console.error(f"Pass the {what} once, as {hint}; not both.")
        raise typer.Exit(1)
    if file is not None:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as exc:
            console.error(f"Cannot read {what} file {file}: {exc}")
            raise typer.Exit(1) from None
    elif value == "-":
        text = sys.stdin.read()
    elif value is not None:
        text = value
    else:
        console.error(f"A {what} is required: pass it as {hint}.")
        raise typer.Exit(1)
    if not text.strip():
        console.error(f"The {what} is empty.")
        raise typer.Exit(1)
    return text.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _show_turn(turn: ConverseTurn) -> None:
    sys.stderr.write(f"\n{turn.text or '(empty reply)'}\n\n")
    _stderr(
        f"[dim]      {turn.speaker} · STATUS: {turn.status.value} · {turn.duration_seconds:.0f}s[/]"
    )
    sys.stderr.flush()


def _interrupted(result: ConversationResult) -> bool:
    from crossby.services.converse import StopReason

    return result.stop_reason is StopReason.INTERRUPTED


def _summarize(
    result: ConversationResult,
    transcript_path: Path,
    tool: AIToolID,
    project_root: Path,
    *,
    quiet: bool,
) -> tuple[str | None, str | None, bool]:
    """Return ``(summary, error, interrupted)``; the transcript is already written."""
    from crossby.services.converse import ConverseSummaryError, summarize_conversation

    def warn_truncation(total: int, kept: int) -> None:
        console.warn(f"Summary input truncated: kept the last {kept} of {total} turns.")

    if not quiet:
        _stderr(f"  [step]▸[/] Summarizing with {tool.value}…")
    try:
        body = summarize_conversation(
            result,
            transcript_path=transcript_path,
            tool=tool,
            working_dir=project_root,
            on_truncate=warn_truncation,
        )
    except ConverseSummaryError as exc:
        console.warn(f"Summary failed: {exc}")
        return None, str(exc), False
    except KeyboardInterrupt:
        console.warn("Summary interrupted; the transcript is saved.")
        return None, "interrupted", True
    return body, None, False


def _report(
    result: ConversationResult,
    *,
    transcript_path: Path,
    summary_path: Path | None,
    summary_body: str | None,
    summary_error: str | None,
    as_json: bool,
) -> None:
    from crossby.services.converse import describe_stop

    if as_json:
        _print_json(
            {
                "status": result.stop_reason.value,
                "agreed": result.agreed,
                "outcome": describe_stop(result),
                "error": result.error,
                "turns": len(result.turns),
                "max_turns": result.max_turns,
                "participants": [
                    {"label": p.label, "tool": p.tool.value, "model": p.model}
                    for p in result.participants
                ],
                "transcript_path": str(transcript_path),
                "summary_path": str(summary_path) if summary_path else None,
                "summary": summary_body,
                "summary_error": summary_error,
            }
        )
        return

    # stdout carries the result: the summary, or the last reply without one.
    if summary_body is not None:
        print(summary_body.strip())
    elif result.turns:
        print(result.turns[-1].text)

    outcome = describe_stop(result)
    if result.agreed:
        _stderr(f"  [success]✓[/] {_escape(outcome)}")
    elif result.exit_code == 1:
        _stderr(f"  [error]✗ Error:[/] {_escape(outcome)}")
    else:
        _stderr(f"  [warning]![/] {_escape(outcome)}")
    _stderr(f"[dim]      transcript: {_escape(str(transcript_path))}[/]")
    if summary_path is not None:
        _stderr(f"[dim]      summary:    {_escape(str(summary_path))}[/]")


def _stderr(markup: str) -> None:
    # soft_wrap keeps long paths on one line so they can be copied.
    console.err.print(markup, soft_wrap=True)


def _escape(text: str) -> str:
    from rich.markup import escape

    return escape(text)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))
