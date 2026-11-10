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

    The question is sent as is. The agent runs unattended and read-only in the
    project: it can read and search but not change files, unexpected questions
    fail the session and unapproved permissions are denied. Each call is a
    fresh session, so include the context the agent needs in the question.

    Examples::

        crossby ask codex "Is there a race in src/cache.py invalidate()?"
        git diff | crossby ask claude -
    """
    from crossby.services.converse import ConverseError, ask_agent

    project_root = _project_root(path)
    tool_id = _headless_tool(tool, "TOOL")
    prompt = _read_text_input(
        question, question_file, what="question", hint="an argument, '-' for stdin, or --file"
    )
    try:
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
    asker_tool: str = typer.Argument(
        ..., metavar="ASKER", help="Agent that holds the goal and asks (e.g. claude)."
    ),
    responder_tool: str = typer.Argument(
        ..., metavar="RESPONDER", help="Agent that answers (e.g. codex)."
    ),
    goal: str | None = typer.Option(
        None, "--goal", "-g", help="What the asker should find out. Only the asker sees it."
    ),
    goal_file: Path | None = typer.Option(
        None, "--goal-file", help="Read the goal from this file instead."
    ),
    max_rounds: int = typer.Option(
        5, "--max-rounds", help="Most questions the asker may send before it must answer."
    ),
    timeout: float = typer.Option(
        1800.0, "--timeout", help="Seconds for the whole consultation (summary excluded)."
    ),
    turn_timeout: float = typer.Option(600.0, "--turn-timeout", help="Seconds per message."),
    asker_model: str | None = typer.Option(None, "--asker-model", help="Model for the asker."),
    responder_model: str | None = typer.Option(
        None, "--responder-model", help="Model for the responder."
    ),
    summary: bool = typer.Option(
        False, "--summary", help="Also write a neutral summary.md of the whole consultation."
    ),
    summarizer_tool: str | None = typer.Option(
        None,
        "--summarizer-tool",
        help="Tool that writes the --summary (defaults to the asker's tool).",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory for transcript.md (default: .crossby/conversations/<timestamp>/).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON result object."),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Do not print messages as they arrive."
    ),
    responder_path: Path | None = typer.Option(
        None,
        "--responder-path",
        help="Folder the responder runs in, with its own files and setup (default: --path).",
    ),
    path: Path = typer.Option(Path("."), "--path", help="Project root directory (the asker's)."),
) -> None:
    """Let one AI coding agent consult another, headless, until it has its answer.

    The ASKER holds the goal. It decides what to ask, judges each answer,
    asks follow-up questions, and ends with its final answer. The RESPONDER
    is never sent the goal; it gets the asker's messages as written and only
    answers, from its own folder (--responder-path), read-only: it can read and
    search but not change files. Each side keeps its own session, resumed every
    turn and sent only the newest message, whole; a CLI build whose resume was
    not verified is refused before anything starts. The asker's final answer
    goes to stdout; messages and progress go to stderr; transcript.md is saved
    after every message.

    Exit codes: 0 answered, 2 stopped by the round or time limit, 1 failed,
    130 interrupted.

    Examples::

        crossby converse claude codex --goal "Check whether this diagnosis is sound: ..."
        crossby converse claude cursor --goal-file goal.md --max-rounds 3 --json
    """
    from crossby.services import converse as engine

    project_root = _project_root(path)
    responder_root = (
        _project_root(responder_path, "--responder-path") if responder_path is not None else None
    )
    asker_id = _headless_tool(asker_tool, "ASKER")
    responder_id = _headless_tool(responder_tool, "RESPONDER")
    goal_text = _read_text_input(goal, goal_file, what="goal", hint="--goal or --goal-file")
    summarizer_id: AIToolID | None = None
    if summary:
        summarizer_id = (
            _headless_tool(summarizer_tool, "--summarizer-tool")
            if summarizer_tool is not None
            else asker_id
        )
    asker, responder = engine.make_participants(
        asker_id, responder_id, asker_model=asker_model, responder_model=responder_model
    )

    started_at = datetime.now(tz=UTC)
    if output_dir is not None:
        out_dir = output_dir.resolve()
    else:
        out_dir = engine.conversation_dir(project_root, started_at)
    transcript_path = out_dir / "transcript.md"
    completed: list[ConverseTurn] = []

    def on_turn_start(number: int, speaker: Participant) -> None:
        if not quiet:
            _stderr(f"[step.count]\\[{number}][/] {speaker.label} is writing…")

    def on_turn(turn: ConverseTurn) -> None:
        # Persist before printing, so a crash or Ctrl-C keeps every completed turn.
        completed.append(turn)
        partial = engine.render_partial_transcript(
            goal=goal_text,
            asker=asker,
            responder=responder,
            max_rounds=max_rounds,
            started_at=started_at,
            turns=completed,
        )
        try:
            _write(transcript_path, partial)
        except OSError as exc:
            # The previous transcript is intact (the write is atomic), and the
            # next turn or the final write tries again; don't end the consultation.
            console.warn(f"Could not save the transcript after turn {turn.number}: {exc}")
        if not quiet:
            _show_turn(turn)

    try:
        result = engine.run_conversation(
            goal_text,
            (asker, responder),
            working_dir=project_root,
            responder_dir=responder_root,
            max_rounds=max_rounds,
            timeout_seconds=timeout,
            turn_timeout_seconds=turn_timeout,
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


def _project_root(path: Path, flag: str = "--path") -> Path:
    root = path.resolve()
    if not root.is_dir():
        console.error(f"{flag} is not a directory: {root}")
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
    """Replace ``path`` atomically, so a failed write keeps the previous content.

    The transcript is rewritten after every turn; a truncating write that
    failed part-way would erase every turn saved so far.
    """
    from crossby.config.json_utils import atomic_write_text

    # On POSIX, invalid bytes in argv (e.g. --goal) arrive as lone surrogates,
    # which strict UTF-8 cannot encode; replace them rather than fail.
    atomic_write_text(path, text.encode("utf-8", "replace").decode("utf-8"))


def _show_turn(turn: ConverseTurn) -> None:
    from crossby.services.converse import turn_footer

    sys.stderr.write(f"\n{turn.text or '(empty message)'}\n\n")
    title = "final answer" if turn.is_final_answer else f"{turn.role.value} ({turn.tool.value})"
    _stderr(f"[dim]      {_escape(f'{title} · {turn_footer(turn)}')}[/]")
    sys.stderr.flush()


def _side_json(result: ConversationResult, participant: Participant) -> dict[str, Any]:
    """One side's tool, model and session facts for ``--json``."""
    own = [turn for turn in result.turns if turn.role is participant.role]
    return {
        "tool": participant.tool.value,
        "model": participant.model,
        # The native session every turn after the first resumed.
        "session_id": result.sessions.get(participant.role),
        "denials": [denial for turn in own for denial in turn.denials],
    }


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

    if not quiet:
        _stderr(f"  [step]▸[/] Summarizing with {tool.value}…")
    try:
        body = summarize_conversation(
            result,
            transcript_path=transcript_path,
            tool=tool,
            working_dir=project_root,
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
                "answered": result.answered,
                "final_answer": result.final_answer,
                "outcome": describe_stop(result),
                "error": result.error,
                "rounds": result.rounds,
                "max_rounds": result.max_rounds,
                "turns": len(result.turns),
                "asker": _side_json(result, result.asker),
                "responder": _side_json(result, result.responder),
                "transcript_path": str(transcript_path),
                "summary_path": str(summary_path) if summary_path else None,
                "summary": summary_body,
                "summary_error": summary_error,
            }
        )
        return

    # stdout carries the result: the asker's final answer, and the summary when asked.
    if result.final_answer is not None:
        print(result.final_answer)
    if summary_body is not None:
        if result.final_answer is not None:
            print()
        print(summary_body.strip())

    outcome = describe_stop(result)
    if result.answered:
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
