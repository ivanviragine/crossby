"""crossby handoff — cross-tool AI session handoff via a structured markdown file."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import typer

from crossby.ui.console import console

_UNSUPPORTED_SOURCES = {"gemini", "opencode", "antigravity", "vscode"}
_UNSUPPORTED_REASONS = {
    "gemini": "Gemini CLI does not persist full transcripts to disk (see follow-up issue).",
    "opencode": (
        "OpenCode stores sessions in a SQLite database that is not yet supported "
        "(see follow-up issue)."
    ),
    "antigravity": "Antigravity sessions are not on disk in a readable form.",
    "vscode": "VS Code chat sessions are not exposed on disk.",
}

_INITIAL_PROMPT_TEMPLATE = (
    "Read the handoff context at {path} and continue the work described there."
)

# Safety cap: the target's CLI argument must stay well under OS argv limits.
# macOS ARG_MAX is ~256KB; 4KB is a comfortable ceiling for a path-based prompt.
_MAX_INITIAL_MESSAGE_BYTES = 4 * 1024


def handoff(
    from_tool: str | None = typer.Option(
        None, "--from", help="Source tool to read the session from."
    ),
    to_tool: str | None = typer.Option(
        None, "--to", help="Target tool to launch with the handoff as initial prompt."
    ),
    session_id: str | None = typer.Option(
        None, "--session-id", help="Override the latest-session heuristic."
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Write the handoff file to this path instead of the default."
    ),
    no_launch: bool = typer.Option(
        False, "--no-launch", help="Write the handoff file but do not launch the target."
    ),
    summarizer_tool: str | None = typer.Option(
        None,
        "--summarizer-tool",
        help="Tool to run the summarization pass (defaults to the source tool).",
    ),
    token_budget: int = typer.Option(
        32_000, "--token-budget", help="Approximate token budget for the transcript."
    ),
    path: Path = typer.Option(Path("."), "--path", help="Project root directory."),
) -> None:
    """Carry session context from one AI CLI to another via a handoff file.

    Examples::

        crossby handoff --from claude --to codex
        crossby handoff --from cursor --to copilot --no-launch
        crossby handoff --from codex --to claude --session-id 019cb497-ec14-...
    """
    from crossby.ai_tools.base import AbstractAITool
    from crossby.handoff.picker import pick_latest_session
    from crossby.handoff.summarizer import (
        HandoffSummarizer,
        SummarizerParseError,
        SummarizerToolNotInstalled,
    )
    from crossby.handoff.writer import HandoffWriter
    from crossby.models.ai import AIToolID

    project_root = path.resolve()

    # --- Interactive wizard fills in missing flags -------------------------
    if from_tool is None or to_tool is None:
        wizard = _run_wizard(
            from_tool=from_tool,
            to_tool=to_tool,
            project_root=project_root,
        )
        if wizard is None:
            raise typer.Exit(1)
        from_tool, to_tool = wizard

    source_id = _parse_tool_id(from_tool, "from")
    target_id = _parse_tool_id(to_tool, "to")

    if source_id.value in _UNSUPPORTED_SOURCES:
        reason = _UNSUPPORTED_REASONS.get(
            source_id.value,
            f"{source_id} is not yet supported as a handoff source.",
        )
        console.error(reason)
        raise typer.Exit(1)

    # --- Locate source session --------------------------------------------
    source_adapter = AbstractAITool.get(source_id)
    try:
        sessions = source_adapter.locate_sessions(project_root)
    except NotImplementedError:
        console.error(f"{source_id} does not support session handoff as a source.")
        raise typer.Exit(1) from None

    if session_id is not None:
        chosen = next((s for s in sessions if s.session_id == session_id), None)
        if chosen is None:
            console.error(f"No {source_id} session with id '{session_id}' in {project_root}.")
            raise typer.Exit(1)
    else:
        chosen = pick_latest_session(sessions, project_root)
        if chosen is None:
            console.error(
                f"No {source_id} sessions found for {project_root}. "
                "Start a session in this project first, or pass --session-id."
            )
            raise typer.Exit(1)

    console.step(f"Using {source_id} session {chosen.session_id} from {chosen.started_at}")

    # --- Read & summarize -------------------------------------------------
    transcript = source_adapter.read_session(chosen)
    if not transcript.turns:
        console.error(
            f"{source_id} session {chosen.session_id} has no readable turns — nothing to hand off."
        )
        raise typer.Exit(1)

    summarizer_id = _parse_tool_id(summarizer_tool, "summarizer-tool") if summarizer_tool else source_id
    summarizer_adapter = AbstractAITool.get(summarizer_id)
    summarizer = HandoffSummarizer(summarizer_adapter, token_budget=token_budget)

    def _warn_truncation(total: int, kept: int) -> None:
        console.warn(
            f"Transcript truncated: kept {kept} of {total} turns to fit token budget."
        )

    try:
        console.step(f"Summarizing with {summarizer_id} (budget={token_budget} tokens)...")
        doc = summarizer.summarize(
            transcript,
            source_tool=source_id,
            target_tool=target_id,
            on_truncate=_warn_truncation,
        )
    except SummarizerToolNotInstalled as exc:
        console.error(str(exc))
        raise typer.Exit(1) from None
    except SummarizerParseError as exc:
        console.error(f"Summarizer failed: {exc}")
        raise typer.Exit(1) from None

    # --- Write markdown --------------------------------------------------
    writer = HandoffWriter(project_root)
    handoff_path = writer.write(doc, output_path=output)
    console.success(f"Wrote handoff file: {handoff_path}")

    if no_launch:
        console.info("--no-launch set; target tool not launched.")
        return

    # --- Launch target ---------------------------------------------------
    target_adapter = AbstractAITool.get(target_id)
    prompt = _INITIAL_PROMPT_TEMPLATE.format(path=handoff_path)
    encoded_len = len(prompt.encode("utf-8"))
    if encoded_len > _MAX_INITIAL_MESSAGE_BYTES:
        console.error(
            f"Initial-message prompt is {encoded_len} bytes, exceeding safety cap "
            f"of {_MAX_INITIAL_MESSAGE_BYTES}. This should never happen — please file a bug."
        )
        raise typer.Exit(1)
    args = target_adapter.initial_message_args(prompt)
    if not args:
        console.warn(
            f"{target_id} has no initial-message support; handoff file written but "
            "target cannot be launched with the context pre-loaded. Open it manually: "
            f"{handoff_path}"
        )
        return

    cmd = target_adapter.build_launch_command(initial_message=prompt)
    console.step(f"Launching {target_id}: {shlex.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
    except FileNotFoundError:
        console.error(f"{target_id} binary not found in PATH.")
        raise typer.Exit(1) from None
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


def _parse_tool_id(value: str, flag: str):  # type: ignore[no-untyped-def]
    from crossby.models.ai import AIToolID

    try:
        return AIToolID(value)
    except ValueError:
        valid = ", ".join(t.value for t in AIToolID)
        console.error(f"Unknown tool for --{flag}: {value!r}. Valid: {valid}")
        raise typer.Exit(1) from None


def _run_wizard(
    from_tool: str | None,
    to_tool: str | None,
    project_root: Path,
) -> tuple[str, str] | None:
    """Prompt for missing ``--from``/``--to`` values."""
    from crossby.ai_tools.base import AbstractAITool
    from crossby.ui import prompts

    installed = [str(t) for t in AbstractAITool.detect_installed()]
    if not installed:
        console.error("No AI tools detected in PATH. Install a supported tool first.")
        return None

    console.step(f"Handoff wizard — project root: {project_root}")
    console.detail(f"Detected tools: {', '.join(installed)}")

    sources = [t for t in installed if t not in _UNSUPPORTED_SOURCES]
    if not sources:
        console.error("None of the installed tools are supported as handoff sources yet.")
        return None

    if from_tool is None:
        idx = prompts.select("Source tool (read session from)?", items=sources)
        from_tool = sources[idx]

    if to_tool is None:
        targets = [t for t in installed if t != from_tool]
        if not targets:
            console.error("No alternate installed tool to hand off to.")
            return None
        idx = prompts.select("Target tool (launch with handoff prompt)?", items=targets)
        to_tool = targets[idx]

    if not prompts.confirm(f"Hand off {from_tool} → {to_tool}?", default=True):
        console.info("Cancelled.")
        return None
    return from_tool, to_tool
