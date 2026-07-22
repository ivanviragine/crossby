"""crossby stats — parse session transcripts for token usage."""

from __future__ import annotations

from pathlib import Path

import typer

from crossby.ui.console import console


def stats(
    transcript_path: Path = typer.Argument(help="Path to session transcript file."),
    tool: str | None = typer.Option(
        None, "--tool", "-t", help="AI tool that produced the transcript."
    ),
) -> None:
    """Parse a session transcript and display token usage statistics.

    Automatically detects the transcript format (Claude, Copilot, Codex, or
    generic). Optionally specify --tool for tool-specific parsing.
    """
    from crossby.ai_tools.base import AbstractAITool
    from crossby.ai_tools.transcript import (
        format_count,
        parse_copilot_transcript,
        parse_transcript_common,
    )

    if not transcript_path.exists():
        console.error(f"File not found: {transcript_path}")
        raise typer.Exit(1)

    # Use tool-specific parser if specified
    if tool:
        try:
            adapter = AbstractAITool.get(tool)
            usage = adapter.parse_transcript(transcript_path)
        except ValueError as err:
            console.error(f"Unknown AI tool: {tool}")
            raise typer.Exit(1) from err
    elif "copilot" in transcript_path.name.lower():
        # Heuristic: use copilot parser for copilot-named files
        usage = parse_copilot_transcript(transcript_path)
    else:
        usage = parse_transcript_common(transcript_path)

    if not usage.total_tokens and not usage.input_tokens and not usage.output_tokens:
        console.warn("No token usage data found in transcript.")
        raise typer.Exit(0)

    # Display results
    console.header("Session Statistics")

    rows: list[tuple[str, str]] = []
    if usage.total_tokens:
        rows.append(("Total tokens", format_count(usage.total_tokens)))
    if usage.input_tokens:
        rows.append(("Input tokens", format_count(usage.input_tokens)))
    if usage.output_tokens:
        rows.append(("Output tokens", format_count(usage.output_tokens)))
    if usage.cached_tokens:
        rows.append(("Cached tokens", format_count(usage.cached_tokens)))
    if usage.premium_requests:
        rows.append(("Premium requests", str(usage.premium_requests)))
    if usage.session_id:
        rows.append(("Session ID", usage.session_id))

    console.summary_table(rows, title="Token Usage")

    # Model breakdown
    if usage.model_breakdown:
        console.empty()
        breakdown_rows: list[tuple[str, str]] = []
        for mb in usage.model_breakdown:
            tokens = f"in:{format_count(mb.input_tokens)} out:{format_count(mb.output_tokens)}"
            if mb.cached_tokens:
                tokens += f" cached:{format_count(mb.cached_tokens)}"
            if mb.premium_requests:
                tokens += f" ({mb.premium_requests} premium)"
            breakdown_rows.append((mb.model, tokens))
        console.summary_table(breakdown_rows, title="Per-Model Breakdown")
