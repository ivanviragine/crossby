"""crossby launch — launch an AI tool with resolved config."""

from __future__ import annotations

from pathlib import Path

import typer

from crossby.ui.console import console


def launch(
    path: Path = typer.Argument(Path("."), help="Working directory for the AI tool."),
    tool: str | None = typer.Option(None, "--tool", "-t", help="AI tool to use."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model to use."),
    effort: str | None = typer.Option(None, "--effort", "-e", help="Effort level."),
    yolo: bool | None = typer.Option(None, "--yolo", help="Skip permission prompts."),
    command: str | None = typer.Option(
        None, "--command", "-c", help="Command name for config lookup."
    ),
    prompt: str | None = typer.Option(None, "--prompt", "-p", help="Initial prompt to send."),
    complexity: str | None = typer.Option(
        None, "--complexity", help="Task complexity for model selection."
    ),
    transcript: Path | None = typer.Option(
        None, "--transcript", help="Path to save session transcript."
    ),
) -> None:
    """Launch an AI tool with resolved configuration.

    Resolves tool, model, effort, and yolo from CLI flags, config file,
    and auto-detection. Optionally captures session transcript.
    """
    from crossby.ai_tools.base import AbstractAITool
    from crossby.config.loader import load_config
    from crossby.services.ai_resolution import (
        confirm_ai_selection,
        resolve_ai_tool,
        resolve_effort,
        resolve_model,
        resolve_yolo,
    )
    from crossby.services.prompt_delivery import deliver_prompt_if_needed

    work_dir = path.resolve()
    config = load_config(work_dir)

    # Resolve AI selection
    resolved_tool = resolve_ai_tool(tool, config, command or "default")
    if not resolved_tool:
        console.error("No AI tool specified or detected.")
        console.hint("Install an AI tool or specify --tool")
        raise typer.Exit(1)

    resolved_model = resolve_model(
        model, config, command or "default", tool=resolved_tool, complexity=complexity
    )
    resolved_effort = resolve_effort(effort, config, command or "default", tool=resolved_tool)
    resolved_yolo = resolve_yolo(yolo, config, command or "default", tool=resolved_tool)

    # Interactive confirmation
    resolved_tool, resolved_model, resolved_effort, resolved_yolo = confirm_ai_selection(
        resolved_tool,
        resolved_model,
        tool_explicit=tool is not None,
        model_explicit=model is not None,
        resolved_effort=resolved_effort,
        effort_explicit=effort is not None,
        resolved_yolo=resolved_yolo,
        yolo_explicit=yolo is not None,
    )

    if not resolved_tool:
        console.error("No AI tool selected.")
        raise typer.Exit(1)

    adapter = AbstractAITool.get(resolved_tool)
    caps = adapter.capabilities()

    # Display selection
    console.kv("AI tool", caps.display_name)
    if resolved_model:
        console.kv("Model", resolved_model)
    if resolved_effort:
        console.kv("Effort", resolved_effort.value)
    if resolved_yolo:
        console.kv("YOLO mode", "on")
    console.empty()

    # Deliver prompt if tool doesn't support initial messages
    if prompt:
        deliver_prompt_if_needed(adapter, prompt)

    # Build allowed commands from config
    allowed_commands = (
        config.permissions.allowed_commands if config.permissions.allowed_commands else None
    )

    # Launch
    exit_code = adapter.launch(
        working_dir=work_dir,
        model=resolved_model,
        prompt=prompt if caps.supports_initial_message else None,
        transcript_path=transcript,
        effort=resolved_effort,
        allowed_commands=allowed_commands,
        yolo=resolved_yolo,
    )

    if exit_code != 0:
        console.warn(f"AI tool exited with code {exit_code}")

    # Parse transcript if captured
    if transcript and transcript.exists():
        usage = adapter.parse_transcript(transcript)
        if usage.total_tokens:
            console.kv("Tokens", f"{usage.total_tokens:,}")
        if usage.session_id:
            console.kv("Session ID", usage.session_id)

    raise typer.Exit(exit_code)
