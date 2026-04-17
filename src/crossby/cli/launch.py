"""crossby launch — launch an AI tool with resolved config."""

from __future__ import annotations

from pathlib import Path

import typer

from crossby.ui.console import console


def launch(
    path: Path = typer.Argument(Path("."), help="Working directory or profile name."),
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
    profile: str | None = typer.Option(
        None, "--profile", "-P", help="Named launch profile from .crossby.yml."
    ),
    resume: str | None = typer.Option(
        None, "--resume", help="Resume a previous session by ID."
    ),
    trusted_dirs: list[str] = typer.Option(
        [], "--trusted-dir", help="Pre-authorize a directory (repeatable)."
    ),
) -> None:
    """Launch an AI tool with resolved configuration.

    Resolves tool, model, effort, and yolo from CLI flags, profiles,
    config file, and auto-detection. Works without any config file.

    Examples::

        crossby launch                     # auto-detect everything
        crossby launch --tool claude       # specific tool
        crossby launch --profile ccyolo    # use saved profile
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
    from crossby.utils.process import run_with_transcript

    # Apply profile overrides (--profile or positional profile name)
    profile_name = profile
    path_str = str(path)
    if (
        not profile_name
        and not path.exists()
        and path_str != "."
        and not path.is_absolute()
        and len(path.parts) == 1
    ):
        # Simple name without path structure — treat as profile name
        profile_name = path_str
        work_dir = Path(".").resolve()
    else:
        work_dir = path.resolve()

    config = load_config(work_dir)

    if profile_name:
        prof = config.get_profile(profile_name)
        if prof is None:
            console.error(f"Unknown profile: {profile_name!r}")
            console.hint("Check .crossby.yml profiles section")
            raise typer.Exit(1)
        # Profile values serve as defaults — explicit CLI flags take precedence
        if tool is None and prof.tool:
            tool = prof.tool
        if model is None and prof.model:
            model = prof.model
        if effort is None and prof.effort:
            effort = prof.effort
        if yolo is None and prof.yolo is not None:
            yolo = prof.yolo

    # Resolve AI selection
    resolved_tool = resolve_ai_tool(tool, config, command or "default")
    if not resolved_tool:
        console.error("No AI tool specified or detected.")
        console.hint("Install an AI tool or specify --tool")
        raise typer.Exit(1)

    try:
        resolved_model = resolve_model(
            model,
            config,
            command or "default",
            tool=resolved_tool,
            complexity=complexity,
            strict=model is not None,
        )
        resolved_effort = resolve_effort(
            effort,
            config,
            command or "default",
            tool=resolved_tool,
            strict=effort is not None,
        )
        resolved_yolo = resolve_yolo(
            yolo,
            config,
            command or "default",
            tool=resolved_tool,
            strict=yolo is not None,
        )
    except ValueError as e:
        console.error(str(e))
        raise typer.Exit(1) from e

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

    try:
        adapter = AbstractAITool.get(resolved_tool)
    except (ValueError, KeyError) as e:
        console.error(str(e))
        raise typer.Exit(1) from e
    caps = adapter.capabilities()

    # --resume path: build and run the resume command, then exit early
    if resume:
        if not caps.supports_resume:
            console.error(f"{caps.display_name} does not support session resume.")
            raise typer.Exit(1)
        resume_cmd = adapter.build_resume_command(resume)
        if resume_cmd is None:
            console.error(
                f"{caps.display_name}.build_resume_command returned None despite supports_resume=True."
            )
            raise typer.Exit(1)
        console.kv("AI tool", caps.display_name)
        console.kv("Session", resume)
        console.empty()
        if transcript:
            try:
                transcript.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                console.error(f"Cannot create transcript directory: {e}")
                raise typer.Exit(1) from e
        exit_code = run_with_transcript(resume_cmd, transcript, cwd=work_dir)
        if exit_code != 0:
            console.warn(f"AI tool exited with code {exit_code}")
        if transcript and transcript.exists():
            usage = adapter.parse_transcript(transcript)
            if usage.total_tokens:
                console.kv("Tokens", f"{usage.total_tokens:,}")
            if usage.session_id:
                console.kv("Session ID", usage.session_id)
        raise typer.Exit(exit_code)

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

    # Ensure transcript parent directory exists
    if transcript:
        try:
            transcript.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            console.error(f"Cannot create transcript directory: {e}")
            raise typer.Exit(1) from e

    # Launch
    exit_code = adapter.launch(
        working_dir=work_dir,
        model=resolved_model,
        prompt=prompt if caps.supports_initial_message else None,
        transcript_path=transcript,
        trusted_dirs=trusted_dirs if trusted_dirs else None,
        effort=resolved_effort,
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
