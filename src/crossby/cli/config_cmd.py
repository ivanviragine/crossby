"""crossby config — display resolved configuration."""

from __future__ import annotations

import typer

from crossby.ui.console import console

config_app = typer.Typer(help="Configuration commands.", no_args_is_help=True)


@config_app.command("show")
def show() -> None:
    """Display the resolved CROSSBY configuration.

    Reads .crossby.yml from the current directory (or parent directories)
    and displays the resolved configuration.
    """
    from crossby.config.loader import find_config_file, load_config

    config_path = find_config_file()
    if not config_path:
        console.warn("No .crossby.yml found.")
        console.hint("Run 'crossby init' to create one.")
        raise typer.Exit(1)

    config = load_config()

    console.header("CROSSBY Configuration")
    console.kv("Config file", str(config_path))
    console.empty()

    # AI defaults
    rows: list[tuple[str, str]] = []
    if config.ai.default_tool:
        rows.append(("Default tool", config.ai.default_tool))
    if config.ai.default_model:
        rows.append(("Default model", config.ai.default_model))
    if config.ai.effort:
        rows.append(("Default effort", config.ai.effort))
    if config.ai.yolo is not None:
        rows.append(("YOLO mode", "on" if config.ai.yolo else "off"))

    if rows:
        console.summary_table(rows, title="AI Defaults")
        console.empty()

    # Per-command overrides
    if config.ai.commands:
        cmd_rows: list[tuple[str, str]] = []
        for cmd_name, cmd_config in config.ai.commands.items():
            parts: list[str] = []
            if cmd_config.tool:
                parts.append(f"tool={cmd_config.tool}")
            if cmd_config.model:
                parts.append(f"model={cmd_config.model}")
            if cmd_config.effort:
                parts.append(f"effort={cmd_config.effort}")
            if cmd_config.yolo is not None:
                parts.append(f"yolo={'on' if cmd_config.yolo else 'off'}")
            if parts:
                cmd_rows.append((cmd_name, ", ".join(parts)))
        if cmd_rows:
            console.summary_table(cmd_rows, title="Command Overrides")
            console.empty()

    # Model mappings
    if config.models:
        for tool_name, mapping in config.models.items():
            model_rows: list[tuple[str, str]] = []
            if mapping.easy:
                model_rows.append(("easy", mapping.easy))
            if mapping.medium:
                model_rows.append(("medium", mapping.medium))
            if mapping.complex:
                model_rows.append(("complex", mapping.complex))
            if mapping.very_complex:
                model_rows.append(("very_complex", mapping.very_complex))
            if model_rows:
                console.summary_table(model_rows, title=f"Models: {tool_name}")
                console.empty()

    # Permissions
    if config.permissions.allowed_commands:
        console.section("Permissions")
        for cmd in config.permissions.allowed_commands:
            console.detail(cmd)
