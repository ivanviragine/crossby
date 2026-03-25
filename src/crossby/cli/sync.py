"""crossby sync — sync .crossby.yml sections to tool-specific config files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from crossby.ui.console import console

sync_app = typer.Typer(
    name="sync",
    help="Sync .crossby.yml sections to tool-specific config files.",
    no_args_is_help=True,
)

_VALID_TOOLS = {"claude", "cursor", "copilot", "gemini", "codex"}


@sync_app.command("mcp")
def sync_mcp(
    project_path: Annotated[
        Path,
        typer.Argument(help="Project directory (defaults to current directory)."),
    ] = Path("."),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview changes without writing files."),
    ] = False,
    tool: Annotated[
        str | None,
        typer.Option("--tool", help="Sync only this tool (claude/cursor/copilot/gemini/codex)."),
    ] = None,
) -> None:
    """Sync MCP server configuration from .crossby.yml to all installed tools."""
    from crossby.config.loader import ConfigError, load_config
    from crossby.sync.mcp import MCP_WRITERS

    project_root = project_path.resolve()

    if tool is not None and tool not in _VALID_TOOLS:
        console.error(f"Unknown tool '{tool}'. Valid: {', '.join(sorted(_VALID_TOOLS))}")
        raise typer.Exit(1)

    try:
        config = load_config(project_root)
    except ConfigError as e:
        console.error(str(e))
        raise typer.Exit(1)

    if not config.mcp_servers:
        console.warn("No mcp_servers defined in .crossby.yml — nothing to sync.")
        console.hint("Add an mcp_servers: section to .crossby.yml to get started.")
        raise typer.Exit(0)

    if dry_run:
        console.step("[dim](dry run — no files will be written)[/]")

    writers = list(MCP_WRITERS.values())
    if tool is not None:
        writers = [w for w in writers if w.tool_id == tool]

    any_written = False
    any_error = False

    for writer in writers:
        results = writer.write(config.mcp_servers, project_root, dry_run=dry_run)
        for result in results:
            rel = result.path.relative_to(project_root) if result.path.is_relative_to(project_root) else result.path
            if result.action == "skipped":
                console.info(f"[dim]{result.tool}[/]  {rel}  (already up to date)")
            elif result.action == "error":
                console.warn(f"{result.tool}  {rel}  skipped — {result.message}")
                any_error = True
            else:
                verb = "would write" if dry_run else result.action
                console.success(f"{result.tool}  {rel}  {verb}")
                any_written = True

    if not any_written and not any_error:
        console.info("All MCP configs are already up to date.")
    elif any_written and not dry_run:
        console.hint("Run 'crossby sync mcp --dry-run' to preview future changes.")
