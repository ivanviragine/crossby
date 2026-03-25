"""crossby sync — sync .crossby.yml config to tool-specific files."""

from __future__ import annotations

from pathlib import Path

import typer

from crossby.ui.console import console


def sync(
    concern: str | None = typer.Argument(
        None,
        help="Sync concern: permissions, rules, mcp, agents. Omit for all.",
    ),
    tool: str | None = typer.Option(
        None, "--tool", "-t", help="Sync only for this tool (e.g. claude, cursor)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without writing any files."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing target directories (backs up first)."
    ),
    path: Path = typer.Option(Path("."), "--path", help="Project root directory."),
) -> None:
    """Sync .crossby.yml config to tool-specific files.

    By default syncs all concerns for all installed tools. Use positional
    CONCERN to restrict to one concern, ``--tool`` to restrict to one tool,
    or ``--dry-run`` to preview without writing.

    Examples::

        crossby sync                        # all concerns, all installed tools
        crossby sync permissions            # permissions only
        crossby sync mcp                    # MCP servers only
        crossby sync --tool claude          # all concerns for Claude only
        crossby sync --tool claude rules    # rules for Claude only
        crossby sync --dry-run              # preview all changes
    """
    from rich.table import Table

    from crossby.ai_tools.base import AbstractAITool
    from crossby.config.loader import load_config
    from crossby.models.ai import AIToolID
    from crossby.sync import run_sync
    from crossby.sync.base import SyncConcern

    start_path = path.resolve()
    config = load_config(start_path)
    project_root = Path(config.project_root).resolve() if config.project_root else start_path

    sync_concern: SyncConcern | None = None
    if concern:
        try:
            sync_concern = SyncConcern(concern.lower())
        except ValueError as err:
            valid = ", ".join(c.value for c in SyncConcern)
            console.error(f"Unknown concern: {concern!r}. Valid values: {valid}")
            raise typer.Exit(1) from err

    sync_tool: AIToolID | None = None
    if tool:
        try:
            sync_tool = AIToolID(tool.lower())
        except ValueError as err:
            console.error(f"Unknown tool: {tool!r}")
            raise typer.Exit(1) from err

    if dry_run:
        console.info("Dry-run mode — no files will be written")

    installed_tools: list[AIToolID] | None = None
    if sync_tool is None:
        installed_tools = AbstractAITool.detect_installed()

    try:
        results = run_sync(
            config,
            project_root,
            tool_id=sync_tool,
            concern=sync_concern,
            dry_run=dry_run,
            force=force,
            installed_tools=installed_tools,
        )
    except ValueError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from exc

    if not results:
        console.info("No sync writers matched the given filters.")
        return

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Tool", style="dim")
    table.add_column("Concern")
    table.add_column("Action")
    table.add_column("Detail", style="dim")

    action_styles = {
        "created": "[success]created[/]",
        "updated": "[success]updated[/]",
        "skipped": "[dim]skipped[/]",
        "error": "[error]error[/]",
    }

    has_error = False
    for result in results:
        styled_action = action_styles.get(result.action, result.action)
        if result.action == "error":
            has_error = True
            detail = result.message or ""
        elif result.file_path:
            try:
                detail = str(result.file_path.relative_to(project_root))
            except ValueError:
                detail = str(result.file_path)
        else:
            detail = result.message or ""
        table.add_row(
            str(result.tool_id) if result.tool_id is not None else "crossby",
            result.concern.value,
            styled_action,
            detail,
        )

    console.out.print(table)

    if has_error:
        raise typer.Exit(1)
