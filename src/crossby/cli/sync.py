"""crossby sync — sync tool-specific files from a canonical source."""

from __future__ import annotations

from pathlib import Path

import typer

from crossby.ui.console import console

sync_app = typer.Typer(
    name="sync",
    help="Sync tool-specific files from a canonical source.",
    no_args_is_help=True,
)


@sync_app.command()
def rules(
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without writing."),
    force: bool = typer.Option(False, "--force", help="Overwrite pre-existing unmanaged files."),
    tool: str | None = typer.Option(
        None, "--tool", help="Sync rules for a specific tool only (claude, cursor, copilot, gemini, codex)."
    ),
) -> None:
    """Sync project instructions to all configured AI tool formats."""
    from crossby.config.loader import load_config
    from crossby.models.config import RulesConfig
    from crossby.sync.base import SyncAction
    from crossby.sync.rules import sync_rules

    config = load_config()
    project_root = Path(config.project_root) if config.project_root else Path.cwd()

    rules_config = config.rules
    if rules_config is None:
        console.error("No 'rules' section in .crossby.yml")
        console.hint("Run 'crossby init' or add a rules section to .crossby.yml")
        raise typer.Exit(1)

    if tool and tool not in ("claude", "cursor", "copilot", "gemini", "codex"):
        console.error(f"Unknown tool: {tool}")
        console.hint("Valid tools: claude, cursor, copilot, gemini, codex")
        raise typer.Exit(1)

    if dry_run:
        console.step("Dry run — no files will be written")

    results = sync_rules(
        project_root,
        rules_config,
        dry_run=dry_run,
        force=force,
        tool_filter=tool,
    )

    has_errors = False
    for r in results:
        if r.action == SyncAction.CREATED:
            console.success(f"{r.target} — created{' (dry run)' if r.dry_run else ''}")
        elif r.action == SyncAction.UPDATED:
            console.success(f"{r.target} — updated{' (dry run)' if r.dry_run else ''}")
        elif r.action == SyncAction.UP_TO_DATE:
            console.info(f"{r.target} — up to date")
        elif r.action == SyncAction.SKIPPED:
            console.warn(f"{r.target} — skipped: {r.message}")
        elif r.action == SyncAction.ERROR:
            console.error(f"{r.target} — {r.message}")
            has_errors = True

    if has_errors:
        raise typer.Exit(1)
