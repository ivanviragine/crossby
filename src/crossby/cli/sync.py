"""crossby sync — stateless sync wizard that reads directly from tool configs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from crossby.ui.console import console

if TYPE_CHECKING:
    from crossby.sync.base import SyncResult


def sync(
    concern: str | None = typer.Argument(
        None,
        help="Sync concern: permissions, rules, mcp, agents, hooks. Omit for all.",
    ),
    from_tool: str | None = typer.Option(
        None, "--from", help="Source tool to read configs from."
    ),
    to_tool: str | None = typer.Option(
        None, "--to", help="Target tool to sync to (default: all installed)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without writing any files."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing target files (backs up first)."
    ),
    path: Path = typer.Option(Path("."), "--path", help="Project root directory."),
) -> None:
    """Sync AI tool configs across tools — no config file needed.

    Reads directly from tool config files (instruction files, MCP servers,
    hooks, permissions, agent dirs) and ports them to other installed tools.

    In interactive mode (default), presents a wizard showing what was found
    and asks for confirmation per concern.

    Examples::

        crossby sync                          # interactive wizard
        crossby sync --from claude            # read from Claude, sync to all
        crossby sync --from claude --to cursor  # Claude → Cursor only
        crossby sync rules                    # rules concern only
        crossby sync mcp --from claude        # MCP from Claude only
    """
    from crossby.ai_tools.base import AbstractAITool
    from crossby.models.ai import AIToolID
    from crossby.sync import run_sync
    from crossby.sync.base import SyncConcern, SyncData
    from crossby.sync.readers import (
        build_sync_data,
        discover_hooks,
        discover_mcp,
        discover_permissions,
        scan_project,
        suggest_agents_source,
        suggest_rules_source,
        suggest_skills_source,
    )

    project_root = path.resolve()

    # Validate concern argument
    sync_concern: SyncConcern | None = None
    if concern:
        try:
            sync_concern = SyncConcern(concern)
        except ValueError:
            valid = ", ".join(c.value for c in SyncConcern)
            console.error(f"Unknown concern: {concern!r}. Valid values: {valid}")
            raise typer.Exit(1) from None

    # Validate from/to arguments
    source_tool: AIToolID | None = None
    if from_tool:
        try:
            source_tool = AIToolID(from_tool)
        except ValueError:
            console.error(f"Unknown tool: {from_tool!r}")
            raise typer.Exit(1) from None

    target_tool: AIToolID | None = None
    if to_tool:
        try:
            target_tool = AIToolID(to_tool)
        except ValueError:
            console.error(f"Unknown tool: {to_tool!r}")
            raise typer.Exit(1) from None

    if target_tool is not None and source_tool is None:
        console.error("--to requires --from; omit --to for the interactive wizard.")
        raise typer.Exit(1)

    # Detect installed tools
    installed_tools = AbstractAITool.detect_installed()
    if not installed_tools:
        console.error("No AI tools found in PATH.")
        console.hint("Install at least one AI tool (claude, copilot, gemini, codex, cursor, etc.)")
        raise typer.Exit(1)

    if dry_run:
        console.info("Dry-run mode — no files will be written")

    results: list[SyncResult] = []

    # Non-interactive mode: --from is specified
    if source_tool is not None:
        data = build_sync_data(project_root, from_tool=source_tool)
        target_tools = (
            [target_tool]
            if target_tool
            else [t for t in installed_tools if t != source_tool]
        )
        results = run_sync(
            data,
            project_root,
            tool_id=target_tool,
            concern=sync_concern,
            dry_run=dry_run,
            force=force,
            installed_tools=target_tools,
        )
        _display_results(results)
        if any(r.action == "error" for r in results):
            raise typer.Exit(1)
        return

    # Interactive wizard mode
    scan = scan_project(project_root, installed_tools)

    console.step(f"Detected tools: {', '.join(str(t) for t in installed_tools)}")
    console.empty()
    console.step("Scanning configs...")
    console.detail(f"  Rules:       {scan.rules.summary}")
    console.detail(f"  Agents:      {scan.agents.summary}")
    console.detail(f"  Skills:      {scan.skills.summary}")
    console.detail(f"  MCP:         {scan.mcp.summary}")
    console.detail(f"  Hooks:       {scan.hooks.summary}")
    console.detail(f"  Permissions: {scan.permissions.summary}")
    console.empty()

    # Check if anything was found
    has_data = any([
        scan.rules.found,
        scan.agents.found,
        scan.skills.found,
        scan.mcp.found,
        scan.hooks.found,
        scan.permissions.found,
    ])
    if not has_data:
        console.info("No tool configs found to sync.")
        return

    # Build SyncData from wizard selections
    from crossby.ui import prompts

    data = SyncData()
    rules_src_tool: AIToolID | None = None
    agents_src_tool: AIToolID | None = None
    skills_src_tool: AIToolID | None = None

    # Rules
    if scan.rules.found and (sync_concern is None or sync_concern == SyncConcern.RULES):
        source: AIToolID | None
        if len(scan.rules.found) > 1:
            tools = list(scan.rules.found)
            suggested = suggest_rules_source(scan.rules.found)
            default_idx = tools.index(suggested) if suggested in tools else 0
            idx = prompts.select(
                "Multiple rules files found — which is the canonical source?",
                items=[str(t) for t in tools],
                hints=[scan.rules.found[t] for t in tools],
                default=default_idx,
            )
            source = tools[idx]
        else:
            source = suggest_rules_source(scan.rules.found)
        if source:
            source_path = scan.rules.found[source]
            other_tools = [str(t) for t in installed_tools if t != source]
            question = f"Port rules ({source_path}) to {', '.join(other_tools)}?"
            if other_tools and prompts.confirm(question, default=True):
                data.rules_source = source_path
                rules_src_tool = source

    # Agents
    if scan.agents.found and (sync_concern is None or sync_concern == SyncConcern.AGENTS):
        source = None
        if len(scan.agents.found) > 1:
            tools = list(scan.agents.found)
            suggested = suggest_agents_source(scan.agents.found)
            default_idx = tools.index(suggested) if suggested in tools else 0
            idx = prompts.select(
                "Multiple agents directories found — which is the canonical source?",
                items=[str(t) for t in tools],
                hints=[scan.agents.found[t] for t in tools],
                default=default_idx,
            )
            source = tools[idx]
        else:
            source = suggest_agents_source(scan.agents.found)
        if source:
            source_path = scan.agents.found[source]
            other_tools = [str(t) for t in installed_tools if t != source]
            question = f"Port agents ({source_path}) to {', '.join(other_tools)}?"
            if other_tools and prompts.confirm(question, default=True):
                data.agents_source = source_path
                agents_src_tool = source

    # Skills
    if scan.skills.found and (sync_concern is None or sync_concern == SyncConcern.SKILLS):
        source = None
        if len(scan.skills.found) > 1:
            tools = list(scan.skills.found)
            suggested = suggest_skills_source(scan.skills.found)
            default_idx = tools.index(suggested) if suggested in tools else 0
            idx = prompts.select(
                "Multiple skills directories found — which is the canonical source?",
                items=[str(t) for t in tools],
                hints=[scan.skills.found[t] for t in tools],
                default=default_idx,
            )
            source = tools[idx]
        else:
            source = suggest_skills_source(scan.skills.found)
        if source:
            source_path = scan.skills.found[source]
            other_tools = [str(t) for t in installed_tools if t != source]
            question = f"Port skills ({source_path}) to {', '.join(other_tools)}?"
            if other_tools and prompts.confirm(question, default=True):
                data.skills_source = source_path
                skills_src_tool = source

    # MCP
    if scan.mcp.found and (sync_concern is None or sync_concern == SyncConcern.MCP):
        servers = discover_mcp(project_root)
        if servers and prompts.confirm(
            f"Port {len(servers)} MCP server(s) to all tools?", default=True
        ):
            data.mcp_servers = servers

    # Permissions
    if scan.permissions.found and (sync_concern is None or sync_concern == SyncConcern.PERMISSIONS):
        patterns = discover_permissions(project_root)
        if patterns and prompts.confirm(
            f"Port {len(patterns)} permission pattern(s)?", default=True
        ):
            data.allowed_commands = patterns

    # Hooks
    if scan.hooks.found and (sync_concern is None or sync_concern == SyncConcern.HOOKS):
        hooks = discover_hooks(project_root)
        if hooks and prompts.confirm(f"Port {len(hooks)} hook(s) to all tools?", default=True):
            data.hooks = hooks

    # Check if user confirmed anything
    has_sync = any([
        data.rules_source,
        data.agents_source,
        data.skills_source,
        data.mcp_servers,
        data.allowed_commands,
        data.hooks,
    ])
    if not has_sync:
        console.info("Nothing to sync.")
        return

    # Execute per-concern to avoid writing back to the source tool.
    # Rules and agents have an explicit source; exclude it from their targets.
    # MCP, permissions, and hooks are merged from all tools — write to all.
    results = []
    if data.rules_source and (sync_concern is None or sync_concern == SyncConcern.RULES):
        rules_targets = [t for t in installed_tools if t != rules_src_tool]
        results += run_sync(data, project_root, concern=SyncConcern.RULES,
                            dry_run=dry_run, force=force, installed_tools=rules_targets)
    if data.agents_source and (sync_concern is None or sync_concern == SyncConcern.AGENTS):
        agents_targets = [t for t in installed_tools if t != agents_src_tool]
        results += run_sync(data, project_root, concern=SyncConcern.AGENTS,
                            dry_run=dry_run, force=force, installed_tools=agents_targets)
    if data.skills_source and (sync_concern is None or sync_concern == SyncConcern.SKILLS):
        skills_targets = [t for t in installed_tools if t != skills_src_tool]
        results += run_sync(data, project_root, concern=SyncConcern.SKILLS,
                            dry_run=dry_run, force=force, installed_tools=skills_targets)
    if data.mcp_servers and (sync_concern is None or sync_concern == SyncConcern.MCP):
        results += run_sync(data, project_root, concern=SyncConcern.MCP,
                            dry_run=dry_run, force=force, installed_tools=installed_tools)
    if data.allowed_commands and (sync_concern is None or sync_concern == SyncConcern.PERMISSIONS):
        results += run_sync(data, project_root, concern=SyncConcern.PERMISSIONS,
                            dry_run=dry_run, force=force, installed_tools=installed_tools)
    if data.hooks and (sync_concern is None or sync_concern == SyncConcern.HOOKS):
        results += run_sync(data, project_root, concern=SyncConcern.HOOKS,
                            dry_run=dry_run, force=force, installed_tools=installed_tools)

    if not results:
        console.info("No sync writers matched the given filters.")
        return

    _display_results(results)

    synced = sum(1 for r in results if r.action in ("created", "updated"))
    console.empty()
    console.success(f"Done. {synced} config(s) synced.")

    if any(r.action == "error" for r in results):
        raise typer.Exit(1)


def _display_results(results: list[SyncResult]) -> None:
    """Display sync results in a Rich table."""
    from rich.table import Table

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Tool", style="dim")
    table.add_column("Concern")
    table.add_column("Action")
    table.add_column("Detail", style="dim")

    _action_styles = {
        "created": "[success]created[/]",
        "updated": "[success]updated[/]",
        "skipped": "[dim]skipped[/]",
        "error": "[error]error[/]",
    }

    for r in results:
        styled_action = _action_styles.get(r.action, r.action)
        if r.action == "error":
            detail = r.message or ""
        elif r.file_path:
            detail = str(r.file_path)
        else:
            detail = r.message or ""
        table.add_row(
            str(r.tool_id) if r.tool_id is not None else "crossby",
            r.concern.value,
            styled_action,
            detail,
        )

    console.out.print(table)
