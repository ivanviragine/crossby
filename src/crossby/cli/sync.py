"""crossby sync — port configs across AI tools."""

from __future__ import annotations

from pathlib import Path

import typer

from crossby.models.ai import AIToolID
from crossby.models.sync import SyncResult, SyncStrategy
from crossby.ui.console import console

# Tools that support sync (have instructions/skills/allowlist).
_SYNCABLE_TOOLS = [
    AIToolID.CLAUDE,
    AIToolID.CURSOR,
    AIToolID.COPILOT,
    AIToolID.GEMINI,
    AIToolID.CODEX,
]

_DISPLAY_NAMES: dict[AIToolID, str] = {
    AIToolID.CLAUDE: "Claude",
    AIToolID.CURSOR: "Cursor",
    AIToolID.COPILOT: "Copilot",
    AIToolID.GEMINI: "Gemini",
    AIToolID.CODEX: "Codex",
}

_STRATEGY_STYLE: dict[SyncStrategy, str] = {
    SyncStrategy.LINK: "success",
    SyncStrategy.CONVERT: "step",
    SyncStrategy.WARN: "warning",
    SyncStrategy.UNSUPPORTED: "warning",
}


def sync(
    from_tool: str | None = typer.Option(None, "--from", help="Source tool."),
    to: list[str] | None = typer.Option(None, "--to", help="Target tool(s). Repeat for multiple."),
    all_tools: bool = typer.Option(False, "--all", help="Sync to all installed tools."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview sync plan without applying."),
    instructions: bool = typer.Option(False, "--instructions", help="Sync instructions only."),
    skills: bool = typer.Option(False, "--skills", help="Sync skills only."),
    allowlist: bool = typer.Option(False, "--allowlist", help="Sync allowlist only."),
) -> None:
    """Sync configs between AI tools.

    Wizard mode (no args): interactive prompts.
    Direct mode: --from and (--to or --all) required.

    Examples:

        crossby sync --from claude --to cursor
        crossby sync --from claude --all
        crossby sync --from claude --to cursor --dry-run
    """
    root = Path.cwd()

    # Determine config types to sync.
    if not any([instructions, skills, allowlist]):
        instructions = skills = allowlist = True

    to = to or []

    if from_tool is None and not to and not all_tools:
        _wizard(root, instructions, skills, allowlist)
        return

    # Direct mode — validate args.
    if from_tool is None:
        console.error("--from is required in direct mode")
        raise typer.Exit(1)

    source = _parse_tool_id(from_tool)
    if source is None:
        console.error(f"Unknown tool: {from_tool}")
        raise typer.Exit(1)

    targets = _resolve_targets(source, to, all_tools)
    if not targets:
        console.error("No targets specified. Use --to or --all.")
        raise typer.Exit(1)

    _run_sync(
        source,
        targets,
        root,
        dry_run=dry_run,
        force=True,  # direct mode overwrites silently
        sync_instructions=instructions,
        sync_skills=skills,
        sync_allowlist=allowlist,
    )


def _wizard(root: Path, sync_instr: bool, sync_sk: bool, sync_al: bool) -> None:
    """Interactive wizard mode."""
    from crossby.ui import prompts

    if not prompts.is_tty():
        console.error("Wizard requires a TTY. Use --from and --to flags.")
        raise typer.Exit(1)

    console.header("crossby sync")

    # Select source tool.
    labels = [_DISPLAY_NAMES[t] for t in _SYNCABLE_TOOLS]
    idx = prompts.select("Source tool:", labels)
    source = _SYNCABLE_TOOLS[idx]

    # Select target tools.
    remaining = [t for t in _SYNCABLE_TOOLS if t != source]
    remaining_labels = [_DISPLAY_NAMES[t] for t in remaining]
    selected_indices = prompts.multi_select("Target tools:", remaining_labels)
    if not selected_indices:
        console.warn("No targets selected.")
        raise typer.Exit(0)
    targets = [remaining[i] for i in selected_indices]

    # Preview (dry run, force=False to surface conflicts as warnings).
    console.empty()
    result = _run_sync(
        source,
        targets,
        root,
        dry_run=True,
        force=False,
        sync_instructions=sync_instr,
        sync_skills=sync_sk,
        sync_allowlist=sync_al,
    )

    if result.linked == 0 and result.converted == 0 and not result.warnings:
        console.info("Nothing to sync.")
        raise typer.Exit(0)

    # Confirm.
    console.empty()
    if not prompts.confirm("Apply sync plan?"):
        console.info("Cancelled.")
        raise typer.Exit(0)

    # Execute with force=True — user has reviewed the plan and confirmed.
    console.empty()
    _run_sync(
        source,
        targets,
        root,
        dry_run=False,
        force=True,
        sync_instructions=sync_instr,
        sync_skills=sync_sk,
        sync_allowlist=sync_al,
    )


def _run_sync(
    source: AIToolID,
    targets: list[AIToolID],
    root: Path,
    *,
    dry_run: bool,
    force: bool,
    sync_instructions: bool,
    sync_skills: bool,
    sync_allowlist: bool,
) -> SyncResult:
    """Run sync and display results."""
    from crossby.services.sync import sync_configs

    label = "Sync plan" if dry_run else "Sync"
    target_names = ", ".join(_DISPLAY_NAMES.get(t, t.value) for t in targets)
    console.step(f"{label}: {_DISPLAY_NAMES.get(source, source.value)} -> {target_names}")
    console.empty()

    result = sync_configs(
        source,
        targets,
        root,
        dry_run=dry_run,
        force=force,
        sync_instructions=sync_instructions,
        sync_skills=sync_skills,
        sync_allowlist=sync_allowlist,
    )

    _display_actions(result, dry_run)
    return result


def _display_actions(result: SyncResult, dry_run: bool) -> None:
    """Format and display sync actions."""
    for action in result.actions:
        style = _STRATEGY_STYLE.get(action.strategy, "info")
        tag = action.strategy.value.upper()

        if action.strategy in (SyncStrategy.WARN, SyncStrategy.UNSUPPORTED):
            console.warn(f"[{style}]{tag:<11}[/] {action.config_type:<14} {action.message}")
        else:
            console.step(f"[{style}]{tag:<11}[/] {action.config_type:<14} {action.message}")

    console.empty()

    if dry_run:
        counts = []
        if result.linked:
            counts.append(f"{result.linked} would link")
        if result.converted:
            counts.append(f"{result.converted} would convert")
        if counts:
            console.info(", ".join(counts))
    else:
        counts = []
        if result.linked:
            counts.append(f"{result.linked} linked")
        if result.converted:
            counts.append(f"{result.converted} converted")
        if counts:
            console.success(", ".join(counts))
        elif not result.warnings:
            console.info("Everything already in sync.")


def _parse_tool_id(name: str) -> AIToolID | None:
    """Parse a tool name to AIToolID."""
    try:
        return AIToolID(name.lower())
    except ValueError:
        return None


def _resolve_targets(
    source: AIToolID,
    to: list[str],
    all_tools: bool,
) -> list[AIToolID]:
    """Resolve target tool list from CLI args."""
    if all_tools:
        from crossby.ai_tools.base import AbstractAITool

        installed = AbstractAITool.detect_installed()
        return [t for t in installed if t != source]

    targets: list[AIToolID] = []
    for name in to:
        tid = _parse_tool_id(name)
        if tid is None:
            console.warn(f"Unknown tool: {name}, skipping")
            continue
        if tid != source:
            targets.append(tid)
    return targets
