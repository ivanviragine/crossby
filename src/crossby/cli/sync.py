"""crossby sync — stateless sync wizard that reads directly from tool configs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import typer

from crossby.ui.console import console

if TYPE_CHECKING:
    from crossby.models.ai import AIToolID
    from crossby.sync.base import SyncConcern, SyncData, SyncResult
    from crossby.sync.validate import ValidationFinding


def sync(
    concern: str | None = typer.Argument(
        None,
        help="Sync concern: permissions, rules, mcp, agents, skills, hooks, plugins. Omit for all.",
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
    strategy: str | None = typer.Option(
        None,
        "--strategy",
        help=(
            "File strategy for rules and skills: 'symlink' (default), 'copy', "
            "or 'translate' (skills-only — per-tool SKILL.md rewrite with "
            "manual-fix notes for non-Claude targets, plus Claude slash "
            "commands as namespaced skills)."
        ),
    ),
    validate_target: bool = typer.Option(
        False,
        "--validate-target",
        help="Re-parse synced files and report structural issues; no writes.",
    ),
    plan: bool = typer.Option(
        False,
        "--plan",
        help="Show a stage-by-concern dry-run summary; no writes.",
    ),
    doctor: bool = typer.Option(
        False,
        "--doctor",
        help="Show a readiness summary (plan + validate-target); no writes.",
    ),
    report_format: str = typer.Option(
        "table",
        "--report-format",
        help="Format for the on-screen result table: 'table' (Rich) or 'markdown-table'.",
    ),
    no_persist_report: bool = typer.Option(
        False,
        "--no-persist-report",
        help="Skip writing .crossby/sync-report.md after the run.",
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
    from crossby.config.loader import load_config
    from crossby.models.ai import AIToolID
    from crossby.services.sync_resolution import (
        resolve_sync_concern,
        resolve_sync_from,
        resolve_sync_to,
    )
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
    config = load_config(project_root)

    # Validate enum-style flag values up-front so the error is descriptive
    # rather than a silent fallback.
    valid_report_formats = {"table", "markdown-table"}
    if report_format not in valid_report_formats:
        console.error(
            f"Unknown --report-format: {report_format!r}. "
            f"Valid: {', '.join(sorted(valid_report_formats))}."
        )
        raise typer.Exit(1)
    valid_strategies = {"symlink", "copy", "translate"}
    if strategy is not None and strategy not in valid_strategies:
        console.error(
            f"Unknown --strategy: {strategy!r}. "
            f"Valid: {', '.join(sorted(valid_strategies))}."
        )
        raise typer.Exit(1)

    # Pre-write inspection modes are mutually exclusive.
    inspect_modes = sum(1 for flag in (validate_target, plan, doctor) if flag)
    if inspect_modes > 1:
        console.error(
            "--validate-target, --plan, and --doctor are mutually exclusive."
        )
        raise typer.Exit(1)

    if validate_target:
        from crossby.sync.validate import has_errors, validate_target as _do_validate

        findings = _do_validate(project_root)
        _display_validation(findings)
        if has_errors(findings):
            raise typer.Exit(1)
        return

    if plan or doctor:
        _run_inspection(
            project_root=project_root,
            config=config,
            from_tool=from_tool,
            to_tool=to_tool,
            concern=concern,
            mode="doctor" if doctor else "plan",
            strategy=strategy,
        )
        return

    concern_explicit = concern is not None
    from_explicit = from_tool is not None
    to_explicit = to_tool is not None

    # Validate explicit CLI values with friendly error messages before
    # handing off to the resolvers (which raise raw ValueError).
    if concern is not None:
        try:
            SyncConcern(concern)
        except ValueError:
            valid = ", ".join(c.value for c in SyncConcern)
            console.error(f"Unknown concern: {concern!r}. Valid values: {valid}")
            raise typer.Exit(1) from None
    if from_tool is not None:
        try:
            AIToolID(from_tool)
        except ValueError:
            console.error(f"Unknown tool: {from_tool!r}")
            raise typer.Exit(1) from None
    if to_tool is not None:
        try:
            AIToolID(to_tool)
        except ValueError:
            console.error(f"Unknown tool: {to_tool!r}")
            raise typer.Exit(1) from None

    # Resolve CLI → config → auto-detect. Auto-detect is *off* for sync_from
    # because today's "crossby sync" with no flags falls through to the
    # per-concern wizard — auto-detect would silently change that.
    source_tool = resolve_sync_from(from_tool, config, auto_detect=False)
    target_tool = resolve_sync_to(to_tool, config)
    sync_concern = resolve_sync_concern(concern, config)

    if to_explicit and not from_explicit:
        console.error("--to requires --from; omit --to for the interactive wizard.")
        raise typer.Exit(1)

    # Detect installed tools
    installed_tools = AbstractAITool.detect_installed()
    if not installed_tools:
        console.error("No AI tools found in PATH.")
        console.hint("Install at least one AI tool (claude, copilot, gemini, codex, cursor, etc.)")
        raise typer.Exit(1)

    # Let users review resolved defaults via the shared Proceed / Change loop.
    source_tool, target_tool, sync_concern = _confirm_sync_defaults(
        source_tool=source_tool,
        target_tool=target_tool,
        sync_concern=sync_concern,
        installed_tools=installed_tools,
        from_explicit=from_explicit,
        to_explicit=to_explicit,
        concern_explicit=concern_explicit,
    )

    if dry_run:
        console.info("Dry-run mode — no files will be written")

    results: list[SyncResult] = []

    # Non-interactive mode: a confirmed source bypasses the per-concern wizard.
    # ``source_tool`` may come from --from, ``sync_defaults.from`` in
    # ``.crossby.yml``, or the user's Proceed/Change choice in the confirm
    # loop above. The "no source anywhere" case still falls through to the
    # interactive wizard so the existing zero-config UX is preserved.
    if source_tool is not None:
        data = build_sync_data(project_root, from_tool=source_tool)
        _apply_strategy(data, strategy)
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
        _display_results(results, report_format=report_format, project_root=project_root)
        if not dry_run and not no_persist_report:
            _persist_report(results, project_root)
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
    console.detail(f"  Plugins:     {scan.plugins.summary}")
    console.empty()

    # Check if anything was found
    has_data = any([
        scan.rules.found,
        scan.agents.found,
        scan.skills.found,
        scan.mcp.found,
        scan.hooks.found,
        scan.permissions.found,
        scan.plugins.found,
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

    _display_results(results, report_format=report_format, project_root=project_root)

    synced = sum(1 for r in results if r.action in ("created", "updated"))
    console.empty()
    console.success(f"Done. {synced} config(s) synced.")

    if not dry_run and not no_persist_report:
        _persist_report(results, project_root)

    if any(r.action == "error" for r in results):
        raise typer.Exit(1)


def _confirm_sync_defaults(
    *,
    source_tool: AIToolID | None,
    target_tool: AIToolID | None,
    sync_concern: SyncConcern | None,
    installed_tools: list[AIToolID],
    from_explicit: bool,
    to_explicit: bool,
    concern_explicit: bool,
) -> tuple[AIToolID | None, AIToolID | None, SyncConcern | None]:
    """Show resolved sync defaults and let the user Proceed or Change.

    Returns ``(source_tool, target_tool, sync_concern)`` after any user edits.
    A no-op on non-TTY stdin (handled inside ``confirm_defaults``).
    """
    from crossby.services.confirm import ConfirmField, confirm_defaults
    from crossby.ui import prompts

    tool_names = [str(t) for t in installed_tools]

    def _change_from(current: AIToolID | None, _state: dict[str, Any]) -> dict[str, Any]:
        current_name = str(current) if current is not None else tool_names[0]
        default_idx = tool_names.index(current_name) if current_name in tool_names else 0
        idx = prompts.select("Source tool", tool_names, default=default_idx)
        return {"from": AIToolID(tool_names[idx])}

    def _change_to(current: AIToolID | None, state: dict[str, Any]) -> dict[str, Any]:
        _ = current
        source = state.get("from")
        choices = ["(all installed)", *[n for n in tool_names if AIToolID(n) != source]]
        idx = prompts.select("Target tool", choices)
        if idx == 0:
            return {"to": None}
        return {"to": AIToolID(choices[idx])}

    def _change_concern(
        current: SyncConcern | None, _state: dict[str, Any]
    ) -> dict[str, Any]:
        _ = current
        concerns = [c.value for c in SyncConcern]
        choices = ["(all concerns)", *concerns]
        idx = prompts.select("Concern", choices)
        if idx == 0:
            return {"concern": None}
        return {"concern": SyncConcern(choices[idx])}

    fields = [
        ConfirmField(
            name="from",
            label="Source tool",
            current_value=source_tool,
            explicit=from_explicit,
            change_fn=_change_from,
            render_value=lambda v: str(v) if v is not None else "(choose in wizard)",
        ),
        ConfirmField(
            name="to",
            label="Target tool",
            current_value=target_tool,
            explicit=to_explicit,
            change_fn=_change_to,
            render_value=lambda v: str(v) if v is not None else "all installed",
        ),
        ConfirmField(
            name="concern",
            label="Concern",
            current_value=sync_concern,
            explicit=concern_explicit,
            change_fn=_change_concern,
            render_value=lambda v: v.value if v is not None else "all",
        ),
    ]

    result = confirm_defaults(fields, title="Confirm sync defaults")
    return result["from"], result["to"], result["concern"]


def _display_results(
    results: list[SyncResult],
    *,
    report_format: str = "table",
    project_root: Path | None = None,
) -> None:
    """Display sync results in a Rich table or as a portable markdown table."""
    if report_format == "markdown-table":
        from crossby.sync.report import render_markdown_table

        rendered = render_markdown_table(results, project_root=project_root)
        if rendered:
            console.out.print(rendered)
        return

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


def _persist_report(results: list[SyncResult], project_root: Path) -> None:
    """Write .crossby/sync-report.md after a real (non-dry-run) sync."""
    from crossby.sync.report import write_persistent_report

    if not results:
        return
    path = write_persistent_report(results, project_root)
    console.detail(f"  report: {path.relative_to(project_root)}")


def _apply_strategy(data: "SyncData", strategy: str | None) -> None:
    """Override SyncData strategy fields from the CLI ``--strategy`` flag.

    ``translate`` only meaningfully affects skills; rules + agents fall back
    to ``copy`` (rules' foreign-marker detection still triggers per-target
    copy on its own when needed). ``None`` means "leave SyncData defaults".
    """
    if strategy is None:
        return
    if strategy == "translate":
        data.skills_strategy = "translate"
        data.rules_strategy = "copy"
        data.agents_strategy = "copy"
    elif strategy == "copy":
        data.skills_strategy = "copy"
        data.rules_strategy = "copy"
        data.agents_strategy = "copy"
    else:
        data.skills_strategy = "symlink"
        data.rules_strategy = "symlink"
        data.agents_strategy = "symlink"


def _display_validation(findings: list[ValidationFinding]) -> None:
    """Render validation findings as a Rich table."""
    from rich.table import Table

    typed_findings: list[ValidationFinding] = list(findings)
    if not typed_findings:
        console.info("Nothing to validate — no synced files found under the project.")
        return

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Tool", style="dim")
    table.add_column("Concern")
    table.add_column("Level")
    table.add_column("Path")
    table.add_column("Detail", style="dim")

    level_styles = {
        "ok": "[success]ok[/]",
        "warning": "[warn]warning[/]",
        "error": "[error]error[/]",
    }

    for finding in typed_findings:
        table.add_row(
            str(finding.tool_id) if finding.tool_id is not None else "crossby",
            finding.concern.value if finding.concern is not None else "",
            level_styles.get(finding.level, finding.level),
            str(finding.path),
            finding.detail,
        )

    console.out.print(table)
    error_count = sum(1 for f in typed_findings if f.level == "error")
    warn_count = sum(1 for f in typed_findings if f.level == "warning")
    ok_count = sum(1 for f in typed_findings if f.level == "ok")
    console.empty()
    console.detail(
        f"  {ok_count} ok · {warn_count} warning · {error_count} error"
    )


def _run_inspection(
    *,
    project_root: Path,
    config: Any,
    from_tool: str | None,
    to_tool: str | None,
    concern: str | None,
    mode: Literal["plan", "doctor"],
    strategy: str | None = None,
) -> None:
    """Run a dry-run sync and print a plan or doctor summary."""
    from crossby.ai_tools.base import AbstractAITool
    from crossby.services.sync_resolution import (
        resolve_sync_concern,
        resolve_sync_from,
        resolve_sync_to,
    )
    from crossby.sync import run_sync
    from crossby.sync.plan import (
        build_doctor,
        render_doctor,
        render_plan,
        summarize_plan,
    )
    from crossby.sync.readers import build_sync_data
    from crossby.sync.validate import validate_target as _do_validate

    source = resolve_sync_from(from_tool, config, auto_detect=True)
    target = resolve_sync_to(to_tool, config)
    sync_concern = resolve_sync_concern(concern, config)

    if source is None:
        console.error(
            "--plan/--doctor needs a source tool. Pass --from <tool> or set "
            "sync_defaults.from in .crossby.yml."
        )
        raise typer.Exit(1)

    installed_tools = AbstractAITool.detect_installed()
    if not installed_tools:
        console.error("No AI tools found in PATH.")
        raise typer.Exit(1)

    target_tools = (
        [target] if target else [t for t in installed_tools if t != source]
    )
    if not target_tools:
        # When the user has only the source tool installed (e.g. just
        # Claude on PATH), there's nothing to sync to. Plugin discovery
        # still runs below, but a clearer message up front prevents the
        # misleading "no sync rows produced — check source/target/concern
        # flags" output.
        console.warn(
            f"No target tools detected besides the source ({source}). "
            "Install another AI tool or pass --to <tool> explicitly."
        )
    data = build_sync_data(project_root, from_tool=source)
    _apply_strategy(data, strategy)
    results = run_sync(
        data,
        project_root,
        tool_id=target,
        concern=sync_concern,
        dry_run=True,
        force=False,
        installed_tools=target_tools,
    )

    summary = summarize_plan(results)
    plan_text = render_plan(summary)
    console.out.print(plan_text)

    if mode == "doctor":
        validation = _do_validate(project_root)
        report = build_doctor(summary, validation)
        console.empty()
        console.out.print(render_doctor(report))
        if report.readiness == "low":
            raise typer.Exit(1)
