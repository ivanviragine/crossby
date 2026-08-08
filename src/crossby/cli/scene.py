"""``crossby scene`` — activate task-shaped capability bundles.

A *scene* names a filtered slice of the project's skills, agents, MCP servers,
hooks and permissions (defined under ``scenes:`` in ``.crossby.yml``). This group
drives the scene activation engine:

- ``list``   — the scenes defined for this project, with per-concern counts.
- ``show``   — what one scene resolves to, and the mechanism each tool would use.
- ``use``    — apply a scene to the installed tools (switching off any active one).
- ``clear``  — revert the active scene to the pre-scene baseline.
- ``status`` — the active scene, its per-tool mechanism, and any drift.

Reversion is driven by the ownership ledger (``owned.json``); ``.crossby/
scene-state.json`` is a companion bookkeeping file that records which scene is
active and a content hash per managed file so ``status`` can detect drift. See
:mod:`crossby.scenes.state`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from crossby.ui.console import console

if TYPE_CHECKING:
    from collections.abc import Callable

    from crossby.models.ai import AIToolID
    from crossby.models.config import CrossbyConfig, SceneConfig, SceneSelector
    from crossby.scenes.authoring import CrossChannelMove, SelectorEdit
    from crossby.scenes.state import SceneState, SceneToolRecord
    from crossby.services.scene_resolution import ResolvedScene
    from crossby.sync.base import SyncResult
    from crossby.sync.readers import ProjectScan

scene_app = typer.Typer(
    name="scene",
    help="Activate task-shaped scenes — apply, clear, and inspect capability bundles.",
    no_args_is_help=True,
)

# The concern order used across every scene surface (matches SCENE_CONCERNS).
_CONCERN_ORDER: tuple[str, ...] = ("skills", "agents", "mcp", "hooks", "permissions")

_TOOL_OPTION = typer.Option(None, "--tool", help="Limit to a single tool (e.g. claude, cursor).")
_PATH_OPTION = typer.Option(Path("."), "--path", help="Project root directory.")

# Selector flags shared by create / add / remove. Each singular flag maps to a
# plural concern; both the include and the exclude channel are first-class,
# because excludes are first-class in the schema and starters rely on them.
_SKILL_OPT = typer.Option([], "--skill", help="Skill glob to include (repeatable).")
_EXCLUDE_SKILL_OPT = typer.Option([], "--exclude-skill", help="Skill glob to exclude (repeatable).")
_AGENT_OPT = typer.Option([], "--agent", help="Agent glob to include (repeatable).")
_EXCLUDE_AGENT_OPT = typer.Option([], "--exclude-agent", help="Agent glob to exclude (repeatable).")
_MCP_OPT = typer.Option([], "--mcp", help="MCP server glob to include (repeatable).")
_EXCLUDE_MCP_OPT = typer.Option(
    [], "--exclude-mcp", help="MCP server glob to exclude (repeatable)."
)
_HOOK_OPT = typer.Option([], "--hook", help="Hook glob to include (repeatable).")
_EXCLUDE_HOOK_OPT = typer.Option([], "--exclude-hook", help="Hook glob to exclude (repeatable).")
_PERMISSION_OPT = typer.Option([], "--permission", help="Permission glob to include (repeatable).")
_EXCLUDE_PERMISSION_OPT = typer.Option(
    [], "--exclude-permission", help="Permission glob to exclude (repeatable)."
)
_DESCRIPTION_OPT = typer.Option(None, "--description", help="One-line scene description.")
_EXTENDS_OPT = typer.Option(None, "--extends", help="Parent scene to compose from (single parent).")
_PROFILE_OPT = typer.Option(None, "--profile", help="Default launch profile for this scene.")
_PRINT_OPT = typer.Option(False, "--print", help="Print the scene block to stdout; write nothing.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_config_or_exit(project_root: Path) -> CrossbyConfig:
    from crossby.config.loader import ConfigError, load_config

    try:
        return load_config(project_root)
    except ConfigError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from exc


def _validate_tool(tool: str | None) -> AIToolID | None:
    """Parse ``--tool`` into an :class:`AIToolID`, exiting 1 on an unknown value."""
    if tool is None:
        return None
    from crossby.models.ai import AIToolID

    try:
        return AIToolID(tool)
    except ValueError:
        valid = ", ".join(t.value for t in AIToolID)
        console.error(f"Unknown tool: {tool!r}. Valid values: {valid}")
        raise typer.Exit(1) from None


def _installed_or_exit() -> list[AIToolID]:
    from crossby.ai_tools.base import AbstractAITool

    installed = AbstractAITool.detect_installed()
    if not installed:
        console.error("No AI tools found in PATH.")
        console.hint("Install at least one AI tool (claude, cursor, codex, ...).")
        raise typer.Exit(1)
    return installed


def _get_scene_or_exit(config: CrossbyConfig, name: str) -> SceneConfig:
    """Flatten and return scene *name*, exiting 1 if it is unknown or invalid."""
    from crossby.config.loader import ConfigError

    try:
        scene = config.get_scene(name)
    except ConfigError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from exc
    if scene is None:
        console.error(f"Unknown scene: {name!r}")
        available = sorted(config.scenes)
        if available:
            console.hint(f"Available scenes: {', '.join(available)}")
        else:
            console.hint("No scenes are defined in .crossby.yml.")
        raise typer.Exit(1)
    return scene


def _resolve(
    config: CrossbyConfig,
    name: str,
    project_root: Path,
    installed: list[AIToolID],
    *,
    tool_id: AIToolID | None = None,
) -> ResolvedScene:
    from crossby.services.scene_resolution import resolve_scene
    from crossby.sync.readers import scan_project

    scene = _get_scene_or_exit(config, name)
    scan = scan_project(project_root, installed)
    return resolve_scene(scene, scan, project_root, tool_id=tool_id)


def _display_results(results: list[SyncResult]) -> None:
    """Render scene results in the same table shape as ``crossby sync``."""
    if not results:
        console.info("No actions taken.")
        return
    from rich.table import Table

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Tool", style="dim")
    table.add_column("Concern")
    table.add_column("Action")
    table.add_column("Detail", style="dim")

    styles = {
        "created": "[success]created[/]",
        "updated": "[success]updated[/]",
        "skipped": "[dim]skipped[/]",
        "error": "[error]error[/]",
    }
    for r in results:
        detail = r.message or (str(r.file_path) if r.file_path else "")
        table.add_row(
            str(r.tool_id) if r.tool_id is not None else "crossby",
            r.concern.value,
            styles.get(r.action, r.action),
            detail,
        )
    console.out.print(table)


def _has_error(results: list[SyncResult]) -> bool:
    return any(r.action == "error" for r in results)


def _refuse_if_ledger_corrupt(project_root: Path) -> None:
    """Exit cleanly when ``owned.json`` exists but is unreadable.

    We cannot revert from provenance we cannot parse, and proceeding would let the
    engine's ``finally: save_ledger`` overwrite the corrupt ledger with an empty
    *valid* one — destroying any chance of hand-recovery (an empty ledger owns
    nothing, so a later ``clear`` reverts nothing yet deletes ``scene-state.json``,
    leaving the settings applied with no record). Every ledger-reverting path
    routes through this one guard so the refusal can't be forgotten on one of them.
    """
    from crossby.sync.ownership import LEDGER_PATH, load_ledger_checked

    if not load_ledger_checked(project_root).corrupt:
        return
    console.error(f"{LEDGER_PATH.as_posix()} is unreadable — cannot determine what to revert.")
    console.hint(
        "Restore a valid owned.json from backup, or manually revert the applied "
        "settings. Do NOT delete owned.json — an empty ledger would let 'clear' "
        "drop the recovery state while leaving settings applied."
    )
    raise typer.Exit(1)


def _scope_for(tool_id: AIToolID | None, installed: list[AIToolID]) -> list[AIToolID]:
    """The installed tools an operation targets: one tool, or all installed."""
    if tool_id is None:
        return installed
    if tool_id not in installed:
        console.error(f"Tool {tool_id} is not installed.")
        console.hint(f"Installed tools: {', '.join(str(t) for t in installed)}")
        raise typer.Exit(1)
    return [tool_id]


def _expand_shared_scope(scope: list[AIToolID], candidates: list[AIToolID]) -> list[AIToolID]:
    """Add any *candidate* that shares a skills directory with a scoped tool.

    Codex and Antigravity CLI both resolve skills to ``.agents/skills``, so
    re-pointing (or restoring) it for one necessarily affects the other. Scoping
    to a single one of them would silently reach the other; expanding the scope
    keeps the operation and the recorded state honest.
    """
    from crossby.config.skills import SKILLS_DIR

    scoped_dirs = {SKILLS_DIR.get(tool) for tool in scope} - {None}
    expanded = list(scope)
    for tool in candidates:
        if tool not in expanded and SKILLS_DIR.get(tool) in scoped_dirs:
            expanded.append(tool)
    return expanded


def _inform_shared_expansion(base: list[AIToolID], expanded: list[AIToolID]) -> None:
    extra = [tool for tool in expanded if tool not in base]
    if extra:
        shared = ", ".join(str(tool) for tool in extra)
        console.info(f"--tool also affects {shared} (shared skills directory).")


def _warn_removed_hooks_permissions(results: list[SyncResult]) -> None:
    """Flag scene-removed hooks/permissions — ``clear`` does not restore them.

    Those entries are crossby-owned synced items; a scene narrows them through
    the revocable-sync removal channel, which the ledger-driven revert cannot put
    back. Re-running ``crossby sync`` restores them.
    """
    removed = [r for r in results if r.concern.value in ("hooks", "permissions") and r.revoked > 0]
    if removed:
        console.warn(
            "This scene removed hook(s)/permission(s); 'crossby scene clear' does not restore them."
        )
        console.hint("Re-run 'crossby sync' to restore them after clearing the scene.")


def _confirm_scene_defaults(
    *, action: str, scene: str | None, tool_id: AIToolID | None, installed: list[AIToolID]
) -> AIToolID | None:
    """Review the scene action's target tool, letting the user change it.

    Mirrors ``crossby sync``'s confirmation: a no-op on non-TTY stdin and when the
    tool was passed explicitly, an interactive Proceed/Change menu otherwise.
    Returns the (possibly changed) tool scope.
    """
    from typing import Any, cast

    from crossby.models.ai import AIToolID
    from crossby.services.confirm import ConfirmField, confirm_defaults
    from crossby.ui import prompts

    tool_names = [str(t) for t in installed]

    def _change_tool(current: AIToolID | None, _state: dict[str, Any]) -> dict[str, Any]:
        choices = ["(all installed)", *tool_names]
        idx = prompts.select("Target tool", choices)
        if idx == 0:
            return {"tool": None}
        return {"tool": AIToolID(choices[idx])}

    fields = [
        ConfirmField(
            name="tool",
            label="Target tool",
            current_value=tool_id,
            explicit=tool_id is not None,
            change_fn=_change_tool,
            render_value=lambda v: str(v) if v is not None else "all installed",
        ),
    ]
    title = f"Confirm scene {action}" + (f" ({scene})" if scene else "")
    result = confirm_defaults(fields, title=title)
    return cast("AIToolID | None", result["tool"])


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@scene_app.command("list")
def list_scenes(
    tool: str | None = _TOOL_OPTION,
    path: Path = _PATH_OPTION,
) -> None:
    """List the scenes defined in ``.crossby.yml`` with per-concern counts."""
    from crossby.services.scene_resolution import scene_root

    project_root = path.resolve()
    config = _load_config_or_exit(project_root)
    root = scene_root(project_root)
    tool_id = _validate_tool(tool)

    if not config.scenes:
        console.info("No scenes defined in .crossby.yml.")
        console.hint("Add a 'scenes:' section to define task-shaped bundles.")
        return

    from rich.table import Table

    from crossby.ai_tools.base import AbstractAITool
    from crossby.sync.readers import scan_project

    installed = AbstractAITool.detect_installed()
    # Scan the project once and reuse it for every scene's resolution.
    scan = scan_project(root, installed)

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Scene")
    table.add_column("Description", style="dim")
    table.add_column("Concerns")

    for name in sorted(config.scenes):
        description = config.scenes[name].description or ""
        counts = _concern_counts(config, name, root, scan, tool_id)
        table.add_row(name, description, counts)
    console.out.print(table)


def _concern_counts(
    config: CrossbyConfig,
    name: str,
    project_root: Path,
    scan: ProjectScan,
    tool_id: AIToolID | None,
) -> str:
    """A compact ``skills:3 agents:1`` summary of what *name* resolves to."""
    from crossby.config.loader import ConfigError
    from crossby.services.scene_resolution import resolve_scene

    try:
        scene = config.get_scene(name)
    except ConfigError as exc:
        return f"[error]invalid ({exc})[/]"
    if scene is None:
        return ""
    resolved = resolve_scene(scene, scan, project_root, tool_id=tool_id)
    parts = [
        f"{concern}:{len(resolved.names(concern))}"
        for concern in _CONCERN_ORDER
        if resolved.names(concern)
    ]
    return "  ".join(parts) if parts else "(nothing selected)"


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@scene_app.command("show")
def show_scene(
    name: str = typer.Argument(..., help="Scene name to inspect."),
    tool: str | None = _TOOL_OPTION,
    path: Path = _PATH_OPTION,
) -> None:
    """Show what a scene resolves to per tool, and the mechanism each would use."""
    from crossby.services.scene_resolution import scene_root

    project_root = path.resolve()
    config = _load_config_or_exit(project_root)
    root = scene_root(project_root)
    tool_id = _validate_tool(tool)
    installed = _installed_or_exit()

    resolved = _resolve(config, name, root, installed, tool_id=tool_id)

    from rich.table import Table

    from crossby.scenes.mechanism import plan_units

    units = plan_units(resolved)
    if not units:
        console.info(f"Scene {name!r} selects nothing for the detected project.")
    else:
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Concern")
        table.add_column("Tools", style="dim")
        table.add_column("Mechanism")
        table.add_column("Selected")
        for unit in units:
            tools = ", ".join(str(t) for t in unit.tools)
            selected = ", ".join(unit.names) if unit.names else "(none)"
            table.add_row(unit.concern, tools, _mechanism_label(unit.mechanism), selected)
        console.out.print(table)

    for warning in resolved.warnings:
        console.warn(warning)


def _mechanism_label(mechanism: object) -> str:
    value = getattr(mechanism, "value", str(mechanism))
    if value == "unsupported":
        return f"[warning]{value}[/]"
    return value


# ---------------------------------------------------------------------------
# use
# ---------------------------------------------------------------------------


@scene_app.command("use")
def use_scene(
    name: str = typer.Argument(..., help="Scene name to apply."),
    tool: str | None = _TOOL_OPTION,
    plan: bool = typer.Option(False, "--plan", help="Preview the apply; write nothing."),
    force: bool = typer.Option(
        False, "--force", help="Proceed despite drift in the outgoing scene (bypass the refusal)."
    ),
    path: Path = _PATH_OPTION,
) -> None:
    """Apply a scene — resolve it, switch off any active scene, and record state."""
    from crossby.scenes.engine import apply_scene
    from crossby.scenes.state import (
        detect_drift,
        load_scene_state,
        save_scene_state,
    )
    from crossby.services.scene_resolution import scene_root

    project_root = path.resolve()
    config = _load_config_or_exit(project_root)
    root = scene_root(project_root)
    tool_id = _validate_tool(tool)
    installed = _installed_or_exit()
    scope = _scope_for(tool_id, installed)
    if tool_id is not None:
        expanded = _expand_shared_scope(scope, installed)
        _inform_shared_expansion(scope, expanded)
        scope = expanded

    # Resolve the full union (every tool) so the disable sets stay anchored on the
    # real inventory; the apply is narrowed to `scope` via the tools= argument.
    resolved = _resolve(config, name, root, installed)

    loaded = load_scene_state(root)
    if loaded.warning:
        console.warn(loaded.warning)
    active = loaded.state

    # Fail closed on a corrupt ledger before ANY engine call — covering first-time
    # apply, a switch, and --plan uniformly. Reverting (or previewing a revert)
    # from an empty-loaded ledger is misleading, and the engine's finally:
    # save_ledger would overwrite the corrupt bytes with an empty valid ledger.
    _refuse_if_ledger_corrupt(root)

    # --plan writes nothing, so it always previews — even against a drifted scene.
    if plan:
        results = apply_scene(resolved, root, dry_run=True, force=force, tools=scope)
        _display_results(results)
        console.info("(--plan) no changes written.")
        if _has_error(results):
            raise typer.Exit(1)
        return

    # The review may change the target tool (interactive only); honour it, and
    # do the scope-safety checks below against the *final* scope.
    new_tool_id = _confirm_scene_defaults(
        action="use", scene=name, tool_id=tool_id, installed=installed
    )
    if new_tool_id != tool_id:
        tool_id = new_tool_id
        scope = _scope_for(tool_id, installed)
        if tool_id is not None:
            scope = _expand_shared_scope(scope, installed)

    scope_strs = {str(t) for t in scope}

    # A scoped switch to a *different* scene would silently leave the other tools
    # on the active scene — the single-active-scene state can't represent that.
    if active is not None and active.scene != name and tool_id is not None:
        others = [t for t in active.tool_ids if t not in scope_strs]
        if others:
            console.error(
                f"Scene {active.scene!r} is active on {', '.join(sorted(others))}; "
                f"switching to {name!r} with --tool would strand them on {active.scene!r}."
            )
            console.hint("Run 'crossby scene clear' first, or re-run without --tool.")
            raise typer.Exit(1)

    # Revert only the scope tools' current state (not the whole active scene), so
    # a same-scene scoped re-apply repairs just those and leaves the rest alone.
    # Drift on those tools is checked first (the revert would discard hand edits),
    # and a failed revert aborts before applying, leaving the state intact.
    if active is not None:
        recorded = _recorded_tools(active)
        # An unscoped use reverts every recorded tool, including one that is no
        # longer installed — otherwise its owned keys stay applied but unrecorded.
        revert = recorded if tool_id is None else [t for t in recorded if str(t) in scope_strs]
        if revert:
            drifted = detect_drift(root, active, tools=[str(t) for t in revert])
            if drifted and not force:
                _report_drift_refusal(active.scene, drifted, verb="switch from")
                raise typer.Exit(1)
            if not _revert_tools(root, revert):
                console.error(f"Could not revert {active.scene!r} — aborting; state left intact.")
                raise typer.Exit(1)

    scene = _get_scene_or_exit(config, name)
    try:
        results = apply_scene(resolved, root, force=force, tools=scope)
    except Exception as exc:
        # apply_scene persists provenance in a finally, so anything already
        # written is revertible via the ledger. Record a minimal recovery state
        # so `clear` knows a scene is active and reverts those tools.
        _save_recovery_state(root, name, scene, scope)
        console.error(f"Scene apply failed: {exc}")
        console.hint("'crossby scene clear' can revert changes crossby recorded.")
        raise typer.Exit(1) from exc
    _display_results(results)
    _warn_removed_hooks_permissions(results)

    state = _build_state(root, name, scene, scope, results)
    # A same-scene re-apply keeps records for tools outside this scope.
    if active is not None and active.scene == name:
        merged = dict(active.tools)
        merged.update(state.tools)
        state.tools = merged
    try:
        save_scene_state(root, state)
    except OSError as exc:
        # The scene is applied (the ledger has provenance) but its state could
        # not be recorded — surface it cleanly so the user can re-run rather than
        # being left with an active-but-untracked scene.
        console.error(f"Scene applied but its state could not be recorded: {exc}")
        console.hint(f"Re-run 'crossby scene use {name}' once the path is writable.")
        raise typer.Exit(1) from exc

    if _has_error(results):
        console.error("Scene applied partially — some tools failed (see rows above).")
        console.hint("'crossby scene clear' will revert exactly the tools that succeeded.")
        raise typer.Exit(1)
    console.success(f"Applied scene {name!r}.")


def _recorded_tools(active: SceneState) -> list[AIToolID]:
    """The recorded tools as :class:`AIToolID`, skipping any unknown id.

    Guards against a corrupt-but-parseable state file carrying a tool key this
    build doesn't recognise — a clear should never crash on that.
    """
    from crossby.models.ai import AIToolID

    tools: list[AIToolID] = []
    for tool in active.tool_ids:
        try:
            tools.append(AIToolID(tool))
        except ValueError:
            continue
    return tools


def _revert_tools(project_root: Path, tools: list[AIToolID]) -> bool:
    """Revert *tools* via the engine. Returns success (no ``error`` rows).

    An empty list is a no-op — never passed to the engine as ``None``, which it
    reads as "every installed tool". A revert that errors returns ``False`` so a
    switch can abort and keep the old state.
    """
    if not tools:
        return True
    from crossby.scenes.engine import clear_scene

    return not _has_error(_call_engine_or_exit(clear_scene, project_root, tools=tools))


def _save_recovery_state(
    project_root: Path, name: str, scene: SceneConfig, scope: list[AIToolID]
) -> None:
    """Best-effort record of a scene whose apply raised part-way.

    The ledger already holds provenance (``apply_scene`` saves it in a finally),
    so recording the scope tools lets ``clear`` revert exactly what was written.
    Failure here must not mask the original error, so it is suppressed.
    """
    import contextlib

    from crossby.scenes.state import save_scene_state

    with contextlib.suppress(Exception):
        state = _build_state(project_root, name, scene, scope, [])
        state.status = "partial"
        save_scene_state(project_root, state)


def _call_engine_or_exit(
    fn: Callable[..., list[SyncResult]], *args: object, **kwargs: object
) -> list[SyncResult]:
    """Run an engine call, turning an unexpected exception into a clean exit.

    The DECLARE handlers persist provenance in a ``finally``, so anything crossby
    already wrote stays revertible via ``clear`` — this just avoids surfacing a
    raw traceback for a genuinely exceptional failure (e.g. a read-only config).
    """
    try:
        return fn(*args, **kwargs)
    except typer.Exit:
        raise
    except Exception as exc:
        console.error(f"Scene engine error: {exc}")
        console.hint("'crossby scene clear' can revert any partial changes crossby recorded.")
        raise typer.Exit(1) from exc


def _build_state(
    project_root: Path,
    name: str,
    scene: SceneConfig,
    scope: list[AIToolID],
    results: list[SyncResult],
) -> SceneState:
    from crossby.scenes.state import SceneState, compute_hashes, now_iso

    tools = _tool_mechanisms(scene, scope)
    hashes_by_tool = compute_hashes(project_root, results)
    for tool_str, record in tools.items():
        record.hashes = hashes_by_tool.get(tool_str, {})
    _replicate_shared_hashes(tools, scope)
    for r in results:
        if r.action == "error" and r.tool_id is not None and str(r.tool_id) in tools:
            tools[str(r.tool_id)].status = "failed"

    status = "partial" if _has_error(results) else "applied"
    return SceneState(scene=name, applied_at=now_iso(), status=status, tools=tools)


def _replicate_shared_hashes(tools: dict[str, SceneToolRecord], scope: list[AIToolID]) -> None:
    """Mirror a shared skills-dir hash onto every tool that resolves to it.

    A shared re-point (Codex + Antigravity CLI on ``.agents/skills``) is reported
    once, so its hash lands under a single tool. Copy it to the co-sharers so
    ``status --tool <either>`` and a scoped drift check both see it.
    """
    from crossby.config.skills import SKILLS_DIR

    for tool in scope:
        shared_dir = SKILLS_DIR.get(tool)
        if shared_dir is None or str(tool) not in tools:
            continue
        if shared_dir in tools[str(tool)].hashes:
            continue
        for other in scope:
            other_hashes = tools.get(str(other))
            if other_hashes is not None and shared_dir in other_hashes.hashes:
                tools[str(tool)].hashes[shared_dir] = other_hashes.hashes[shared_dir]
                break


def _tool_mechanisms(scene: SceneConfig, scope: list[AIToolID]) -> dict[str, SceneToolRecord]:
    """The mechanism each scope tool uses, per concern the scene declares.

    Mirrors the engine's *installed-tools + matrix* model rather than the
    resolver's per-source-directory attribution, so a PROJECT target with no
    source directory of its own (e.g. Cursor's skills) is still recorded.
    """
    from crossby.scenes.mechanism import base_mechanism
    from crossby.scenes.state import SceneToolRecord

    declared = [concern for concern in _CONCERN_ORDER if getattr(scene, concern) is not None]
    return {
        str(tool): SceneToolRecord(
            mechanisms={concern: base_mechanism(tool, concern).value for concern in declared}
        )
        for tool in scope
    }


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


@scene_app.command("clear")
def clear_active(
    tool: str | None = _TOOL_OPTION,
    plan: bool = typer.Option(False, "--plan", help="Preview the revert; write nothing."),
    force: bool = typer.Option(
        False, "--force", help="Proceed despite drift in the active scene (bypass the refusal)."
    ),
    path: Path = _PATH_OPTION,
) -> None:
    """Revert the active scene to the pre-scene baseline (from the state file)."""
    from crossby.scenes.engine import clear_scene
    from crossby.scenes.state import (
        SCENE_STATE_PATH,
        clear_scene_state,
        detect_drift,
        load_scene_state,
        save_scene_state,
    )
    from crossby.services.scene_resolution import scene_root

    project_root = path.resolve()
    root = scene_root(project_root)
    tool_id = _validate_tool(tool)

    loaded = load_scene_state(root)
    if loaded.warning:
        console.warn(loaded.warning)
    active = loaded.state
    if active is None:
        console.info("No active scene — nothing to clear.")
        return

    recorded = _recorded_tools(active)
    if not recorded:
        # The state exists but records only tool ids this build doesn't know —
        # we can't revert them, so leave the file in place rather than delete it
        # (deleting could orphan a ledger-owned setting).
        console.warn(f"{SCENE_STATE_PATH.as_posix()} records only unknown tools; left in place.")
        console.hint("Delete it manually if the scene is no longer applied.")
        return
    if tool_id is not None and tool_id not in recorded:
        console.info(f"Tool {tool_id} is not part of the active scene {active.scene!r}.")
        console.hint(f"Recorded tools: {', '.join(str(t) for t in recorded) or '(none)'}")
        return
    scope = _clear_scope(tool_id, recorded)

    # Fail closed on a corrupt ledger before BOTH the --plan preview and the real
    # clear: an empty-loaded ledger reverts nothing yet the engine's finally:
    # save_ledger would overwrite the corrupt bytes, and the CLI would then delete
    # scene-state.json — leaving settings applied with no recovery record.
    _refuse_if_ledger_corrupt(root)

    # --plan writes nothing, so it always previews — even against a drifted scene.
    if plan:
        results = clear_scene(root, dry_run=True, tools=scope)
        _display_results(results)
        console.info("(--plan) no changes written.")
        if _has_error(results):
            raise typer.Exit(1)
        return

    # Drift is scoped to the tools being reverted, so an unrelated tool's drift
    # neither blocks nor is reported by a --tool clear.
    drifted = detect_drift(root, active, tools=[str(t) for t in scope])
    if drifted and not force:
        _report_drift_refusal(active.scene, drifted, verb="clear")
        raise typer.Exit(1)

    # The review may change the target tool (interactive only); honour it.
    new_tool_id = _confirm_scene_defaults(
        action="clear", scene=active.scene, tool_id=tool_id, installed=recorded
    )
    if new_tool_id != tool_id:
        tool_id = new_tool_id
        scope = _clear_scope(tool_id, recorded)

    results = _call_engine_or_exit(clear_scene, root, tools=scope)
    _display_results(results)

    # A failed clear leaves the state untouched so the revert can be retried —
    # never delete the only record of what to revert on a partial failure.
    if _has_error(results):
        console.error("Clear failed for some tools — state left intact for retry.")
        raise typer.Exit(1)

    # On success update the state file: a full clear removes it; a scoped clear
    # drops the cleared tools (their per-tool hashes go with them). The engine has
    # already reverted the in-scope tools, so any OSError here (read-only dir,
    # permission on unlink) is "cleared but state stale" — surface it cleanly,
    # never a traceback, mirroring `use`'s "applied but state could not be
    # recorded." All three finalize call sites route through this one handler
    # (clear_scene_state only suppresses FileNotFoundError, so a PermissionError
    # on unlink would otherwise escape).
    try:
        if tool_id is None:
            clear_scene_state(root)
        else:
            for cleared in scope:
                active.tools.pop(str(cleared), None)
            if active.tools:
                save_scene_state(root, active)
            else:
                clear_scene_state(root)
    except OSError as exc:
        console.error(f"Scene cleared but its state could not be updated: {exc}")
        console.hint("Re-run 'crossby scene clear' once the path is writable.")
        raise typer.Exit(1) from exc

    console.success(f"Cleared scene {active.scene!r}.")


def _clear_scope(tool_id: AIToolID | None, recorded: list[AIToolID]) -> list[AIToolID]:
    """The recorded tools a clear targets: one tool (plus shared-dir co-sharers),
    or every recorded tool."""
    if tool_id is None:
        return recorded
    return _expand_shared_scope([tool_id], recorded)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@scene_app.command("status")
def scene_status(
    tool: str | None = _TOOL_OPTION,
    path: Path = _PATH_OPTION,
) -> None:
    """Report the active scene, its per-tool mechanism, and any drift."""
    from crossby.config.loader import ConfigError, load_config
    from crossby.scenes.state import detect_drift, load_scene_state
    from crossby.services.scene_resolution import scene_root

    project_root = path.resolve()
    root = scene_root(project_root)
    tool_id = _validate_tool(tool)

    loaded = load_scene_state(root)
    if loaded.warning:
        console.warn(loaded.warning)
    active = loaded.state
    if active is None:
        console.info("No active scene.")
        return

    # Load the config only to answer "is this scene still defined?" — status must
    # not crash when the scene was renamed or deleted after it was applied.
    scene_defined = False
    try:
        config = load_config(project_root)
        scene_defined = config.get_scene(active.scene) is not None
    except ConfigError:
        scene_defined = False

    console.section(f"Active scene: {active.scene}")
    console.kv("Applied", active.applied_at)
    console.kv("Status", active.status)
    if not scene_defined:
        console.warn(f"scene {active.scene!r} is no longer defined in .crossby.yml")

    _render_status_tools(active, tool_id)

    # With --tool, report drift only for that tool.
    drift_tools = None if tool_id is None else [str(tool_id)]
    drifted = detect_drift(root, active, tools=drift_tools)
    if drifted:
        console.warn("Drift detected — scene-managed files changed since apply:")
        for rel in drifted:
            console.detail(rel)
        console.hint("Re-run 'crossby scene use <name> --force', or 'crossby scene clear'.")
    else:
        console.success("No drift — scene-managed files match apply time.")


def _render_status_tools(active: SceneState, tool_id: AIToolID | None) -> None:
    from rich.table import Table

    tools = active.tools
    wanted = None if tool_id is None else str(tool_id)

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Tool", style="dim")
    table.add_column("Concern")
    table.add_column("Mechanism")
    table.add_column("Status")
    rows = 0
    for tool_name in sorted(tools):
        if wanted is not None and tool_name != wanted:
            continue
        record = tools[tool_name]
        if not record.mechanisms:
            table.add_row(tool_name, "(none)", "", record.status)
            rows += 1
            continue
        for concern in _CONCERN_ORDER:
            if concern in record.mechanisms:
                table.add_row(
                    tool_name,
                    concern,
                    _mechanism_label_str(record.mechanisms[concern]),
                    record.status,
                )
                rows += 1
    if rows:
        console.out.print(table)
    elif wanted is not None:
        console.info(f"Scene records no state for tool {wanted!r}.")


def _mechanism_label_str(value: str) -> str:
    return f"[warning]{value}[/]" if value == "unsupported" else value


# ---------------------------------------------------------------------------
# Shared drift reporting
# ---------------------------------------------------------------------------


def _report_drift_refusal(scene: str, drifted: list[str], *, verb: str) -> None:
    console.error(f"Refusing to {verb} scene {scene!r}: its managed files have drifted.")
    for rel in drifted:
        console.detail(rel)
    console.hint("Re-run with --force to overwrite the drift.")


# ---------------------------------------------------------------------------
# Authoring — shared helpers (create / add / remove / delete / install-starters)
# ---------------------------------------------------------------------------


def _collect_selector_edits(
    *,
    skill: list[str],
    exclude_skill: list[str],
    agent: list[str],
    exclude_agent: list[str],
    mcp: list[str],
    exclude_mcp: list[str],
    hook: list[str],
    exclude_hook: list[str],
    permission: list[str],
    exclude_permission: list[str],
) -> list[SelectorEdit]:
    """Turn the raw selector flags into SelectorEdits, dropping empty channels."""
    from crossby.scenes.authoring import SelectorEdit

    spec: tuple[tuple[str, list[str], bool], ...] = (
        ("skills", skill, False),
        ("skills", exclude_skill, True),
        ("agents", agent, False),
        ("agents", exclude_agent, True),
        ("mcp", mcp, False),
        ("mcp", exclude_mcp, True),
        ("hooks", hook, False),
        ("hooks", exclude_hook, True),
        ("permissions", permission, False),
        ("permissions", exclude_permission, True),
    )
    return [
        SelectorEdit(concern, tuple(values), exclude) for concern, values, exclude in spec if values
    ]


def _apply_scalar_updates(
    scene: SceneConfig,
    description: str | None,
    extends: str | None,
    profile: str | None,
) -> SceneConfig:
    """Override description/extends/profile for any flag that was given.

    An empty string clears the field (``--description ""`` drops it), so a scalar
    can be removed as well as set.
    """
    updates: dict[str, str | None] = {}
    if description is not None:
        updates["description"] = description or None
    if extends is not None:
        updates["extends"] = extends or None
    if profile is not None:
        updates["profile"] = profile or None
    return scene.model_copy(update=updates) if updates else scene


def _report_moves(moves: list[CrossChannelMove], *, print_: bool = False) -> None:
    # In --print mode stdout must stay pure YAML, so defer these info notices
    # (they describe a side effect of the edit, not the rendered block itself).
    if print_:
        return
    # Escape: glob patterns can contain [...] (fnmatch char classes), which Rich
    # would otherwise interpret as markup and strip from the message.
    from rich.markup import escape

    for move in moves:
        console.info(escape(move.describe()))


def _report_missing(missing: list[str]) -> None:
    from rich.markup import escape

    for pattern in missing:
        console.warn(escape(f"selector {pattern!r} was not present — nothing removed."))


def _require_resolves(name: str) -> Callable[[CrossbyConfig], None]:
    """A write_config_checked validator asserting scene *name* still resolves.

    A structural parse accepts an ``extends`` that names a missing scene or forms
    a cycle — those surface only in ``get_scene``. Failing here lets
    write_config_checked roll the file back before the user is left with a config
    that parses but can never be applied.
    """

    def _validate(config: CrossbyConfig) -> None:
        from crossby.config.loader import ConfigError

        if config.get_scene(name) is None:  # pragma: no cover — just written
            raise ConfigError(f"scene {name!r} did not round-trip through the loader")

    return _validate


def _config_target(config: CrossbyConfig, project_root: Path) -> Path:
    """The ``.crossby.yml`` an authoring command should write.

    Prefer the loaded config's own path so an edit run from a subdirectory
    targets the same file it *read* (``load_config`` walks up), instead of
    writing a shadow config into the subdirectory. Falls back to
    ``project_root/.crossby.yml`` only when no config was found (a fresh create).
    """
    return Path(config.config_path) if config.config_path else project_root / ".crossby.yml"


def _write_scene_entry(target: Path, name: str, scene: SceneConfig, *, print_: bool) -> None:
    """Splice *scene* into *target* (or print it), reporting the outcome.

    With ``--print`` the rendered entry goes to stdout and nothing is written;
    otherwise the write is backed up, re-parsed, resolution-checked, and rolled
    back byte-for-byte on any error.
    """
    from crossby.config.safe_write import ConfigWriteError, write_config_checked
    from crossby.scenes.authoring import (
        SceneAuthoringError,
        render_scene_entry,
        splice_scene_text,
    )

    if print_:
        # Raw YAML: disable Rich markup (flow lists like ``[a, b]`` read as tags)
        # and wrapping so the block stays pasteable.
        console.out.print(
            render_scene_entry(name, scene).rstrip("\n"),
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
        return

    # read_bytes().decode() rather than read_text() so a CRLF file's line endings
    # are not silently normalised to LF, which would rewrite every line.
    base = target.read_bytes().decode("utf-8") if target.exists() else "version: 1\n"
    try:
        new_text = splice_scene_text(base, name, scene)
    except SceneAuthoringError as exc:
        console.error(f"Cannot edit {target.name}: {exc}")
        raise typer.Exit(1) from exc
    try:
        write_config_checked(target, new_text, validate=_require_resolves(name))
    except ConfigWriteError as exc:
        console.error(f"Refusing to write scene {name!r}: {exc.original}")
        console.hint(
            ".crossby.yml was left unchanged." if exc.restored else "No config file was written."
        )
        raise typer.Exit(1) from exc
    console.success(f"Wrote scene {name!r} to {target.name}.")


# ---------------------------------------------------------------------------
# add / remove — non-interactive selector editing of an existing scene
# ---------------------------------------------------------------------------


@scene_app.command("add")
def add_to_scene(
    name: str = typer.Argument(..., help="Existing scene to append selectors to."),
    skill: list[str] = _SKILL_OPT,
    exclude_skill: list[str] = _EXCLUDE_SKILL_OPT,
    agent: list[str] = _AGENT_OPT,
    exclude_agent: list[str] = _EXCLUDE_AGENT_OPT,
    mcp: list[str] = _MCP_OPT,
    exclude_mcp: list[str] = _EXCLUDE_MCP_OPT,
    hook: list[str] = _HOOK_OPT,
    exclude_hook: list[str] = _EXCLUDE_HOOK_OPT,
    permission: list[str] = _PERMISSION_OPT,
    exclude_permission: list[str] = _EXCLUDE_PERMISSION_OPT,
    description: str | None = _DESCRIPTION_OPT,
    extends: str | None = _EXTENDS_OPT,
    profile: str | None = _PROFILE_OPT,
    print_: bool = _PRINT_OPT,
    path: Path = _PATH_OPTION,
) -> None:
    """Append selectors (and optionally set description/extends/profile) on a scene.

    Adding a pattern to one channel removes it from the other, so include and
    exclude can never contradict; each such move is reported. Idempotent — adding
    an already-present selector changes nothing.
    """
    from crossby.scenes.authoring import add_selectors

    project_root = path.resolve()
    config = _load_config_or_exit(project_root)
    if name not in config.scenes:
        console.error(f"Unknown scene: {name!r}")
        console.hint("Use 'crossby scene create' to define a new scene.")
        raise typer.Exit(1)

    edits = _collect_selector_edits(
        skill=skill,
        exclude_skill=exclude_skill,
        agent=agent,
        exclude_agent=exclude_agent,
        mcp=mcp,
        exclude_mcp=exclude_mcp,
        hook=hook,
        exclude_hook=exclude_hook,
        permission=permission,
        exclude_permission=exclude_permission,
    )
    if not edits and description is None and extends is None and profile is None:
        console.error("Nothing to add.")
        console.hint(
            "Pass a selector flag (e.g. --skill review-*) or --description/--extends/--profile."
        )
        raise typer.Exit(1)

    # Edit the *raw* entry, not the extends-flattened scene, so a child keeps
    # inheriting from its parent instead of the parent being inlined.
    scene, moves = add_selectors(config.scenes[name], edits)
    scene = _apply_scalar_updates(scene, description, extends, profile)
    _report_moves(moves, print_=print_)
    _write_scene_entry(_config_target(config, project_root), name, scene, print_=print_)


@scene_app.command("remove")
def remove_from_scene(
    name: str = typer.Argument(..., help="Existing scene to remove selectors from."),
    skill: list[str] = _SKILL_OPT,
    exclude_skill: list[str] = _EXCLUDE_SKILL_OPT,
    agent: list[str] = _AGENT_OPT,
    exclude_agent: list[str] = _EXCLUDE_AGENT_OPT,
    mcp: list[str] = _MCP_OPT,
    exclude_mcp: list[str] = _EXCLUDE_MCP_OPT,
    hook: list[str] = _HOOK_OPT,
    exclude_hook: list[str] = _EXCLUDE_HOOK_OPT,
    permission: list[str] = _PERMISSION_OPT,
    exclude_permission: list[str] = _EXCLUDE_PERMISSION_OPT,
    print_: bool = _PRINT_OPT,
    path: Path = _PATH_OPTION,
) -> None:
    """Remove selector patterns from a scene's include/exclude channels.

    Idempotent — removing a pattern that is not present changes nothing and is
    reported as a no-op.
    """
    from crossby.scenes.authoring import remove_selectors

    project_root = path.resolve()
    config = _load_config_or_exit(project_root)
    if name not in config.scenes:
        console.error(f"Unknown scene: {name!r}")
        raise typer.Exit(1)

    edits = _collect_selector_edits(
        skill=skill,
        exclude_skill=exclude_skill,
        agent=agent,
        exclude_agent=exclude_agent,
        mcp=mcp,
        exclude_mcp=exclude_mcp,
        hook=hook,
        exclude_hook=exclude_hook,
        permission=permission,
        exclude_permission=exclude_permission,
    )
    if not edits:
        console.error("Nothing to remove.")
        console.hint("Pass a selector flag (e.g. --skill review-* or --exclude-mcp linear).")
        raise typer.Exit(1)

    scene, missing = remove_selectors(config.scenes[name], edits)
    _report_missing(missing)
    _write_scene_entry(_config_target(config, project_root), name, scene, print_=print_)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@scene_app.command("delete")
def delete_scene(
    name: str = typer.Argument(..., help="Scene to delete."),
    force: bool = typer.Option(
        False, "--force", help="Delete even if the scene is active or extended by another."
    ),
    path: Path = _PATH_OPTION,
) -> None:
    """Drop a scene's entry from ``.crossby.yml``.

    Refuses to delete the currently active scene (clear it first, or pass
    ``--force``) and refuses when another scene ``extends`` it, since that would
    leave the child unresolvable.
    """
    from crossby.config.safe_write import ConfigWriteError, write_config_checked
    from crossby.scenes.authoring import SceneAuthoringError, remove_scene_text
    from crossby.scenes.state import load_scene_state
    from crossby.services.scene_resolution import scene_root

    project_root = path.resolve()
    config = _load_config_or_exit(project_root)
    root = scene_root(project_root)
    if name not in config.scenes:
        console.error(f"Unknown scene: {name!r}")
        available = sorted(config.scenes)
        if available:
            console.hint(f"Available scenes: {', '.join(available)}")
        raise typer.Exit(1)

    active = load_scene_state(root).state
    is_active = active is not None and active.scene == name
    if is_active and not force:
        console.error(f"Scene {name!r} is currently active; refusing to delete it.")
        console.hint("Run 'crossby scene clear' first, or pass --force to delete anyway.")
        raise typer.Exit(1)

    dependents = sorted(s for s, sc in config.scenes.items() if sc.extends == name and s != name)
    if dependents and not force:
        joined = ", ".join(dependents)
        console.error(f"Scene(s) {joined} extend {name!r}; deleting it would break them.")
        console.hint("Update or delete those first, or pass --force.")
        raise typer.Exit(1)

    # --force skips the refusals above; spell out the state it leaves behind so the
    # user is not silently left with a dangling scene reference.
    if force:
        if is_active:
            console.warn(
                f"Scene {name!r} is active; scene-state.json still records it — "
                "run 'crossby scene clear' to revert the tools it applied."
            )
        if dependents:
            joined = ", ".join(dependents)
            console.warn(
                f"Scene(s) {joined} still extend {name!r}; their 'extends' is now "
                "unresolvable until you update or delete them."
            )

    target = _config_target(config, project_root)
    text = target.read_bytes().decode("utf-8")
    try:
        new_text, found = remove_scene_text(text, name)
    except SceneAuthoringError as exc:
        console.error(f"Cannot edit {target.name}: {exc}")
        raise typer.Exit(1) from exc
    if not found:  # pragma: no cover — guarded by the config.scenes check above
        console.error(f"Scene {name!r} was not found in {target.name}.")
        raise typer.Exit(1)
    try:
        write_config_checked(target, new_text)
    except ConfigWriteError as exc:
        console.error(f"Could not delete scene {name!r}: {exc.original}")
        raise typer.Exit(1) from exc
    console.success(f"Deleted scene {name!r}.")


# ---------------------------------------------------------------------------
# install-starters
# ---------------------------------------------------------------------------


@scene_app.command("install-starters")
def install_starters(
    print_: bool = _PRINT_OPT,
    path: Path = _PATH_OPTION,
) -> None:
    """Install the bundled starter scenes, skipping any name already defined.

    Idempotent: a re-run installs nothing new. Same-named user scenes are left
    untouched. Each starter uses glob selectors, so it stays valid even in a
    project that has none of the skills/servers it names.
    """
    from crossby.config.safe_write import ConfigWriteError, write_config_checked
    from crossby.scenes.authoring import (
        SceneAuthoringError,
        render_scene_entry,
        splice_scene_text,
    )
    from crossby.scenes.starters import load_starter_scenes

    project_root = path.resolve()
    starters = load_starter_scenes()
    config = _load_config_or_exit(project_root)
    target = _config_target(config, project_root)
    existing = set(config.scenes)
    installed = [name for name in starters if name not in existing]
    skipped = [name for name in starters if name in existing]

    if print_:
        # --print never edits the file, so skip splicing entirely (splicing a
        # flow-style ``scenes:`` block would raise even though we only render) and
        # keep skipped/all-present notices off stdout — emit only valid YAML.
        for name in installed:
            console.out.print(
                render_scene_entry(name, starters[name]).rstrip("\n"),
                markup=False,
                highlight=False,
                soft_wrap=True,
            )
        return

    text = target.read_bytes().decode("utf-8") if target.exists() else "version: 1\n"
    try:
        for name in installed:
            text = splice_scene_text(text, name, starters[name])
    except SceneAuthoringError as exc:
        console.error(f"Cannot edit {target.name}: {exc}")
        raise typer.Exit(1) from exc

    for name in skipped:
        console.info(f"Skipped {name!r} — a scene with that name already exists.")
    if not installed:
        console.info("All starter scenes are already present.")
        return
    try:
        write_config_checked(target, text)
    except ConfigWriteError as exc:
        console.error(f"Could not install starters: {exc.original}")
        raise typer.Exit(1) from exc
    console.success(f"Installed starter scene(s): {', '.join(installed)}.")


# ---------------------------------------------------------------------------
# create — interactive wizard (TTY) or flag-driven (non-TTY)
# ---------------------------------------------------------------------------


@scene_app.command("create")
def create_scene(
    name: str | None = typer.Argument(None, help="Name for the new scene (prompted if omitted)."),
    skill: list[str] = _SKILL_OPT,
    exclude_skill: list[str] = _EXCLUDE_SKILL_OPT,
    agent: list[str] = _AGENT_OPT,
    exclude_agent: list[str] = _EXCLUDE_AGENT_OPT,
    mcp: list[str] = _MCP_OPT,
    exclude_mcp: list[str] = _EXCLUDE_MCP_OPT,
    hook: list[str] = _HOOK_OPT,
    exclude_hook: list[str] = _EXCLUDE_HOOK_OPT,
    permission: list[str] = _PERMISSION_OPT,
    exclude_permission: list[str] = _EXCLUDE_PERMISSION_OPT,
    description: str | None = _DESCRIPTION_OPT,
    extends: str | None = _EXTENDS_OPT,
    profile: str | None = _PROFILE_OPT,
    print_: bool = _PRINT_OPT,
    path: Path = _PATH_OPTION,
) -> None:
    """Create a new scene — an interactive wizard on a TTY, flags otherwise.

    On a TTY this multi-selects over the skills/agents/servers/hooks/permissions
    discovered in the project, then a confirm-and-review step. With stdin not a
    TTY it builds the scene from the selector flags alone and never prompts — and
    refuses (non-zero) if no selector flag was given, rather than silently
    selecting every item.

    With ``--print`` on a TTY, the entire interactive phase (name prompt through
    the wizard) runs with stdout redirected to stderr — questionary/prompt-toolkit
    resolves ``sys.stdout`` at call time regardless of what ``is_tty()`` (which
    gates on stdin) reports, so a piped stdout would otherwise get interactive
    menus interleaved with the final YAML. Only the final rendered block, printed
    after this function returns to its caller, reaches real stdout.
    """
    from crossby.models.config import SceneConfig
    from crossby.scenes.authoring import add_selectors
    from crossby.services.scene_resolution import scene_root
    from crossby.ui import prompts

    project_root = path.resolve()
    config = _load_config_or_exit(project_root)
    root = scene_root(project_root)
    seed_edits = _collect_selector_edits(
        skill=skill,
        exclude_skill=exclude_skill,
        agent=agent,
        exclude_agent=exclude_agent,
        mcp=mcp,
        exclude_mcp=exclude_mcp,
        hook=hook,
        exclude_hook=exclude_hook,
        permission=permission,
        exclude_permission=exclude_permission,
    )

    if prompts.is_tty():
        import contextlib
        import sys

        redirect = contextlib.redirect_stdout(sys.stderr) if print_ else contextlib.nullcontext()
        with redirect:
            if name is None:
                name = _prompt_new_scene_name(config)
                if name is None:
                    raise typer.Exit(1)
            _refuse_if_scene_exists(config, name)
            scene = _run_create_wizard(
                root, config, seed_edits, description, extends, profile, print_=print_
            )
    else:
        if name is None:
            console.error("A scene name is required.")
            console.hint("Usage: crossby scene create <name> [--skill ... --exclude-mcp ...].")
            raise typer.Exit(1)
        _refuse_if_scene_exists(config, name)
        # Non-TTY guard sits here, before any prompt could run: require explicit
        # selectors rather than fall through to multi_select's "select everything".
        if not seed_edits:
            console.error("Interactive prompts require a TTY.")
            console.hint(
                "Pass selectors (e.g. --skill review-* --exclude-mcp linear) "
                "to build a scene non-interactively."
            )
            raise typer.Exit(1)
        base = SceneConfig(
            description=description or None, extends=extends or None, profile=profile or None
        )
        scene, moves = add_selectors(base, seed_edits)
        _report_moves(moves, print_=print_)

    _write_scene_entry(_config_target(config, project_root), name, scene, print_=print_)


def _refuse_if_scene_exists(config: CrossbyConfig, name: str) -> None:
    if name in config.scenes:
        console.error(f"Scene {name!r} already exists.")
        console.hint("Use 'crossby scene add' to modify it, or choose another name.")
        raise typer.Exit(1)


def _prompt_new_scene_name(config: CrossbyConfig) -> str | None:
    from crossby.ui import prompts

    name = prompts.input_prompt("New scene name", allow_empty=True).strip()
    if not name:
        console.info("No name given — aborting.")
        return None
    if name in config.scenes:
        console.error(f"Scene {name!r} already exists.")
        return None
    return name


def _pick_optional(title: str, names: list[str]) -> str | None:
    """Pick one of *names* or ``(none)``; ``None`` when there is nothing to pick."""
    from crossby.ui import prompts

    if not names:
        return None
    choices = ["(none)", *names]
    idx = prompts.select(title, choices)
    return None if idx == 0 else choices[idx]


def _scene_from_selectors(
    *,
    description: str | None,
    extends: str | None,
    profile: str | None,
    selectors: dict[str, SceneSelector | None],
) -> SceneConfig:
    """Build a :class:`SceneConfig` from scalar fields and a per-concern map."""
    from crossby.models.config import SceneConfig

    return SceneConfig(
        description=description,
        extends=extends,
        profile=profile,
        skills=selectors.get("skills"),
        agents=selectors.get("agents"),
        mcp=selectors.get("mcp"),
        hooks=selectors.get("hooks"),
        permissions=selectors.get("permissions"),
    )


def _concern_visible(
    universe: dict[str, tuple[str, ...]], concern: str
) -> Callable[[dict[str, object]], bool]:
    """A ``visible_when`` predicate: show a concern only if it has candidates.

    Built as a factory so each concern binds its own value (a loop-local lambda
    would capture the last concern instead).
    """
    return lambda _state: bool(universe[concern])


def _render_names(value: object) -> str | None:
    """Render a selected-name list for the review's ``kv`` line (``None`` skips)."""
    from typing import cast

    if not value:
        return None
    return ", ".join(cast("list[str]", value))


def _run_create_wizard(
    root: Path,
    config: CrossbyConfig,
    seed_edits: list[SelectorEdit],
    description: str | None,
    extends: str | None,
    profile: str | None,
    *,
    print_: bool = False,
) -> SceneConfig:
    """Walk discovered items with multi-select, then a confirm/review step.

    Only ever called on a TTY (guarded by the caller). Any selector flags already
    given are folded in on top of the interactive picks, so ``create --exclude-mcp
    linear`` still works interactively. *root* is the config-rooted scene root
    (see :func:`crossby.services.scene_resolution.scene_root`), not necessarily
    the invocation directory.
    """
    from crossby.ai_tools.base import AbstractAITool
    from crossby.models.config import SCENE_CONCERNS, SceneSelector
    from crossby.scenes.authoring import add_selectors
    from crossby.services.scene_resolution import concern_universe
    from crossby.sync.readers import scan_project
    from crossby.ui import prompts

    installed = AbstractAITool.detect_installed()
    scan = scan_project(root, installed)
    universe = concern_universe(scan, root)

    selectors: dict[str, SceneSelector | None] = {}
    for concern in SCENE_CONCERNS:
        items = list(universe[concern])
        if not items:
            console.info(f"No {concern} detected in the project — skipping.")
            continue
        picked = prompts.multi_select(f"Include which {concern}?", items)
        chosen = [items[i] for i in picked]
        if chosen:
            selectors[concern] = SceneSelector(include=chosen)
        # Selecting nothing leaves the concern unset (an unfiltered concern —
        # every item kept). This is deliberate: scenes *remove* capability, so a
        # concern the user skipped must stay whole rather than be emptied. To
        # build a scene that selects none of a concern, use `--exclude-<concern>`.

    if description is None:
        description = prompts.input_prompt("Description (optional)", allow_empty=True) or None
    else:
        description = description or None
    extends = (
        extends or None
        if extends is not None
        else _pick_optional("Extend a parent scene?", sorted(config.scenes))
    )
    profile = (
        profile or None
        if profile is not None
        else _pick_optional("Default launch profile?", sorted(config.profiles))
    )

    scene = _scene_from_selectors(
        description=description, extends=extends, profile=profile, selectors=selectors
    )
    if seed_edits:
        scene, moves = add_selectors(scene, seed_edits)
        _report_moves(moves, print_=print_)

    return _review_scene(config, scene, universe)


def _review_scene(
    config: CrossbyConfig,
    initial: SceneConfig,
    universe: dict[str, tuple[str, ...]],
) -> SceneConfig:
    """The ``confirm_defaults`` review — mirror of ``crossby init``'s final step."""
    from typing import Any, cast

    from crossby.models.config import SCENE_CONCERNS, SceneSelector
    from crossby.services.confirm import ConfirmField, confirm_defaults
    from crossby.ui import prompts

    state: dict[str, Any] = {
        "description": initial.description,
        "extends": initial.extends,
        "profile": initial.profile,
    }
    excludes: dict[str, list[str]] = {}
    for concern in SCENE_CONCERNS:
        sel: SceneSelector | None = getattr(initial, concern)
        state[concern] = list(sel.include) if sel and sel.include is not None else None
        excludes[concern] = list(sel.exclude) if sel else []

    def _desc_change(_current: Any, _state: dict[str, Any]) -> dict[str, Any]:
        return {"description": prompts.input_prompt("Description", allow_empty=True) or None}

    def _extends_change(_current: Any, _state: dict[str, Any]) -> dict[str, Any]:
        return {"extends": _pick_optional("Extend a parent scene?", sorted(config.scenes))}

    def _profile_change(_current: Any, _state: dict[str, Any]) -> dict[str, Any]:
        return {"profile": _pick_optional("Default launch profile?", sorted(config.profiles))}

    def _concern_change(concern: str) -> Callable[[Any, dict[str, Any]], dict[str, Any]]:
        def _change(_current: Any, _state: dict[str, Any]) -> dict[str, Any]:
            items = list(universe[concern])
            picked = prompts.multi_select(f"Include which {concern}?", items)
            chosen = [items[i] for i in picked]
            return {concern: chosen or None}

        return _change

    fields: list[ConfirmField] = [
        ConfirmField(
            name="description",
            label="Description",
            current_value=state["description"],
            explicit=False,
            change_fn=_desc_change,
        ),
        ConfirmField(
            name="extends",
            label="Extends",
            current_value=state["extends"],
            explicit=False,
            change_fn=_extends_change,
            visible_when=lambda _s: bool(config.scenes),
        ),
        ConfirmField(
            name="profile",
            label="Profile",
            current_value=state["profile"],
            explicit=False,
            change_fn=_profile_change,
            visible_when=lambda _s: bool(config.profiles),
        ),
    ]
    for concern in SCENE_CONCERNS:
        fields.append(
            ConfirmField(
                name=concern,
                label=f"{concern.capitalize()} include",
                current_value=state[concern],
                explicit=False,
                change_fn=_concern_change(concern),
                visible_when=_concern_visible(universe, concern),
                render_value=_render_names,
            )
        )

    result = confirm_defaults(fields, title="Confirm scene")

    selectors: dict[str, SceneSelector | None] = {}
    for concern in SCENE_CONCERNS:
        include = cast("list[str] | None", result[concern])
        exclude = excludes[concern]
        # The user may have re-selected (into include) an item carried in the
        # exclude channel from a seed flag. Keep the cross-channel invariant by
        # letting the interactive include win, so the two never contradict.
        if include is not None:
            exclude = [pattern for pattern in exclude if pattern not in include]
        selectors[concern] = (
            SceneSelector(include=include, exclude=exclude)
            if include is not None or exclude
            else None
        )
    return _scene_from_selectors(
        description=cast("str | None", result["description"]) or None,
        extends=cast("str | None", result["extends"]),
        profile=cast("str | None", result["profile"]),
        selectors=selectors,
    )
