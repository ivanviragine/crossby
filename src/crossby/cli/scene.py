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
    from crossby.models.ai import AIToolID
    from crossby.models.config import CrossbyConfig, SceneConfig
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


def _scope_for(tool_id: AIToolID | None, installed: list[AIToolID]) -> list[AIToolID]:
    """The installed tools an operation targets: one tool, or all installed."""
    if tool_id is None:
        return installed
    if tool_id not in installed:
        console.error(f"Tool {tool_id} is not installed.")
        console.hint(f"Installed tools: {', '.join(str(t) for t in installed)}")
        raise typer.Exit(1)
    return [tool_id]


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
    project_root = path.resolve()
    config = _load_config_or_exit(project_root)
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
    scan = scan_project(project_root, installed)

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Scene")
    table.add_column("Description", style="dim")
    table.add_column("Concerns")

    for name in sorted(config.scenes):
        description = config.scenes[name].description or ""
        counts = _concern_counts(config, name, project_root, scan, tool_id)
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
    project_root = path.resolve()
    config = _load_config_or_exit(project_root)
    tool_id = _validate_tool(tool)
    installed = _installed_or_exit()

    resolved = _resolve(config, name, project_root, installed, tool_id=tool_id)

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
        False, "--force", help="Proceed despite drift in the outgoing scene, overwriting it."
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

    project_root = path.resolve()
    config = _load_config_or_exit(project_root)
    tool_id = _validate_tool(tool)
    installed = _installed_or_exit()
    scope = _scope_for(tool_id, installed)

    # Resolve the full union (every tool) so the disable sets stay anchored on the
    # real inventory; the apply is narrowed to `scope` via the tools= argument.
    resolved = _resolve(config, name, project_root, installed)

    loaded = load_scene_state(project_root)
    if loaded.warning:
        console.warn(loaded.warning)
    active = loaded.state

    # --plan writes nothing, so it always previews — even against a drifted scene.
    if plan:
        results = apply_scene(resolved, project_root, dry_run=True, force=force, tools=scope)
        _display_results(results)
        console.info("(--plan) no changes written.")
        if _has_error(results):
            raise typer.Exit(1)
        return

    # Drift on the outgoing scene must be checked before we revert it — reverting
    # a hand-edited scene would silently discard that work.
    if active is not None:
        drifted = detect_drift(project_root, active)
        if drifted and not force:
            _report_drift_refusal(active.scene, drifted, verb="switch from")
            raise typer.Exit(1)

    # The review may change the target tool (interactive only); honour it.
    tool_id = _confirm_scene_defaults(
        action="use", scene=name, tool_id=tool_id, installed=installed
    )
    scope = _scope_for(tool_id, installed)

    # Switch: revert the active scene first, scoped to exactly the tools it was
    # applied to, so the new scene applies from the true pre-scene baseline.
    if active is not None:
        _revert_active(project_root, active)

    results = apply_scene(resolved, project_root, force=force, tools=scope)
    _display_results(results)

    scene = _get_scene_or_exit(config, name)
    state = _build_state(project_root, name, scene, scope, results)
    save_scene_state(project_root, state)

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


def _revert_active(project_root: Path, active: SceneState) -> None:
    from crossby.scenes.engine import clear_scene

    # Revert exactly the recorded tools (the engine filters to installed itself
    # for the source re-points). An empty list must mean "revert nothing" — never
    # collapse to None, which the engine reads as "every installed tool".
    revert_tools = _recorded_tools(active)
    if revert_tools:
        clear_scene(project_root, tools=revert_tools)


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
    for r in results:
        if r.action == "error" and r.tool_id is not None and str(r.tool_id) in tools:
            tools[str(r.tool_id)].status = "failed"

    status = "partial" if _has_error(results) else "applied"
    return SceneState(scene=name, applied_at=now_iso(), status=status, tools=tools)


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
        False, "--force", help="Proceed despite drift in the active scene, overwriting it."
    ),
    path: Path = _PATH_OPTION,
) -> None:
    """Revert the active scene to the pre-scene baseline (from the state file)."""
    from crossby.scenes.engine import clear_scene
    from crossby.scenes.state import (
        clear_scene_state,
        detect_drift,
        load_scene_state,
        save_scene_state,
    )

    project_root = path.resolve()
    tool_id = _validate_tool(tool)

    loaded = load_scene_state(project_root)
    if loaded.warning:
        console.warn(loaded.warning)
    active = loaded.state
    if active is None:
        console.info("No active scene — nothing to clear.")
        return

    recorded = _recorded_tools(active)
    if tool_id is not None and tool_id not in recorded:
        console.info(f"Tool {tool_id} is not part of the active scene {active.scene!r}.")
        console.hint(f"Recorded tools: {', '.join(str(t) for t in recorded) or '(none)'}")
        return
    scope = [tool_id] if tool_id is not None else recorded

    # --plan writes nothing, so it always previews — even against a drifted scene.
    if plan:
        results = clear_scene(project_root, dry_run=True, tools=scope)
        _display_results(results)
        console.info("(--plan) no changes written.")
        if _has_error(results):
            raise typer.Exit(1)
        return

    drifted = detect_drift(project_root, active)
    if drifted and not force:
        _report_drift_refusal(active.scene, drifted, verb="clear")
        raise typer.Exit(1)

    # The review may change the target tool (interactive only); honour it.
    tool_id = _confirm_scene_defaults(
        action="clear", scene=active.scene, tool_id=tool_id, installed=recorded
    )
    scope = [tool_id] if tool_id is not None else recorded

    results = clear_scene(project_root, tools=scope)
    _display_results(results)

    # Update the state file: a full clear removes it; a scoped clear drops just
    # that tool (its per-tool hashes go with it, so no re-baseline is needed).
    if tool_id is None:
        clear_scene_state(project_root)
    else:
        active.tools.pop(str(tool_id), None)
        if active.tools:
            save_scene_state(project_root, active)
        else:
            clear_scene_state(project_root)

    if _has_error(results):
        raise typer.Exit(1)
    console.success(f"Cleared scene {active.scene!r}.")


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

    project_root = path.resolve()
    tool_id = _validate_tool(tool)

    loaded = load_scene_state(project_root)
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

    drifted = detect_drift(project_root, active)
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
