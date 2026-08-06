"""Scene apply / clear orchestration.

:func:`apply_scene` enacts a resolved scene on every installed tool, choosing the
least-invasive mechanism per concern: write a DECLARE key (disabling the
deselected remainder), re-point a tool directory at a projected filtered source,
drive the revocable-sync removal channel (hooks / permissions), or report an
unsupported cell. :func:`clear_scene` reverts every crossby-owned DECLARE key,
removes the projection, and re-points each tool back at its unfiltered source.

The engine is driven by *installed tools + the mechanism matrix + the union of
selected names* (``ResolvedScene.names``), not by the resolver's per-directory
groups. That matters for scene **switching**: once scene A re-points a tool's
directory at a filtered tree, re-resolving scene B would enumerate that filtered
view and miss items — but the union selection stays correct because it is
anchored on the real canonical source, and the projection always materialises
from that source. The result: apply is idempotent, partial-failure safe, and
switch-safe, and both entry points return ``list[SyncResult]`` so the CLI reuses
``sync/report`` unchanged.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from crossby.config.skills import SKILLS_DIR, list_skills
from crossby.models.ai import AIToolID
from crossby.scenes import declare, projection, trust, versioning
from crossby.scenes.mechanism import SceneMechanism, base_mechanism
from crossby.services.scene_resolution import ResolvedScene
from crossby.sync import run_sync
from crossby.sync.agents import _AGENT_TARGET_PATHS
from crossby.sync.base import SyncConcern, SyncData, SyncResult
from crossby.sync.ownership import OwnershipLedger, load_ledger, save_ledger
from crossby.sync.readers import build_sync_data

logger = structlog.get_logger()

# Per-tool MCP DECLARE activators (Codex takes an extra trust flag, handled inline).
_MCP_DECLARE = {
    AIToolID.CLAUDE: declare.apply_claude_disabled_mcp,
    AIToolID.ANTIGRAVITY_CLI: declare.apply_antigravity_disabled_mcp,
}


def apply_scene(
    resolved: ResolvedScene,
    project_root: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> list[SyncResult]:
    """Apply *resolved* to every installed tool, least-invasive mechanism first.

    Returns one :class:`SyncResult` per action taken. ``dry_run`` computes every
    change and writes nothing; ``force`` reaches only the skills/agents symlink
    re-point of a real, non-crossby directory (crossby's own symlinks always
    re-point).
    """
    ctx = _context(project_root, resolved, dry_run=dry_run, force=force)
    results: list[SyncResult] = []

    # 1. DECLARE surfaces (record scene provenance into the in-memory ledger).
    results.extend(_declare_skills(ctx))
    results.extend(_declare_agents(ctx))
    results.extend(_declare_mcp(ctx))

    # Persist scene DECLARE provenance BEFORE the hooks/permissions run_sync
    # calls: those load the ledger from disk and re-save it (owned section);
    # load_ledger/to_json round-trip the scene section, so an early save keeps it.
    if not dry_run:
        save_ledger(project_root, ctx.ledger)

    # 2. PROJECT the skills/agents directories at the filtered source tree.
    results.extend(_project_concern(ctx, "skills", ctx.base.skills_source))
    results.extend(_project_concern(ctx, "agents", ctx.base.agents_source))
    _ensure_scene_gitignore(ctx)

    # 3. hooks / permissions: revocable-sync removal channel, once each.
    results.extend(_filter_removable(ctx, "hooks", SyncConcern.HOOKS))
    results.extend(_filter_removable(ctx, "permissions", SyncConcern.PERMISSIONS))

    # 4. Plugin-provided skills are reachable by neither mechanism.
    results.extend(_report_plugin_skills(project_root))
    return results


def clear_scene(project_root: Path, *, dry_run: bool = False) -> list[SyncResult]:
    """Revert to the pre-scene state — nothing crossby didn't write is touched.

    Every crossby-owned DECLARE key is reverted (``disable`` is the empty set, so
    the provenance diff removes all owned entries), the projection is removed,
    and each tool's skills/agents directory is re-pointed at its unfiltered
    source. A human-authored ``skillOverrides`` / ``deny`` / MCP ``disabled``
    entry crossby never recorded survives untouched.
    """
    base = build_sync_data(project_root)
    ledger = load_ledger(project_root)
    version = versioning.detect_tool_version(AIToolID.CLAUDE)
    results: list[SyncResult] = []

    # 1. Revert DECLARE keys — an empty desired set reverts everything owned.
    results.append(
        declare.apply_claude_skill_overrides(
            project_root, set(), ledger, dry_run=dry_run, version=version
        )
    )
    results.append(declare.apply_claude_deny_agents(project_root, set(), ledger, dry_run=dry_run))
    results.append(declare.apply_claude_disabled_mcp(project_root, set(), ledger, dry_run=dry_run))
    results.append(declare.apply_codex_disabled_mcp(project_root, set(), ledger, dry_run=dry_run))
    results.append(
        declare.apply_antigravity_disabled_mcp(project_root, set(), ledger, dry_run=dry_run)
    )
    if not dry_run:
        save_ledger(project_root, ledger)

    # 2. Re-point skills/agents back at the unfiltered source (before removing the
    #    projection, so the tools never briefly resolve to a deleted tree).
    results.extend(_restore_sources(project_root, base, dry_run=dry_run))

    # 3. Remove the projection tree.
    removed = projection.clear_projection(project_root, dry_run=dry_run)
    if removed is not None:
        results.append(removed)
    return results


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class _Context:
    """Shared, precomputed inputs threaded through the per-concern handlers."""

    def __init__(
        self,
        *,
        project_root: Path,
        base: SyncData,
        ledger: OwnershipLedger,
        installed: list[AIToolID],
        selected: dict[str, set[str]],
        claude_version: tuple[int, int, int] | None,
        codex_trusted: bool,
        dry_run: bool,
        force: bool,
    ) -> None:
        self.project_root = project_root
        self.base = base
        self.ledger = ledger
        self.installed = installed
        self.selected = selected
        self.claude_version = claude_version
        self.codex_trusted = codex_trusted
        self.dry_run = dry_run
        self.force = force
        self._trees: dict[str, projection.ProjectionTree] = {}

    def tree(self, kind: str, source_rel: str) -> projection.ProjectionTree:
        if kind not in self._trees:
            self._trees[kind] = projection.plan_tree(
                self.project_root, kind, source_rel, self.selected[kind], dry_run=self.dry_run
            )
        return self._trees[kind]


def _context(
    project_root: Path, resolved: ResolvedScene, *, dry_run: bool, force: bool
) -> _Context:
    from crossby.ai_tools.base import AbstractAITool
    from crossby.models.config import SCENE_CONCERNS

    return _Context(
        project_root=project_root,
        base=build_sync_data(project_root),
        ledger=load_ledger(project_root),
        installed=AbstractAITool.detect_installed(),
        selected={concern: set(resolved.names(concern)) for concern in SCENE_CONCERNS},
        claude_version=versioning.detect_tool_version(AIToolID.CLAUDE),
        codex_trusted=trust.codex_trusts_project(project_root),
        dry_run=dry_run,
        force=force,
    )


# ---------------------------------------------------------------------------
# DECLARE handlers
# ---------------------------------------------------------------------------


def _declare_skills(ctx: _Context) -> list[SyncResult]:
    if AIToolID.CLAUDE not in ctx.installed:
        return []
    if base_mechanism(AIToolID.CLAUDE, "skills") != SceneMechanism.DECLARE:
        return []
    universe = set(list_skills(ctx.project_root / SKILLS_DIR[AIToolID.CLAUDE]))
    disable = universe - ctx.selected["skills"]
    return [
        declare.apply_claude_skill_overrides(
            ctx.project_root, disable, ctx.ledger, dry_run=ctx.dry_run, version=ctx.claude_version
        )
    ]


def _declare_agents(ctx: _Context) -> list[SyncResult]:
    if AIToolID.CLAUDE not in ctx.installed:
        return []
    if base_mechanism(AIToolID.CLAUDE, "agents") != SceneMechanism.DECLARE:
        return []
    target = _AGENT_TARGET_PATHS.get(str(AIToolID.CLAUDE))
    if target is None:
        return []
    universe = projection.scene_names(ctx.project_root, target, "agents")
    disable = universe - ctx.selected["agents"]
    return [
        declare.apply_claude_deny_agents(ctx.project_root, disable, ctx.ledger, dry_run=ctx.dry_run)
    ]


def _declare_mcp(ctx: _Context) -> list[SyncResult]:
    universe = set(ctx.base.mcp_servers)
    disable = universe - ctx.selected["mcp"]
    results: list[SyncResult] = []
    for tool in ctx.installed:
        mechanism = base_mechanism(tool, "mcp")
        if mechanism == SceneMechanism.UNSUPPORTED:
            if disable:
                results.append(
                    SyncResult(
                        tool_id=tool,
                        concern=SyncConcern.MCP,
                        action="skipped",
                        message=(
                            f"{tool} has no per-server disable key; "
                            f"{len(disable)} deselected server(s) remain enabled"
                        ),
                    )
                )
            continue
        if mechanism != SceneMechanism.DECLARE:
            continue
        if tool == AIToolID.CODEX:
            results.append(
                declare.apply_codex_disabled_mcp(
                    ctx.project_root,
                    disable,
                    ctx.ledger,
                    dry_run=ctx.dry_run,
                    trusted=ctx.codex_trusted,
                )
            )
        elif tool in _MCP_DECLARE:
            results.append(
                _MCP_DECLARE[tool](ctx.project_root, disable, ctx.ledger, dry_run=ctx.dry_run)
            )
    return results


# ---------------------------------------------------------------------------
# PROJECT handler
# ---------------------------------------------------------------------------


def _project_concern(ctx: _Context, kind: str, source_rel: str | None) -> list[SyncResult]:
    concern = SyncConcern.SKILLS if kind == "skills" else SyncConcern.AGENTS
    paths = _project_paths(ctx.installed, kind)
    if not paths:
        return []
    if source_rel is None:
        return [
            SyncResult(
                tool_id=tools[0],
                concern=concern,
                action="skipped",
                message=f"no {kind} source detected; nothing to project",
            )
            for tools in paths.values()
        ]

    tree = ctx.tree(kind, source_rel)
    results: list[SyncResult] = []
    for target_rel, tools in paths.items():
        results.append(_repoint_path(ctx, tree, target_rel, tuple(tools), source_rel))
    return results


def _repoint_path(
    ctx: _Context,
    tree: projection.ProjectionTree,
    target_rel: str,
    tools: tuple[AIToolID, ...],
    source_rel: str,
) -> SyncResult:
    # Re-pointing the canonical source onto a tree that links back into it would
    # be circular; that directory is left unfiltered (its tool filters via DECLARE
    # where it can) and the skip is reported rather than corrupting it.
    if projection.is_source_dir(ctx.project_root, target_rel, source_rel):
        shared = ", ".join(sorted(str(t) for t in tools))
        return SyncResult(
            tool_id=tools[0],
            concern=tree.concern,
            action="skipped",
            message=(
                f"{target_rel} is the canonical {tree.concern.value} source ({shared}); "
                "left unfiltered to avoid a circular re-point"
            ),
        )
    if ctx.dry_run:
        return projection.preview_repoint(tree, tools)
    # crossby's own symlink (or a missing dir) always re-points; a real,
    # non-crossby directory honours the user's --force (refused otherwise).
    target = ctx.project_root / target_rel
    force = True if (target.is_symlink() or not target.exists()) else ctx.force
    return projection.repoint(ctx.project_root, tree, tools[0], tools, force=force)


def _project_paths(installed: list[AIToolID], kind: str) -> dict[str, list[AIToolID]]:
    """Group the PROJECT tools' target directories, collapsing shared paths.

    Tools sharing a resolved directory (codex + antigravity-cli on
    ``.agents/skills``) collapse to one entry so the re-point happens — and is
    reported — once for both.
    """
    paths: dict[str, list[AIToolID]] = {}
    for tool in installed:
        if base_mechanism(tool, kind) != SceneMechanism.PROJECT:
            continue
        target = SKILLS_DIR.get(tool) if kind == "skills" else _AGENT_TARGET_PATHS.get(str(tool))
        if target is None:
            continue
        paths.setdefault(target, []).append(tool)
    return {path: sorted(tools, key=str) for path, tools in paths.items()}


# ---------------------------------------------------------------------------
# hooks / permissions removal channel
# ---------------------------------------------------------------------------


def _filter_removable(ctx: _Context, concern_key: str, concern: SyncConcern) -> list[SyncResult]:
    """Narrow hooks / permissions to the selected set via run_sync's removal channel.

    Runs only when the scene actively narrows the concern (selected is a strict
    subset of what the project already has); when the scene selects everything
    there is nothing to remove and re-syncing would needlessly propagate.
    """
    if concern == SyncConcern.HOOKS:
        universe = {f"{h.event}:{h.command}" for h in ctx.base.hooks}
    else:
        universe = set(ctx.base.allowed_commands)
    if not universe:
        return []
    selected = ctx.selected[concern_key]
    if selected == universe:
        return []

    if concern == SyncConcern.HOOKS:
        kept = [h for h in ctx.base.hooks if f"{h.event}:{h.command}" in selected]
        data = SyncData(hooks=kept)
    else:
        data = SyncData(allowed_commands=[p for p in ctx.base.allowed_commands if p in selected])
    return run_sync(data, ctx.project_root, concern=concern, dry_run=ctx.dry_run)


# ---------------------------------------------------------------------------
# clear helpers
# ---------------------------------------------------------------------------


def _restore_sources(project_root: Path, base: SyncData, *, dry_run: bool) -> list[SyncResult]:
    """Re-point every installed tool's skills/agents dir at the unfiltered source."""
    from crossby.ai_tools.base import AbstractAITool

    installed = AbstractAITool.detect_installed()
    results: list[SyncResult] = []
    for concern, source_rel in (
        (SyncConcern.SKILLS, base.skills_source),
        (SyncConcern.AGENTS, base.agents_source),
    ):
        if source_rel is None:
            continue
        if dry_run:
            results.append(
                SyncResult(
                    tool_id=None,
                    concern=concern,
                    action="updated",
                    message=f"(dry-run) would restore {concern.value} to {source_rel}",
                )
            )
            continue
        seen: set[str] = set()
        for tool in installed:
            target = _target_for(tool, concern)
            if target is None or target in seen:
                continue
            seen.add(target)
            if projection.is_source_dir(project_root, target, source_rel):
                continue
            # Mirror the apply-path gate (_repoint_path): re-point only crossby's
            # own symlink or a missing target. A real, non-crossby directory is
            # refused here too — clear must not back up and replace a user's own
            # directory without an explicit opt-in.
            target_path = project_root / target
            force = target_path.is_symlink() or not target_path.exists()
            results.append(
                projection.restore_source(project_root, concern, source_rel, tool, force=force)
            )
    return results


_SCENE_GITIGNORE_BLOCK = "scene projection"


def _ensure_scene_gitignore(ctx: _Context) -> None:
    """Gitignore ``.crossby/scene/`` once a projection exists (generated output).

    Uses the same managed-block helper the rest of crossby uses, so the entry
    sits inside crossby's delimited block rather than loose in ``.gitignore``.
    """
    if ctx.dry_run:
        return
    if not (ctx.project_root / projection.SCENE_PROJECTION_ROOT).exists():
        return
    from crossby.sync.gitignore_utils import update_managed_block

    update_managed_block(
        ctx.project_root,
        _SCENE_GITIGNORE_BLOCK,
        [projection.SCENE_PROJECTION_ROOT.as_posix() + "/"],
    )


def _target_for(tool: AIToolID, concern: SyncConcern) -> str | None:
    if concern == SyncConcern.SKILLS:
        return SKILLS_DIR.get(tool)
    return _AGENT_TARGET_PATHS.get(str(tool))


def _report_plugin_skills(project_root: Path) -> list[SyncResult]:
    """Report plugin-provided skills as always-on and outside scene control.

    Neither DECLARE (``skillOverrides`` doesn't touch plugin skills) nor PROJECT
    (they live under ``.claude/plugins/<plugin>/skills/``, outside the project
    skills tree) can exclude them, so a scene's "exactly the selected skills"
    promise is reported honestly rather than quietly failing.
    """
    plugins_dir = project_root / ".claude" / "plugins"
    if not plugins_dir.is_dir():
        return []
    names: set[str] = set()
    for plugin in sorted(plugins_dir.iterdir()):
        skills_dir = plugin / "skills"
        if skills_dir.is_dir():
            names.update(list_skills(skills_dir))
    if not names:
        return []
    return [
        SyncResult(
            tool_id=AIToolID.CLAUDE,
            concern=SyncConcern.SKILLS,
            action="skipped",
            message=f"plugin skill(s) always on, outside scene control: {', '.join(sorted(names))}",
        )
    ]
