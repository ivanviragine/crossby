"""Scene apply / clear orchestration.

:func:`apply_scene` enacts a resolved scene on every installed tool, choosing the
least-invasive mechanism per concern: write a DECLARE key (disabling the
deselected remainder), re-point a tool directory at a projected filtered source,
drive the revocable-sync removal channel (hooks / permissions), or report an
unsupported cell. :func:`clear_scene` reverts every crossby-owned DECLARE key,
restores each PROJECT path from its exact ownership-ledger baseline, and removes
the projection once nothing still depends on it.

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

import os
from collections.abc import Iterable, Sequence
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
from crossby.sync.file_utils import has_managed_marker
from crossby.sync.ownership import (
    LEDGER_PATH,
    OwnershipLedger,
    ScenePathRestore,
    ScenePathRestoreKind,
    load_ledger_checked,
    save_ledger,
)
from crossby.sync.readers import build_sync_data
from crossby.sync.safe_write import (
    ProjectScope,
    SyncContainmentError,
    assert_ancestors,
    safe_rmtree,
    safe_unlink,
)

logger = structlog.get_logger()

# Per-tool MCP DECLARE activators (Codex takes an extra trust flag, handled inline).
_MCP_DECLARE = {
    AIToolID.CLAUDE: declare.apply_claude_disabled_mcp,
    AIToolID.ANTIGRAVITY_CLI: declare.apply_antigravity_disabled_mcp,
}


class SceneApplyError(RuntimeError):
    """An exceptional apply failure together with actions completed before it."""

    def __init__(self, cause: Exception, results: Sequence[SyncResult]) -> None:
        super().__init__(str(cause))
        self.results = tuple(results)


def apply_scene(
    resolved: ResolvedScene,
    project_root: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    tools: Iterable[AIToolID] | None = None,
) -> list[SyncResult]:
    """Apply *resolved* to every installed tool, least-invasive mechanism first.

    Returns one :class:`SyncResult` per action taken. ``dry_run`` computes every
    change and writes nothing; ``force`` reaches only the skills/agents symlink
    re-point of a real, non-crossby directory (crossby's own symlinks always
    re-point). ``tools`` narrows the apply to a subset of the installed tools
    (``crossby scene use --tool``); ``None`` means every installed tool.
    """
    ctx = _context(project_root, resolved, dry_run=dry_run, force=force, tools=tools)
    results: list[SyncResult] = []

    try:
        # 1. DECLARE surfaces. Each handler commits provenance for a (tool, key) into
        #    the in-memory ledger only AFTER its file write succeeds, or inline on a
        #    verified no-write path — so a handler that raises leaves that key's prior
        #    ownership untouched, and the in-memory ledger only ever reflects writes
        #    that actually landed. The provenance save runs in a finally so a writer
        #    raising part-way still persists a ledger consistent with disk — `clear`
        #    can always revert an on-disk setting crossby recorded. It also lands
        #    BEFORE the hooks/permissions run_sync calls: those reload the ledger from
        #    disk and re-save it (owned section), and load_ledger/to_json round-trip
        #    the scene section, so this early save is preserved.
        try:
            results.extend(_declare_skills(ctx))
            results.extend(_declare_agents(ctx))
            results.extend(_declare_mcp(ctx))
        finally:
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
    except Exception as exc:
        # The caller must retain both reversible provenance and irreversible
        # revocations completed before a later phase failed.
        raise SceneApplyError(exc, results) from exc


def clear_scene(
    project_root: Path,
    *,
    dry_run: bool = False,
    tools: Iterable[AIToolID] | None = None,
    force: bool = False,
) -> list[SyncResult]:
    """Revert to the pre-scene state — nothing crossby didn't write is touched.

    Every crossby-owned DECLARE key is reverted (``disable`` is the empty set, so
    the provenance diff removes all owned entries), each PROJECT target returns
    to its recorded absent/symlink/directory state, and the projection is removed.
    A human-authored ``skillOverrides`` / ``deny`` / MCP ``disabled`` entry
    crossby never recorded survives untouched.

    ``tools`` narrows the revert to a subset of the installed tools
    (``crossby scene clear --tool``, and the per-tool revert of a switch): only
    those tools' DECLARE keys and source directories are touched, and the shared
    projection tree is removed only once no out-of-scope tool still points at it.
    ``None`` reverts every installed tool.
    """
    loaded = load_ledger_checked(project_root)
    if loaded.corrupt:
        raise ValueError(
            f"{LEDGER_PATH.as_posix()} has corrupt scene provenance; refusing to clear"
        )
    ledger = loaded.ledger
    version = versioning.detect_tool_version(AIToolID.CLAUDE)
    scope = set(tools) if tools is not None else None
    results: list[SyncResult] = []

    def _in_scope(tool: AIToolID) -> bool:
        return scope is None or tool in scope

    # 1. Revert DECLARE keys — an empty desired set reverts everything owned.
    #    Each key is gated by scope so a --tool clear leaves other tools' keys
    #    alone (a tool never applied has nothing owned, so its revert is a no-op
    #    regardless, but gating keeps the result rows scoped too).
    # Each handler commits its narrowed ownership only AFTER a successful write,
    # or inline on a verified no-write path, and Codex retains ownership of any
    # entry whose revert splice failed. The ledger save runs in a finally so a
    # revert that raises part-way still persists ownership consistent with disk —
    # a (tool, key) crossby could not revert stays owned, so a re-run retries
    # exactly it rather than dropping the record of a setting still applied.
    try:
        if _in_scope(AIToolID.CLAUDE):
            results.append(
                declare.apply_claude_skill_overrides(
                    project_root, set(), ledger, dry_run=dry_run, version=version
                )
            )
            results.append(
                declare.apply_claude_deny_agents(project_root, set(), ledger, dry_run=dry_run)
            )
            results.append(
                declare.apply_claude_disabled_mcp(project_root, set(), ledger, dry_run=dry_run)
            )
        if _in_scope(AIToolID.CODEX):
            results.append(
                declare.apply_codex_disabled_mcp(project_root, set(), ledger, dry_run=dry_run)
            )
        if _in_scope(AIToolID.ANTIGRAVITY_CLI):
            results.append(
                declare.apply_antigravity_disabled_mcp(project_root, set(), ledger, dry_run=dry_run)
            )
    finally:
        if not dry_run:
            save_ledger(project_root, ledger)

    # 2. Restore exact physical-path baselines from the ownership ledger. Never
    #    rediscover a source after activation: projected targets can outrank the
    #    original source and a forced directory backup cannot be inferred safely.
    restore_results = _restore_paths(
        project_root, ledger, dry_run=dry_run, tools=scope, force=force
    )
    results.extend(restore_results)

    # 3. Remove the projection tree, but only when (a) every in-scope tool was
    #    re-pointed cleanly and (b) no tool left out of this scope still resolves
    #    to it — otherwise a tool would be left dangling at a deleted path.
    restore_failed = any(r.action == "error" for r in restore_results)
    if not restore_failed and _should_clear_projection(project_root, scope):
        removed = projection.clear_projection(project_root, dry_run=dry_run)
        if removed is not None:
            results.append(removed)
    return results


def _should_clear_projection(project_root: Path, scope: set[AIToolID] | None) -> bool:
    """True when removing ``.crossby/scene/`` is safe for the current clear scope.

    An unscoped clear always removes it. A scoped clear removes it only when no
    tool *outside* the scope still points its skills/agents directory at the
    projection tree — otherwise that tool would be left dangling at a deleted
    path. Every scene-participating tool's directory is checked on disk (not just
    the currently-installed ones), so a recorded-but-uninstalled tool whose
    symlink still resolves into the projection also protects it.
    """
    if scope is None:
        return True
    if not (project_root / projection.SCENE_PROJECTION_ROOT).exists():
        return False

    for tool in _scene_tools():
        if tool in scope:
            continue
        for kind, concern in (("skills", SyncConcern.SKILLS), ("agents", SyncConcern.AGENTS)):
            target = _target_for(tool, concern)
            if target is not None and projection.tool_points_at_projection(
                project_root, target, kind
            ):
                return False
    return True


def _scene_tools() -> set[AIToolID]:
    """Every tool that has a skills or agents directory a scene could re-point."""
    tools = set(SKILLS_DIR)
    tools.update(tool for tool in AIToolID if str(tool) in _AGENT_TARGET_PATHS)
    return tools


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
    project_root: Path,
    resolved: ResolvedScene,
    *,
    dry_run: bool,
    force: bool,
    tools: Iterable[AIToolID] | None = None,
) -> _Context:
    from crossby.ai_tools.base import AbstractAITool
    from crossby.models.config import SCENE_CONCERNS

    installed = AbstractAITool.detect_installed()
    if tools is not None:
        scope = set(tools)
        installed = [tool for tool in installed if tool in scope]

    loaded = load_ledger_checked(project_root)
    if loaded.corrupt:
        raise ValueError(
            f"{LEDGER_PATH.as_posix()} has corrupt scene provenance; refusing to apply a scene"
        )

    return _Context(
        project_root=project_root,
        base=build_sync_data(project_root),
        ledger=loaded.ledger,
        installed=installed,
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
                        unsupported=True,
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

    try:
        tree = ctx.tree(kind, source_rel)
    except SyncContainmentError as exc:
        # Materialising the projection tree is refused — e.g. a symlinked
        # ``.crossby`` ancestor makes the managed-marker write escape the root.
        # Mirror run_sync's containment contract: surface an ``error`` row per
        # target path rather than letting the exception escape apply_scene, so a
        # direct engine caller gets the same reported-row result the CLI wraps.
        return [
            SyncResult(tool_id=tools[0], concern=concern, action="error", message=str(exc))
            for tools in paths.values()
        ]
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
    # where it can) and the skip is reported rather than corrupting it. Record
    # that no-op durably: a normal sync may have left the source as a managed
    # marker-backed directory, which otherwise looks like an unrecoverable
    # legacy projection during clear.
    if projection.is_source_dir(ctx.project_root, target_rel, source_rel):
        if not ctx.dry_run and ctx.ledger.scene_restore(target_rel) is None:
            ctx.ledger.record_scene_restore(target_rel, ScenePathRestore.unchanged())
            save_ledger(ctx.project_root, ctx.ledger)
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
        preview = projection.preview_repoint(tree, tools)
        try:
            baseline = _describe_repoint_baseline(
                ctx.project_root,
                target_rel,
                ctx.force,
                ctx.ledger.scene_restore(target_rel),
            )
        except (OSError, ValueError, SyncContainmentError) as exc:
            baseline = f"error:{exc}"
        if baseline.startswith("error:"):
            preview.action = "error"
            preview.file_path = ctx.project_root / target_rel
            preview.message = baseline.removeprefix("error:")
        else:
            preview.message = f"{preview.message}; {baseline}"
        return preview

    target = ctx.project_root / target_rel
    descriptor = ctx.ledger.scene_restore(target_rel)
    if descriptor is None:
        # Capturing and durably persisting the exact baseline is a prerequisite,
        # not a recoverable per-writer error. Escalate so launch fallbacks abort
        # instead of starting a child with a path whose baseline is unrecorded.
        try:
            descriptor = _capture_path_baseline(ctx, target_rel)
        except (ValueError, SyncContainmentError) as exc:
            return _path_error(tools, tree.concern, target, str(exc))
    try:
        if descriptor.kind == ScenePathRestoreKind.DIRECTORY:
            backup_rel = descriptor.backup_path
            assert backup_rel is not None
            backup = ctx.project_root / backup_rel
            # A retry after the durable descriptor write may find either side of
            # the displacement. Move only in the unambiguous pre-mutation state;
            # a missing backup otherwise does not prove the directory was ever
            # displaced and must not authorize replacing a new target.
            if os.path.lexists(backup):
                if backup.is_symlink() or not backup.is_dir():
                    return _path_error(
                        tools,
                        tree.concern,
                        target,
                        f"recorded backup is not the displaced real directory: {backup_rel}",
                    )
                if (
                    descriptor.directory_device is not None
                    and not _matches_recorded_directory_identity(backup, descriptor)
                ):
                    return _path_error(
                        tools,
                        tree.concern,
                        target,
                        f"recorded backup is not the displaced real directory: {backup_rel}",
                    )
                if descriptor.directory_displaced is not True:
                    descriptor = ctx.ledger.record_scene_directory_displaced(target_rel)
                    save_ledger(ctx.project_root, ctx.ledger)
                if target.is_dir() and not target.is_symlink():
                    # Translate/copy writers (notably Codex and Copilot agents)
                    # materialise their scene output as a marker-backed directory.
                    # That active target legitimately coexists with the saved
                    # pre-scene directory, and can be re-synced without replacing
                    # either path. An unmarked directory is still drift.
                    if not has_managed_marker(target):
                        return _path_error(
                            tools,
                            tree.concern,
                            target,
                            f"recorded backup is occupied: {backup_rel}",
                        )
                    return projection.repoint(ctx.project_root, tree, tools[0], tools, force=False)
            elif descriptor.directory_displaced is True:
                return _path_error(
                    tools,
                    tree.concern,
                    target,
                    f"recorded directory backup is missing: {backup_rel}; "
                    "refusing to replace the target",
                )
            elif not (target.is_dir() and not target.is_symlink()):
                return _path_error(
                    tools,
                    tree.concern,
                    target,
                    f"recorded directory baseline was not displaced: {backup_rel}; "
                    "refusing to replace the target",
                )
            elif descriptor.directory_displaced is None:
                return _path_error(
                    tools,
                    tree.concern,
                    target,
                    f"legacy directory displacement state is unknown for {backup_rel}; "
                    "refusing to replace the target",
                )
            elif not _matches_recorded_directory_identity(target, descriptor):
                return _path_error(
                    tools,
                    tree.concern,
                    target,
                    f"recorded directory baseline changed before displacement: {target_rel}; "
                    "refusing to replace the target",
                )
            else:
                projection.displace_directory(ctx.project_root, target_rel, backup_rel)
                descriptor = ctx.ledger.record_scene_directory_displaced(target_rel)
                save_ledger(ctx.project_root, ctx.ledger)
        elif target.is_dir() and not target.is_symlink():
            # Some writers materialise a managed directory rather than a
            # symlink. Re-run those without force; an unmarked directory is
            # drift and must not be preserved at an unrecorded .bak path.
            if not has_managed_marker(target):
                return _path_error(
                    tools,
                    tree.concern,
                    target,
                    f"{target_rel} drifted to a real directory after its baseline was recorded; "
                    "restore the recorded baseline first",
                )
            return projection.repoint(ctx.project_root, tree, tools[0], tools, force=False)
        # The scene owns replacement only after the baseline is durable. A real
        # directory has already been displaced, so the generic writer cannot
        # allocate an independent .bak path.
        return projection.repoint(ctx.project_root, tree, tools[0], tools, force=True)
    except (OSError, ValueError, SyncContainmentError) as exc:
        return _path_error(tools, tree.concern, target, str(exc))


def _capture_path_baseline(ctx: _Context, target_rel: str) -> ScenePathRestore:
    """Durably record a target's first pre-scene state before mutating it."""
    target = ctx.project_root / target_rel
    # Baseline inspection itself establishes durable recovery authority, so it
    # must obey the same ancestor-containment policy as the later writer. Without
    # this check, a symlinked target ancestor could leak an outside-tree state
    # into the ledger even though the writer would refuse to mutate it.
    assert_ancestors(ProjectScope(ctx.project_root), target)
    if target.is_symlink():
        descriptor = ScenePathRestore.symlink(os.readlink(target))
    elif not os.path.lexists(target):
        descriptor = ScenePathRestore.absent()
    elif target.is_dir():
        contents = [child for child in target.iterdir() if child.name != ".crossby-managed"]
        if contents and not has_managed_marker(target) and not ctx.force:
            raise ValueError(
                f"{target_rel} exists as a directory; migrate its contents first, "
                "or use --force to preserve it at a recorded backup"
            )
        backup_rel = projection.allocate_directory_backup(ctx.project_root, target_rel)
        identity = target.stat()
        descriptor = ScenePathRestore.directory(
            backup_rel,
            displaced=False,
            device=identity.st_dev,
            inode=identity.st_ino,
        )
    else:
        raise ValueError(
            f"{target_rel} exists but is not a directory or symlink; refusing scene projection"
        )
    ctx.ledger.record_scene_restore(target_rel, descriptor)
    # The descriptor must reach durable storage before the corresponding path
    # changes. A save failure propagates with the target untouched.
    save_ledger(ctx.project_root, ctx.ledger)
    return descriptor


def _describe_repoint_baseline(
    project_root: Path,
    target_rel: str,
    force: bool,
    descriptor: ScenePathRestore | None,
) -> str:
    """Describe the recovery authority a dry-run re-point would use."""
    target = project_root / target_rel
    assert_ancestors(ProjectScope(project_root), target)
    if descriptor is None:
        return _describe_prospective_baseline(project_root, target_rel, force)

    if descriptor.kind == ScenePathRestoreKind.DIRECTORY:
        backup_rel = descriptor.backup_path
        assert backup_rel is not None
        backup = project_root / backup_rel
        if os.path.lexists(backup):
            if backup.is_symlink() or not backup.is_dir():
                return f"error:recorded backup is not the displaced real directory: {backup_rel}"
            if descriptor.directory_device is not None and not _matches_recorded_directory_identity(
                backup, descriptor
            ):
                return f"error:recorded backup is not the displaced real directory: {backup_rel}"
            if target.is_dir() and not target.is_symlink() and not has_managed_marker(target):
                return f"error:recorded backup is occupied: {backup_rel}"
        elif descriptor.directory_displaced is True:
            return (
                f"error:recorded directory backup is missing: {backup_rel}; "
                "refusing to replace the target"
            )
        elif not (target.is_dir() and not target.is_symlink()):
            return (
                f"error:recorded directory baseline was not displaced: {backup_rel}; "
                "refusing to replace the target"
            )
        elif descriptor.directory_displaced is None:
            return (
                f"error:legacy directory displacement state is unknown for {backup_rel}; "
                "refusing to replace the target"
            )
        elif not _matches_recorded_directory_identity(target, descriptor):
            return (
                f"error:recorded directory baseline changed before displacement: {target_rel}; "
                "refusing to replace the target"
            )
        else:
            return f"would displace the recorded directory baseline at {backup_rel}"
        return f"would retain the recorded directory baseline at {backup_rel}"

    if target.is_dir() and not target.is_symlink() and not has_managed_marker(target):
        return (
            f"error:{target_rel} drifted to a real directory after its baseline was recorded; "
            "restore the recorded baseline first"
        )

    if descriptor.kind == ScenePathRestoreKind.ABSENT:
        return "would retain the recorded absent baseline"
    return f"would retain the recorded literal symlink target {descriptor.link_target!r}"


def _describe_prospective_baseline(project_root: Path, target_rel: str, force: bool) -> str:
    target = project_root / target_rel
    if target.is_symlink():
        return f"would record literal symlink target {os.readlink(target)!r}"
    if not os.path.lexists(target):
        return "would record that the target was absent"
    if target.is_dir():
        contents = [child for child in target.iterdir() if child.name != ".crossby-managed"]
        if contents and not has_managed_marker(target) and not force:
            return (
                f"error:{target_rel} exists as a directory; migrate its contents first, "
                "or use --force to preserve it at a recorded backup"
            )
        backup_rel = projection.allocate_directory_backup(project_root, target_rel)
        return f"would preserve the directory at {backup_rel}"
    return f"error:{target_rel} is not a directory or symlink"


def _path_error(
    tools: tuple[AIToolID, ...], concern: SyncConcern, target: Path, message: str
) -> SyncResult:
    return SyncResult(
        tool_id=tools[0], concern=concern, action="error", file_path=target, message=message
    )


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
    # Pass the (possibly scoped) installed set so a --tool run narrows the
    # removal channel to the same subset the DECLARE/PROJECT handlers touched.
    return run_sync(
        data,
        ctx.project_root,
        concern=concern,
        dry_run=ctx.dry_run,
        installed_tools=ctx.installed,
    )


# ---------------------------------------------------------------------------
# clear helpers
# ---------------------------------------------------------------------------


def _restore_paths(
    project_root: Path,
    ledger: OwnershipLedger,
    *,
    dry_run: bool,
    tools: set[AIToolID] | None,
    force: bool,
) -> list[SyncResult]:
    """Restore each in-scope physical path from durable pre-scene provenance."""
    candidates = _restore_candidates(ledger, tools)
    results: list[SyncResult] = []
    for target_rel, path_tools in candidates.items():
        descriptor = ledger.scene_restore(target_rel)
        concern = _concern_for_target(target_rel)
        target = project_root / target_rel
        representative = path_tools[0] if path_tools else None
        if descriptor is None:
            kind = "skills" if concern == SyncConcern.SKILLS else "agents"
            # Legacy scenes did not record exact path baselines.  Most legacy
            # projections are directory symlinks into ``.crossby/scene``, but
            # Codex and Copilot agents can instead be materialised into a
            # marker-backed directory.  Both shapes are active scene output;
            # without provenance, clearing either one would orphan its
            # pre-scene state while reporting a successful clear.
            legacy_projection = projection.tool_points_at_projection(
                project_root, target_rel, kind
            ) or projection.tool_symlink_points_into_projection(project_root, target_rel)
            marker_projection = (
                target.is_dir() and not target.is_symlink() and has_managed_marker(target)
            )
            if legacy_projection or marker_projection:
                results.append(
                    SyncResult(
                        tool_id=representative,
                        concern=concern,
                        action="error",
                        file_path=target,
                        message=(
                            f"{target_rel} has an active scene projection but no exact path "
                            "provenance (legacy activation); restore it manually and retry. "
                            "Crossby will not infer a source or adopt a nearby .bak path."
                        ),
                    )
                )
            continue
        if dry_run:
            try:
                _validate_restore_one_path(
                    project_root, target_rel, concern, descriptor, force=force
                )
            except (OSError, ValueError, SyncContainmentError) as exc:
                results.append(
                    SyncResult(
                        tool_id=representative,
                        concern=concern,
                        action="error",
                        file_path=target,
                        message=f"could not restore {target_rel}: {exc}",
                    )
                )
                continue
            results.append(
                SyncResult(
                    tool_id=representative,
                    concern=concern,
                    action="updated",
                    file_path=target,
                    message=(
                        f"(dry-run) would restore {target_rel} {_restore_description(descriptor)}"
                    ),
                )
            )
            continue
        try:
            descriptor = _restore_one_path(
                project_root, ledger, target_rel, concern, descriptor, force=force
            )
            ledger.clear_scene_restore(target_rel)
            # Cleanup is commit-like: descriptor removal becomes authoritative
            # only after the exact baseline is on disk. A directory descriptor
            # persists the moved directory's identity before its rename, so a
            # later retry can authenticate a completed rename if this save fails.
            try:
                save_ledger(project_root, ledger)
            except Exception:
                # Keep the shared in-memory ledger consistent too; a later path
                # may save it successfully during this same partial clear.
                ledger.record_scene_restore(target_rel, descriptor)
                raise
        except (OSError, ValueError, SyncContainmentError) as exc:
            results.append(
                SyncResult(
                    tool_id=representative,
                    concern=concern,
                    action="error",
                    file_path=target,
                    message=f"could not restore {target_rel}: {exc}",
                )
            )
            continue
        shared = ""
        if len(path_tools) > 1:
            shared = f" (shared by {', '.join(sorted(str(tool) for tool in path_tools))})"
        results.append(
            SyncResult(
                tool_id=representative,
                concern=concern,
                action="updated",
                file_path=target,
                message=f"restored exact pre-scene baseline{shared}",
                revoked=1,
            )
        )
    return results


def _restore_candidates(
    ledger: OwnershipLedger, tools: set[AIToolID] | None
) -> dict[str, list[AIToolID]]:
    if tools is None:
        candidates = {path: _tools_for_target(path) for path in ledger.scene_restores()}
        # Legacy records have no descriptors, so also inspect currently-installed
        # PROJECT paths for a projection that requires manual recovery.
        from crossby.ai_tools.base import AbstractAITool

        scoped_tools = AbstractAITool.detect_installed()
    else:
        candidates = {}
        scoped_tools = sorted(tools, key=str)
    for kind in ("skills", "agents"):
        for path, path_tools in _project_paths(scoped_tools, kind).items():
            candidates.setdefault(path, path_tools)
    return dict(sorted(candidates.items()))


def _tools_for_target(target_rel: str) -> list[AIToolID]:
    return sorted(
        [
            tool
            for tool in _scene_tools()
            if _target_for(tool, _concern_for_target(target_rel)) == target_rel
        ],
        key=str,
    )


def _concern_for_target(target_rel: str) -> SyncConcern:
    return SyncConcern.SKILLS if target_rel in set(SKILLS_DIR.values()) else SyncConcern.AGENTS


def _restore_description(descriptor: ScenePathRestore) -> str:
    if descriptor.kind == ScenePathRestoreKind.UNCHANGED:
        return "without changing the canonical source"
    if descriptor.kind == ScenePathRestoreKind.ABSENT:
        return "to absence"
    if descriptor.kind == ScenePathRestoreKind.SYMLINK:
        return f"to literal symlink {descriptor.link_target!r}"
    return f"from recorded backup {descriptor.backup_path}"


def _restore_one_path(
    project_root: Path,
    ledger: OwnershipLedger,
    target_rel: str,
    concern: SyncConcern,
    descriptor: ScenePathRestore,
    *,
    force: bool,
) -> ScenePathRestore:
    target = project_root / target_rel
    kind = "skills" if concern == SyncConcern.SKILLS else "agents"
    _validate_restore_one_path(project_root, target_rel, concern, descriptor, force=force)

    if descriptor.kind == ScenePathRestoreKind.UNCHANGED:
        return descriptor

    if descriptor.kind == ScenePathRestoreKind.ABSENT:
        if os.path.lexists(target):
            _remove_scene_output(project_root, target_rel, kind, force=force)
        return descriptor

    if descriptor.kind == ScenePathRestoreKind.SYMLINK:
        literal = descriptor.link_target
        assert literal is not None
        if target.is_symlink() and os.readlink(target) == literal:
            return descriptor
        if os.path.lexists(target):
            _remove_scene_output(project_root, target_rel, kind, force=force)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(literal, target)
        return descriptor

    backup_rel = descriptor.backup_path
    assert backup_rel is not None
    backup = project_root / backup_rel
    if not os.path.lexists(backup):
        # Validation accepts only a target that matches the durable identity
        # captured immediately before the backup's rename, so there is nothing
        # left to do when cleanup persistence was interrupted after os.replace.
        return descriptor
    stat = backup.stat()
    descriptor = ledger.record_scene_restore_directory_identity(
        target_rel, device=stat.st_dev, inode=stat.st_ino
    )
    # This identity must be durable before the rename: a retry can then prove
    # that the backup reached its target instead of adopting a recreated path.
    save_ledger(project_root, ledger)
    _validate_restore_one_path(project_root, target_rel, concern, descriptor, force=force)
    if os.path.lexists(target):
        _remove_scene_output(project_root, target_rel, kind, force=force)
    os.replace(backup, target)
    return descriptor


def _validate_restore_one_path(
    project_root: Path,
    target_rel: str,
    concern: SyncConcern,
    descriptor: ScenePathRestore,
    *,
    force: bool,
) -> None:
    """Raise when restoring *descriptor* cannot safely succeed without writing."""
    if descriptor.kind == ScenePathRestoreKind.UNCHANGED:
        return
    target = project_root / target_rel
    kind = "skills" if concern == SyncConcern.SKILLS else "agents"
    scope = ProjectScope(project_root)
    assert_ancestors(scope, target)

    if descriptor.kind == ScenePathRestoreKind.ABSENT:
        if not os.path.lexists(target):
            return
        _validate_scene_output_removal(project_root, target_rel, kind, force=force)
        return

    if descriptor.kind == ScenePathRestoreKind.SYMLINK:
        literal = descriptor.link_target
        assert literal is not None
        if target.is_symlink() and os.readlink(target) == literal:
            return
        if os.path.lexists(target):
            _validate_scene_output_removal(project_root, target_rel, kind, force=force)
        return

    backup_rel = descriptor.backup_path
    assert backup_rel is not None
    backup = project_root / backup_rel
    assert_ancestors(scope, backup)
    backup_exists = os.path.lexists(backup)
    if not backup_exists:
        if _matches_recorded_directory_identity(target, descriptor):
            return
        # A real target directory cannot authenticate that the exact recorded
        # backup reached it: the backup could have been lost while an unrelated
        # directory was recreated at the target. Keep recovery authority intact
        # until the recorded backup is available for an explicit restore.
        raise FileNotFoundError(f"recorded backup is missing: {backup_rel}")
    if backup.is_symlink() or not backup.is_dir():
        raise ValueError(f"recorded backup is not the displaced real directory: {backup_rel}")
    if os.path.lexists(target):
        _validate_scene_output_removal(project_root, target_rel, kind, force=force)


def _matches_recorded_directory_identity(path: Path, descriptor: ScenePathRestore) -> bool:
    """Whether *path* is the exact directory authenticated by the descriptor."""
    device = descriptor.directory_device
    inode = descriptor.directory_inode
    if device is None or inode is None or path.is_symlink() or not path.is_dir():
        return False
    try:
        identity = path.stat()
    except OSError:
        return False
    return (identity.st_dev, identity.st_ino) == (device, inode)


def _validate_scene_output_removal(
    project_root: Path, target_rel: str, kind: str, *, force: bool
) -> None:
    """Raise unless *target_rel* is a removable scene projection output."""
    target = project_root / target_rel
    if target.is_symlink():
        if not force and not projection.tool_points_at_projection(project_root, target_rel, kind):
            raise ValueError("target symlink drifted away from the active scene projection")
        return
    if target.is_dir() and has_managed_marker(target):
        return
    raise ValueError("target is not removable scene-owned projection output")


def _remove_scene_output(project_root: Path, target_rel: str, kind: str, *, force: bool) -> None:
    target = project_root / target_rel
    scope = ProjectScope(project_root)
    _validate_scene_output_removal(project_root, target_rel, kind, force=force)
    if target.is_symlink():
        safe_unlink(scope, target, missing_ok=False)
        return
    if target.is_dir() and has_managed_marker(target):
        safe_rmtree(scope, target)
        return


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
