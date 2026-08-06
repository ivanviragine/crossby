"""PROJECT mechanism — materialise a scene-filtered source and re-point tools.

For skills and agents, a scene selection is enacted by building a small tree of
relative symlinks to *only* the selected items under ``.crossby/scene/active/``
and pointing each PROJECT tool's directory at that tree via the existing sync
writers. No new per-tool path knowledge is duplicated — the 25 registered
writers already know where each tool reads::

    .crossby/scene/active/
    ├── skills/   # symlinks to the selected skill directories
    └── agents/   # symlinks to the selected agent files

The real source directory is never re-pointed onto a tree that references it
(that would be circular): the anchor tool filters itself via DECLARE
(Claude ``skillOverrides`` / ``permissions.deny``) and the tree links straight
into its still-real directory.

**Dry-run never materialises.** Rather than build a scratch tree — which some
writers (``agents_strategy="translate"``) would read real content from — a
dry-run hand-computes the result rows and leaves the filesystem byte-identical.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import structlog

from crossby.config.linker import create_symlink
from crossby.models.ai import AIToolID
from crossby.sync import run_sync
from crossby.sync.base import SyncConcern, SyncData, SyncResult
from crossby.sync.file_utils import MANAGED_MARKER_NAME, is_same_path, write_managed_marker

logger = structlog.get_logger()

SCENE_PROJECTION_ROOT = Path(".crossby") / "scene"
_ACTIVE = SCENE_PROJECTION_ROOT / "active"

# Agent files are named by stem; longest suffix first so ``foo.agent.md``
# resolves to ``foo`` rather than ``foo.agent`` (mirrors scene_resolution).
_AGENT_SUFFIXES: tuple[str, ...] = (".agent.md", ".md", ".toml")


@dataclass(frozen=True)
class ProjectionTree:
    """A materialised (or previewed) scene tree for one concern."""

    concern: SyncConcern
    #: Relative POSIX path of the tree, used as the run_sync ``*_source``.
    source_rel: str
    #: Original basenames symlinked into the tree (skills dirs / agent files).
    linked: tuple[str, ...]


def _kind_dir(kind: str) -> Path:
    return _ACTIVE / kind


def _scene_name(child: Path, kind: str) -> str | None:
    """The scene item name a *child* of a source dir contributes, or ``None``.

    Skills are one directory per skill (name == dir name); agents are files
    named by stem. Anything that fits neither shape is ignored.
    """
    if kind == "skills":
        return child.name if child.is_dir() else None
    if not child.is_file():
        return None
    for suffix in _AGENT_SUFFIXES:
        if child.name.endswith(suffix):
            return child.name[: -len(suffix)]
    return None


def selected_basenames(
    project_root: Path, source_rel: str, kind: str, selected: set[str]
) -> list[str]:
    """Original basenames under *source_rel* whose scene name is in *selected*.

    Used for both the real materialisation and the dry-run preview so the two
    agree on exactly what the tree would contain.
    """
    source_dir = project_root / source_rel
    out: list[str] = []
    if not source_dir.is_dir():
        return out
    for child in sorted(source_dir.iterdir()):
        if child.name == MANAGED_MARKER_NAME:
            continue
        name = _scene_name(child, kind)
        if name is not None and name in selected:
            out.append(child.name)
    return out


def scene_names(project_root: Path, source_rel: str, kind: str) -> set[str]:
    """Every scene item name present under *source_rel* (the concern universe)."""
    source_dir = project_root / source_rel
    out: set[str] = set()
    if not source_dir.is_dir():
        return out
    for child in source_dir.iterdir():
        if child.name == MANAGED_MARKER_NAME:
            continue
        name = _scene_name(child, kind)
        if name is not None:
            out.add(name)
    return out


def plan_tree(
    project_root: Path, kind: str, source_rel: str, selected: set[str], *, dry_run: bool
) -> ProjectionTree:
    """Materialise the scene tree, or (dry-run) compute what it would contain.

    In dry-run nothing is written — the returned tree reflects the basenames a
    real run would link, so the preview rows are accurate without touching disk.
    """
    if not dry_run:
        return materialise_tree(project_root, kind, source_rel, selected)
    concern = SyncConcern.SKILLS if kind == "skills" else SyncConcern.AGENTS
    linked = selected_basenames(project_root, source_rel, kind, selected)
    return ProjectionTree(
        concern=concern, source_rel=_kind_dir(kind).as_posix(), linked=tuple(linked)
    )


def materialise_tree(
    project_root: Path, kind: str, source_rel: str, selected: set[str]
) -> ProjectionTree:
    """Build ``.crossby/scene/active/<kind>`` of relative symlinks to *selected*.

    Idempotent: existing links are fixed in place, entries no longer selected are
    pruned, and the crossby-managed marker is (re)written so a filtered tree —
    shape-identical to a native one — is still recognised as crossby's own.
    """
    concern = SyncConcern.SKILLS if kind == "skills" else SyncConcern.AGENTS
    tree = project_root / _kind_dir(kind)
    tree.mkdir(parents=True, exist_ok=True)
    write_managed_marker(tree)

    source_dir = project_root / source_rel
    wanted = selected_basenames(project_root, source_rel, kind, selected)
    for basename in wanted:
        create_symlink(source_dir / basename, tree / basename, force=True)

    # Prune anything the previous scene left behind that this one doesn't want.
    keep = set(wanted) | {MANAGED_MARKER_NAME}
    for child in tree.iterdir():
        if child.name in keep:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    return ProjectionTree(
        concern=concern, source_rel=_kind_dir(kind).as_posix(), linked=tuple(wanted)
    )


def repoint(
    project_root: Path,
    tree: ProjectionTree,
    representative: AIToolID,
    tools: tuple[AIToolID, ...],
    *,
    force: bool = False,
) -> SyncResult:
    """Point *representative*'s directory at the projected *tree* via run_sync.

    Runs a single writer (scoped by ``tool_id`` + ``concern``) so ``force=True``
    only ever reaches this symlink re-point, never unrelated managed content.
    For a shared target path the caller passes one representative and the full
    *tools* tuple, so the row is reported once but names every sharer.
    """
    data = _source_data(tree)
    results = run_sync(
        data,
        project_root,
        tool_id=representative,
        concern=tree.concern,
        force=force,
    )
    result = _first_for(results, tree.concern, representative)
    if len(tools) > 1:
        shared = ", ".join(sorted(str(t) for t in tools))
        result.message = f"{result.message or 're-pointed'} (shared by {shared})"
    return result


def preview_repoint(tree: ProjectionTree, tools: tuple[AIToolID, ...]) -> SyncResult:
    """The dry-run row for a re-point — no filesystem touched."""
    label = ", ".join(sorted(str(t) for t in tools))
    return SyncResult(
        tool_id=tools[0],
        concern=tree.concern,
        action="updated",
        message=(
            f"(dry-run) would re-point {label} to {len(tree.linked)} selected "
            f"{tree.concern.value} via {tree.source_rel}"
        ),
        added=len(tree.linked),
    )


def restore_source(
    project_root: Path,
    concern: SyncConcern,
    source_rel: str,
    representative: AIToolID,
    *,
    force: bool = True,
) -> SyncResult:
    """Re-point *representative*'s directory back at the unfiltered *source_rel*.

    Used by ``clear_scene`` to undo a projection: the tool's directory resolves
    to the full source again rather than the (now-removed) scene tree.
    """
    data = _source_data_for(concern, source_rel)
    results = run_sync(data, project_root, tool_id=representative, concern=concern, force=force)
    return _first_for(results, concern, representative)


def is_source_dir(project_root: Path, target_rel: str, source_rel: str) -> bool:
    """True when *target_rel* resolves to the same real dir as *source_rel*.

    Re-pointing such a directory would be circular (the tree links back into it),
    so the engine filters that tool via DECLARE instead and reports the skip.
    """
    return is_same_path(project_root / source_rel, project_root / target_rel)


def clear_projection(project_root: Path, *, dry_run: bool = False) -> SyncResult | None:
    """Remove ``.crossby/scene/`` entirely. Returns a row only if it existed."""
    root = project_root / SCENE_PROJECTION_ROOT
    if not root.exists():
        return None
    if not dry_run:
        shutil.rmtree(root)
    return SyncResult(
        tool_id=None,
        concern=SyncConcern.SKILLS,
        action="updated",
        file_path=root,
        message="(dry-run) would remove scene projection"
        if dry_run
        else "removed scene projection",
        revoked=1,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _source_data(tree: ProjectionTree) -> SyncData:
    return _source_data_for(tree.concern, tree.source_rel)


def _source_data_for(concern: SyncConcern, source_rel: str) -> SyncData:
    if concern == SyncConcern.SKILLS:
        return SyncData(skills_source=source_rel, skills_strategy="symlink", skills_gitignore=False)
    return SyncData(agents_source=source_rel, agents_strategy="symlink", agents_gitignore=False)


def _first_for(results: list[SyncResult], concern: SyncConcern, tool: AIToolID) -> SyncResult:
    for result in results:
        if result.concern == concern and result.tool_id == tool:
            return result
    # run_sync always returns the scoped writer's row; fall back defensively.
    return SyncResult(tool_id=tool, concern=concern, action="skipped", message="no writer ran")
