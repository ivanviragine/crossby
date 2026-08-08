"""Scene resolver — turn a flat :class:`SceneConfig` into concrete selections.

Given an already-``extends``-flattened scene (see
:meth:`crossby.models.config.CrossbyConfig.get_scene`) and a
:class:`~crossby.sync.readers.ProjectScan`, :func:`resolve_scene` produces the
per-concern set of items each selector actually matched against the detected
project inventory.

The result is deliberately **tool-agnostic**: it is a set of concern selections
keyed by resolved target path, not a per-tool artefact plan. Translating those
selections into files/flags for a specific tool is the activation engine's and
the launch adapters' job — not the resolver's. Passing ``tool_id`` only *narrows*
the same structure to a single tool.

The module is pure in the sense that matters here: it never writes to the
filesystem and never mutates its inputs. It does *read* the project (to enumerate
skills/agents on disk and to run the MCP/permission/hook discoverers), because
:class:`ProjectScan` records where those live but not their individual names.
"""

from __future__ import annotations

import contextlib
import fnmatch
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from crossby.config.skills import list_skills
from crossby.models.ai import AIToolID
from crossby.models.config import SCENE_CONCERNS, SceneConfig, SceneSelector
from crossby.sync.readers import (
    ProjectScan,
    discover_hooks,
    discover_mcp,
    discover_permissions,
)

# Files whose stem names an agent, longest suffix first so ``foo.agent.md``
# resolves to ``foo`` rather than ``foo.agent``.
_AGENT_SUFFIXES: tuple[str, ...] = (".agent.md", ".md", ".toml")


@dataclass(frozen=True)
class ResolvedGroup:
    """One resolved ``(concern, target_path)`` bucket.

    ``target_path`` is the relative path multiple tools may share (e.g.
    ``.agents/skills`` for both ``codex`` and ``antigravity-cli``), or ``None``
    for concerns that have no per-tool directory (mcp / permissions / hooks).
    ``tools`` are the tools attributed to this bucket; ``names`` are the selected
    canonical item names present in it. Both are sorted for determinism.
    """

    concern: str
    target_path: str | None
    tools: tuple[AIToolID, ...]
    names: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedScene:
    """The outcome of resolving one scene against a project scan.

    ``groups`` is the generic path-keyed structure: tools sharing a resolved
    target path collapse into a single group, so any current or future path
    collision is handled uniformly. ``warnings`` lists selectors that matched no
    detected item — surfaced, never raised.
    """

    tool_id: AIToolID | None
    groups: tuple[ResolvedGroup, ...]
    warnings: tuple[str, ...]

    def groups_for(self, concern: str) -> tuple[ResolvedGroup, ...]:
        """Return the resolved groups for a single concern."""
        return tuple(g for g in self.groups if g.concern == concern)

    def names(self, concern: str) -> tuple[str, ...]:
        """Return the union of selected item names for a concern, sorted."""
        seen: dict[str, None] = {}
        for group in self.groups:
            if group.concern == concern:
                for name in group.names:
                    seen.setdefault(name, None)
        return tuple(sorted(seen))


@dataclass(frozen=True)
class _RawGroup:
    """A pre-selection bucket: everything detected at one target path."""

    target_path: str | None
    tools: tuple[AIToolID, ...]
    names: tuple[str, ...]


def scene_root(project_root: Path) -> Path:
    """The dir the scene subsystem operates in — the found config's parent.

    Scene state (``.crossby/scene-state.json``), the ``owned.json`` ledger, the
    projection tree, and the project scan must all root at the SAME directory
    the config loaded from. :func:`~crossby.config.loader.load_config` walks
    *up* to a parent ``.crossby.yml``, so a command run from a subdirectory
    must resolve state/scan against ``config.config_path.parent``, not the
    invocation dir. Falls back to *project_root* itself when no config exists.

    Uses :func:`~crossby.config.loader.find_config_file` (no parse) rather than
    ``load_config``, so callers that must stay robust to a malformed config
    (``clear``/``status``, which revert from the ledger) can still resolve a
    root without a parse error getting in the way.

    NOTE: pre-existing shadow ``.crossby/`` state left by older buggy subdir
    runs is not migrated — this only stops NEW shadows and always operates on
    the config-rooted state.
    """
    from crossby.config.loader import find_config_file

    found = find_config_file(project_root)
    return found.parent if found is not None else project_root


def resolve_scene(
    scene: SceneConfig,
    scan: ProjectScan,
    project_root: Path,
    *,
    tool_id: AIToolID | None = None,
) -> ResolvedScene:
    """Resolve *scene* against *scan* into concrete per-concern selections.

    *scene* must already be flat (``extends`` folded in). ``tool_id=None`` — the
    normal mode — resolves across every tool; passing a ``tool_id`` narrows the
    result to buckets that tool participates in. Selectors that match nothing
    become :attr:`ResolvedScene.warnings`, never exceptions; the function reads
    the project but never writes to it.
    """
    groups: list[ResolvedGroup] = []
    warnings: list[str] = []

    for concern in SCENE_CONCERNS:
        selector: SceneSelector | None = getattr(scene, concern)
        raw_groups = _collect_groups(concern, scan, project_root)

        universe = sorted({name for group in raw_groups for name in group.names})
        selected, concern_warnings = _match_selector(universe, selector, concern)
        warnings.extend(concern_warnings)

        for raw in raw_groups:
            chosen = tuple(name for name in raw.names if name in selected)
            if not chosen:
                continue
            tools = raw.tools
            if tool_id is not None:
                if tool_id not in tools:
                    continue
                tools = (tool_id,)
            groups.append(
                ResolvedGroup(
                    concern=concern,
                    target_path=raw.target_path,
                    tools=tools,
                    names=chosen,
                )
            )

    return ResolvedScene(tool_id=tool_id, groups=tuple(groups), warnings=tuple(warnings))


def concern_universe(scan: ProjectScan, project_root: Path) -> dict[str, tuple[str, ...]]:
    """Every candidate item name per concern, before any selector is applied.

    This is exactly the inventory :func:`resolve_scene` matches globs against —
    exposed directly so the authoring wizard and the ``--skill`` / ``--agent`` /
    … selectors can offer only items that actually exist in the project, without
    having to resolve an all-inclusive scene to enumerate them. Keys are the
    members of :data:`SCENE_CONCERNS`; each value is sorted and de-duplicated.
    """
    return {
        concern: tuple(
            sorted(
                {
                    name
                    for group in _collect_groups(concern, scan, project_root)
                    for name in group.names
                }
            )
        )
        for concern in SCENE_CONCERNS
    }


# ---------------------------------------------------------------------------
# Selector matching
# ---------------------------------------------------------------------------


def _match_selector(
    universe: list[str], selector: SceneSelector | None, concern: str
) -> tuple[set[str], list[str]]:
    """Apply one selector's include/exclude globs to *universe*.

    Returns the selected names and a list of warnings for globs that matched
    nothing. Semantics: an omitted selector (``None``) applies no filter;
    ``include`` absent means "start from everything"; ``include: []`` means
    "start from nothing"; ``exclude`` is applied last and always wins.
    """
    warnings: list[str] = []
    if selector is None:
        return set(universe), warnings

    if selector.include is None:
        selected = set(universe)
    else:
        selected = set()
        for pattern in selector.include:
            matched = _match_pattern(universe, pattern)
            if not matched:
                warnings.append(_no_match_warning(concern, "include", pattern))
            selected |= matched

    for pattern in selector.exclude:
        matched = _match_pattern(universe, pattern)
        if not matched:
            warnings.append(_no_match_warning(concern, "exclude", pattern))
        selected -= matched

    return selected, warnings


def _match_pattern(universe: Iterable[str], pattern: str) -> set[str]:
    """Return the names in *universe* the fnmatch *pattern* matches (case-sensitive)."""
    return {name for name in universe if fnmatch.fnmatchcase(name, pattern)}


def _no_match_warning(concern: str, kind: str, pattern: str) -> str:
    return f"scene {concern} {kind} selector {pattern!r} matched no detected item"


# ---------------------------------------------------------------------------
# Per-concern inventory normalisation
#
# ``ProjectScan``'s ``found`` payload differs by concern (skills/agents record a
# per-tool *directory*; mcp/permissions/hooks are name lists). build_sync_data
# does not enumerate individual skill/agent names — it only picks a source
# directory — so this is genuinely new unpacking, done here into a uniform
# ``_RawGroup`` list keyed by resolved target path.
# ---------------------------------------------------------------------------


def _collect_groups(concern: str, scan: ProjectScan, project_root: Path) -> list[_RawGroup]:
    if concern == "skills":
        return _dir_groups(scan.skills.found, project_root, list_skills)
    if concern == "agents":
        return _dir_groups(scan.agents.found, project_root, _list_agent_names)
    if concern == "mcp":
        return _global_group(sorted(discover_mcp(project_root)), scan)
    if concern == "permissions":
        # sorted() like mcp/hooks: discover_permissions preserves reader order,
        # so sort here to keep every concern's group.names deterministic.
        return _global_group(sorted(discover_permissions(project_root)), scan)
    if concern == "hooks":
        names = [f"{hook.event}:{hook.command}" for hook in discover_hooks(project_root)]
        return _global_group(sorted(names), scan)
    return []  # unreachable — SCENE_CONCERNS is the single source of truth


def _dir_groups(
    found: object,
    project_root: Path,
    lister: Callable[[Path], list[str]],
) -> list[_RawGroup]:
    """Group a per-tool directory map by resolved path, enumerating each once.

    *found* is ``ProjectScan``'s ``{tool: relative_dir}`` payload (typed ``Any``
    on the scan). Tools sharing a directory (codex + antigravity-cli both map to
    ``.agents/skills``) collapse into one group; the directory is enumerated a
    single time per distinct path via *lister*.
    """
    if not isinstance(found, dict):
        return []

    tools_by_path: dict[str, list[AIToolID]] = {}
    for tool, rel in found.items():
        if isinstance(tool, AIToolID) and isinstance(rel, str):
            tools_by_path.setdefault(rel, []).append(tool)

    groups: list[_RawGroup] = []
    for rel, tools in tools_by_path.items():
        names = lister(project_root / rel)
        groups.append(
            _RawGroup(
                target_path=rel,
                tools=tuple(sorted(tools, key=str)),
                names=tuple(names),
            )
        )
    return groups


def _global_group(names: list[str], scan: ProjectScan) -> list[_RawGroup]:
    """Build the single path-less group for a project-wide concern.

    *names* is expected already deduplicated and sorted by the caller.
    Attribution is tool-agnostic: these items apply across every installed tool,
    so the group carries ``target_path=None`` and all installed tools.
    """
    if not names:
        return []
    return [
        _RawGroup(
            target_path=None,
            tools=tuple(sorted(scan.installed_tools, key=str)),
            names=tuple(names),
        )
    ]


def _list_agent_names(directory: Path) -> list[str]:
    """Return agent names in *directory*, stripping known agent-file suffixes.

    Mirrors how the rest of crossby detects agents (``*.md`` files), and also
    accepts codex ``*.toml`` and antigravity ``*.agent.md`` so a name matches
    regardless of which tool's directory it lives in. Sorted; unreadable or
    missing directories yield an empty list.
    """
    names: set[str] = set()
    with contextlib.suppress(OSError):
        if directory.is_dir():
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                for suffix in _AGENT_SUFFIXES:
                    if path.name.endswith(suffix):
                        names.add(path.name[: -len(suffix)])
                        break
    return sorted(names)
