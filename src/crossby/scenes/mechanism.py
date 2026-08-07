"""The ``(tool, concern) -> SceneMechanism`` matrix and shared-path resolution.

Each cell records the *least-invasive* mechanism crossby can use to make one
tool honour a scene's selection for one concern, with the rationale documented
inline. The base matrix is per-cell, but the final choice is made **after**
grouping by resolved target path: when several tools share one physical
directory and any of them lacks a DECLARE lever, PROJECT re-points that
directory for all of them, so a best-effort DECLARE can never contradict the
authoritative re-point (the "PROJECT wins for a shared path" rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from crossby.models.ai import AIToolID
from crossby.services.scene_resolution import ResolvedScene


class SceneMechanism(StrEnum):
    """How one ``(tool, concern)`` cell enacts a scene selection."""

    #: Write the tool's own disable key. Non-destructive, instantly reversible.
    DECLARE = "declare"
    #: Materialise a filtered source / drive run_sync's revocable removal channel.
    PROJECT = "project"
    #: No per-item lever exists for this tool+concern — reported, never faked.
    UNSUPPORTED = "unsupported"


# ---------------------------------------------------------------------------
# Base matrix. Concern strings match ``models.config.SCENE_CONCERNS``:
# ("skills", "agents", "mcp", "hooks", "permissions"). Tools without any sync
# writer for a concern (or any lever at all) map to UNSUPPORTED. Only the five
# scene-participating tools appear; vscode / antigravity (IDE) are GUI tools
# dropped at launch time. opencode has no scene lever either (no launch flag,
# no sync writer), so it isn't listed — base_mechanism() returns UNSUPPORTED for
# it, and a scene narrowing its MCP set is reported "deselected server(s) remain
# enabled" rather than silently dropped.
# ---------------------------------------------------------------------------

_BASE_MATRIX: dict[tuple[AIToolID, str], SceneMechanism] = {
    # -- skills ------------------------------------------------------------
    # Claude honours a per-skill `skillOverrides: {"<name>": "off"}` key
    # (>= 2.1.129), leaving `.claude/skills` fully populated but filtered.
    (AIToolID.CLAUDE, "skills"): SceneMechanism.DECLARE,
    # Codex `[[skills.config]] enabled = false` exists in principle, but it is an
    # array-of-tables the surgical TOML editor (sync/toml_edit) can't splice, and
    # it is non-authoritative anyway: Codex shares `.agents/skills` with
    # Antigravity CLI (no lever), so the shared-path rule makes PROJECT
    # authoritative for that directory. Declaring PROJECT here keeps the two
    # sharers consistent and sidesteps the array-of-tables gap.
    (AIToolID.CODEX, "skills"): SceneMechanism.PROJECT,
    # No per-skill lever — re-point the whole skills directory.
    (AIToolID.ANTIGRAVITY_CLI, "skills"): SceneMechanism.PROJECT,
    (AIToolID.CURSOR, "skills"): SceneMechanism.PROJECT,
    (AIToolID.COPILOT, "skills"): SceneMechanism.PROJECT,
    # -- agents ------------------------------------------------------------
    # Claude blocks a named subagent with `permissions.deny: ["Agent(<name>)"]`
    # (verified honoured on 2.1.x; `Task` was renamed `Agent` in 2.1.63).
    (AIToolID.CLAUDE, "agents"): SceneMechanism.DECLARE,
    # No agent-disable lever on the rest — re-point/re-translate a filtered
    # agents source (each tool has its own agents directory; no path sharing).
    (AIToolID.CODEX, "agents"): SceneMechanism.PROJECT,
    (AIToolID.CURSOR, "agents"): SceneMechanism.PROJECT,
    (AIToolID.COPILOT, "agents"): SceneMechanism.PROJECT,
    (AIToolID.ANTIGRAVITY_CLI, "agents"): SceneMechanism.PROJECT,
    # -- mcp ---------------------------------------------------------------
    # Each of these has a native per-server disable key we write in place;
    # the user's actual server definitions are left untouched.
    (AIToolID.CLAUDE, "mcp"): SceneMechanism.DECLARE,  # disabledMcpjsonServers
    (AIToolID.CODEX, "mcp"): SceneMechanism.DECLARE,  # mcp_servers.<id>.enabled=false
    (AIToolID.ANTIGRAVITY_CLI, "mcp"): SceneMechanism.DECLARE,  # mcpServers.<n>.disabled
    # Cursor / Copilot expose no per-server disable key; deleting the server
    # would be destructive, so a scene cannot narrow their MCP set.
    (AIToolID.CURSOR, "mcp"): SceneMechanism.UNSUPPORTED,
    (AIToolID.COPILOT, "mcp"): SceneMechanism.UNSUPPORTED,
    # -- hooks -------------------------------------------------------------
    # No tool crossby supports has a per-hook disable lever, and the
    # all-or-nothing `disableAllHooks` would take out user hooks too. The only
    # safe narrowing is removal, bounded to crossby-owned entries via the
    # revocable-sync ledger — that is what PROJECT means for hooks here.
    (AIToolID.CLAUDE, "hooks"): SceneMechanism.PROJECT,
    (AIToolID.CURSOR, "hooks"): SceneMechanism.PROJECT,
    (AIToolID.CODEX, "hooks"): SceneMechanism.PROJECT,
    (AIToolID.ANTIGRAVITY_CLI, "hooks"): SceneMechanism.PROJECT,
    (AIToolID.COPILOT, "hooks"): SceneMechanism.PROJECT,
    # -- permissions -------------------------------------------------------
    # Only Claude and Cursor have a permissions writer / allowlist file; a
    # scene narrows the allowlist by revoking crossby-owned entries no longer
    # selected. Codex / Copilot / Antigravity autonomy is a launch flag, not a
    # per-project policy file, so there is nothing to narrow.
    (AIToolID.CLAUDE, "permissions"): SceneMechanism.PROJECT,
    (AIToolID.CURSOR, "permissions"): SceneMechanism.PROJECT,
    (AIToolID.CODEX, "permissions"): SceneMechanism.UNSUPPORTED,
    (AIToolID.COPILOT, "permissions"): SceneMechanism.UNSUPPORTED,
    (AIToolID.ANTIGRAVITY_CLI, "permissions"): SceneMechanism.UNSUPPORTED,
}


def base_mechanism(tool: AIToolID, concern: str) -> SceneMechanism:
    """The per-cell mechanism before shared-path resolution.

    Unknown ``(tool, concern)`` pairs (a tool with no writer for the concern)
    default to :attr:`SceneMechanism.UNSUPPORTED`.
    """
    return _BASE_MATRIX.get((tool, concern), SceneMechanism.UNSUPPORTED)


@dataclass(frozen=True)
class ActivationUnit:
    """One dispatchable slice of a scene apply.

    A *directory* concern (skills / agents — ``target_path`` set) yields one
    unit per shared path, so ``tools`` may hold several tools that resolve to
    that path and ``mechanism`` is the single resolved choice for all of them
    (reported once). A *global* concern (mcp / hooks / permissions —
    ``target_path`` is ``None``) yields one unit per tool, each carrying its own
    base mechanism. ``names`` are the scene-*selected* item names for the unit.
    """

    concern: str
    target_path: str | None
    tools: tuple[AIToolID, ...]
    names: tuple[str, ...]
    mechanism: SceneMechanism


def _resolve_shared(tools: tuple[AIToolID, ...], concern: str) -> SceneMechanism:
    """Resolve a directory concern shared by *tools* into one mechanism.

    DECLARE holds only if **every** tool sharing the path can DECLARE; if any
    tool must PROJECT (or is UNSUPPORTED, which for a directory concern still
    means "re-point the physical dir"), PROJECT is authoritative for all of
    them — otherwise a best-effort DECLARE would claim to control an outcome the
    re-point already decided.
    """
    mechs = {base_mechanism(tool, concern) for tool in tools}
    if mechs == {SceneMechanism.DECLARE}:
        return SceneMechanism.DECLARE
    return SceneMechanism.PROJECT


def plan_units(resolved: ResolvedScene) -> list[ActivationUnit]:
    """Turn a :class:`ResolvedScene` into ordered, dispatch-ready units.

    Directory concerns collapse per shared ``target_path`` (the resolver already
    grouped them), so the shared-path precedence rule is applied once per group.
    Global concerns split per tool so each tool's independent mechanism is
    preserved. Order follows ``resolved.groups`` (concern order from
    ``SCENE_CONCERNS``), which keeps reports deterministic.
    """
    units: list[ActivationUnit] = []
    for group in resolved.groups:
        if group.target_path is not None:
            units.append(
                ActivationUnit(
                    concern=group.concern,
                    target_path=group.target_path,
                    tools=group.tools,
                    names=group.names,
                    mechanism=_resolve_shared(group.tools, group.concern),
                )
            )
        else:
            for tool in group.tools:
                units.append(
                    ActivationUnit(
                        concern=group.concern,
                        target_path=None,
                        tools=(tool,),
                        names=group.names,
                        mechanism=base_mechanism(tool, group.concern),
                    )
                )
    return units
