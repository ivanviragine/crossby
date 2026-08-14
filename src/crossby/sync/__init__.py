"""Sync framework — global registry and run_sync() orchestrator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import structlog

from crossby.models.ai import AIToolID
from crossby.sync.agents import (
    AntigravityCLIAgentsWriter,
    ClaudeAgentsWriter,
    CodexAgentsWriter,
    CopilotAgentsWriter,
    CursorAgentsWriter,
    update_agents_gitignore,
)
from crossby.sync.base import (
    AbstractSyncWriter,
    SyncConcern,
    SyncData,
    SyncRegistry,
    SyncResult,
)
from crossby.sync.hooks import (
    AntigravityCLIHooksWriter,
    ClaudeHooksWriter,
    CodexHooksWriter,
    CopilotHooksWriter,
    CursorHooksWriter,
)
from crossby.sync.mcp import (
    AntigravityCLIMCPWriter,
    ClaudeMCPWriter,
    CodexMCPWriter,
    CopilotMCPWriter,
    CursorMCPWriter,
)
from crossby.sync.permissions import (
    ClaudePermissionWriter,
    CursorPermissionWriter,
)
from crossby.sync.rules import (
    AntigravityCLIRulesWriter,
    ClaudeRulesWriter,
    CodexRulesWriter,
    CopilotRulesWriter,
    CursorRulesWriter,
    update_rules_gitignore,
)
from crossby.sync.safe_write import SyncContainmentError
from crossby.sync.skills import (
    AntigravityCLISkillsWriter,
    ClaudeSkillsWriter,
    CodexSkillsWriter,
    CopilotSkillsWriter,
    CursorSkillsWriter,
    update_skills_gitignore,
)

# Global default registry — one writer per (tool, concern) pair.
# Neither Codex nor Antigravity CLI has a PERMISSIONS writer: their autonomy is
# mode-based launch flags (Codex sandbox mode; agy --mode/--sandbox/
# --dangerously-skip-permissions), not a per-project policy file to sync. Both
# DO have a HOOKS writer — CodexHooksWriter (.codex/hooks.json) and
# AntigravityCLIHooksWriter (.agents/hooks.json), registered below — each with a
# matching reader in sync/readers.py, so every hooks writer round-trips.
logger = structlog.get_logger()

# .gitignore managed-block id for the per-machine ownership ledger.
_LEDGER_GITIGNORE_BLOCK_ID = "sync ownership"

_registry = SyncRegistry()
_registry.register(ClaudePermissionWriter())
_registry.register(CursorPermissionWriter(scope="project"))
_registry.register(ClaudeAgentsWriter())
_registry.register(CopilotAgentsWriter())
_registry.register(CursorAgentsWriter())
_registry.register(CodexAgentsWriter())
_registry.register(AntigravityCLIAgentsWriter())
_registry.register(ClaudeRulesWriter())
_registry.register(CursorRulesWriter())
_registry.register(CopilotRulesWriter())
_registry.register(CodexRulesWriter())
_registry.register(AntigravityCLIRulesWriter())
_registry.register(ClaudeMCPWriter())
_registry.register(CursorMCPWriter())
_registry.register(CopilotMCPWriter())
_registry.register(CodexMCPWriter())
_registry.register(AntigravityCLIMCPWriter())
_registry.register(ClaudeHooksWriter())
_registry.register(CursorHooksWriter())
_registry.register(CopilotHooksWriter())
_registry.register(CodexHooksWriter())
_registry.register(AntigravityCLIHooksWriter())
_registry.register(ClaudeSkillsWriter())
_registry.register(CursorSkillsWriter())
_registry.register(CodexSkillsWriter())
_registry.register(AntigravityCLISkillsWriter())
_registry.register(CopilotSkillsWriter())


def run_sync(
    data: SyncData,
    project_root: Path,
    *,
    tool_id: AIToolID | None = None,
    concern: SyncConcern | None = None,
    dry_run: bool = False,
    force: bool = False,
    installed_tools: list[AIToolID] | None = None,
    registry: SyncRegistry | None = None,
    include_user_scope: bool = False,
    skip_concerns: frozenset[SyncConcern] | set[SyncConcern] | None = None,
) -> list[SyncResult]:
    """Run all matching sync writers, collecting results.

    Continue-on-error: if one writer raises, the error is recorded in
    ``SyncResult(action="error")`` and the next writer proceeds.

    Args:
        data: Sync input data (from readers or wizard).
        project_root: Project root directory.
        tool_id: When set, only writers for this tool run, and the
            installed-tools filter is bypassed.  When None, all installed
            tools run.
        concern: When set, only writers for this concern run.
        dry_run: Compute results without writing any files.
        force: If True, overwrite existing target directories (with backup).
        installed_tools: Override the installed-tools list.  Detected
            automatically when None.  Ignored when ``tool_id`` is set.
        registry: Custom registry (defaults to the global ``_registry``).
        include_user_scope: Whether to include user-scope ``~/.claude.json``
            in MCP discovery and validation.
        skip_concerns: Concerns whose writers must not run at all. Used by the
            single-source ``--from`` paths to drop a concern the source tool has
            no reader for: an empty source there is *unknown*, not *emptied*, so
            letting the ownership diff run would revoke previously-synced entries
            from targets. Distinct from ``concern`` (which selects one to run) —
            this excludes.

    Returns:
        List of SyncResult, one per writer that ran.
    """
    reg = registry or _registry
    writers = reg.get_writers(tool_id=tool_id, concern=concern)

    if skip_concerns:
        writers = [w for w in writers if w.concern not in skip_concerns]

    # When no specific tool is requested, restrict to installed (or provided) tools.
    if tool_id is None:
        if installed_tools is None:
            from crossby.ai_tools.base import AbstractAITool

            installed_tools = AbstractAITool.detect_installed()

        writers = [w for w in writers if w.tool_id in installed_tools]

    # Group whole-file *overwrite* writers by their physical target path so a
    # collision (Codex + Antigravity CLI both own ``AGENTS.md`` /
    # ``.agents/skills/``) collapses to a single deterministic winner instead of
    # churning the file on every run and letting registry iteration order decide
    # the final bytes. Only writers that opt in via ``target_path`` (rules/
    # agents/skills) are grouped; merge writers (permissions/MCP/hooks) return
    # ``None`` and are never collapsed — they co-write shared files by key.
    #
    # The grouping key canonicalises the *parent* (mirroring
    # ``file_utils.is_same_path``) without following a target symlink, so
    # identical ``_target_rel`` strings always collide even when the project
    # root is reached through a symlink; the displayed ``file_path`` stays the
    # plain ``project_root / rel``. The winner is the first group member in the
    # original ``writers`` sequence — i.e. registration order in this module —
    # which is the documented ownership precedence (Codex before Antigravity CLI).
    def _grouping_key(writer: AbstractSyncWriter) -> Path | None:
        tp = writer.target_path(project_root)
        if tp is None:
            return None
        # Canonicalising the parent can fail — ``Path.resolve()`` raises
        # ``RuntimeError`` on a symlink loop in the parent chain (documented
        # behaviour on Python < 3.13), and other resolution failures surface as
        # ``OSError``. This runs *before* the per-writer try/except below, so an
        # unguarded failure here would abort the whole ``run_sync`` instead of
        # leaving the bad target to the writer's own containment guard (which
        # reports an ``error`` row and lets the other writers proceed). Fall back
        # to the un-resolved path so identical literal targets still collide.
        try:
            return tp.parent.resolve() / tp.name
        except (OSError, RuntimeError):
            return tp

    writer_keys: dict[int, Path | None] = {}
    group_winner: dict[Path, AbstractSyncWriter] = {}
    for w in writers:
        key = _grouping_key(w)
        writer_keys[id(w)] = key
        if key is not None and key not in group_winner:
            group_winner[key] = w
    # The winner's actual result, recorded when it runs so covered rows can
    # mirror the artifact path it produced (see below). The winner is the
    # first group member in ``writers`` order, so it is always processed
    # before any covered member of the same group.
    group_winner_result: dict[Path, SyncResult] = {}

    # Ownership ledger — revocation is computed *here*, never inferred by a
    # writer. For each hooks/permissions/MCP writer we diff what crossby wrote
    # last time (the ledger) against the current sync data and hand the writer an
    # explicit, provenance-bounded removal set. A missing/malformed ledger loads
    # empty, so this degrades to purely additive behaviour.
    from crossby.sync.ownership import LEDGER_PATH, load_ledger, save_ledger

    ledger = load_ledger(project_root)
    current_hooks = {(h.event, h.command) for h in data.hooks}
    current_perms = set(data.allowed_commands)
    disabled_mcp = {name for name, s in data.mcp_servers.items() if not s.enabled}
    # tool → the identity set crossby should own after a successful write.
    # Applied to the ledger only for writers that did not error.
    pending_hooks: dict[AIToolID, set[tuple[str, str]]] = {}
    pending_perms: dict[AIToolID, set[str]] = {}
    pending_mcp: dict[AIToolID, set[str]] = {}

    results: list[SyncResult] = []
    agents_writers_ran = False
    rules_writers_ran = False
    skills_writers_ran = False
    for writer in writers:
        # Covered-by short-circuit: a non-winner of a whole-file target group is
        # emitted as ``skipped "covered by <winner>"`` in its ORIGINAL position
        # without running ``.sync()``. Iterating the original sequence (rather
        # than reordering winners-first) keeps covered rows where a custom or
        # interleaved registry placed them. Grouping applies identically under
        # ``dry_run`` — the winner computes its dry-run result, the rest are
        # covered. Covered rows carry no ``created`` identities, so the ownership
        # ledger is unaffected.
        group_key = writer_keys[id(writer)]
        if group_key is not None and group_winner[group_key] is not writer:
            winner = group_winner[group_key]
            # Mirror the winner's *produced* artifact path, not this writer's
            # static ``target_path()``. When the winner wrote nothing (e.g. it
            # returned ``skipped`` with ``file_path=None`` because no source was
            # configured), the covered row must also carry ``None`` so the report
            # classifies it as "Not Added" rather than "Added". The winner ran in
            # an earlier iteration, so its result is already recorded.
            winner_result = group_winner_result[group_key]
            results.append(
                SyncResult(
                    tool_id=writer.tool_id,
                    concern=writer.concern,
                    action="skipped",
                    file_path=winner_result.file_path,
                    message=f"covered by {winner.tool_id}",
                )
            )
            if writer.concern == SyncConcern.AGENTS:
                agents_writers_ran = True
            elif writer.concern == SyncConcern.RULES:
                rules_writers_ran = True
            elif writer.concern == SyncConcern.SKILLS:
                skills_writers_ran = True
            continue

        writer_data = data
        hooks_owned: frozenset[tuple[str, str]] = frozenset()
        perms_owned: frozenset[str] = frozenset()
        mcp_owned: frozenset[str] = frozenset()
        if writer.concern == SyncConcern.HOOKS:
            hooks_owned = ledger.hooks(writer.tool_id)
            writer_data = replace(
                data,
                hooks_remove=sorted(hooks_owned - current_hooks),
                hooks_owned=hooks_owned,
            )
        elif writer.concern == SyncConcern.PERMISSIONS:
            perms_owned = ledger.permissions(writer.tool_id)
            writer_data = replace(data, permissions_remove=sorted(perms_owned - current_perms))
        elif writer.concern == SyncConcern.MCP:
            mcp_owned = ledger.mcp(writer.tool_id)
            # Only ledger-owned *and* explicitly-disabled servers may be removed,
            # so a same-named server crossby never wrote survives.
            writer_data = replace(data, mcp_remove=frozenset(disabled_mcp & mcp_owned))

        try:
            result = writer.safe_sync(writer_data, project_root, dry_run=dry_run, force=force)
        except Exception as exc:
            result = SyncResult(
                tool_id=writer.tool_id,
                concern=writer.concern,
                action="error",
                message=str(exc),
            )
        results.append(result)
        # Record the winner's result so later covered rows in the same group can
        # mirror its produced artifact path. Only a group winner reaches here (a
        # non-winner is short-circuited above), so this never overwrites.
        if group_key is not None:
            group_winner_result[group_key] = result

        # Record intended new ownership, gated on writer success. New ownership =
        # what crossby still owns that is still in the source (previously-owned ∩
        # current, i.e. previously-owned minus what was revoked) PLUS what the
        # writer wrote **fresh** this run (``result.created``). Crucially this is
        # NOT the whole source set: a human entry that merely shares an identity
        # with a source entry is never in ``created`` (the writer found it already
        # present), so crossby never claims — and thus never narrows or revokes —
        # it. An ``error`` row leaves the ledger untouched entirely.
        if result.action != "error":
            if writer.concern == SyncConcern.HOOKS:
                created_hooks = cast(
                    "set[tuple[str, str]]", {c for c in result.created if isinstance(c, tuple)}
                )
                pending_hooks[writer.tool_id] = set(hooks_owned & current_hooks) | created_hooks
            elif writer.concern == SyncConcern.PERMISSIONS:
                created_perms = {c for c in result.created if isinstance(c, str)}
                pending_perms[writer.tool_id] = set(perms_owned & current_perms) | created_perms
            elif writer.concern == SyncConcern.MCP:
                created_mcp = {c for c in result.created if isinstance(c, str)}
                pending_mcp[writer.tool_id] = set(mcp_owned - disabled_mcp) | created_mcp

        if writer.concern == SyncConcern.AGENTS:
            agents_writers_ran = True
        if writer.concern == SyncConcern.RULES:
            rules_writers_ran = True
        if writer.concern == SyncConcern.SKILLS:
            skills_writers_ran = True

    # After all agents/rules/skills writers, update the .gitignore managed block
    # once each. These post-writer side effects run OUTSIDE the per-writer
    # try/except above, so a ``SyncContainmentError`` (a symlinked ``.gitignore``)
    # would otherwise escape ``run_sync`` entirely instead of becoming an
    # ``error`` row. Wrap each in the same error model. Skip when a specific tool
    # filter is active to avoid cross-tool side effects during --tool runs.
    def _side_effect_error(concern: SyncConcern, exc: SyncContainmentError) -> SyncResult:
        return SyncResult(tool_id=None, concern=concern, action="error", message=str(exc))

    if agents_writers_ran and tool_id is None:
        try:
            gi_result = update_agents_gitignore(
                data, project_root, dry_run=dry_run, installed_tools=installed_tools
            )
            if gi_result is not None:
                results.append(gi_result)
        except SyncContainmentError as exc:
            results.append(_side_effect_error(SyncConcern.AGENTS, exc))

    if rules_writers_ran and tool_id is None:
        try:
            gi_result = update_rules_gitignore(
                data, project_root, dry_run=dry_run, installed_tools=installed_tools
            )
            if gi_result is not None:
                results.append(gi_result)
        except SyncContainmentError as exc:
            results.append(_side_effect_error(SyncConcern.RULES, exc))

    if skills_writers_ran and tool_id is None:
        try:
            gi_result = update_skills_gitignore(
                data, project_root, dry_run=dry_run, installed_tools=installed_tools
            )
            if gi_result is not None:
                results.append(gi_result)
        except SyncContainmentError as exc:
            results.append(_side_effect_error(SyncConcern.SKILLS, exc))

    # Persist the ownership ledger for every hooks/permissions/MCP writer that
    # succeeded. ``dry_run`` computes revocations above but writes neither the
    # targets nor the ledger, matching the report-only contract of --plan.
    if (pending_hooks or pending_perms or pending_mcp) and not dry_run:
        for hook_tool, hook_pairs in pending_hooks.items():
            ledger.record_hooks(hook_tool, hook_pairs)
        for perm_tool, perm_pats in pending_perms.items():
            ledger.record_permissions(perm_tool, perm_pats)
        for mcp_tool, mcp_names in pending_mcp.items():
            ledger.record_mcp(mcp_tool, mcp_names)
        # The ledger is advisory: a persistence failure (read-only dir, no
        # permission, full disk) must not discard the SyncResults for writes
        # that already succeeded, so mirror run_sync's per-writer isolation
        # here. load_ledger already degrades to "own nothing" on a missing or
        # malformed file, so a dropped save simply retries next run.
        try:
            save_ledger(project_root, ledger)
            # Ensure the ledger is gitignored whenever the file exists on disk —
            # decoupled from save_ledger's change detection so a prior transient
            # .gitignore failure self-heals on the next sync (rather than being
            # skipped forever because the unchanged ledger makes save_ledger
            # return False). update_managed_block is itself a no-op when the
            # block is already present, so this is cheap on the common path; an
            # empty ledger is never materialised, so there's nothing to ignore.
            if (project_root / LEDGER_PATH).is_file():
                from crossby.sync.gitignore_utils import update_managed_block

                update_managed_block(
                    project_root,
                    _LEDGER_GITIGNORE_BLOCK_ID,
                    [LEDGER_PATH.as_posix()],
                )
        except OSError as exc:
            logger.warning("ownership.persist_failed", path=str(project_root), error=str(exc))
        except SyncContainmentError as exc:
            # A symlinked ``.crossby/owned.json`` or ``.gitignore`` on the
            # post-writer path: surface as an ``error`` row (the writes already
            # succeeded) rather than escaping run_sync. Attribute the failure to
            # the concern(s) whose ownership actually failed to persist — a
            # hooks-only or permissions-only sync must not mis-report a ledger
            # containment failure under MCP (the ledger holds all three, but only
            # the concerns with pending ownership were being recorded this run).
            logger.warning("ownership.persist_refused", path=str(project_root), error=str(exc))
            for affected_concern, pending in (
                (SyncConcern.HOOKS, pending_hooks),
                (SyncConcern.PERMISSIONS, pending_perms),
                (SyncConcern.MCP, pending_mcp),
            ):
                if pending:
                    results.append(
                        SyncResult(
                            tool_id=None,
                            concern=affected_concern,
                            action="error",
                            message=str(exc),
                        )
                    )

    # Plugin discovery — append manual-fix rows when scoped to all tools or
    # when the user explicitly asked for the plugins concern. We don't run
    # for narrow --tool runs because plugins aren't a per-target output.
    if tool_id is None and (concern is None or concern == SyncConcern.PLUGINS):
        from crossby.sync.plugins import report_plugins

        results.extend(report_plugins(project_root))

    # MCP oauth-config discovery — same detect-only shape as plugins above:
    # append manual-fix rows for source MCP servers with an `oauth` block
    # that no writer ports across tools.
    if tool_id is None and (concern is None or concern == SyncConcern.MCP):
        from crossby.sync.mcp import report_dropped_default_fallbacks
        from crossby.sync.mcp_discovery import (
            report_duplicate_claude_servers,
            report_oauth_configs,
        )

        results.extend(report_oauth_configs(project_root, include_user_scope=include_user_scope))
        results.extend(report_duplicate_claude_servers(project_root))
        results.extend(report_dropped_default_fallbacks(data.mcp_servers))

    return results


__all__ = [
    "SyncConcern",
    "SyncData",
    "SyncRegistry",
    "SyncResult",
    "_registry",
    "run_sync",
]
