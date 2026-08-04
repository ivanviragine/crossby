"""Sync framework — global registry and run_sync() orchestrator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

from crossby.models.ai import AIToolID
from crossby.sync.agents import (
    AntigravityCLIAgentsWriter,
    ClaudeAgentsWriter,
    CodexAgentsWriter,
    CopilotAgentsWriter,
    CursorAgentsWriter,
    update_agents_gitignore,
)
from crossby.sync.base import SyncConcern, SyncData, SyncRegistry, SyncResult
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
from crossby.sync.skills import (
    AntigravityCLISkillsWriter,
    ClaudeSkillsWriter,
    CodexSkillsWriter,
    CopilotSkillsWriter,
    CursorSkillsWriter,
    update_skills_gitignore,
)

# Global default registry — one writer per (tool, concern) pair.
# Antigravity CLI has no (ANTIGRAVITY_CLI, PERMISSIONS) or (ANTIGRAVITY_CLI,
# HOOKS) writer: its permission model is mode-based (--mode/--sandbox/
# --dangerously-skip-permissions launch flags, no per-project policy file)
# and it has no hook system at all — same absence pattern as Codex having
# no permission writer (sandbox mode is inherent, not a file to write).
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

    Returns:
        List of SyncResult, one per writer that ran.
    """
    reg = registry or _registry
    writers = reg.get_writers(tool_id=tool_id, concern=concern)

    # When no specific tool is requested, restrict to installed (or provided) tools.
    if tool_id is None:
        if installed_tools is None:
            from crossby.ai_tools.base import AbstractAITool

            installed_tools = AbstractAITool.detect_installed()

        writers = [w for w in writers if w.tool_id in installed_tools]

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
            result = writer.sync(writer_data, project_root, dry_run=dry_run, force=force)
        except Exception as exc:
            result = SyncResult(
                tool_id=writer.tool_id,
                concern=writer.concern,
                action="error",
                message=str(exc),
            )
        results.append(result)

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

    # After all agents writers, update .gitignore managed block once.
    # Skip when a specific tool filter is active to avoid cross-tool side effects
    # and misattributed results during --tool runs.
    if agents_writers_ran and tool_id is None:
        gi_result = update_agents_gitignore(
            data,
            project_root,
            dry_run=dry_run,
            installed_tools=installed_tools,
        )
        if gi_result is not None:
            results.append(gi_result)

    # After all rules writers, update .gitignore managed block once.
    if rules_writers_ran and tool_id is None:
        gi_result = update_rules_gitignore(
            data,
            project_root,
            dry_run=dry_run,
            installed_tools=installed_tools,
        )
        if gi_result is not None:
            results.append(gi_result)

    # After all skills writers, update .gitignore managed block once.
    if skills_writers_ran and tool_id is None:
        gi_result = update_skills_gitignore(
            data,
            project_root,
            dry_run=dry_run,
            installed_tools=installed_tools,
        )
        if gi_result is not None:
            results.append(gi_result)

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
        # Only touch .gitignore when the ledger file was actually created or
        # changed — save_ledger returns False on an idempotent no-op.
        if save_ledger(project_root, ledger):
            from crossby.sync.gitignore_utils import update_managed_block

            update_managed_block(
                project_root,
                _LEDGER_GITIGNORE_BLOCK_ID,
                [LEDGER_PATH.as_posix()],
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
