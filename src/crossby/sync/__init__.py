"""Sync framework — global registry and run_sync() orchestrator."""

from __future__ import annotations

from pathlib import Path

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

    results: list[SyncResult] = []
    agents_writers_ran = False
    rules_writers_ran = False
    skills_writers_ran = False
    for writer in writers:
        try:
            result = writer.sync(data, project_root, dry_run=dry_run, force=force)
        except Exception as exc:
            result = SyncResult(
                tool_id=writer.tool_id,
                concern=writer.concern,
                action="error",
                message=str(exc),
            )
        results.append(result)
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
        from crossby.sync.mcp_discovery import report_oauth_configs

        results.extend(report_oauth_configs(project_root))

    return results


__all__ = [
    "SyncConcern",
    "SyncData",
    "SyncRegistry",
    "SyncResult",
    "_registry",
    "run_sync",
]
