"""Sync framework — global registry and run_sync() orchestrator."""

from __future__ import annotations

from pathlib import Path

from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern, SyncData, SyncRegistry, SyncResult
from crossby.sync.agents import (
    ClaudeAgentsWriter,
    CodexAgentsWriter,
    CopilotAgentsWriter,
    CursorAgentsWriter,
    GeminiAgentsWriter,
    update_agents_gitignore,
)
from crossby.sync.mcp import (
    ClaudeMCPWriter,
    CodexMCPWriter,
    CopilotMCPWriter,
    CursorMCPWriter,
    GeminiMCPWriter,
)
from crossby.sync.hooks import (
    ClaudeHooksWriter,
    CopilotHooksWriter,
    CursorHooksWriter,
    GeminiHooksWriter,
)
from crossby.sync.permissions import (
    ClaudePermissionWriter,
    CursorPermissionWriter,
    GeminiPermissionWriter,
)
from crossby.sync.rules import (
    ClaudeRulesWriter,
    CodexRulesWriter,
    CopilotRulesWriter,
    CursorRulesWriter,
    GeminiRulesWriter,
    update_rules_gitignore,
)

# Global default registry — one writer per (tool, concern) pair.
_registry = SyncRegistry()
_registry.register(ClaudePermissionWriter())
_registry.register(CursorPermissionWriter(scope="project"))
_registry.register(GeminiPermissionWriter())
_registry.register(ClaudeAgentsWriter())
_registry.register(CopilotAgentsWriter())
_registry.register(CursorAgentsWriter())
_registry.register(GeminiAgentsWriter())
_registry.register(CodexAgentsWriter())
_registry.register(ClaudeRulesWriter())
_registry.register(CursorRulesWriter())
_registry.register(CopilotRulesWriter())
_registry.register(GeminiRulesWriter())
_registry.register(CodexRulesWriter())
_registry.register(ClaudeMCPWriter())
_registry.register(CursorMCPWriter())
_registry.register(CopilotMCPWriter())
_registry.register(GeminiMCPWriter())
_registry.register(CodexMCPWriter())
_registry.register(ClaudeHooksWriter())
_registry.register(CursorHooksWriter())
_registry.register(CopilotHooksWriter())
_registry.register(GeminiHooksWriter())


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
    for writer in writers:
        try:
            result = writer.sync(data, project_root, dry_run=dry_run, force=force)
        except Exception as exc:  # noqa: BLE001
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

    return results


__all__ = [
    "run_sync",
    "SyncConcern",
    "SyncData",
    "SyncRegistry",
    "SyncResult",
    "_registry",
]
