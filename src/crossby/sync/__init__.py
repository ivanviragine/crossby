"""Sync framework — global registry and run_sync() orchestrator."""

from __future__ import annotations

from pathlib import Path

from crossby.models.ai import AIToolID
from crossby.models.config import CrossbyConfig
from crossby.sync.base import SyncConcern, SyncRegistry, SyncResult
from crossby.sync.agents import (
    ClaudeAgentsWriter,
    CodexAgentsWriter,
    CopilotAgentsWriter,
    CursorAgentsWriter,
    GeminiAgentsWriter,
    update_agents_gitignore,
)
from crossby.sync.permissions import ClaudePermissionWriter, CursorPermissionWriter
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


def run_sync(
    config: CrossbyConfig,
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
        config: Loaded CrossbyConfig.
        project_root: Project root directory.
        tool_id: When set, only writers for this tool run, and the
            installed-tools filter is bypassed (useful for forcing a sync on a
            specific tool regardless of whether it is currently detected as
            installed).  When None, all installed tools run (subject to the
            ``config.sync.tools`` filter).
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

        # Apply sync.tools config filter (empty list = all installed tools).
        config_tools = config.sync.tools if config.sync.tools else None
        if config_tools:
            try:
                allowed = {AIToolID(t) for t in config_tools}
            except ValueError as exc:
                raise ValueError(
                    f"Invalid tool ID in config.sync.tools: {exc}"
                ) from exc
            installed_tools = [t for t in installed_tools if t in allowed]

        writers = [w for w in writers if w.tool_id in installed_tools]

    results: list[SyncResult] = []
    agents_writers_ran = False
    rules_writers_ran = False
    rules_synced_targets: list[str] = []
    for writer in writers:
        try:
            result = writer.sync(config, project_root, dry_run=dry_run, force=force)
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
            if result.action in ("created", "updated") and result.file_path:
                rel = str(result.file_path.relative_to(project_root))
                rules_synced_targets.append(rel)

    # After all agents writers, update .gitignore managed block once.
    # Skip when a specific tool filter is active to avoid cross-tool side effects
    # and misattributed results during --tool runs.
    if agents_writers_ran and tool_id is None:
        gi_result = update_agents_gitignore(
            config,
            project_root,
            dry_run=dry_run,
            installed_tools=installed_tools,
        )
        if gi_result is not None:
            results.append(gi_result)

    # After all rules writers, update .gitignore managed block once.
    if rules_writers_ran and tool_id is None:
        gi_result = update_rules_gitignore(
            config,
            project_root,
            dry_run=dry_run,
            synced_targets=rules_synced_targets,
        )
        if gi_result is not None:
            results.append(gi_result)

    return results


__all__ = [
    "run_sync",
    "SyncConcern",
    "SyncRegistry",
    "SyncResult",
    "_registry",
]
