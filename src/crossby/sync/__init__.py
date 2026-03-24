"""Sync framework — global registry and run_sync() orchestrator."""

from __future__ import annotations

from pathlib import Path

from crossby.models.ai import AIToolID
from crossby.models.config import CrossbyConfig
from crossby.sync.base import SyncConcern, SyncRegistry, SyncResult
from crossby.sync.permissions import ClaudePermissionWriter, CursorPermissionWriter

# Global default registry — one writer per (tool, concern) pair.
_registry = SyncRegistry()
_registry.register(ClaudePermissionWriter())
_registry.register(CursorPermissionWriter(scope="project"))


def run_sync(
    config: CrossbyConfig,
    project_root: Path,
    *,
    tool_id: AIToolID | None = None,
    concern: SyncConcern | None = None,
    dry_run: bool = False,
    installed_tools: list[AIToolID] | None = None,
    registry: SyncRegistry | None = None,
) -> list[SyncResult]:
    """Run all matching sync writers, collecting results.

    Continue-on-error: if one writer raises, the error is recorded in
    ``SyncResult(action="error")`` and the next writer proceeds.

    Args:
        config: Loaded CrossbyConfig.
        project_root: Project root directory.
        tool_id: When set, only writers for this tool run.  When None, all
            installed tools run (subject to ``config.sync.tools`` filter).
        concern: When set, only writers for this concern run.
        dry_run: Compute results without writing any files.
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
    for writer in writers:
        try:
            result = writer.sync(config, project_root, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            result = SyncResult(
                tool_id=writer.tool_id,
                concern=writer.concern,
                action="error",
                message=str(exc),
            )
        results.append(result)

    return results


__all__ = [
    "run_sync",
    "SyncConcern",
    "SyncRegistry",
    "SyncResult",
    "_registry",
]
