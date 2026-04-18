"""Copilot .github/hooks/hooks.json hook management."""

from __future__ import annotations

from pathlib import Path

from crossby.models.config import HookEntry
from crossby.sync.base import SyncData
from crossby.sync.hooks import CopilotHooksWriter

__all__ = ["configure_plan_hooks"]


def configure_plan_hooks(worktree_path: Path, guard_path: Path) -> None:
    """Install a plan-mode write-guard hook into .github/hooks/hooks.json.

    Registers ``guard_path`` as a ``preToolUse`` hook. Idempotent — calling
    twice does not duplicate the entry. Preserves any existing hooks already
    in the file.

    **Copilot limitation**: Copilot's hook format has no per-tool filter field.
    The guard will fire on *all* tool calls, not just file-write tools (Edit,
    Write). This is a known Copilot limitation and cannot be worked around
    without Copilot adding tool-filter support.

    If ``.github/hooks/hooks.json`` contains invalid JSON, the underlying writer
    emits a ``warnings.warn()`` and returns without writing — no exception is raised.

    Args:
        worktree_path: Root of the worktree (directory that contains ``.github/``).
        guard_path: Path to the guard script to run before tool calls.
    """
    # tools is intentionally left empty: Copilot has no per-tool filter.
    hook = HookEntry(event="pre_tool_use", tools=[], command=str(guard_path))
    CopilotHooksWriter().sync(SyncData(hooks=[hook]), worktree_path)
