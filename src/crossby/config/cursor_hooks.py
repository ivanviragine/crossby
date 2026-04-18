"""Cursor .cursor/hooks.json hook management."""

from __future__ import annotations

from pathlib import Path

from crossby.models.config import HookEntry
from crossby.sync.base import SyncData
from crossby.sync.hooks import CursorHooksWriter

__all__ = ["configure_plan_hooks"]


def configure_plan_hooks(worktree_path: Path, guard_path: Path) -> None:
    """Install a plan-mode write-guard hook into .cursor/hooks.json.

    Registers ``guard_path`` as a ``preToolUse`` hook scoped to Edit and Write
    tools. Idempotent — calling twice does not duplicate the entry. Preserves
    any existing hooks already in the file.

    If ``.cursor/hooks.json`` contains invalid JSON, the underlying writer
    emits a ``warnings.warn()`` and returns without writing — no exception is raised.

    Args:
        worktree_path: Root of the worktree (directory that contains ``.cursor/``).
        guard_path: Path to the guard script to run before file writes.
    """
    hook = HookEntry(event="pre_tool_use", tools=["Edit", "Write"], command=str(guard_path))
    CursorHooksWriter().sync(SyncData(hooks=[hook]), worktree_path)
