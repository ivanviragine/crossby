"""Copilot .github/hooks/hooks.json hook management."""

from __future__ import annotations

from pathlib import Path

from crossby.models.config import HookEntry
from crossby.sync.base import SyncData
from crossby.sync.hooks import CopilotHooksWriter

__all__ = ["configure_plan_hooks", "configure_worktree_hooks"]


def configure_plan_hooks(worktree_path: Path, guard_path: Path) -> None:
    """Install a plan-mode write-guard hook into .github/hooks/hooks.json.

    Registers ``guard_path`` as a ``preToolUse`` hook. Idempotent — calling
    twice does not duplicate the entry. Preserves any existing hooks already
    in the file.

    **Unscoped**: the guard fires on *all* tool calls, not just file-write tools
    (Edit, Write). Copilot does support a `matcher` regex on its tool events;
    crossby simply doesn't wire it up yet (tracked in #88 §6), so this is a
    crossby gap rather than a Copilot limitation. Firing too often is safe here —
    the guard decides per call from its own payload.

    If ``.github/hooks/hooks.json`` contains invalid JSON, the underlying writer
    emits a ``warnings.warn()`` and returns without writing — no exception is raised.

    Args:
        worktree_path: Root of the worktree (directory that contains ``.github/``).
        guard_path: Path to the guard script to run before tool calls.
    """
    # tools is intentionally left empty: crossby writes Copilot hooks unscoped.
    hook = HookEntry(event="pre_tool_use", tools=[], command=str(guard_path))
    CopilotHooksWriter().sync(SyncData(hooks=[hook]), worktree_path)


def configure_worktree_hooks(worktree_path: Path, guard_path: Path) -> None:
    """Install a worktree-isolation write-guard hook into .github/hooks/hooks.json.

    Registers ``guard_path`` as a ``preToolUse`` hook. Idempotent — calling
    twice does not duplicate the entry. Preserves any existing hooks already
    in the file.

    **Unscoped**: the guard fires on *all* tool calls, not just file-write tools
    (Edit, Write). Copilot does support a `matcher` regex on its tool events;
    crossby simply doesn't wire it up yet (tracked in #88 §6), so this is a
    crossby gap rather than a Copilot limitation. Firing too often is safe here —
    the guard decides per call from its own payload.

    If ``.github/hooks/hooks.json`` contains invalid JSON, the underlying writer
    emits a ``warnings.warn()`` and returns without writing — no exception is raised.

    Args:
        worktree_path: Root of the worktree (directory that contains ``.github/``).
        guard_path: Path to the guard script to run before tool calls.
    """
    # tools is intentionally left empty: crossby writes Copilot hooks unscoped.
    hook = HookEntry(event="pre_tool_use", tools=[], command=str(guard_path))
    CopilotHooksWriter().sync(SyncData(hooks=[hook]), worktree_path)
