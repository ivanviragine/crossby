"""Claude Code .claude/settings.json allowlist management.

Configures the Claude Code permission allowlist to include project commands
and scripts, so agents can run them without manual approval.

Backward-compatible shim — allowlist logic lives in
``crossby.sync.permissions.ClaudePermissionWriter``.  This module preserves
the original public API for existing callers (e.g. wade).
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from crossby.models.config import HookEntry
from crossby.sync.base import SyncData
from crossby.sync.hooks import ClaudeHooksWriter
from crossby.sync.permissions import (
    ClaudePermissionWriter,
    canonical_to_claude,
)

# Re-export for callers that import canonical_to_claude from here.
__all__ = [
    "canonical_to_claude",
    "configure_allowlist",
    "configure_plan_hooks",
    "configure_worktree_hooks",
    "is_allowlist_configured",
    "read_allowlist",
]


def read_allowlist(project_root: Path) -> list[str]:
    """Read Claude allowlist and return canonical command patterns.

    Only extracts ``Bash(…)`` entries — other permission patterns
    (``Read``, ``Edit``, …) are tool-specific and not portable.
    Returns ``[]`` if the file is missing or malformed.
    """
    settings_path = project_root / ".claude" / "settings.json"
    if not settings_path.is_file():
        return []
    with contextlib.suppress(json.JSONDecodeError, OSError):
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            allow = raw.get("permissions", {}).get("allow", [])
            if isinstance(allow, list):
                return [
                    p[5:-1]
                    for p in allow
                    if isinstance(p, str) and p.startswith("Bash(") and p.endswith(")")
                ]
    return []


def is_allowlist_configured(project_root: Path, patterns: list[str]) -> bool:
    """Return True if ALL given patterns are present in the allowlist at project_root.

    Args:
        project_root: Project directory containing ``.claude/``.
        patterns: Canonical command patterns to check for.
    """
    return ClaudePermissionWriter.check(project_root, patterns)


def configure_allowlist(
    project_root: Path,
    patterns: list[str],
) -> None:
    """Add command patterns to .claude/settings.json permissions allowlist.

    Args:
        project_root: Project directory containing ``.claude/``.
        patterns: Canonical command patterns to ensure are present
            (e.g. ``["myapp:*", "./scripts/check.sh:*"]``).
            Translated to Claude syntax and merged into the allowlist.

    Idempotent — each pattern is added at most once.  Non-destructive
    merge with existing settings.
    """
    ClaudePermissionWriter.write(project_root, patterns)


def configure_plan_hooks(worktree_path: Path, guard_path: Path) -> None:
    """Install a plan-mode write-guard hook into .claude/settings.json.

    Registers ``guard_path`` as a ``PreToolUse`` hook scoped to Edit and Write
    tools. Idempotent — calling twice does not duplicate the entry. Preserves
    any existing hooks already in the file.

    If ``.claude/settings.json`` contains invalid JSON, the underlying writer
    emits a ``warnings.warn()`` and returns without writing — no exception is raised.

    Args:
        worktree_path: Root of the worktree (directory that contains ``.claude/``).
        guard_path: Path to the guard script to run before file writes.
    """
    hook = HookEntry(event="pre_tool_use", tools=["Edit", "Write"], command=str(guard_path))
    ClaudeHooksWriter().sync(SyncData(hooks=[hook]), worktree_path)


def configure_worktree_hooks(worktree_path: Path, guard_path: Path) -> None:
    """Install a worktree-isolation write-guard hook into .claude/settings.json.

    Registers ``guard_path`` as a ``PreToolUse`` hook scoped to Edit and Write
    tools. Idempotent — calling twice does not duplicate the entry. Preserves
    any existing hooks already in the file.

    If ``.claude/settings.json`` contains invalid JSON, the underlying writer
    emits a ``warnings.warn()`` and returns without writing — no exception is raised.

    Args:
        worktree_path: Root of the worktree (directory that contains ``.claude/``).
        guard_path: Path to the guard script to run before file writes.
    """
    hook = HookEntry(event="pre_tool_use", tools=["Edit", "Write"], command=str(guard_path))
    ClaudeHooksWriter().sync(SyncData(hooks=[hook]), worktree_path)
