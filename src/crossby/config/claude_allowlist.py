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

import structlog

from crossby.sync.permissions import (
    ClaudePermissionWriter,
    canonical_to_claude,
)

logger = structlog.get_logger()

# Re-export for callers that import canonical_to_claude from here.
__all__ = [
    "canonical_to_claude",
    "configure_allowlist",
    "configure_plan_hooks",
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


def configure_plan_hooks(working_dir: Path, guard_script: Path) -> None:
    """Add PreToolUse hooks to .claude/settings.json for plan-session guard.

    Merges a ``hooks.PreToolUse`` entry into the existing settings.
    Idempotent — re-running with the same guard_script path is a no-op.
    """
    settings_path = working_dir / ".claude" / "settings.json"

    existing: dict[str, object] = {}
    if settings_path.is_file():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            raw = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw

    hooks = existing.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = {}
        existing["hooks"] = hooks

    pre_list: list[object] = hooks.setdefault("PreToolUse", [])
    if not isinstance(pre_list, list):
        pre_list = []
        hooks["PreToolUse"] = pre_list

    guard_entry: dict[str, object] = {
        "matcher": "Edit|Write|NotebookEdit",
        "hooks": [{"type": "command", "command": f"python3 {guard_script}"}],
    }

    # Check if already present (by hook command)
    guard_cmd = f"python3 {guard_script}"
    for entry in pre_list:
        if isinstance(entry, dict):
            entry_hooks = entry.get("hooks", [])
            if isinstance(entry_hooks, list):
                for hook in entry_hooks:
                    # Check for object format: {"type": "command", "command": "..."}
                    if isinstance(hook, dict) and hook.get("command") == guard_cmd:
                        return  # Already configured
                    # Legacy fallback: plain string format
                    if isinstance(hook, str) and hook == guard_cmd:
                        return  # Already configured

    pre_list.append(guard_entry)

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(existing, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("claude_plan_hooks.configured", path=str(settings_path))
