"""Cursor CLI permission allowlist management.

Configures the Cursor CLI permission allowlist to include project commands
and scripts, so agents can run them without manual approval.

Cursor supports two config locations:

- **Per-project**: ``<project>/.cursor/cli.json`` (preferred)
- **Global**: ``~/.cursor/cli-config.json`` (fallback / ``crossby init``)

When a ``project_root`` is provided, the per-project config is used.
When ``project_root`` is ``None``, the global config is used.

Backward-compatible shim — allowlist logic lives in
``crossby.sync.permissions.CursorPermissionWriter``.  This module preserves
the original public API for existing callers (e.g. wade).
"""

from __future__ import annotations

from pathlib import Path

import crossby.sync.permissions as _perm
from crossby.sync.permissions import (
    CursorPermissionWriter,
    canonical_to_cursor,
)

# Re-export for callers that import canonical_to_cursor from here.
__all__ = [
    "canonical_to_cursor",
    "configure_allowlist",
    "is_allowlist_configured",
]

# Module-level alias kept for backward compatibility.  Tests that monkeypatch
# the global config path should patch
# ``crossby.sync.permissions._GLOBAL_CURSOR_CONFIG_PATH`` instead.
_GLOBAL_CONFIG_PATH = _perm._GLOBAL_CURSOR_CONFIG_PATH


def _config_path(project_root: Path | None) -> Path:
    """Return the Cursor CLI config path for the given scope.

    Per-project: ``<project_root>/.cursor/cli.json``
    Global:      ``~/.cursor/cli-config.json``
    """
    return _perm._cursor_config_path(project_root)


def is_allowlist_configured(
    project_root: Path | None = None,
    patterns: list[str] | None = None,
) -> bool:
    """Return True if ALL given patterns are present in the Cursor allowlist.

    When ``project_root`` is given, checks the per-project config.
    Otherwise checks the global config.

    Args:
        project_root: Project directory, or None for global config.
        patterns: Canonical command patterns to check for.
    """
    return CursorPermissionWriter.check(project_root, patterns)


def configure_allowlist(
    project_root: Path | None = None,
    patterns: list[str] | None = None,
) -> None:
    """Add command patterns to the Cursor CLI permissions allowlist.

    Args:
        project_root: When provided, writes to the per-project config
            ``<project_root>/.cursor/cli.json``.  When ``None``, writes
            to the global ``~/.cursor/cli-config.json``.
        patterns: Canonical command patterns to ensure are present
            (e.g. ``["myapp:*", "./scripts/check.sh:*"]``).
            Translated to Cursor syntax and merged into the allowlist.

    Idempotent — each pattern is added at most once.  Non-destructive
    merge with existing config.
    """
    CursorPermissionWriter.write(project_root, patterns)
