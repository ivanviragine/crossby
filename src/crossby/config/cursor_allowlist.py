"""Cursor CLI permission allowlist management.

Configures the Cursor CLI permission allowlist to include project commands
and scripts, so agents can run them without manual approval.

Cursor supports two config locations:

- **Per-project**: ``<project>/.cursor/cli.json`` (preferred)
- **Global**: ``~/.cursor/cli-config.json`` (fallback / ``crossby init``)

When a ``project_root`` is provided, the per-project config is used.
When ``project_root`` is ``None``, the global config is used.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import structlog

logger = structlog.get_logger()

_GLOBAL_CONFIG_PATH = Path.home() / ".cursor" / "cli-config.json"


def _config_path(project_root: Path | None) -> Path:
    """Return the Cursor CLI config path for the given scope.

    Per-project: ``<project_root>/.cursor/cli.json``
    Global:      ``~/.cursor/cli-config.json``
    """
    if project_root is not None:
        return project_root / ".cursor" / "cli.json"
    return _GLOBAL_CONFIG_PATH


def read_allowlist(project_root: Path) -> list[str]:
    """Read Cursor allowlist and return canonical command patterns.

    Only extracts ``Shell(…)`` entries.
    Returns ``[]`` if the file is missing or malformed.
    """
    config_file = _config_path(project_root)
    if not config_file.is_file():
        return []
    with contextlib.suppress(json.JSONDecodeError, OSError):
        raw = json.loads(config_file.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            allow = raw.get("permissions", {}).get("allow", [])
            if isinstance(allow, list):
                return [
                    p[6:-1]
                    for p in allow
                    if isinstance(p, str) and p.startswith("Shell(") and p.endswith(")")
                ]
    return []


def canonical_to_cursor(pattern: str) -> str:
    """Convert a canonical command pattern to Cursor CLI allowlist syntax.

    Canonical patterns use ``"cmd:args"`` notation (colon-separated).
    Cursor expects ``"Shell(cmd:args)"`` — the command string wrapped in
    ``Shell(…)``.

    Examples::

        "myapp:*"                 → "Shell(myapp:*)"
        "./scripts/check.sh:*"    → "Shell(./scripts/check.sh:*)"
    """
    return f"Shell({pattern})"


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
    if not patterns:
        return True
    config_file = _config_path(project_root)
    if not config_file.is_file():
        return False
    with contextlib.suppress(json.JSONDecodeError, OSError):
        raw = json.loads(config_file.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            allow = raw.get("permissions", {}).get("allow", [])
            if not isinstance(allow, list):
                return False
            cursor_patterns = [canonical_to_cursor(p) for p in patterns]
            return all(cp in allow for cp in cursor_patterns)
    return False


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
    if not patterns:
        return

    config_file = _config_path(project_root)

    existing: dict[str, object] = {}
    if config_file.is_file():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            raw = json.loads(config_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw

    permissions = existing.setdefault("permissions", {})
    if not isinstance(permissions, dict):
        permissions = {}
        existing["permissions"] = permissions

    allow_list = permissions.setdefault("allow", [])
    if not isinstance(allow_list, list):
        allow_list = []
        permissions["allow"] = allow_list

    changed = False

    # Build the full set of Cursor-syntax patterns to ensure
    all_patterns = [canonical_to_cursor(p) for p in patterns]

    for pat in all_patterns:
        if pat not in allow_list:
            allow_list.append(pat)
            changed = True

    if not changed:
        return  # All patterns already present

    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps(existing, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("cursor_allowlist.configured", path=str(config_file))
