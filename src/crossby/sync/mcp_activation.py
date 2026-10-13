"""Read audited native MCP activation metadata without resolving credentials.

See docs/mcp-activation-discovery.md for native evidence and scope limitations.
Unknown activation is distinct from an explicit disable: it cannot authorize
either provisioning or revocation.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_GLOBAL_CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
_CURSOR_DATA_PATH = Path.home() / ".cursor"


def claude_config_paths(default_global: Path) -> tuple[Path, Path]:
    """Match Claude 2.1.296's config-dir override and legacy config preference."""
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    settings = (
        Path(override) / "settings.json" if override is not None else _GLOBAL_CLAUDE_SETTINGS_PATH
    )
    global_config = Path(override) / ".claude.json" if override else default_global
    legacy = settings.parent / ".config.json"
    if legacy.exists() or legacy.is_symlink():
        global_config = legacy
    return global_config, settings


def native_project_root(project_root: Path) -> Path:
    """Native CLI project records use the Git root, or the non-repo directory."""
    root = project_root.resolve()
    return next(
        (ancestor for ancestor in (root, *root.parents) if (ancestor / ".git").exists()), root
    )


def read_activation_json(path: Path, errors: list[str]) -> Any:
    """Missing files mean no override; present unreadable files are errors."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw is None:
            errors.append(f"{path}: MCP activation configuration must not be null")
        return raw
    except FileNotFoundError:
        # A dangling source symlink is not an absent activation decision.
        if path.is_symlink():
            errors.append(f"{path}: unreadable MCP activation configuration")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        errors.append(f"{path}: unreadable or invalid MCP activation JSON")
    return None


def activation_names(value: object, label: str, errors: list[str]) -> set[str]:
    """Reject malformed lists wholesale; never turn partial data into authority."""
    if not isinstance(value, list) or any(not isinstance(name, str) for name in value):
        errors.append(f"{label}: expected a list of MCP server names")
        return set()
    return set(value)


def claude_disabled_names(
    project_root: Path, global_config: Path, errors: list[str]
) -> tuple[set[str], set[str]]:
    """Return (project rejection names, regular-server toggle names).

    Project approval/rejection is separate from regular-server toggles. Read
    project metadata in ~/.claude.json even without importing user definitions.
    Default-off built-ins and unaudited policy fields have no inferred semantics.
    """
    global_config, user_settings = claude_config_paths(global_config)
    rejected: set[str] = set()
    for path in (
        user_settings,
        project_root / ".claude" / "settings.json",
        project_root / ".claude" / "settings.local.json",
    ):
        raw = read_activation_json(path, errors)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            errors.append(f"{path}: MCP settings must be an object")
            continue
        if "disabledMcpjsonServers" in raw:
            rejected |= activation_names(
                raw["disabledMcpjsonServers"], f"{path}: disabledMcpjsonServers", errors
            )

    toggled: set[str] = set()
    raw = read_activation_json(global_config, errors)
    if raw is not None:
        if not isinstance(raw, dict) or not isinstance(raw.get("projects", {}), dict):
            errors.append(f"{global_config}: Claude projects must be an object")
        else:
            project = raw.get("projects", {}).get(str(native_project_root(project_root)), {})
            if not isinstance(project, dict):
                errors.append(f"{global_config}: Claude project activation must be an object")
            elif "disabledMcpServers" in project:
                toggled = activation_names(
                    project["disabledMcpServers"],
                    f"{global_config}: project disabledMcpServers",
                    errors,
                )
    return rejected, toggled


def cursor_disabled_path(project_root: Path) -> Path:
    """Match Cursor CLI's project-paths.ts and workspace-paths.js (2026.09.10)."""
    root = native_project_root(project_root)
    key = re.sub(r"[^a-zA-Z0-9]+", "-", str(root)).strip("-")
    override = os.environ.get("CURSOR_DATA_DIR", "")
    data_root = Path(override) if override.strip() else _CURSOR_DATA_PATH
    return data_root / "projects" / key / "mcp-disabled.json"


def cursor_disabled_names(project_root: Path, errors: list[str]) -> set[str]:
    path = cursor_disabled_path(project_root)
    raw = read_activation_json(path, errors)
    return activation_names(raw, str(path), errors) if raw is not None else set()
