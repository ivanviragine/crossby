"""JSON read-modify-write utilities for sync writers.

Provides atomic read-modify-write with consistent formatting (2-space indent,
sorted keys) and safe malformed-file handling.  Used by both MCP and
permissions sync modules.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal

SyncAction = Literal["created", "updated", "skipped", "error"]


def read_json_file(path: Path) -> tuple[dict[str, Any] | None, str | None, bool]:
    """Read a JSON file, returning (data, error_message, was_new).

    Returns:
        (dict, None, False) on success for an existing file.
        (None, error_message, False) if file is malformed.
        ({}, None, True) if file does not exist.
    """
    if not path.exists():
        return {}, None, True
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return None, f"contains invalid JSON: {e}", False
    except (OSError, UnicodeDecodeError) as e:
        return None, f"could not be read: {e}", False
    if not isinstance(raw, dict):
        return None, "root value is not a JSON object", False
    return raw, None, False


def read_merge_write_json(
    path: Path,
    key: str,
    updates: dict[str, Any],
    removals: set[str],
    dry_run: bool = False,
) -> tuple[SyncAction, str]:
    """Atomic read-modify-write for a JSON config file.

    Merges ``updates`` into ``file[key]`` and removes ``removals`` from it.
    All other keys in the file and in ``file[key]`` are preserved.
    Writes with 2-space indent and sorted keys.

    Args:
        path: Path to the JSON file.
        key: The top-level key to update (e.g. ``"mcpServers"``).
        updates: Mapping of server_name → server_dict to add/update.
        removals: Set of server names to remove from ``file[key]``.
        dry_run: If True, compute action but do not write.

    Returns:
        Tuple of (action, message) where action is one of:
        ``"created"``, ``"updated"``, ``"skipped"``, ``"error"``.
    """
    data, error, was_new = read_json_file(path)
    if error is not None:
        msg = f"{path} {error} — skipping sync. Fix the file manually or delete it."
        warnings.warn(msg, stacklevel=2)
        return "error", msg

    existing = data or {}

    section: dict[str, Any] = existing.get(key, {})
    if not isinstance(section, dict):
        section = {}

    changed = False

    for name, entry in updates.items():
        if section.get(name) != entry:
            section[name] = entry
            changed = True

    for name in removals:
        if name in section:
            del section[name]
            changed = True

    if not changed:
        return "skipped", ""

    if dry_run:
        action: str = "created" if was_new else "updated"
        return action, ""

    existing[key] = section
    write_json_file(path, existing)
    return ("created" if was_new else "updated"), ""


def write_json_file(path: Path, data: dict[str, Any]) -> None:
    """Atomic write of a JSON dict with consistent formatting.

    Uses 2-space indent, sorted keys, and a tmp+replace pattern to avoid
    partial writes on crash.
    """
    json_text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json_text, encoding="utf-8")
    tmp.replace(path)
