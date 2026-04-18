"""JSON read-modify-write utilities for sync writers.

Provides atomic read-modify-write with consistent formatting (2-space indent,
sorted keys) and safe malformed-file handling.  Used by MCP and hooks sync
modules, with ``read_json_file`` and ``write_json_file`` re-exported here as a
sync-layer compatibility shim.

``read_json_file`` and ``write_json_file`` live in ``crossby.config.json_utils``
(a neutral, import-side-effect-free module) and are re-exported here for
backward compatibility with sync-layer callers.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

from crossby.config.json_utils import read_json_file
from crossby.config.json_utils import write_json_file

SyncAction = Literal["created", "updated", "skipped", "error"]

__all__ = ["SyncAction", "read_json_file", "write_json_file", "read_merge_write_json"]


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
