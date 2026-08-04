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

from crossby.config.json_utils import atomic_write_text, read_json_file, write_json_file

SyncAction = Literal["created", "updated", "skipped", "error"]

__all__ = [
    "SyncAction",
    "atomic_write_text",
    "read_json_file",
    "read_merge_write_json",
    "write_json_file",
]


def read_merge_write_json(
    path: Path,
    key: str,
    updates: dict[str, Any],
    removals: set[str],
    dry_run: bool = False,
) -> tuple[SyncAction, str, list[str], int]:
    """Atomic read-modify-write for a JSON config file.

    Merges ``updates`` into ``file[key]`` and removes ``removals`` from it.
    All other keys in the file and in ``file[key]`` are preserved.
    Writes with 2-space indent and sorted keys.

    Args:
        path: Path to the JSON file.
        key: The top-level key to update (e.g. ``"mcpServers"``).
        updates: Mapping of server_name → server_dict to add/update.
        removals: Set of server names to remove from ``file[key]``. Callers pass
            only names crossby is permitted to delete (ledger-bounded), so a
            same-named hand-authored server is never dropped.
        dry_run: If True, compute action but do not write.

    Returns:
        Tuple of (action, message, written_names, removed_count) where
        ``written_names`` are the names crossby wrote or overwrote this call
        (used to record ownership) and action is one of ``"created"``,
        ``"updated"``, ``"skipped"``, ``"error"``.
    """
    data, error, was_new = read_json_file(path)
    if error is not None:
        msg = f"{path} {error} — skipping sync. Fix the file manually or delete it."
        warnings.warn(msg, stacklevel=2)
        return "error", msg, [], 0

    existing = data or {}

    section: dict[str, Any] = existing.get(key, {})
    if not isinstance(section, dict):
        section = {}

    written: list[str] = []
    removed = 0

    for name, entry in updates.items():
        if section.get(name) != entry:
            section[name] = entry
            written.append(name)

    for name in removals:
        if name in section:
            del section[name]
            removed += 1

    if not written and not removed:
        return "skipped", "", [], 0

    if dry_run:
        action: SyncAction = "created" if was_new else "updated"
        return action, "", written, removed

    existing[key] = section
    write_json_file(path, existing)
    return ("created" if was_new else "updated"), "", written, removed
