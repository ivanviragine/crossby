"""Shared file utilities for sync writers."""

from __future__ import annotations

from pathlib import Path

# Marker file written into managed target directories (agents/skills) so
# subsequent syncs can distinguish a directory crossby owns from one a user
# (or another tool) populated by hand. Without this marker, a native-looking
# layout — e.g. ``.claude/agents/*.md`` or ``.claude/skills/<name>/SKILL.md``
# — is treated as user-owned and refused without ``--force``.
MANAGED_MARKER_NAME = ".crossby-managed"

_MANAGED_MARKER_BODY = (
    "# crossby managed marker\n"
    "#\n"
    "# This directory is managed by crossby (https://github.com/anthropics/crossby).\n"
    "# Edits will be overwritten on the next `crossby sync` run.\n"
    "# Delete this marker to take manual ownership of the directory.\n"
)


def backup_path(target: Path) -> Path:
    """Return the next available numbered backup path for *target*.

    Numbering scheme: ``.bak``, ``.bak2``, ``.bak3``, etc.
    Works for both files and directories.
    """
    candidate = Path(str(target) + ".bak")
    counter = 2
    while candidate.exists() or candidate.is_symlink():
        candidate = Path(str(target) + f".bak{counter}")
        counter += 1
    return candidate


def has_managed_marker(target_dir: Path) -> bool:
    """Return True if ``target_dir`` carries the crossby ownership marker."""
    return (target_dir / MANAGED_MARKER_NAME).is_file()


def write_managed_marker(target_dir: Path) -> None:
    """Idempotently write the crossby ownership marker into ``target_dir``.

    Safe to call repeatedly — the marker content is fixed, so rewriting it
    on every sync is a no-op for git. Callers should invoke this whenever a
    write-bearing sync (copy or translate) produces or refreshes content in
    ``target_dir``.
    """
    if not target_dir.is_dir():
        return
    marker = target_dir / MANAGED_MARKER_NAME
    if marker.is_file() and marker.read_text(encoding="utf-8") == _MANAGED_MARKER_BODY:
        return
    marker.write_text(_MANAGED_MARKER_BODY, encoding="utf-8")
