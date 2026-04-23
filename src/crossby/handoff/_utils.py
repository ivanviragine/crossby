"""Shared helpers for handoff readers.

Kept private — consumers should import from the reader modules.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def safe_resolve(path: Path) -> Path:
    """Resolve ``path`` without raising on unresolvable symlinks or missing dirs."""
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path


def mtime_as_utc(path: Path) -> datetime:
    """Return the file's mtime as a timezone-aware UTC datetime.

    Readers use this as a fallback when a session file has no embedded
    timestamp. All `SessionRef.started_at` values across every reader must
    be timezone-aware so :func:`pick_latest_session` can sort them together
    without raising ``TypeError``.
    """
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
