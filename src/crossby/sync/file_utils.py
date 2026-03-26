"""Shared file utilities for sync writers."""

from __future__ import annotations

from pathlib import Path


def backup_path(target: Path) -> Path:
    """Return the next available numbered backup path for *target*.

    Numbering scheme: ``.bak``, ``.bak2``, ``.bak3``, etc.
    Works for both files and directories.
    """
    candidate = Path(str(target) + ".bak")
    counter = 2
    while candidate.exists():
        candidate = Path(str(target) + f".bak{counter}")
        counter += 1
    return candidate
