"""Skills directory detection and mapping."""

from __future__ import annotations

import contextlib
from pathlib import Path

from crossby.models.ai import AIToolID

# Skills directory per tool (relative to project root).
SKILLS_DIR: dict[AIToolID, str] = {
    AIToolID.CLAUDE: ".claude/skills",
    AIToolID.CURSOR: ".cursor/skills",
    AIToolID.CODEX: ".agents/skills",
    AIToolID.GEMINI: ".gemini/skills",
    AIToolID.COPILOT: ".github/skills",
}

# Scan order for detecting the real (non-symlinked) source directory.
_SCAN_ORDER = [
    AIToolID.CLAUDE,
    AIToolID.GEMINI,
    AIToolID.CODEX,
]


def detect_skills_source(root: Path) -> Path | None:
    """Find the first real (non-symlinked) skills directory.

    Scans ``.claude/skills/``, ``.gemini/skills/``, ``.agents/skills/``
    in order.  Returns None if no real skills directory exists.
    """
    for tool_id in _SCAN_ORDER:
        source = get_skills_source(tool_id, root)
        if source is not None:
            return source
    return None


def get_skills_source(tool_id: AIToolID, root: Path) -> Path | None:
    """Return the real skills directory for *tool_id*.

    If the tool's skills path is a symlink, this returns its resolved target
    when that target exists and is a directory.
    """
    rel = SKILLS_DIR.get(tool_id)
    if rel is None:
        return None

    candidate = root / rel
    if candidate.is_dir() and not candidate.is_symlink():
        return candidate

    if candidate.is_symlink():
        with contextlib.suppress(OSError):
            resolved = candidate.resolve(strict=True)
            if resolved.is_dir():
                return resolved

    return None


def get_skills_target(tool_id: AIToolID, root: Path) -> Path | None:
    """Return the skills directory path for *tool_id*."""
    rel = SKILLS_DIR.get(tool_id)
    if rel is None:
        return None
    return root / rel
