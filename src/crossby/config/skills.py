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

# Scan order for detecting the canonical (non-symlinked) source directory.
# CLAUDE, GEMINI, CODEX are checked first — they are most commonly the real source.
# CURSOR and COPILOT are appended so all five tools can serve as a source,
# while preserving the original three-tool detect_skills_source() priority.
_SCAN_ORDER = [
    AIToolID.CLAUDE,
    AIToolID.GEMINI,
    AIToolID.CODEX,
    AIToolID.CURSOR,
    AIToolID.COPILOT,
]


def detect_skills_source(root: Path) -> Path | None:
    """Find the first real (non-symlinked) skills directory.

    Scans ``.claude/skills/``, ``.gemini/skills/``, ``.agents/skills/``
    in order.  Returns None if no real skills directory exists.
    """
    for tool_id in _SCAN_ORDER:
        rel = SKILLS_DIR[tool_id]
        candidate = root / rel
        if candidate.is_dir() and not candidate.is_symlink():
            return candidate
    return None


def count_skills(directory: Path) -> int:
    """Count skill subdirectories (each containing a SKILL.md file) in directory."""
    count = 0
    with contextlib.suppress(OSError):
        resolved = directory.resolve() if directory.is_symlink() else directory
        count = sum(1 for d in resolved.iterdir() if d.is_dir() and (d / "SKILL.md").is_file())
    return count


def get_skills_target(tool_id: AIToolID, root: Path) -> Path | None:
    """Return the skills directory path for *tool_id*."""
    rel = SKILLS_DIR.get(tool_id)
    if rel is None:
        return None
    return root / rel
