"""Instruction file mappings for AI tools."""

from __future__ import annotations

from pathlib import Path

from crossby.models.ai import AIToolID

# Primary instruction file per tool (relative to project root).
INSTRUCTIONS_FILE: dict[AIToolID, str] = {
    AIToolID.CLAUDE: "CLAUDE.md",
    AIToolID.CURSOR: ".cursorrules",
    AIToolID.COPILOT: ".github/copilot-instructions.md",
    AIToolID.GEMINI: "GEMINI.md",
    AIToolID.CODEX: "AGENTS.md",
}

UNSUPPORTED_TOOLS = {AIToolID.VSCODE, AIToolID.OPENCODE, AIToolID.ANTIGRAVITY}


def get_instructions_source(tool_id: AIToolID, root: Path) -> Path | None:
    """Return the instruction file for *tool_id*, or None if it does not exist."""
    rel = INSTRUCTIONS_FILE.get(tool_id)
    if rel is None:
        return None
    path = root / rel
    if path.is_file():
        return path
    if path.is_symlink():
        # Only accept symlinks that resolve to an existing file.
        try:
            target = path.resolve(strict=True)
        except FileNotFoundError:
            return None
        if target.is_file():
            return path
    return None


def get_instructions_target(tool_id: AIToolID, root: Path) -> Path | None:
    """Return where the instruction symlink should be placed for *tool_id*."""
    if tool_id in UNSUPPORTED_TOOLS:
        return None
    rel = INSTRUCTIONS_FILE.get(tool_id)
    if rel is None:
        return None
    return root / rel


def is_instructions_supported(tool_id: AIToolID) -> bool:
    """Return True if *tool_id* supports instruction files."""
    return tool_id not in UNSUPPORTED_TOOLS and tool_id in INSTRUCTIONS_FILE
