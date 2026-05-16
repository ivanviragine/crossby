"""Detect tool-specific markers in instruction content.

Crossby's default rules strategy is to symlink the canonical instruction
file (e.g. ``CLAUDE.md``) to every target tool's expected path. That works
when the content is provider-neutral. It silently leaks when content
references one tool's lifecycle, hook system, or permission model.

This module owns the detection. Each tool has a list of markers — file
paths, type names, mode values — that strongly imply the content is meant
for that tool. When a writer is about to symlink a source file to a target
that doesn't own the markers it found, it should fall back to copy strategy
and embed a :mod:`crossby.sync.manual_fix` block.

Only Claude has a meaningfully large surface today (hooks, subagents,
``ExitPlanMode``, etc.). Other tools have shorter marker lists; we still
keep them so direction symmetry holds.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from crossby.models.ai import AIToolID
from crossby.sync.manual_fix import ManualFixNote

# Map of tool → list of (regex pattern, human-readable description) tuples.
# Patterns are lowercased before matching for case-insensitivity.
_MARKERS: Mapping[AIToolID, tuple[tuple[str, str], ...]] = {
    AIToolID.CLAUDE: (
        (r"\.claude/agents/", "Claude subagent paths"),
        (r"\.claude/settings", "Claude settings files"),
        (r"\.claude/skills/", "Claude skills paths"),
        (r"\bexitplanmode\b", "Claude ExitPlanMode tool"),
        (r"\bpermissionmode\b", "Claude permissionMode setting"),
        (r"\bsubagent[s]?\b", "Claude subagents"),
        (r"\btodowrite\b", "Claude TodoWrite tool"),
        (r"/hooks\b", "Claude /hooks slash command"),
    ),
    AIToolID.CODEX: (
        (r"\.codex/", "Codex config paths"),
        (r"\bsandbox_mode\b", "Codex sandbox_mode"),
        (r"\bdeveloper_instructions\b", "Codex developer_instructions"),
        (r"\[mcp_servers", "Codex [mcp_servers] TOML table"),
        (r"\[features\]", "Codex [features] TOML table"),
        (r"\bmodel_reasoning_effort\b", "Codex model_reasoning_effort"),
        (r"\bcodex_hooks\b", "Codex codex_hooks feature flag"),
        (r"\.agents/skills/", "Codex .agents/skills paths"),
    ),
    AIToolID.CURSOR: (
        (r"\.cursorrules\b", "Cursor rules file"),
        (r"\.cursor/", "Cursor config paths"),
        (r"\.cursor/agents/", "Cursor agents paths"),
        (r"\.cursor/commands/", "Cursor commands paths"),
        (r"\.cursor/skills/", "Cursor skills paths"),
        (r"\.cursor/cli\.json\b", "Cursor cli.json"),
    ),
    AIToolID.COPILOT: (
        (r"\.github/copilot-instructions", "Copilot instructions file"),
        (r"\bcopilot-instructions\b", "Copilot instructions filename"),
        (r"@workspace\b", "Copilot @workspace participant"),
        (r"@github\b", "Copilot @github participant"),
        (r"\.github/agents/", "Copilot .github/agents paths"),
        (r"\.github/skills/", "Copilot .github/skills paths"),
        (r"\.github/hooks/", "Copilot .github/hooks paths"),
        (r"\.vscode/mcp\.json\b", "Copilot .vscode/mcp.json"),
    ),
    AIToolID.GEMINI: (
        (r"\.gemini/", "Gemini config paths"),
        (r"\bapproval-mode\b", "Gemini approval-mode flag"),
        (r"\.gemini/agents/", "Gemini agents paths"),
        (r"\.gemini/skills/", "Gemini skills paths"),
        (r"\.gemini/commands/", "Gemini commands paths"),
        (r"\bbeforetool\b", "Gemini BeforeTool hook event"),
        (r"\baftertool\b", "Gemini AfterTool hook event"),
    ),
}


def detect_tool_markers(content: str) -> dict[AIToolID, list[str]]:
    """Return tools whose markers appear in ``content``, with the descriptions
    of the markers that matched.

    The return is a dict so callers can ask "which tools claim this content?"
    and "what specifically did we see?" in one pass.
    """
    lowered = content.lower()
    found: dict[AIToolID, list[str]] = {}
    for tool, marker_set in _MARKERS.items():
        descriptions: list[str] = []
        for pattern, description in marker_set:
            if re.search(pattern, lowered):
                descriptions.append(description)
        if descriptions:
            found[tool] = descriptions
    return found


def is_neutral_for_target(content: str, target: AIToolID) -> bool:
    """True if ``content`` has no markers specific to a tool other than
    ``target``.

    Markers belonging to ``target`` itself are fine — they're expected. What
    we want to flag is content that names some *other* tool's surfaces.
    """
    found = detect_tool_markers(content)
    return all(tool == target for tool in found)


def foreign_markers(content: str, target: AIToolID) -> dict[AIToolID, list[str]]:
    """Return the marker sets in ``content`` that don't belong to ``target``."""
    return {
        tool: descriptions
        for tool, descriptions in detect_tool_markers(content).items()
        if tool != target
    }


def manual_fix_notes_for_target(content: str, target: AIToolID) -> list[ManualFixNote]:
    """Build manual-fix notes describing the foreign markers found.

    One note per source-tool that contributed markers; the note enumerates
    the specific surfaces and tells the user to translate them by hand
    because Crossby preserved the literal source content.
    """
    notes: list[ManualFixNote] = []
    for tool, descriptions in foreign_markers(content, target).items():
        joined = ", ".join(descriptions)
        notes.append(
            ManualFixNote(
                category=str(tool),
                message=(
                    f"Source content mentions {tool}-specific surfaces ({joined}) "
                    f"that don't apply to {target}. Translate or remove them before "
                    "relying on this file."
                ),
            )
        )
    return notes


__all__ = [
    "detect_tool_markers",
    "foreign_markers",
    "is_neutral_for_target",
    "manual_fix_notes_for_target",
]
