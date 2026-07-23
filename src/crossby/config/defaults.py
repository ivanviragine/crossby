"""Hardcoded defaults per AI tool — fallback when probing fails."""

from __future__ import annotations

from crossby.models.ai import AIToolID
from crossby.models.config import ComplexityModelMapping

# Default model mappings when tool probing fails or returns no recognized models
TOOL_DEFAULTS: dict[str, ComplexityModelMapping] = {
    AIToolID.CLAUDE: ComplexityModelMapping(
        easy="claude-haiku-4.5",
        medium="claude-sonnet-5",
        complex="claude-sonnet-5",
        very_complex="claude-opus-4.8",
    ),
    AIToolID.COPILOT: ComplexityModelMapping(
        easy="claude-haiku-4.5",
        medium="claude-sonnet-4.6",
        complex="claude-sonnet-4.6",
        very_complex="claude-opus-4.6",
    ),
    AIToolID.ANTIGRAVITY_CLI: ComplexityModelMapping(
        easy="gemini-3.6-flash-low",
        medium="claude-sonnet-4-6",
        complex="claude-sonnet-4-6",
        very_complex="claude-opus-4-6-thinking",
    ),
    AIToolID.CODEX: ComplexityModelMapping(
        easy="gpt-5.4-mini",
        medium="gpt-5.4",
        complex="gpt-5.4",
        very_complex="gpt-5.4",
    ),
    AIToolID.CURSOR: ComplexityModelMapping(
        easy="gemini-3-flash",
        medium="sonnet-4.6",
        complex="sonnet-4.6",
        very_complex="opus-4.6",
    ),
    AIToolID.OPENCODE: ComplexityModelMapping(
        easy="anthropic/claude-haiku-4.5",
        medium="anthropic/claude-sonnet-4.6",
        complex="anthropic/claude-sonnet-4.6",
        very_complex="anthropic/claude-opus-4.6",
    ),
}


def get_defaults(tool: str) -> ComplexityModelMapping:
    """Get default model mapping for a tool.

    Returns empty mapping for unknown tools.
    """
    try:
        from crossby.models.ai import AIToolID

        tool_id = AIToolID(tool)
        return TOOL_DEFAULTS.get(tool_id, ComplexityModelMapping())
    except ValueError:
        return ComplexityModelMapping()
