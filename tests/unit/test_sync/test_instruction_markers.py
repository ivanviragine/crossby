"""Tests for tool-marker detection in instruction content."""

from __future__ import annotations

from crossby.models.ai import AIToolID
from crossby.sync.instruction_markers import (
    detect_tool_markers,
    foreign_markers,
    is_neutral_for_target,
    manual_fix_notes_for_target,
)


class TestDetectToolMarkers:
    def test_neutral_content(self) -> None:
        assert detect_tool_markers("Just a plain instruction file.") == {}

    def test_claude_subagent(self) -> None:
        found = detect_tool_markers("See `.claude/agents/release.md` for context.")
        assert AIToolID.CLAUDE in found
        assert "Claude subagent paths" in found[AIToolID.CLAUDE]

    def test_claude_exit_plan_mode(self) -> None:
        found = detect_tool_markers("Use ExitPlanMode when ready.")
        assert AIToolID.CLAUDE in found

    def test_claude_permission_mode(self) -> None:
        found = detect_tool_markers("Set permissionMode to acceptEdits.")
        assert AIToolID.CLAUDE in found

    def test_codex_sandbox(self) -> None:
        found = detect_tool_markers("Configure sandbox_mode in .codex/config.toml.")
        assert AIToolID.CODEX in found

    def test_cursor(self) -> None:
        found = detect_tool_markers("Edit .cursorrules to allowlist commands.")
        assert AIToolID.CURSOR in found

    def test_copilot(self) -> None:
        found = detect_tool_markers(
            "See .github/copilot-instructions.md for the contract."
        )
        assert AIToolID.COPILOT in found

    def test_gemini(self) -> None:
        found = detect_tool_markers("Run with --approval-mode plan in .gemini/.")
        assert AIToolID.GEMINI in found

    def test_multi_tool_content(self) -> None:
        found = detect_tool_markers(
            "Use ExitPlanMode in Claude or sandbox_mode in .codex/."
        )
        assert AIToolID.CLAUDE in found
        assert AIToolID.CODEX in found

    def test_case_insensitive(self) -> None:
        # Match should not depend on original casing.
        found = detect_tool_markers("EXITPLANMODE")
        assert AIToolID.CLAUDE in found


class TestIsNeutralForTarget:
    def test_pure_neutral(self) -> None:
        assert is_neutral_for_target("Plain content.", AIToolID.CLAUDE) is True

    def test_target_owns_markers(self) -> None:
        # ExitPlanMode is Claude-specific; for a Claude target, that's fine.
        assert is_neutral_for_target("Use ExitPlanMode.", AIToolID.CLAUDE) is True

    def test_target_does_not_own_markers(self) -> None:
        assert is_neutral_for_target("Use ExitPlanMode.", AIToolID.GEMINI) is False

    def test_mixed_with_target_marker_present(self) -> None:
        # Content names both Claude *and* Codex; for a Claude target only the
        # Codex markers are foreign, so it's not neutral.
        text = "Use ExitPlanMode and configure sandbox_mode."
        assert is_neutral_for_target(text, AIToolID.CLAUDE) is False


class TestForeignMarkers:
    def test_returns_only_other_tools(self) -> None:
        text = "Use ExitPlanMode and configure sandbox_mode."
        foreign = foreign_markers(text, AIToolID.CLAUDE)
        assert AIToolID.CODEX in foreign
        assert AIToolID.CLAUDE not in foreign

    def test_empty_when_neutral(self) -> None:
        assert foreign_markers("Plain.", AIToolID.CLAUDE) == {}


class TestManualFixNotes:
    def test_one_note_per_foreign_tool(self) -> None:
        text = "Use ExitPlanMode and configure sandbox_mode."
        notes = manual_fix_notes_for_target(text, AIToolID.GEMINI)
        # Both Claude and Codex are foreign to a Gemini target.
        categories = {note.category for note in notes}
        assert {str(AIToolID.CLAUDE), str(AIToolID.CODEX)} <= categories

    def test_no_notes_when_neutral(self) -> None:
        assert manual_fix_notes_for_target("Plain text.", AIToolID.CLAUDE) == []

    def test_note_text_mentions_target(self) -> None:
        text = "Use ExitPlanMode."
        notes = manual_fix_notes_for_target(text, AIToolID.CURSOR)
        assert len(notes) == 1
        assert "cursor" in notes[0].message
