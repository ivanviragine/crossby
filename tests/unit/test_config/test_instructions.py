"""Tests for instruction file mappings."""

from __future__ import annotations

from pathlib import Path

from crossby.config.instructions import (
    get_instructions_source,
    get_instructions_target,
    is_instructions_supported,
)
from crossby.models.ai import AIToolID


class TestGetInstructionsSource:
    def test_returns_path_when_file_exists(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("instructions")
        result = get_instructions_source(AIToolID.CLAUDE, tmp_path)
        assert result == tmp_path / "CLAUDE.md"

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert get_instructions_source(AIToolID.CLAUDE, tmp_path) is None

    def test_cursor_source(self, tmp_path: Path) -> None:
        (tmp_path / ".cursorrules").write_text("rules")
        result = get_instructions_source(AIToolID.CURSOR, tmp_path)
        assert result == tmp_path / ".cursorrules"

    def test_copilot_source(self, tmp_path: Path) -> None:
        github_dir = tmp_path / ".github"
        github_dir.mkdir()
        (github_dir / "copilot-instructions.md").write_text("instructions")
        result = get_instructions_source(AIToolID.COPILOT, tmp_path)
        assert result == github_dir / "copilot-instructions.md"

    def test_gemini_source(self, tmp_path: Path) -> None:
        (tmp_path / "GEMINI.md").write_text("instructions")
        result = get_instructions_source(AIToolID.GEMINI, tmp_path)
        assert result == tmp_path / "GEMINI.md"

    def test_codex_source(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("instructions")
        result = get_instructions_source(AIToolID.CODEX, tmp_path)
        assert result == tmp_path / "AGENTS.md"

    def test_unsupported_tool_returns_none(self, tmp_path: Path) -> None:
        assert get_instructions_source(AIToolID.VSCODE, tmp_path) is None


class TestGetInstructionsTarget:
    def test_returns_path_for_supported_tool(self, tmp_path: Path) -> None:
        assert get_instructions_target(AIToolID.CURSOR, tmp_path) == tmp_path / ".cursorrules"

    def test_returns_none_for_unsupported_tool(self, tmp_path: Path) -> None:
        assert get_instructions_target(AIToolID.VSCODE, tmp_path) is None
        assert get_instructions_target(AIToolID.OPENCODE, tmp_path) is None
        assert get_instructions_target(AIToolID.ANTIGRAVITY, tmp_path) is None


class TestIsInstructionsSupported:
    def test_supported_tools(self) -> None:
        supported = [
            AIToolID.CLAUDE,
            AIToolID.CURSOR,
            AIToolID.COPILOT,
            AIToolID.GEMINI,
            AIToolID.CODEX,
        ]
        for tool in supported:
            assert is_instructions_supported(tool) is True

    def test_unsupported_tools(self) -> None:
        for tool in [AIToolID.VSCODE, AIToolID.OPENCODE, AIToolID.ANTIGRAVITY]:
            assert is_instructions_supported(tool) is False
