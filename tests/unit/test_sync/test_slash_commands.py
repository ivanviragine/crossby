"""Tests for tool-specific slash-command → Codex/etc skill conversion."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossby.models.ai import AIToolID
from crossby.sync.slash_commands import (
    command_skill_name,
    command_to_skill,
    detect_runtime_caveats,
    discover_commands,
    iter_command_skills,
)

_SOURCE_RELS: dict[AIToolID, str] = {
    AIToolID.CLAUDE: ".claude/commands",
    AIToolID.CURSOR: ".cursor/commands",
    AIToolID.GEMINI: ".gemini/commands",
}


def _make_command(
    project_root: Path,
    rel: str,
    content: str,
    *,
    source_tool: AIToolID = AIToolID.CLAUDE,
) -> Path:
    path = project_root / _SOURCE_RELS[source_tool] / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscoverClaude:
    def test_empty_when_no_commands_dir(self, tmp_path: Path) -> None:
        assert discover_commands(tmp_path, AIToolID.CLAUDE) == []

    def test_lists_md_files(self, tmp_path: Path) -> None:
        _make_command(tmp_path, "alpha.md", "# alpha")
        _make_command(tmp_path, "beta.md", "# beta")
        results = discover_commands(tmp_path, AIToolID.CLAUDE)
        assert [p.name for p in results] == ["alpha.md", "beta.md"]

    def test_recurses_into_subdirs(self, tmp_path: Path) -> None:
        _make_command(tmp_path, "release/cut.md", "# cut")
        results = discover_commands(tmp_path, AIToolID.CLAUDE)
        assert len(results) == 1
        assert results[0].name == "cut.md"


class TestDiscoverCursor:
    def test_lists_cursor_commands(self, tmp_path: Path) -> None:
        _make_command(tmp_path, "fmt.md", "# fmt", source_tool=AIToolID.CURSOR)
        results = discover_commands(tmp_path, AIToolID.CURSOR)
        assert [p.name for p in results] == ["fmt.md"]


class TestDiscoverGemini:
    def test_lists_gemini_commands(self, tmp_path: Path) -> None:
        _make_command(tmp_path, "sum.md", "# sum", source_tool=AIToolID.GEMINI)
        results = discover_commands(tmp_path, AIToolID.GEMINI)
        assert [p.name for p in results] == ["sum.md"]


class TestDiscoverNoCommandPrimitive:
    def test_codex_returns_empty(self, tmp_path: Path) -> None:
        # Codex has no command primitive in _COMMAND_SOURCES.
        assert discover_commands(tmp_path, AIToolID.CODEX) == []


# ---------------------------------------------------------------------------
# Skill naming
# ---------------------------------------------------------------------------


class TestSkillName:
    def test_top_level_claude_command(self, tmp_path: Path) -> None:
        cmd = _make_command(tmp_path, "review.md", "")
        root = tmp_path / ".claude" / "commands"
        assert (
            command_skill_name(cmd, root=root, source_tool=AIToolID.CLAUDE)
            == "claude-command-review"
        )

    def test_nested_claude_command(self, tmp_path: Path) -> None:
        cmd = _make_command(tmp_path, "release/cut.md", "")
        root = tmp_path / ".claude" / "commands"
        assert (
            command_skill_name(cmd, root=root, source_tool=AIToolID.CLAUDE)
            == "claude-command-release-cut"
        )

    def test_cursor_namespace(self, tmp_path: Path) -> None:
        cmd = _make_command(tmp_path, "fmt.md", "", source_tool=AIToolID.CURSOR)
        root = tmp_path / ".cursor" / "commands"
        assert (
            command_skill_name(cmd, root=root, source_tool=AIToolID.CURSOR) == "cursor-command-fmt"
        )

    def test_gemini_namespace(self, tmp_path: Path) -> None:
        cmd = _make_command(tmp_path, "sum.md", "", source_tool=AIToolID.GEMINI)
        root = tmp_path / ".gemini" / "commands"
        assert (
            command_skill_name(cmd, root=root, source_tool=AIToolID.GEMINI) == "gemini-command-sum"
        )


# ---------------------------------------------------------------------------
# Runtime caveats — per source tool
# ---------------------------------------------------------------------------


class TestClaudeRuntimeCaveats:
    def test_detects_arguments(self) -> None:
        notes = detect_runtime_caveats("Run with $ARGUMENTS", source_tool=AIToolID.CLAUDE)
        assert any(n.category == "argument-placeholders" for n in notes)

    def test_detects_positional(self) -> None:
        notes = detect_runtime_caveats("Use $1 and $2", source_tool=AIToolID.CLAUDE)
        assert any(n.category == "argument-placeholders" for n in notes)

    def test_detects_shell_interpolation(self) -> None:
        notes = detect_runtime_caveats("Run !`git status` first.", source_tool=AIToolID.CLAUDE)
        assert any(n.category == "shell-interpolation" for n in notes)

    def test_detects_file_reference(self) -> None:
        notes = detect_runtime_caveats("See @docs/spec.md", source_tool=AIToolID.CLAUDE)
        assert any(n.category == "file-references" for n in notes)

    def test_detects_template_variables(self) -> None:
        notes = detect_runtime_caveats("Hello {{name}}.", source_tool=AIToolID.CLAUDE)
        assert any(n.category == "template-variables" for n in notes)

    def test_clean_body_no_notes(self) -> None:
        assert detect_runtime_caveats("Plain prose.", source_tool=AIToolID.CLAUDE) == []


class TestCursorRuntimeCaveats:
    def test_no_patterns_means_clean(self) -> None:
        # Cursor has an empty pattern list; even $ARGUMENTS-shaped text emits
        # no caveats because Cursor doesn't expand it.
        assert detect_runtime_caveats("Run with $ARGUMENTS", source_tool=AIToolID.CURSOR) == []


class TestGeminiRuntimeCaveats:
    def test_detects_args_template(self) -> None:
        notes = detect_runtime_caveats("Summarize {{args}}", source_tool=AIToolID.GEMINI)
        assert any(n.category == "gemini-args-template" for n in notes)

    def test_clean_body_no_notes(self) -> None:
        assert detect_runtime_caveats("Plain prose.", source_tool=AIToolID.GEMINI) == []


# ---------------------------------------------------------------------------
# Wrapping
# ---------------------------------------------------------------------------


class TestCommandToSkill:
    def test_basic_wrap(self, tmp_path: Path) -> None:
        cmd = _make_command(tmp_path, "review.md", "Review the code changes thoroughly.")
        root = tmp_path / ".claude" / "commands"
        skill = command_to_skill(cmd, root=root, source_tool=AIToolID.CLAUDE)
        assert skill.name == "claude-command-review"
        assert "Review the code changes" in skill.body
        assert "Command Template" in skill.body
        assert any(n.category == "slash-command" for n in skill.manual_fix_notes)

    def test_uses_frontmatter_description(self, tmp_path: Path) -> None:
        cmd = _make_command(
            tmp_path,
            "review.md",
            "---\ndescription: Code review helper.\n---\nBody.",
        )
        root = tmp_path / ".claude" / "commands"
        skill = command_to_skill(cmd, root=root, source_tool=AIToolID.CLAUDE)
        assert skill.description == "Code review helper."

    def test_default_description_when_missing(self, tmp_path: Path) -> None:
        cmd = _make_command(tmp_path, "release/cut.md", "Body.")
        root = tmp_path / ".claude" / "commands"
        skill = command_to_skill(cmd, root=root, source_tool=AIToolID.CLAUDE)
        assert "release/cut" in skill.description

    def test_runtime_caveats_attached(self, tmp_path: Path) -> None:
        cmd = _make_command(
            tmp_path,
            "review.md",
            "Run with $ARGUMENTS and inspect !`git diff`.",
        )
        root = tmp_path / ".claude" / "commands"
        skill = command_to_skill(cmd, root=root, source_tool=AIToolID.CLAUDE)
        cats = {n.category for n in skill.manual_fix_notes}
        assert "argument-placeholders" in cats
        assert "shell-interpolation" in cats


class TestCommandToSkillCursor:
    def test_wraps_cursor_command(self, tmp_path: Path) -> None:
        cmd = _make_command(tmp_path, "fmt.md", "Format the file.", source_tool=AIToolID.CURSOR)
        root = tmp_path / ".cursor" / "commands"
        skill = command_to_skill(cmd, root=root, source_tool=AIToolID.CURSOR)
        assert skill.name == "cursor-command-fmt"
        assert "Format the file" in skill.body
        assert any(n.category == "slash-command" for n in skill.manual_fix_notes)


class TestCommandToSkillGemini:
    def test_wraps_gemini_with_args(self, tmp_path: Path) -> None:
        cmd = _make_command(
            tmp_path,
            "sum.md",
            "Summarize {{args}} succinctly.",
            source_tool=AIToolID.GEMINI,
        )
        root = tmp_path / ".gemini" / "commands"
        skill = command_to_skill(cmd, root=root, source_tool=AIToolID.GEMINI)
        cats = {n.category for n in skill.manual_fix_notes}
        assert skill.name == "gemini-command-sum"
        assert "slash-command" in cats
        assert "gemini-args-template" in cats


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


class TestIterCommandSkills:
    def test_yields_claude_commands(self, tmp_path: Path) -> None:
        _make_command(tmp_path, "a.md", "A")
        _make_command(tmp_path, "release/cut.md", "B")
        triples = list(iter_command_skills(tmp_path))
        names = [definition.name for _, _, definition in triples]
        tools = {tool for _, tool, _ in triples}
        assert names == ["claude-command-a", "claude-command-release-cut"]
        assert tools == {AIToolID.CLAUDE}

    def test_yields_across_source_tools(self, tmp_path: Path) -> None:
        _make_command(tmp_path, "review.md", "C", source_tool=AIToolID.CLAUDE)
        _make_command(tmp_path, "fmt.md", "Cu", source_tool=AIToolID.CURSOR)
        _make_command(tmp_path, "sum.md", "G", source_tool=AIToolID.GEMINI)
        triples = list(iter_command_skills(tmp_path))
        tools = {tool for _, tool, _ in triples}
        names = sorted(definition.name for _, _, definition in triples)
        assert tools == {AIToolID.CLAUDE, AIToolID.CURSOR, AIToolID.GEMINI}
        assert names == [
            "claude-command-review",
            "cursor-command-fmt",
            "gemini-command-sum",
        ]

    def test_empty_when_no_dir(self, tmp_path: Path) -> None:
        assert list(iter_command_skills(tmp_path)) == []

    def test_filter_by_source_tools(self, tmp_path: Path) -> None:
        _make_command(tmp_path, "a.md", "A", source_tool=AIToolID.CLAUDE)
        _make_command(tmp_path, "b.md", "B", source_tool=AIToolID.CURSOR)
        triples = list(iter_command_skills(tmp_path, source_tools=[AIToolID.CURSOR]))
        assert [tool for _, tool, _ in triples] == [AIToolID.CURSOR]


# ---------------------------------------------------------------------------
# Cross-namespace collision regression
# ---------------------------------------------------------------------------


class TestNamespaceCollisions:
    @pytest.mark.parametrize(
        ("tool_a", "tool_b"),
        [
            (AIToolID.CLAUDE, AIToolID.CURSOR),
            (AIToolID.CURSOR, AIToolID.GEMINI),
        ],
    )
    def test_same_command_name_different_tools_distinct_skills(
        self, tmp_path: Path, tool_a: AIToolID, tool_b: AIToolID
    ) -> None:
        _make_command(tmp_path, "pr-review.md", "A", source_tool=tool_a)
        _make_command(tmp_path, "pr-review.md", "B", source_tool=tool_b)
        triples = list(iter_command_skills(tmp_path))
        names = sorted(definition.name for _, _, definition in triples)
        assert len(names) == len(set(names)), "skill names must be unique across sources"
