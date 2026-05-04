"""Tests for Claude slash-command -> Codex/etc skill conversion."""

from __future__ import annotations

from pathlib import Path

from crossby.sync.slash_commands import (
    SKILL_NAME_PREFIX,
    command_skill_name,
    command_to_skill,
    detect_runtime_caveats,
    discover_claude_commands,
    iter_command_skills,
)


def _make_command(project_root: Path, rel: str, content: str) -> Path:
    path = project_root / ".claude" / "commands" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestDiscover:
    def test_empty_when_no_commands_dir(self, tmp_path: Path) -> None:
        assert discover_claude_commands(tmp_path) == []

    def test_lists_md_files(self, tmp_path: Path) -> None:
        _make_command(tmp_path, "alpha.md", "# alpha")
        _make_command(tmp_path, "beta.md", "# beta")
        results = discover_claude_commands(tmp_path)
        assert [p.name for p in results] == ["alpha.md", "beta.md"]

    def test_recurses_into_subdirs(self, tmp_path: Path) -> None:
        _make_command(tmp_path, "release/cut.md", "# cut")
        results = discover_claude_commands(tmp_path)
        assert len(results) == 1
        assert results[0].name == "cut.md"


class TestSkillName:
    def test_top_level_command(self, tmp_path: Path) -> None:
        cmd = _make_command(tmp_path, "review.md", "")
        root = tmp_path / ".claude" / "commands"
        assert command_skill_name(cmd, root=root) == f"{SKILL_NAME_PREFIX}review"

    def test_nested_command(self, tmp_path: Path) -> None:
        cmd = _make_command(tmp_path, "release/cut.md", "")
        root = tmp_path / ".claude" / "commands"
        assert command_skill_name(cmd, root=root) == f"{SKILL_NAME_PREFIX}release-cut"


class TestRuntimeCaveats:
    def test_detects_arguments(self) -> None:
        notes = detect_runtime_caveats("Run with $ARGUMENTS")
        assert any("argument-placeholders" == n.category for n in notes)

    def test_detects_positional(self) -> None:
        notes = detect_runtime_caveats("Use $1 and $2")
        assert any("argument-placeholders" == n.category for n in notes)

    def test_detects_shell_interpolation(self) -> None:
        notes = detect_runtime_caveats("Run !`git status` first.")
        assert any("shell-interpolation" == n.category for n in notes)

    def test_detects_file_reference(self) -> None:
        notes = detect_runtime_caveats("See @docs/spec.md for details.")
        assert any("file-references" == n.category for n in notes)

    def test_detects_template_variables(self) -> None:
        notes = detect_runtime_caveats("Hello {{name}}.")
        assert any("template-variables" == n.category for n in notes)

    def test_clean_body_no_notes(self) -> None:
        assert detect_runtime_caveats("Plain prose without any specials.") == []


class TestCommandToSkill:
    def test_basic_wrap(self, tmp_path: Path) -> None:
        cmd = _make_command(
            tmp_path,
            "review.md",
            "Review the code changes thoroughly.",
        )
        root = tmp_path / ".claude" / "commands"
        skill = command_to_skill(cmd, root=root)
        assert skill.name == f"{SKILL_NAME_PREFIX}review"
        assert "Review the code changes" in skill.body
        assert "Command Template" in skill.body
        # Always emits the slash-command caveat.
        assert any(n.category == "slash-command" for n in skill.manual_fix_notes)

    def test_uses_frontmatter_description(self, tmp_path: Path) -> None:
        cmd = _make_command(
            tmp_path,
            "review.md",
            "---\ndescription: Code review helper.\n---\nBody.",
        )
        root = tmp_path / ".claude" / "commands"
        skill = command_to_skill(cmd, root=root)
        assert skill.description == "Code review helper."

    def test_default_description_when_missing(self, tmp_path: Path) -> None:
        cmd = _make_command(tmp_path, "release/cut.md", "Body.")
        root = tmp_path / ".claude" / "commands"
        skill = command_to_skill(cmd, root=root)
        assert "release/cut" in skill.description

    def test_runtime_caveats_attached(self, tmp_path: Path) -> None:
        cmd = _make_command(
            tmp_path,
            "review.md",
            "Run with $ARGUMENTS and inspect !`git diff`.",
        )
        root = tmp_path / ".claude" / "commands"
        skill = command_to_skill(cmd, root=root)
        cats = {n.category for n in skill.manual_fix_notes}
        assert "argument-placeholders" in cats
        assert "shell-interpolation" in cats


class TestIterCommandSkills:
    def test_yields_all(self, tmp_path: Path) -> None:
        _make_command(tmp_path, "a.md", "A")
        _make_command(tmp_path, "release/cut.md", "B")
        pairs = list(iter_command_skills(tmp_path))
        names = [definition.name for _, definition in pairs]
        assert names == [f"{SKILL_NAME_PREFIX}a", f"{SKILL_NAME_PREFIX}release-cut"]

    def test_empty_when_no_dir(self, tmp_path: Path) -> None:
        assert list(iter_command_skills(tmp_path)) == []
