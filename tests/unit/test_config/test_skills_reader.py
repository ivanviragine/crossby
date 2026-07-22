"""Tests for skills directory detection."""

from __future__ import annotations

import os
from pathlib import Path

from crossby.config.skills import detect_skills_source, get_skills_target
from crossby.models.ai import AIToolID


class TestDetectSkillsSource:
    def test_finds_claude_skills(self, tmp_path: Path) -> None:
        skills = tmp_path / ".claude" / "skills"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text("skill")

        result = detect_skills_source(tmp_path)
        assert result == skills

    def test_finds_agents_skills_when_no_claude(self, tmp_path: Path) -> None:
        skills = tmp_path / ".agents" / "skills"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text("skill")

        result = detect_skills_source(tmp_path)
        assert result == skills

    def test_prefers_claude_over_agents(self, tmp_path: Path) -> None:
        claude_skills = tmp_path / ".claude" / "skills"
        claude_skills.mkdir(parents=True)
        agents_skills = tmp_path / ".agents" / "skills"
        agents_skills.mkdir(parents=True)

        result = detect_skills_source(tmp_path)
        assert result == claude_skills

    def test_returns_none_when_no_skills_dir(self, tmp_path: Path) -> None:
        assert detect_skills_source(tmp_path) is None

    def test_skips_symlinked_directories(self, tmp_path: Path) -> None:
        real_skills = tmp_path / ".agents" / "skills"
        real_skills.mkdir(parents=True)
        (real_skills / "SKILL.md").write_text("skill")

        claude_skills = tmp_path / ".claude" / "skills"
        claude_skills.parent.mkdir(parents=True)
        os.symlink(str(real_skills), str(claude_skills))

        result = detect_skills_source(tmp_path)
        assert result == real_skills  # skips symlinked .claude/skills

    def test_returns_none_when_only_symlinks(self, tmp_path: Path) -> None:
        # Create a dangling symlink
        claude_skills = tmp_path / ".claude" / "skills"
        claude_skills.parent.mkdir(parents=True)
        os.symlink("/nonexistent/skills", str(claude_skills))

        assert detect_skills_source(tmp_path) is None


class TestGetSkillsTarget:
    def test_returns_path_for_known_tool(self, tmp_path: Path) -> None:
        assert get_skills_target(AIToolID.CURSOR, tmp_path) == tmp_path / ".cursor" / "skills"
        assert get_skills_target(AIToolID.CODEX, tmp_path) == tmp_path / ".agents" / "skills"
        assert (
            get_skills_target(AIToolID.ANTIGRAVITY_CLI, tmp_path)
            == tmp_path / ".agents" / "skills"
        )

    def test_returns_none_for_unknown_tool(self, tmp_path: Path) -> None:
        assert get_skills_target(AIToolID.VSCODE, tmp_path) is None
