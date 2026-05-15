"""Tests for crossby-sync skill installation."""

from __future__ import annotations

from pathlib import Path

from crossby.models.ai import AIToolID
from crossby.sync.skill_install import (
    SKILL_NAME,
    bundled_skill_root,
    install_bundle,
    install_for_tools,
)


class TestBundledSkillRoot:
    def test_bundle_contains_skill_md(self) -> None:
        root = bundled_skill_root()
        assert (root / "SKILL.md").is_file()

    def test_bundle_contains_references(self) -> None:
        root = bundled_skill_root()
        assert (root / "references" / "differences.md").is_file()

    def test_bundle_contains_agents_openai_yaml(self) -> None:
        # Required for the Agent Skills standard layout (and so Codex /
        # similar UIs can render the skill in lists and chips).
        root = bundled_skill_root()
        assert (root / "agents" / "openai.yaml").is_file()

    def test_skill_md_has_trigger_shaped_description(self) -> None:
        # Discovery convention: the description must start with "Use when"
        # so the model knows when to invoke the skill, not just what it is.
        text = (bundled_skill_root() / "SKILL.md").read_text(encoding="utf-8")
        assert "description: Use when" in text

    def test_skill_md_carries_metadata_short_description(self) -> None:
        text = (bundled_skill_root() / "SKILL.md").read_text(encoding="utf-8")
        assert "metadata:" in text
        assert "short-description:" in text


class TestInstallBundle:
    def test_creates_when_missing(self, tmp_path: Path) -> None:
        result = install_bundle(tmp_path)
        assert result.action == "created"
        skill_dir = tmp_path / SKILL_NAME
        assert (skill_dir / "SKILL.md").is_file()
        assert (skill_dir / "references" / "differences.md").is_file()
        assert (skill_dir / "agents" / "openai.yaml").is_file()

    def test_skipped_when_byte_for_byte_match(self, tmp_path: Path) -> None:
        install_bundle(tmp_path)
        result = install_bundle(tmp_path)
        assert result.action == "skipped"

    def test_updated_when_existing_differs(self, tmp_path: Path) -> None:
        install_bundle(tmp_path)
        skill_md = tmp_path / SKILL_NAME / "SKILL.md"
        skill_md.write_text("user-edited", encoding="utf-8")
        result = install_bundle(tmp_path)
        assert result.action == "updated"

    def test_does_not_copy_python_init(self, tmp_path: Path) -> None:
        install_bundle(tmp_path)
        skill_dir = tmp_path / SKILL_NAME
        assert not (skill_dir / "__init__.py").exists()
        assert not (skill_dir / "references" / "__init__.py").exists()
        assert not (skill_dir / "agents" / "__init__.py").exists()


class TestInstallForTools:
    def test_installs_per_tool(self, tmp_path: Path) -> None:
        results = install_for_tools(tmp_path, [AIToolID.CLAUDE, AIToolID.CODEX])
        assert len(results) == 2
        assert (tmp_path / ".claude" / "skills" / SKILL_NAME / "SKILL.md").is_file()
        assert (tmp_path / ".agents" / "skills" / SKILL_NAME / "SKILL.md").is_file()

    def test_unknown_tool_skipped(self, tmp_path: Path) -> None:
        # No skills dir for ANTIGRAVITY; the helper should silently skip.
        results = install_for_tools(tmp_path, [AIToolID.ANTIGRAVITY])
        assert results == []
