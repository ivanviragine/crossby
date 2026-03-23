"""Tests for sync service orchestrator."""

from __future__ import annotations

import json
import os
from pathlib import Path

from crossby.models.ai import AIToolID
from crossby.models.sync import SyncStrategy
from crossby.services.sync import sync_configs


def _setup_claude_source(root: Path) -> None:
    """Create a minimal Claude source config."""
    (root / "CLAUDE.md").write_text("# Instructions")
    skills = root / ".claude" / "skills"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("# Skill")
    settings = {"permissions": {"allow": ["Bash(myapp:*)", "Bash(npm:*)"]}}
    (root / ".claude" / "settings.json").write_text(json.dumps(settings), encoding="utf-8")


class TestSyncInstructions:
    def test_creates_symlink_to_cursor(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            force=True,
            sync_skills=False,
            sync_allowlist=False,
        )
        link = tmp_path / ".cursorrules"
        assert link.is_symlink()
        assert os.readlink(link) == "CLAUDE.md"
        assert result.linked == 1

    def test_creates_symlink_to_copilot(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.COPILOT],
            tmp_path,
            force=True,
            sync_skills=False,
            sync_allowlist=False,
        )
        link = tmp_path / ".github" / "copilot-instructions.md"
        assert link.is_symlink()
        assert os.readlink(link) == "../CLAUDE.md"
        assert result.linked == 1

    def test_creates_symlinks_to_multiple_targets(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR, AIToolID.GEMINI, AIToolID.CODEX],
            tmp_path,
            force=True,
            sync_skills=False,
            sync_allowlist=False,
        )
        assert (tmp_path / ".cursorrules").is_symlink()
        assert (tmp_path / "GEMINI.md").is_symlink()
        assert (tmp_path / "AGENTS.md").is_symlink()
        assert result.linked == 3

    def test_idempotent_second_run_noop(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            force=True,
            sync_skills=False,
            sync_allowlist=False,
        )
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            force=True,
            sync_skills=False,
            sync_allowlist=False,
        )
        assert result.linked == 0  # already linked

    def test_missing_source_skips_gracefully(self, tmp_path: Path) -> None:
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            sync_skills=False,
            sync_allowlist=False,
        )
        assert not (tmp_path / ".cursorrules").exists()
        assert any("not found" in w.lower() for w in result.warnings)

    def test_skips_same_tool(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CLAUDE],
            tmp_path,
            sync_skills=False,
            sync_allowlist=False,
        )
        assert result.linked == 0


class TestSyncSkills:
    def test_creates_skills_symlink(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            force=True,
            sync_instructions=False,
            sync_allowlist=False,
        )
        link = tmp_path / ".cursor" / "skills"
        assert link.is_symlink()
        assert result.linked == 1

    def test_skills_idempotent(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            force=True,
            sync_instructions=False,
            sync_allowlist=False,
        )
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            force=True,
            sync_instructions=False,
            sync_allowlist=False,
        )
        assert result.linked == 0

    def test_no_skills_dir_skips(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("# Instructions")
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            sync_instructions=False,
            sync_allowlist=False,
        )
        assert result.linked == 0
        assert any("skills" in w.lower() for w in result.warnings)

    def test_uses_selected_tool_skills_source_when_cursor_is_real_source(
        self, tmp_path: Path
    ) -> None:
        skills = tmp_path / ".cursor" / "skills"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text("# Cursor skill")

        result = sync_configs(
            AIToolID.CURSOR,
            [AIToolID.GEMINI],
            tmp_path,
            force=True,
            sync_instructions=False,
            sync_allowlist=False,
        )

        assert (tmp_path / ".gemini" / "skills").is_symlink()
        assert result.linked == 1


class TestSyncAllowlist:
    def test_converts_claude_to_cursor(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            sync_instructions=False,
            sync_skills=False,
        )
        cli_json = tmp_path / ".cursor" / "cli.json"
        assert cli_json.is_file()
        data = json.loads(cli_json.read_text(encoding="utf-8"))
        allow = data["permissions"]["allow"]
        assert "Shell(myapp:*)" in allow
        assert "Shell(npm:*)" in allow
        assert result.converted == 1

    def test_converts_cursor_to_claude(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir()
        settings = {"permissions": {"allow": ["Shell(myapp:*)"]}}
        (cursor_dir / "cli.json").write_text(json.dumps(settings), encoding="utf-8")

        result = sync_configs(
            AIToolID.CURSOR,
            [AIToolID.CLAUDE],
            tmp_path,
            sync_instructions=False,
            sync_skills=False,
        )
        settings_path = tmp_path / ".claude" / "settings.json"
        assert settings_path.is_file()
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert "Bash(myapp:*)" in data["permissions"]["allow"]
        assert result.converted == 1

    def test_copilot_warns_no_persistent_config(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.COPILOT],
            tmp_path,
            sync_instructions=False,
            sync_skills=False,
        )
        assert result.converted == 0
        assert any("copilot" in w.lower() for w in result.warnings)

    def test_gemini_warns_no_persistent_config(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.GEMINI],
            tmp_path,
            sync_instructions=False,
            sync_skills=False,
        )
        assert result.converted == 0
        assert any("gemini" in w.lower() for w in result.warnings)

    def test_codex_warns_sandbox(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CODEX],
            tmp_path,
            sync_instructions=False,
            sync_skills=False,
        )
        assert result.converted == 0
        assert any("codex" in w.lower() for w in result.warnings)

    def test_no_source_allowlist_skips(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("# Instructions")
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            sync_instructions=False,
            sync_skills=False,
        )
        assert result.converted == 0

    def test_non_allowlist_source_warns(self, tmp_path: Path) -> None:
        (tmp_path / "GEMINI.md").write_text("# Instructions")
        result = sync_configs(
            AIToolID.GEMINI,
            [AIToolID.CURSOR],
            tmp_path,
            sync_instructions=False,
            sync_skills=False,
        )
        assert result.converted == 0
        assert any("allowlist" in w.lower() for w in result.warnings)

    def test_second_allowlist_sync_is_idempotent(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            sync_instructions=False,
            sync_skills=False,
        )

        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            sync_instructions=False,
            sync_skills=False,
        )

        assert result.converted == 0
        assert any("already contains" in action.message for action in result.actions)


class TestSyncUnsupported:
    def test_vscode_unsupported(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(AIToolID.CLAUDE, [AIToolID.VSCODE], tmp_path)
        assert any(a.strategy == SyncStrategy.UNSUPPORTED for a in result.actions)

    def test_opencode_unsupported(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(AIToolID.CLAUDE, [AIToolID.OPENCODE], tmp_path)
        assert any(a.strategy == SyncStrategy.UNSUPPORTED for a in result.actions)

    def test_antigravity_unsupported(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(AIToolID.CLAUDE, [AIToolID.ANTIGRAVITY], tmp_path)
        assert any(a.strategy == SyncStrategy.UNSUPPORTED for a in result.actions)


class TestSyncDryRun:
    def test_dry_run_creates_no_files(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            dry_run=True,
            force=True,
        )
        assert not (tmp_path / ".cursorrules").exists()
        assert not (tmp_path / ".cursor" / "cli.json").exists()
        assert result.linked > 0 or result.converted > 0  # plan has actions


class TestSyncFull:
    def test_full_sync_claude_to_cursor(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = sync_configs(
            AIToolID.CLAUDE,
            [AIToolID.CURSOR],
            tmp_path,
            force=True,
        )
        assert (tmp_path / ".cursorrules").is_symlink()
        assert (tmp_path / ".cursor" / "skills").is_symlink()
        assert (tmp_path / ".cursor" / "cli.json").is_file()
        assert result.linked == 2  # instructions + skills
        assert result.converted == 1  # allowlist
