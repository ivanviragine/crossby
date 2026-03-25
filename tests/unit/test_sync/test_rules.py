"""Tests for rules/instructions sync."""

import os
from pathlib import Path

import pytest

from crossby.models.config import RulesConfig
from crossby.sync.base import SyncAction
from crossby.sync.rules import (
    MANAGED_HEADER,
    TOOL_TARGETS,
    detect_existing_rules,
    suggest_source,
    sync_rules,
)


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A minimal project directory with a source file."""
    (tmp_path / "AGENTS.md").write_text("# Project Rules\nBe helpful.\n")
    return tmp_path


@pytest.fixture()
def default_config() -> RulesConfig:
    return RulesConfig()


class TestSymlinkCreation:
    def test_creates_symlinks_for_all_targets(self, project: Path, default_config: RulesConfig):
        results = sync_rules(project, default_config)

        # Codex target (AGENTS.md) should be skipped (same resolved path)
        non_skipped = [r for r in results if r.action == SyncAction.CREATED]
        assert len(non_skipped) == len(TOOL_TARGETS) - 1  # minus codex (same path)

        assert (project / "CLAUDE.md").is_symlink()
        assert (project / ".cursorrules").is_symlink()
        assert (project / ".github" / "copilot-instructions.md").is_symlink()
        assert (project / "GEMINI.md").is_symlink()

    def test_symlinks_are_relative(self, project: Path, default_config: RulesConfig):
        sync_rules(project, default_config)

        link = os.readlink(project / "CLAUDE.md")
        assert not os.path.isabs(link)
        assert link == "AGENTS.md"

    def test_symlink_relative_for_nested_target(self, project: Path, default_config: RulesConfig):
        sync_rules(project, default_config)

        link = os.readlink(project / ".github" / "copilot-instructions.md")
        assert not os.path.isabs(link)
        assert link == os.path.join("..", "AGENTS.md")

    def test_creates_parent_directory(self, project: Path, default_config: RulesConfig):
        """The .github/ directory is created when needed."""
        assert not (project / ".github").exists()
        sync_rules(project, default_config)
        assert (project / ".github").is_dir()


class TestCopyCreation:
    def test_creates_copies_with_header(self, project: Path):
        config = RulesConfig(strategy="copy")
        sync_rules(project, config)

        content = (project / "CLAUDE.md").read_text()
        assert content.startswith(MANAGED_HEADER)
        assert "# Project Rules" in content

    def test_copy_hash_idempotency(self, project: Path):
        config = RulesConfig(strategy="copy")
        sync_rules(project, config)

        results = sync_rules(project, config)
        non_codex = [r for r in results if r.target != "AGENTS.md"]
        assert all(r.action == SyncAction.UP_TO_DATE for r in non_codex)


class TestIdempotency:
    def test_symlink_idempotent(self, project: Path, default_config: RulesConfig):
        sync_rules(project, default_config)
        results = sync_rules(project, default_config)

        non_codex = [r for r in results if r.target != "AGENTS.md"]
        assert all(r.action == SyncAction.UP_TO_DATE for r in non_codex)

    def test_copy_idempotent(self, project: Path):
        config = RulesConfig(strategy="copy")
        sync_rules(project, config)
        results = sync_rules(project, config)

        non_codex = [r for r in results if r.target != "AGENTS.md"]
        assert all(r.action == SyncAction.UP_TO_DATE for r in non_codex)


class TestCircularSymlinkGuard:
    def test_skip_when_source_equals_target(self, project: Path, default_config: RulesConfig):
        """Codex target is AGENTS.md, same as source — must be skipped."""
        results = sync_rules(project, default_config)

        codex = [r for r in results if r.target == "AGENTS.md"]
        assert len(codex) == 1
        assert codex[0].action == SyncAction.SKIPPED
        assert "same file" in codex[0].message

    def test_skip_with_custom_source_matching_target(self, project: Path):
        (project / "CLAUDE.md").write_text("hello")
        config = RulesConfig(source="CLAUDE.md")
        results = sync_rules(project, config)

        claude = [r for r in results if r.target == "CLAUDE.md"]
        assert len(claude) == 1
        assert claude[0].action == SyncAction.SKIPPED


class TestSourceNotFound:
    def test_error_when_source_missing(self, tmp_path: Path, default_config: RulesConfig):
        results = sync_rules(tmp_path, default_config)

        assert len(results) == 1
        assert results[0].action == SyncAction.ERROR
        assert "not found" in results[0].message


class TestUnmanagedFileCollision:
    def test_skip_unmanaged_file(self, project: Path, default_config: RulesConfig):
        (project / "CLAUDE.md").write_text("my custom rules")
        results = sync_rules(project, default_config)

        claude = [r for r in results if r.target == "CLAUDE.md"]
        assert claude[0].action == SyncAction.SKIPPED
        assert "--force" in claude[0].message

        # Original file is preserved
        assert (project / "CLAUDE.md").read_text() == "my custom rules"

    def test_force_overwrites_with_backup(self, project: Path, default_config: RulesConfig):
        (project / "CLAUDE.md").write_text("my custom rules")
        results = sync_rules(project, default_config, force=True)

        claude = [r for r in results if r.target == "CLAUDE.md"]
        assert claude[0].action in (SyncAction.CREATED, SyncAction.UPDATED)

        # Backup was created
        assert (project / "CLAUDE.md.bak").exists()
        assert (project / "CLAUDE.md.bak").read_text() == "my custom rules"

        # New file is a symlink
        assert (project / "CLAUDE.md").is_symlink()


class TestDryRun:
    def test_dry_run_no_files_written(self, project: Path, default_config: RulesConfig):
        results = sync_rules(project, default_config, dry_run=True)

        created = [r for r in results if r.action == SyncAction.CREATED]
        assert len(created) == 4
        assert all(r.dry_run for r in created)

        # No files actually created
        assert not (project / "CLAUDE.md").exists()
        assert not (project / ".cursorrules").exists()

    def test_dry_run_reports_would_create(self, project: Path, default_config: RulesConfig):
        results = sync_rules(project, default_config, dry_run=True)

        for r in results:
            if r.action == SyncAction.CREATED:
                assert "Would" in r.message


class TestToolFilter:
    def test_filter_single_tool(self, project: Path, default_config: RulesConfig):
        results = sync_rules(project, default_config, tool_filter="claude")

        assert len(results) == 1
        assert results[0].target == "CLAUDE.md"


class TestDetectExisting:
    def test_detect_agents_md(self, project: Path):
        found = detect_existing_rules(project)
        assert "codex" in found

    def test_detect_multiple(self, project: Path):
        (project / "CLAUDE.md").write_text("rules")
        (project / "GEMINI.md").write_text("rules")
        found = detect_existing_rules(project)
        assert "claude" in found
        assert "gemini" in found
        assert "codex" in found

    def test_detect_empty(self, tmp_path: Path):
        found = detect_existing_rules(tmp_path)
        assert found == {}


class TestSuggestSource:
    def test_prefer_agents_md(self, project: Path):
        existing = detect_existing_rules(project)
        assert suggest_source(existing) == "AGENTS.md"

    def test_prefer_claude_if_no_agents(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("rules")
        existing = detect_existing_rules(tmp_path)
        assert suggest_source(existing) == "CLAUDE.md"

    def test_default_agents_when_empty(self):
        assert suggest_source({}) == "AGENTS.md"


class TestCopyDetectsSourceChange:
    def test_copy_updated_after_source_modification(self, project: Path):
        config = RulesConfig(strategy="copy")
        sync_rules(project, config)

        # Modify source
        (project / "AGENTS.md").write_text("# Updated Rules\nNew content.\n")
        results = sync_rules(project, config)

        claude = [r for r in results if r.target == "CLAUDE.md"]
        assert claude[0].action == SyncAction.UPDATED


class TestForceBackupNumbering:
    def test_second_force_creates_numbered_backup(self, project: Path, default_config: RulesConfig):
        (project / "CLAUDE.md").write_text("original")
        sync_rules(project, default_config, force=True)

        # First backup exists
        assert (project / "CLAUDE.md.bak").exists()

        # Create another unmanaged file and force again
        (project / "CLAUDE.md").unlink()
        (project / "CLAUDE.md").write_text("second version")
        sync_rules(project, default_config, force=True)

        # Second backup has numbered suffix
        assert (project / "CLAUDE.md.bak2").exists()
