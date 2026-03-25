"""Tests for rules/instructions sync writers."""

import os
from pathlib import Path

import pytest

from crossby.models.ai import AIToolID
from crossby.models.config import CrossbyConfig, RulesConfig, RulesTargetsConfig
from crossby.sync.base import SyncConcern, SyncRegistry
from crossby.sync.rules import (
    MANAGED_HEADER,
    TOOL_TARGETS,
    ClaudeRulesWriter,
    CodexRulesWriter,
    CopilotRulesWriter,
    CursorRulesWriter,
    GeminiRulesWriter,
    detect_existing_rules,
    suggest_source,
)


def _make_config(
    source: str = "AGENTS.md",
    strategy: str = "symlink",
    targets: RulesTargetsConfig | None = None,
) -> CrossbyConfig:
    return CrossbyConfig(
        rules=RulesConfig(
            enabled=True,
            source=source,
            strategy=strategy,
            targets=targets or RulesTargetsConfig(),
        ),
    )


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A minimal project directory with a source file."""
    (tmp_path / "AGENTS.md").write_text("# Project Rules\nBe helpful.\n")
    return tmp_path


@pytest.fixture()
def config() -> CrossbyConfig:
    return _make_config()


class TestSymlinkCreation:
    def test_creates_symlinks_for_all_targets(self, project: Path, config: CrossbyConfig):
        from crossby.sync import run_sync

        registry = SyncRegistry()
        for writer_cls in (ClaudeRulesWriter, CursorRulesWriter, CopilotRulesWriter, GeminiRulesWriter, CodexRulesWriter):
            registry.register(writer_cls())

        all_tools = [AIToolID.CLAUDE, AIToolID.CURSOR, AIToolID.COPILOT, AIToolID.GEMINI, AIToolID.CODEX]
        results = run_sync(config, project, concern=SyncConcern.RULES, installed_tools=all_tools, registry=registry)

        # Codex target (AGENTS.md) should be skipped (same resolved path)
        # Filter out gitignore result (tool_id=None)
        created = [r for r in results if r.action == "created" and r.tool_id is not None]
        assert len(created) == len(TOOL_TARGETS) - 1  # minus codex (same path)

        assert (project / "CLAUDE.md").is_symlink()
        assert (project / ".cursorrules").is_symlink()
        assert (project / ".github" / "copilot-instructions.md").is_symlink()
        assert (project / "GEMINI.md").is_symlink()

    def test_symlinks_are_relative(self, project: Path, config: CrossbyConfig):
        ClaudeRulesWriter().sync(config, project)

        link = os.readlink(project / "CLAUDE.md")
        assert not os.path.isabs(link)
        assert link == "AGENTS.md"

    def test_symlink_relative_for_nested_target(self, project: Path, config: CrossbyConfig):
        CopilotRulesWriter().sync(config, project)

        link = os.readlink(project / ".github" / "copilot-instructions.md")
        assert not os.path.isabs(link)
        assert link == os.path.join("..", "AGENTS.md")

    def test_creates_parent_directory(self, project: Path, config: CrossbyConfig):
        """The .github/ directory is created when needed."""
        assert not (project / ".github").exists()
        CopilotRulesWriter().sync(config, project)
        assert (project / ".github").is_dir()


class TestCopyCreation:
    def test_creates_copies_with_header(self, project: Path):
        config = _make_config(strategy="copy")
        ClaudeRulesWriter().sync(config, project)

        content = (project / "CLAUDE.md").read_text()
        assert content.startswith(MANAGED_HEADER)
        assert "# Project Rules" in content

    def test_copy_hash_idempotency(self, project: Path):
        config = _make_config(strategy="copy")
        writer = ClaudeRulesWriter()
        writer.sync(config, project)

        result = writer.sync(config, project)
        assert result.action == "skipped"


class TestIdempotency:
    def test_symlink_idempotent(self, project: Path, config: CrossbyConfig):
        writer = ClaudeRulesWriter()
        writer.sync(config, project)
        result = writer.sync(config, project)
        assert result.action == "skipped"

    def test_copy_idempotent(self, project: Path):
        config = _make_config(strategy="copy")
        writer = ClaudeRulesWriter()
        writer.sync(config, project)
        result = writer.sync(config, project)
        assert result.action == "skipped"


class TestCircularSymlinkGuard:
    def test_skip_when_source_equals_target(self, project: Path, config: CrossbyConfig):
        """Codex target is AGENTS.md, same as source — must be skipped."""
        result = CodexRulesWriter().sync(config, project)
        assert result.action == "skipped"
        assert "same file" in result.message

    def test_skip_with_custom_source_matching_target(self, project: Path):
        (project / "CLAUDE.md").write_text("hello")
        config = _make_config(source="CLAUDE.md")
        result = ClaudeRulesWriter().sync(config, project)
        assert result.action == "skipped"


class TestSourceNotFound:
    def test_error_when_source_missing(self, tmp_path: Path, config: CrossbyConfig):
        result = ClaudeRulesWriter().sync(config, tmp_path)
        assert result.action == "error"
        assert "not found" in result.message


class TestUnmanagedFileCollision:
    def test_skip_unmanaged_file(self, project: Path, config: CrossbyConfig):
        (project / "CLAUDE.md").write_text("my custom rules")
        result = ClaudeRulesWriter().sync(config, project)

        assert result.action == "skipped"
        assert "--force" in result.message
        # Original file is preserved
        assert (project / "CLAUDE.md").read_text() == "my custom rules"

    def test_force_overwrites_with_backup(self, project: Path, config: CrossbyConfig):
        (project / "CLAUDE.md").write_text("my custom rules")
        result = ClaudeRulesWriter().sync(config, project, force=True)

        assert result.action == "created"
        # Backup was created
        assert (project / "CLAUDE.md.bak").exists()
        assert (project / "CLAUDE.md.bak").read_text() == "my custom rules"
        # New file is a symlink
        assert (project / "CLAUDE.md").is_symlink()


class TestDryRun:
    def test_dry_run_no_files_written(self, project: Path, config: CrossbyConfig):
        result = ClaudeRulesWriter().sync(config, project, dry_run=True)

        assert result.action == "created"
        assert "dry-run" in result.message
        # No files actually created
        assert not (project / "CLAUDE.md").exists()

    def test_dry_run_reports_would_create(self, project: Path, config: CrossbyConfig):
        result = ClaudeRulesWriter().sync(config, project, dry_run=True)
        assert "would sync" in result.message.lower()


class TestToolFilter:
    def test_filter_single_tool(self, project: Path, config: CrossbyConfig):
        from crossby.sync import run_sync

        registry = SyncRegistry()
        for writer_cls in (ClaudeRulesWriter, CursorRulesWriter, CopilotRulesWriter, GeminiRulesWriter, CodexRulesWriter):
            registry.register(writer_cls())

        results = run_sync(config, project, tool_id=AIToolID.CLAUDE, concern=SyncConcern.RULES, registry=registry)
        assert len(results) == 1
        assert results[0].tool_id == AIToolID.CLAUDE


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
        config = _make_config(strategy="copy")
        writer = ClaudeRulesWriter()
        writer.sync(config, project)

        # Modify source
        (project / "AGENTS.md").write_text("# Updated Rules\nNew content.\n")
        result = writer.sync(config, project)
        assert result.action == "created"


class TestForceBackupNumbering:
    def test_second_force_creates_numbered_backup(self, project: Path, config: CrossbyConfig):
        (project / "CLAUDE.md").write_text("original")
        ClaudeRulesWriter().sync(config, project, force=True)

        assert (project / "CLAUDE.md.bak").exists()

        # Create another unmanaged file and force again
        (project / "CLAUDE.md").unlink()
        (project / "CLAUDE.md").write_text("second version")
        ClaudeRulesWriter().sync(config, project, force=True)

        assert (project / "CLAUDE.md.bak2").exists()


class TestDisabledTarget:
    def test_skips_disabled_target(self, project: Path):
        config = _make_config(targets=RulesTargetsConfig(claude=False))
        result = ClaudeRulesWriter().sync(config, project)
        assert result.action == "skipped"
        assert "not in targets" in result.message


class TestDisabledRules:
    def test_skips_when_not_enabled(self, project: Path):
        config = CrossbyConfig()  # rules.enabled=False by default
        result = ClaudeRulesWriter().sync(config, project)
        assert result.action == "skipped"
