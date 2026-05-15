"""Tests for rules/instructions sync writers."""

import os
from pathlib import Path

import pytest

from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern, SyncData, SyncRegistry
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


def _make_data(
    source: str = "AGENTS.md",
    strategy: str = "symlink",
) -> SyncData:
    return SyncData(
        rules_source=source,
        rules_strategy=strategy,
    )


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A minimal project directory with a source file."""
    (tmp_path / "AGENTS.md").write_text("# Project Rules\nBe helpful.\n")
    return tmp_path


@pytest.fixture()
def data() -> SyncData:
    return _make_data()


class TestSymlinkCreation:
    def test_creates_symlinks_for_all_targets(self, project: Path, data: SyncData):
        from crossby.sync import run_sync

        registry = SyncRegistry()
        for writer_cls in (
            ClaudeRulesWriter,
            CursorRulesWriter,
            CopilotRulesWriter,
            GeminiRulesWriter,
            CodexRulesWriter,
        ):
            registry.register(writer_cls())

        all_tools = [
            AIToolID.CLAUDE,
            AIToolID.CURSOR,
            AIToolID.COPILOT,
            AIToolID.GEMINI,
            AIToolID.CODEX,
        ]
        results = run_sync(
            data,
            project,
            concern=SyncConcern.RULES,
            installed_tools=all_tools,
            registry=registry,
        )

        # Codex target (AGENTS.md) should be skipped (same resolved path)
        # Filter out gitignore result (tool_id=None)
        created = [r for r in results if r.action == "created" and r.tool_id is not None]
        assert len(created) == len(TOOL_TARGETS) - 1  # minus codex (same path)

        assert (project / "CLAUDE.md").is_symlink()
        assert (project / ".cursorrules").is_symlink()
        assert (project / ".github" / "copilot-instructions.md").is_symlink()
        assert (project / "GEMINI.md").is_symlink()

    def test_symlinks_are_relative(self, project: Path, data: SyncData):
        ClaudeRulesWriter().sync(data, project)

        link = os.readlink(project / "CLAUDE.md")
        assert not os.path.isabs(link)
        assert link == "AGENTS.md"

    def test_symlink_relative_for_nested_target(self, project: Path, data: SyncData):
        CopilotRulesWriter().sync(data, project)

        link = os.readlink(project / ".github" / "copilot-instructions.md")
        assert not os.path.isabs(link)
        assert link == os.path.join("..", "AGENTS.md")

    def test_creates_parent_directory(self, project: Path, data: SyncData):
        """The .github/ directory is created when needed."""
        assert not (project / ".github").exists()
        CopilotRulesWriter().sync(data, project)
        assert (project / ".github").is_dir()

    def test_symlink_false_return_falls_back_to_copy(self, project: Path, data: SyncData):
        """When create_symlink returns False (e.g. circular guard), sync falls back to copy."""
        from unittest.mock import patch

        with patch("crossby.sync.rules.create_symlink", return_value=False):
            result = ClaudeRulesWriter().sync(data, project)

        assert result.action == "created"
        assert result.message == "copy (symlink failed)"
        target = project / "CLAUDE.md"
        assert target.exists()
        assert not target.is_symlink()


class TestCopyCreation:
    def test_creates_copies_with_header(self, project: Path):
        data = _make_data(strategy="copy")
        ClaudeRulesWriter().sync(data, project)

        content = (project / "CLAUDE.md").read_text()
        assert content.startswith(MANAGED_HEADER)
        assert "# Project Rules" in content

    def test_copy_hash_idempotency(self, project: Path):
        data = _make_data(strategy="copy")
        writer = ClaudeRulesWriter()
        writer.sync(data, project)

        result = writer.sync(data, project)
        assert result.action == "skipped"


class TestIdempotency:
    def test_symlink_idempotent(self, project: Path, data: SyncData):
        writer = ClaudeRulesWriter()
        writer.sync(data, project)
        result = writer.sync(data, project)
        assert result.action == "skipped"

    def test_copy_idempotent(self, project: Path):
        data = _make_data(strategy="copy")
        writer = ClaudeRulesWriter()
        writer.sync(data, project)
        result = writer.sync(data, project)
        assert result.action == "skipped"


class TestCircularSymlinkGuard:
    def test_skip_when_source_equals_target(self, project: Path, data: SyncData):
        """Codex target is AGENTS.md, same as source — must be skipped."""
        result = CodexRulesWriter().sync(data, project)
        assert result.action == "skipped"
        assert "same file" in result.message

    def test_skip_with_custom_source_matching_target(self, project: Path):
        (project / "CLAUDE.md").write_text("hello")
        data = _make_data(source="CLAUDE.md")
        result = ClaudeRulesWriter().sync(data, project)
        assert result.action == "skipped"


class TestSourceNotFound:
    def test_error_when_source_missing(self, tmp_path: Path, data: SyncData):
        result = ClaudeRulesWriter().sync(data, tmp_path)
        assert result.action == "error"
        assert "not found" in result.message


class TestUnmanagedFileCollision:
    def test_skip_unmanaged_file(self, project: Path, data: SyncData):
        (project / "CLAUDE.md").write_text("my custom rules")
        result = ClaudeRulesWriter().sync(data, project)

        assert result.action == "skipped"
        assert "--force" in result.message
        # Original file is preserved
        assert (project / "CLAUDE.md").read_text() == "my custom rules"

    def test_force_overwrites_with_backup(self, project: Path, data: SyncData):
        (project / "CLAUDE.md").write_text("my custom rules")
        result = ClaudeRulesWriter().sync(data, project, force=True)

        assert result.action == "updated"
        # Backup was created
        assert (project / "CLAUDE.md.bak").exists()
        assert (project / "CLAUDE.md.bak").read_text() == "my custom rules"
        # New file is a symlink
        assert (project / "CLAUDE.md").is_symlink()


class TestDryRun:
    def test_dry_run_no_files_written(self, project: Path, data: SyncData):
        result = ClaudeRulesWriter().sync(data, project, dry_run=True)

        assert result.action == "created"
        assert "dry-run" in result.message
        # No files actually created
        assert not (project / "CLAUDE.md").exists()

    def test_dry_run_reports_would_create(self, project: Path, data: SyncData):
        result = ClaudeRulesWriter().sync(data, project, dry_run=True)
        assert "would sync" in result.message.lower()


class TestToolFilter:
    def test_filter_single_tool(self, project: Path, data: SyncData):
        from crossby.sync import run_sync

        registry = SyncRegistry()
        for writer_cls in (
            ClaudeRulesWriter,
            CursorRulesWriter,
            CopilotRulesWriter,
            GeminiRulesWriter,
            CodexRulesWriter,
        ):
            registry.register(writer_cls())

        results = run_sync(
            data,
            project,
            tool_id=AIToolID.CLAUDE,
            concern=SyncConcern.RULES,
            registry=registry,
        )
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

    def test_detect_broken_symlink(self, tmp_path: Path):
        """A broken symlink (dangling) should still be detected as existing."""
        claude_path = tmp_path / "CLAUDE.md"
        claude_path.symlink_to(tmp_path / "nonexistent_source.md")
        assert not claude_path.exists()  # broken
        found = detect_existing_rules(tmp_path)
        assert "claude" in found


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
        data = _make_data(strategy="copy")
        writer = ClaudeRulesWriter()
        writer.sync(data, project)

        # Modify source
        (project / "AGENTS.md").write_text("# Updated Rules\nNew content.\n")
        result = writer.sync(data, project)
        assert result.action == "updated"

    def test_copy_up_to_date_when_source_starts_with_newline(self, tmp_path: Path):
        """Source content with leading newlines must not cause repeated rewrites."""
        (tmp_path / "AGENTS.md").write_text("\n# Rules\nContent.\n")
        data = _make_data(strategy="copy")
        writer = ClaudeRulesWriter()
        writer.sync(data, tmp_path)
        # Second sync must detect it is already up-to-date
        result = writer.sync(data, tmp_path)
        assert result.action == "skipped"


class TestForceBackupNumbering:
    def test_second_force_creates_numbered_backup(self, project: Path, data: SyncData):
        (project / "CLAUDE.md").write_text("original")
        ClaudeRulesWriter().sync(data, project, force=True)

        assert (project / "CLAUDE.md.bak").exists()

        # Create another unmanaged file and force again
        (project / "CLAUDE.md").unlink()
        (project / "CLAUDE.md").write_text("second version")
        ClaudeRulesWriter().sync(data, project, force=True)

        assert (project / "CLAUDE.md.bak2").exists()


class TestStrategySwitch:
    def test_symlink_to_copy_resync(self, project: Path):
        """Switching from symlink to copy should re-sync (not skip)."""
        ClaudeRulesWriter().sync(_make_data(strategy="symlink"), project)
        assert (project / "CLAUDE.md").is_symlink()

        result = ClaudeRulesWriter().sync(_make_data(strategy="copy"), project)
        assert result.action == "updated"
        assert not (project / "CLAUDE.md").is_symlink()
        assert (project / "CLAUDE.md").read_text().startswith(MANAGED_HEADER)

    def test_copy_to_symlink_resync(self, project: Path):
        """Switching from copy to symlink should re-sync (not skip)."""
        ClaudeRulesWriter().sync(_make_data(strategy="copy"), project)
        assert not (project / "CLAUDE.md").is_symlink()

        result = ClaudeRulesWriter().sync(_make_data(strategy="symlink"), project)
        assert result.action == "updated"
        assert (project / "CLAUDE.md").is_symlink()


class TestBackupPath:
    def test_broken_symlink_slot_is_skipped(self, tmp_path: Path):
        """backup_path must not return a path occupied by a broken symlink."""
        from crossby.sync.file_utils import backup_path

        target = tmp_path / "file.md"
        target.write_text("content")
        bak = tmp_path / "file.md.bak"
        bak.symlink_to(tmp_path / "nonexistent")
        assert not bak.exists()  # broken symlink

        result = backup_path(target)
        assert result == tmp_path / "file.md.bak2"


class TestBackupSymlink:
    def test_force_backup_of_unmanaged_symlink_is_symlink(self, project: Path, data: SyncData):
        """When force-overwriting an unmanaged symlink, the backup should be a symlink."""
        # Create an unmanaged symlink pointing elsewhere
        other_file = project / "other.md"
        other_file.write_text("other content")
        (project / "CLAUDE.md").symlink_to("other.md")

        result = ClaudeRulesWriter().sync(data, project, force=True)
        assert result.action == "updated"

        backup = project / "CLAUDE.md.bak"
        assert backup.exists()
        assert backup.is_symlink()
        assert os.readlink(backup) == "other.md"


class TestDisabledRules:
    def test_skips_when_no_rules_source(self, project: Path):
        data = SyncData()  # rules_source=None by default
        result = ClaudeRulesWriter().sync(data, project)
        assert result.action == "skipped"


class TestForeignMarkerForceCopy:
    """When source content references another tool's surfaces, the writer
    should switch from symlink to copy and embed a manual-fix block.
    """

    def test_claude_only_content_forces_copy_to_gemini(self, tmp_path: Path):
        # CLAUDE.md mentions ExitPlanMode → not neutral for a Gemini target.
        (tmp_path / "CLAUDE.md").write_text("# Rules\nUse ExitPlanMode when planning is done.\n")
        data = _make_data(source="CLAUDE.md", strategy="symlink")

        result = GeminiRulesWriter().sync(data, tmp_path)

        target = tmp_path / "GEMINI.md"
        assert result.action == "created"
        assert target.is_file()
        assert not target.is_symlink()
        text = target.read_text()
        assert text.startswith(MANAGED_HEADER)
        assert "ExitPlanMode" in text
        assert "<!-- crossby:manual-fix:start -->" in text
        assert "<!-- crossby:manual-fix:end -->" in text
        assert result.message == "foreign markers in source"

    def test_neutral_content_still_symlinks(self, tmp_path: Path):
        # Plain content stays on the configured (symlink) strategy.
        (tmp_path / "CLAUDE.md").write_text("# Rules\nBe helpful.\n")
        data = _make_data(source="CLAUDE.md", strategy="symlink")

        result = GeminiRulesWriter().sync(data, tmp_path)

        target = tmp_path / "GEMINI.md"
        assert result.action == "created"
        assert target.is_symlink()
        assert "<!-- crossby:manual-fix" not in target.read_text()

    def test_claude_target_keeps_native_content_as_symlink(self, tmp_path: Path):
        # The same Claude-only content is fine for Claude itself.
        (tmp_path / "CLAUDE.md").write_text("# Rules\nUse ExitPlanMode when planning is done.\n")
        # Self-symlink is guarded against; use a different filename.
        (tmp_path / "AGENTS.md").write_text("# Rules\nUse ExitPlanMode when planning is done.\n")
        data = _make_data(source="AGENTS.md", strategy="symlink")

        result = ClaudeRulesWriter().sync(data, tmp_path)

        target = tmp_path / "CLAUDE.md"
        # The pre-existing CLAUDE.md is unmanaged → skipped without --force.
        assert result.action == "skipped"
        # The pre-existing CLAUDE.md is preserved (no foreign-markers copy
        # was injected over it).
        assert "<!-- crossby:manual-fix" not in target.read_text()

    def test_idempotent_on_force_copy(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("# Rules\nUse ExitPlanMode when planning is done.\n")
        data = _make_data(source="CLAUDE.md", strategy="symlink")

        first = GeminiRulesWriter().sync(data, tmp_path)
        second = GeminiRulesWriter().sync(data, tmp_path)
        assert first.action == "created"
        assert second.action == "skipped"
        assert second.message == "already linked"

    def test_re_translation_replaces_block_after_source_change(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("# Rules\nUse ExitPlanMode.\n")
        data = _make_data(source="CLAUDE.md", strategy="symlink")

        GeminiRulesWriter().sync(data, tmp_path)

        # User edits the source — different Claude-only marker.
        (tmp_path / "CLAUDE.md").write_text("# Rules\nSet permissionMode to acceptEdits.\n")

        result = GeminiRulesWriter().sync(data, tmp_path)
        text = (tmp_path / "GEMINI.md").read_text()

        assert result.action == "updated"
        assert "permissionMode" in text
        # The previous run's content is gone.
        assert "ExitPlanMode" not in text
        # Exactly one manual-fix block.
        assert text.count("<!-- crossby:manual-fix:start -->") == 1

    def test_dry_run_reports_foreign_markers(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("# Rules\nUse ExitPlanMode.\n")
        data = _make_data(source="CLAUDE.md", strategy="symlink")

        result = GeminiRulesWriter().sync(data, tmp_path, dry_run=True)

        assert result.action == "created"
        assert "would sync via copy" in (result.message or "")
        assert "foreign markers" in (result.message or "")
        # No file written.
        assert not (tmp_path / "GEMINI.md").exists()
