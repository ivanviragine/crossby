"""Tests for skills sync writers, gitignore helper, and run_sync integration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

import pytest

from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern, SyncData
from crossby.sync.skills import (
    _GITIGNORE_BLOCK_ID,
    AntigravityCLISkillsWriter,
    ClaudeSkillsWriter,
    CodexSkillsWriter,
    CopilotSkillsWriter,
    CursorSkillsWriter,
    _is_managed_skills_dir,
    update_skills_gitignore,
)

_BLOCK_START = f"# >>> crossby {_GITIGNORE_BLOCK_ID} (generated — do not edit) >>>"
_BLOCK_END = f"# <<< crossby {_GITIGNORE_BLOCK_ID} <<<"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data(
    source: str = ".crossby/skills",
    strategy: Literal["symlink", "copy", "translate"] = "symlink",
    gitignore: bool = True,
) -> SyncData:
    return SyncData(
        skills_source=source,
        skills_strategy=strategy,
        skills_gitignore=gitignore,
    )


def _make_source(tmp_path: Path, skills: list[str] | None = None) -> Path:
    """Create a skills source dir with optional skill subdirectories (each with SKILL.md)."""
    source = tmp_path / ".crossby" / "skills"
    source.mkdir(parents=True)
    for name in skills or []:
        skill_dir = source / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    return source


def _make_skill(directory: Path, name: str) -> Path:
    """Create a skill subdirectory with SKILL.md inside *directory*."""
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    return skill_dir


# ---------------------------------------------------------------------------
# _is_managed_skills_dir
# ---------------------------------------------------------------------------


class TestIsManagedSkillsDir:
    def test_empty_dir_is_managed(self, tmp_path: Path) -> None:
        d = tmp_path / "skills"
        d.mkdir()
        assert _is_managed_skills_dir(d) is True

    def test_dir_with_skill_subdirs_alone_is_not_managed(self, tmp_path: Path) -> None:
        """SKILL.md-bearing subdir shape isn't enough — needs the marker.

        Previously the shape alone qualified, but that's identical to a
        natively-organized user skills tree (every tool uses the same
        ``<name>/SKILL.md`` layout). Without the marker we cannot tell the
        difference.
        """
        d = tmp_path / "skills"
        d.mkdir()
        _make_skill(d, "my-skill")
        _make_skill(d, "another-skill")
        assert _is_managed_skills_dir(d) is False

    def test_dir_with_skill_subdirs_and_marker_is_managed(self, tmp_path: Path) -> None:
        d = tmp_path / "skills"
        d.mkdir()
        _make_skill(d, "my-skill")
        _make_skill(d, "another-skill")
        (d / ".crossby-managed").write_text("", encoding="utf-8")
        assert _is_managed_skills_dir(d) is True

    def test_dir_with_md_file_is_not_managed(self, tmp_path: Path) -> None:
        d = tmp_path / "skills"
        d.mkdir()
        (d / "README.md").write_text("readme", encoding="utf-8")
        assert _is_managed_skills_dir(d) is False

    def test_dir_with_subdir_missing_skill_md_is_not_managed(self, tmp_path: Path) -> None:
        d = tmp_path / "skills"
        d.mkdir()
        (d / "not-a-skill").mkdir()
        assert _is_managed_skills_dir(d) is False

    def test_mixed_content_is_not_managed(self, tmp_path: Path) -> None:
        d = tmp_path / "skills"
        d.mkdir()
        _make_skill(d, "valid-skill")
        (d / "junk.txt").write_text("junk", encoding="utf-8")
        assert _is_managed_skills_dir(d) is False


# ---------------------------------------------------------------------------
# ClaudeSkillsWriter (representative for all _BaseSkillsWriter subclasses)
# ---------------------------------------------------------------------------


class TestClaudeSkillsWriter:
    writer = ClaudeSkillsWriter()

    def test_fresh_symlink(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["skill-a"])
        result = self.writer.sync(_data(), tmp_path)
        assert result.action == "created"
        target = tmp_path / ".claude" / "skills"
        assert target.is_symlink()
        assert not os.path.isabs(os.readlink(target))
        assert target.resolve() == (tmp_path / ".crossby" / "skills").resolve()

    def test_idempotent_re_run(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["skill-a"])
        self.writer.sync(_data(), tmp_path)
        result = self.writer.sync(_data(), tmp_path)
        assert result.action == "skipped"
        assert "already linked" in (result.message or "")

    def test_skips_when_no_source(self, tmp_path: Path) -> None:
        data = SyncData(skills_source=None)
        result = self.writer.sync(data, tmp_path)
        assert result.action == "skipped"

    def test_error_when_source_missing(self, tmp_path: Path) -> None:
        result = self.writer.sync(_data(), tmp_path)
        assert result.action == "error"
        assert "not found" in (result.message or "")

    def test_error_when_source_is_file(self, tmp_path: Path) -> None:
        source = tmp_path / ".crossby" / "skills"
        source.parent.mkdir(parents=True)
        source.write_text("not a dir", encoding="utf-8")
        result = self.writer.sync(_data(), tmp_path)
        assert result.action == "error"
        assert "not a directory" in (result.message or "")

    def test_circular_source_target_skipped(self, tmp_path: Path) -> None:
        """When source and target resolve to the same path, return skipped."""
        source = tmp_path / ".claude" / "skills"
        source.mkdir(parents=True)
        _make_skill(source, "skill-a")
        data = _data(source=".claude/skills")
        result = self.writer.sync(data, tmp_path)
        assert result.action == "skipped"
        assert "same path" in (result.message or "")

    def test_managed_real_dir_re_synced_via_copy(self, tmp_path: Path) -> None:
        """A crossby-marked real dir is replaced via copy without --force."""
        _make_source(tmp_path, ["skill-a"])
        target = tmp_path / ".claude" / "skills"
        target.mkdir(parents=True)
        _make_skill(target, "old-skill")
        (target / ".crossby-managed").write_text("", encoding="utf-8")
        result = self.writer.sync(_data(), tmp_path)
        assert result.action == "created"

    def test_unmanaged_real_dir_blocked_without_force(self, tmp_path: Path) -> None:
        """An unmarked real dir is treated as user-owned and blocked without --force."""
        _make_source(tmp_path, ["skill-a"])
        target = tmp_path / ".claude" / "skills"
        target.mkdir(parents=True)
        # Hand-curated skills tree — no marker → user-owned.
        _make_skill(target, "user-skill")
        result = self.writer.sync(_data(), tmp_path)
        assert result.action == "error"
        assert "--force" in (result.message or "")
        # User skill is untouched.
        assert (target / "user-skill" / "SKILL.md").is_file()

    def test_force_replaces_unmanaged_dir_with_backup(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["skill-a"])
        target = tmp_path / ".claude" / "skills"
        target.mkdir(parents=True)
        (target / "not_a_skill.txt").write_text("junk", encoding="utf-8")
        result = self.writer.sync(_data(), tmp_path, force=True)
        assert result.action == "created"
        assert target.is_symlink()
        backup = tmp_path / ".claude" / "skills.bak"
        assert backup.is_dir()
        assert (backup / "not_a_skill.txt").exists()

    def test_dry_run_does_not_create_symlink(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["skill-a"])
        result = self.writer.sync(_data(), tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".claude" / "skills").exists()

    def test_copy_writes_managed_marker(self, tmp_path: Path) -> None:
        """Copy strategy drops the .crossby-managed marker so the dir is recognized later."""
        _make_source(tmp_path, ["skill-a"])
        self.writer.sync(_data(strategy="copy"), tmp_path)
        marker = tmp_path / ".claude" / "skills" / ".crossby-managed"
        assert marker.is_file()

    def test_symlink_failure_fallback_marks_managed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A copy-fallback after a symlink failure must drop the marker too,
        otherwise the next sync would refuse its own output as user-owned."""
        _make_source(tmp_path, ["skill-a"])

        def _boom(*_args: object, **_kwargs: object) -> bool:
            raise OSError("simulated symlink failure")

        monkeypatch.setattr("crossby.sync.skills.create_symlink", _boom)

        first = self.writer.sync(_data(), tmp_path)
        assert first.action == "created"
        assert first.message == "copy (symlink failed)"
        marker = tmp_path / ".claude" / "skills" / ".crossby-managed"
        assert marker.is_file()

        # Second run, still failing, must not error out on its own output.
        second = self.writer.sync(_data(), tmp_path)
        assert second.action != "error", second.message


# ---------------------------------------------------------------------------
# Copy strategy
# ---------------------------------------------------------------------------


class TestSkillsWriterCopyStrategy:
    writer = ClaudeSkillsWriter()

    def test_copy_strategy_copies_structure(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["skill-a", "skill-b"])
        result = self.writer.sync(_data(strategy="copy"), tmp_path)
        assert result.action == "created"
        target = tmp_path / ".claude" / "skills"
        assert target.is_dir()
        assert not target.is_symlink()
        assert (target / "skill-a" / "SKILL.md").exists()
        assert (target / "skill-b" / "SKILL.md").exists()

    def test_copy_into_symlinked_target_blocked_without_force(self, tmp_path: Path) -> None:
        """copy strategy refuses to write into a symlinked target without --force."""
        _make_source(tmp_path, ["skill-a"])
        # Pre-create a symlink at target
        target = tmp_path / ".claude" / "skills"
        target.parent.mkdir(parents=True, exist_ok=True)
        somewhere = tmp_path / "other"
        somewhere.mkdir()
        os.symlink(os.path.relpath(somewhere, target.parent), target)
        result = self.writer.sync(_data(strategy="copy"), tmp_path)
        assert result.action == "error"
        assert "symlink" in (result.message or "").lower()

    def test_copy_dry_run(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["skill-a"])
        result = self.writer.sync(_data(strategy="copy"), tmp_path, dry_run=True)
        assert result.action == "created"
        assert "dry-run" in (result.message or "")
        assert not (tmp_path / ".claude" / "skills").exists()


# ---------------------------------------------------------------------------
# All five concrete writers have correct tool_id and target_rel
# ---------------------------------------------------------------------------


class TestConcreteWriterAttributes:
    @pytest.mark.parametrize(
        "writer_cls, expected_tool, expected_target",
        [
            (ClaudeSkillsWriter, AIToolID.CLAUDE, ".claude/skills"),
            (CursorSkillsWriter, AIToolID.CURSOR, ".cursor/skills"),
            (CodexSkillsWriter, AIToolID.CODEX, ".agents/skills"),
            (AntigravityCLISkillsWriter, AIToolID.ANTIGRAVITY_CLI, ".agents/skills"),
            (CopilotSkillsWriter, AIToolID.COPILOT, ".github/skills"),
        ],
    )
    def test_writer_metadata(
        self,
        writer_cls: type,
        expected_tool: AIToolID,
        expected_target: str,
    ) -> None:
        w = writer_cls()
        assert w.tool_id == expected_tool
        assert w._target_rel == expected_target
        assert w.concern == SyncConcern.SKILLS

    @pytest.mark.parametrize(
        "writer_cls, expected_target",
        [
            (ClaudeSkillsWriter, ".claude/skills"),
            (CursorSkillsWriter, ".cursor/skills"),
            (CodexSkillsWriter, ".agents/skills"),
            (AntigravityCLISkillsWriter, ".agents/skills"),
            (CopilotSkillsWriter, ".github/skills"),
        ],
    )
    def test_each_writer_creates_symlink(
        self,
        tmp_path: Path,
        writer_cls: type,
        expected_target: str,
    ) -> None:
        _make_source(tmp_path, ["skill-a"])
        w = writer_cls()
        result = w.sync(_data(), tmp_path)
        assert result.action == "created"
        target = tmp_path / expected_target
        assert target.is_symlink()
        assert target.resolve() == (tmp_path / ".crossby" / "skills").resolve()


# ---------------------------------------------------------------------------
# update_skills_gitignore
# ---------------------------------------------------------------------------


class TestUpdateSkillsGitignore:
    def test_creates_gitignore_with_block(self, tmp_path: Path) -> None:
        data = _data()
        result = update_skills_gitignore(data, tmp_path)
        assert result is not None
        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert _BLOCK_START in content
        assert _BLOCK_END in content
        assert ".claude/skills" in content

    def test_none_source_returns_none(self, tmp_path: Path) -> None:
        data = SyncData(skills_source=None)
        result = update_skills_gitignore(data, tmp_path)
        assert result is None
        assert not (tmp_path / ".gitignore").exists()

    def test_gitignore_false_returns_none(self, tmp_path: Path) -> None:
        data = _data(gitignore=False)
        result = update_skills_gitignore(data, tmp_path)
        assert result is None

    def test_installed_tools_filter_entries(self, tmp_path: Path) -> None:
        data = _data()
        update_skills_gitignore(data, tmp_path, installed_tools=[AIToolID.CLAUDE, AIToolID.CURSOR])
        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert ".claude/skills" in content
        assert ".cursor/skills" in content
        assert ".github/skills" not in content
        assert ".agents/skills" not in content

    def test_installed_tools_none_includes_all(self, tmp_path: Path) -> None:
        data = _data()
        update_skills_gitignore(data, tmp_path, installed_tools=None)
        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert ".claude/skills" in content
        assert ".cursor/skills" in content
        assert ".github/skills" in content
        assert ".agents/skills" in content

    def test_idempotent(self, tmp_path: Path) -> None:
        data = _data()
        update_skills_gitignore(data, tmp_path)
        content_before = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        result = update_skills_gitignore(data, tmp_path)
        assert result is None
        content_after = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert content_before == content_after

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        data = _data()
        result = update_skills_gitignore(data, tmp_path, dry_run=True)
        assert result is not None
        assert not (tmp_path / ".gitignore").exists()

    def test_appends_to_existing_gitignore(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n", encoding="utf-8")
        data = _data()
        update_skills_gitignore(data, tmp_path)
        content = gitignore.read_text(encoding="utf-8")
        assert "*.pyc" in content
        assert _BLOCK_START in content

    def test_action_created_when_file_absent(self, tmp_path: Path) -> None:
        data = _data()
        result = update_skills_gitignore(data, tmp_path)
        assert result is not None
        assert result.action == "created"

    def test_action_updated_when_file_exists(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("", encoding="utf-8")
        data = _data()
        result = update_skills_gitignore(data, tmp_path)
        assert result is not None
        assert result.action == "updated"

    def test_tool_id_is_none(self, tmp_path: Path) -> None:
        data = _data()
        result = update_skills_gitignore(data, tmp_path)
        assert result is not None
        assert result.tool_id is None
        assert result.concern == SyncConcern.SKILLS


# ---------------------------------------------------------------------------
# Integration: run_sync with SyncConcern.SKILLS
# ---------------------------------------------------------------------------


class TestRunSyncSkills:
    def test_skills_concern_syncs_to_all_targets(self, tmp_path: Path) -> None:
        from crossby.sync import run_sync
        from crossby.sync.base import SyncRegistry

        _make_source(tmp_path, ["skill-a"])
        reg = SyncRegistry()
        reg.register(ClaudeSkillsWriter())
        reg.register(CursorSkillsWriter())

        data = _data()
        results = run_sync(
            data,
            tmp_path,
            concern=SyncConcern.SKILLS,
            installed_tools=[AIToolID.CLAUDE, AIToolID.CURSOR],
            registry=reg,
        )

        tool_ids = [r.tool_id for r in results]
        assert AIToolID.CLAUDE in tool_ids
        assert AIToolID.CURSOR in tool_ids

        assert (tmp_path / ".claude" / "skills").is_symlink()
        assert (tmp_path / ".cursor" / "skills").is_symlink()

    def test_gitignore_updated_after_skills_writers(self, tmp_path: Path) -> None:
        from crossby.sync import run_sync
        from crossby.sync.base import SyncRegistry

        _make_source(tmp_path, ["skill-a"])
        reg = SyncRegistry()
        reg.register(ClaudeSkillsWriter())

        data = _data()
        results = run_sync(
            data,
            tmp_path,
            concern=SyncConcern.SKILLS,
            installed_tools=[AIToolID.CLAUDE],
            registry=reg,
        )

        # One of the results should be the gitignore update
        gitignore_results = [r for r in results if r.message == "gitignore"]
        assert gitignore_results, "Expected a gitignore update result"
        assert (tmp_path / ".gitignore").exists()
        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert _BLOCK_START in content

    def test_dry_run_no_files_written(self, tmp_path: Path) -> None:
        from crossby.sync import run_sync
        from crossby.sync.base import SyncRegistry

        _make_source(tmp_path, ["skill-a"])
        reg = SyncRegistry()
        reg.register(ClaudeSkillsWriter())

        data = _data()
        run_sync(
            data,
            tmp_path,
            concern=SyncConcern.SKILLS,
            installed_tools=[AIToolID.CLAUDE],
            dry_run=True,
            registry=reg,
        )

        assert not (tmp_path / ".claude" / "skills").exists()
        assert not (tmp_path / ".gitignore").exists()

    def test_gitignore_not_written_when_skills_gitignore_false(self, tmp_path: Path) -> None:
        from crossby.sync import run_sync
        from crossby.sync.base import SyncRegistry

        _make_source(tmp_path, ["skill-a"])
        reg = SyncRegistry()
        reg.register(ClaudeSkillsWriter())

        data = SyncData(skills_source=".crossby/skills", skills_gitignore=False)
        results = run_sync(
            data,
            tmp_path,
            concern=SyncConcern.SKILLS,
            installed_tools=[AIToolID.CLAUDE],
            registry=reg,
        )

        gitignore_results = [r for r in results if r.message == "gitignore"]
        assert not gitignore_results
        assert not (tmp_path / ".gitignore").exists()

    def test_tool_id_filter_skips_gitignore(self, tmp_path: Path) -> None:
        """When tool_id filter is set, gitignore update is skipped."""
        from crossby.sync import run_sync
        from crossby.sync.base import SyncRegistry

        _make_source(tmp_path, ["skill-a"])
        reg = SyncRegistry()
        reg.register(ClaudeSkillsWriter())

        data = _data()
        results = run_sync(
            data,
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            concern=SyncConcern.SKILLS,
            registry=reg,
        )

        gitignore_results = [r for r in results if r.message == "gitignore"]
        assert not gitignore_results


# ---------------------------------------------------------------------------
# Translate strategy
# ---------------------------------------------------------------------------


class TestTranslateStrategy:
    """``translate`` strategy: per-skill copy with target-aware SKILL.md rewriting."""

    def _make_skill_with_frontmatter(
        self,
        source_dir: Path,
        name: str,
        *,
        allowed_tools: list[str] | None = None,
        body: str = "Body.",
    ) -> Path:
        skill_dir = source_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        lines = ["---", f"name: {name}", "description: A skill."]
        if allowed_tools:
            lines.append("allowed-tools:")
            for tool in allowed_tools:
                lines.append(f"  - {tool}")
        lines.extend(["---", body, ""])
        (skill_dir / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
        return skill_dir

    def test_writes_skill_md_with_no_lossy_fields_for_claude_target(self, tmp_path: Path) -> None:
        # Claude target with allowed-tools — no manual-fix block expected.
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "my-skill", allowed_tools=["Read"])

        result = ClaudeSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert result.action == "created"
        out = tmp_path / ".claude" / "skills" / "my-skill" / "SKILL.md"
        text = out.read_text(encoding="utf-8")
        assert "name: my-skill" in text
        assert "<!-- crossby:manual-fix" not in text

    def test_emits_manual_fix_for_non_claude_target(self, tmp_path: Path) -> None:
        # Codex target with allowed-tools — manual-fix block expected.
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "my-skill", allowed_tools=["Read"])

        result = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert result.action == "created"
        out = tmp_path / ".agents" / "skills" / "my-skill" / "SKILL.md"
        text = out.read_text(encoding="utf-8")
        assert "<!-- crossby:manual-fix:start -->" in text
        assert "allowed-tools" in text

    def test_no_lossy_fields_no_manual_fix_anywhere(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "my-skill")

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        text = (tmp_path / ".agents" / "skills" / "my-skill" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        assert "<!-- crossby:manual-fix" not in text

    def test_support_dirs_copied(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path, [])
        skill = self._make_skill_with_frontmatter(source, "my-skill")
        (skill / "scripts").mkdir()
        (skill / "scripts" / "helper.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (skill / "references").mkdir()
        (skill / "references" / "doc.md").write_text("doc", encoding="utf-8")

        ClaudeSkillsWriter().sync(_data(strategy="translate"), tmp_path)

        target = tmp_path / ".claude" / "skills" / "my-skill"
        assert (target / "scripts" / "helper.sh").is_file()
        assert (target / "references" / "doc.md").is_file()

    def test_idempotent_translate(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "my-skill", allowed_tools=["Read"])

        first = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        second = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert first.action == "created"
        assert second.action == "skipped"

    def test_stale_skill_dir_removed(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "keep")
        self._make_skill_with_frontmatter(source, "drop")
        ClaudeSkillsWriter().sync(_data(strategy="translate"), tmp_path)

        # Remove one source skill.
        shutil.rmtree(source / "drop")

        ClaudeSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert (tmp_path / ".claude" / "skills" / "keep").is_dir()
        assert not (tmp_path / ".claude" / "skills" / "drop").exists()

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "my-skill")

        result = ClaudeSkillsWriter().sync(_data(strategy="translate"), tmp_path, dry_run=True)
        assert result.action in {"created", "updated"}
        assert "dry-run" in (result.message or "")
        assert not (tmp_path / ".claude" / "skills").exists()

    def test_dry_run_reports_manual_fix_count(self, tmp_path: Path) -> None:
        """Translate dry-run should surface manual-fix items for --plan."""
        source = _make_source(tmp_path, [])
        # Claude allowed-tools fields produce a manual-fix block when target is Codex.
        self._make_skill_with_frontmatter(source, "needs_fix", allowed_tools=["Read"])
        self._make_skill_with_frontmatter(source, "clean")
        result = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path, dry_run=True)
        assert "1 manual-fix" in (result.message or "")
        from crossby.sync.plan import summarize_plan

        summary = summarize_plan([result])
        assert summary.manual_fix_count == 1

    def test_re_translate_after_source_change(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "my-skill")

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        # Now add allowed-tools to source — manual-fix should appear.
        self._make_skill_with_frontmatter(source, "my-skill", allowed_tools=["Read", "Bash"])

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        text = (tmp_path / ".agents" / "skills" / "my-skill" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        # Exactly one manual-fix block, with the latest content.
        assert text.count("<!-- crossby:manual-fix:start -->") == 1
        assert "Bash" in text


# ---------------------------------------------------------------------------
# Translate strategy: Claude slash commands → skills
# ---------------------------------------------------------------------------


class TestTranslateClaudeSlashCommands:
    """When syncing claude → other tool with translate strategy, .claude/commands/
    files become single-file skills under the target's skills dir."""

    def test_command_appears_as_skill_in_codex_target(self, tmp_path: Path) -> None:
        # Set up a regular skill at the source plus a Claude command.
        source = _make_source(tmp_path, [])
        skill = source / "my-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: A skill.\n---\nBody.\n",
            encoding="utf-8",
        )
        cmd = tmp_path / ".claude" / "commands" / "review.md"
        cmd.parent.mkdir(parents=True)
        cmd.write_text(
            "---\ndescription: Code review.\n---\nReview the diff.\n",
            encoding="utf-8",
        )

        result = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert result.action == "created"
        # Regular skill present.
        assert (tmp_path / ".agents" / "skills" / "my-skill" / "SKILL.md").is_file()
        # Slash-command-derived skill present with the namespaced name.
        cmd_skill = tmp_path / ".agents" / "skills" / "claude-command-review" / "SKILL.md"
        assert cmd_skill.is_file()
        text = cmd_skill.read_text(encoding="utf-8")
        assert "Code review." in text
        assert "Command Template" in text
        assert "Review the diff." in text
        # Slash-command manual-fix note included.
        assert "<!-- crossby:manual-fix:start -->" in text
        assert "Claude slash command" in text

    def test_no_command_skill_for_claude_target(self, tmp_path: Path) -> None:
        # Claude target should NOT receive a converted command skill — it
        # already owns the original command.
        _make_source(tmp_path, ["my-skill"])
        cmd = tmp_path / ".claude" / "commands" / "review.md"
        cmd.parent.mkdir(parents=True)
        cmd.write_text("---\ndescription: x\n---\nBody.", encoding="utf-8")

        result = ClaudeSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert result.action == "created"
        assert not (tmp_path / ".claude" / "skills" / "claude-command-review").exists()

    def test_command_runtime_caveats_in_manual_fix(self, tmp_path: Path) -> None:
        _make_source(tmp_path, [])
        cmd = tmp_path / ".claude" / "commands" / "review.md"
        cmd.parent.mkdir(parents=True)
        cmd.write_text("Run with $ARGUMENTS and !`git diff`.", encoding="utf-8")

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        text = (tmp_path / ".agents" / "skills" / "claude-command-review" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        assert "$ARGUMENTS" in text  # preserved verbatim in the body
        assert "argument" in text  # caveat note about runtime expansion
        assert "shell-interpolation" not in text or "Source uses Claude `!" in text

    def test_command_skill_idempotent(self, tmp_path: Path) -> None:
        _make_source(tmp_path, [])
        cmd = tmp_path / ".claude" / "commands" / "review.md"
        cmd.parent.mkdir(parents=True)
        cmd.write_text("Body.", encoding="utf-8")

        first = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        second = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert first.action == "created"
        assert second.action == "skipped"

    def test_stale_command_skill_removed(self, tmp_path: Path) -> None:
        _make_source(tmp_path, [])
        cmd = tmp_path / ".claude" / "commands" / "review.md"
        cmd.parent.mkdir(parents=True)
        cmd.write_text("Body.", encoding="utf-8")
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        # Remove the source command.
        cmd.unlink()
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert not (tmp_path / ".agents" / "skills" / "claude-command-review").exists()


class TestSkillsTranslateNoDuplicateManualFix:
    """Regression test: a source SKILL.md that already contains a
    `<!-- crossby:manual-fix -->` block (because the user round-tripped it,
    edited a previously-translated artifact, or fed it back) must not
    accumulate a second block on re-translate. The fix strips any existing
    block at parse time so SkillDefinition.body is always clean and
    render_markdown_skill writes exactly one fresh block per target."""

    def test_source_with_manual_fix_block_strips_and_replaces(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path, [])
        skill_dir = source / "leftover"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: leftover\n"
            "description: y\n"
            "allowed-tools:\n"
            "  - Read\n"
            "---\n"
            "Body.\n\n"
            "<!-- crossby:manual-fix:start -->\n"
            "## Manual migration required\n\n"
            "- stale note\n"
            "<!-- crossby:manual-fix:end -->\n",
            encoding="utf-8",
        )
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        out = (tmp_path / ".agents" / "skills" / "leftover" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        assert out.count("<!-- crossby:manual-fix:start -->") == 1
        assert "stale note" not in out
