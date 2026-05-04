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
    ClaudeSkillsWriter,
    CodexSkillsWriter,
    CopilotSkillsWriter,
    CursorSkillsWriter,
    GeminiSkillsWriter,
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

    def test_dir_with_skill_subdirs_is_managed(self, tmp_path: Path) -> None:
        d = tmp_path / "skills"
        d.mkdir()
        _make_skill(d, "my-skill")
        _make_skill(d, "another-skill")
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
        """A managed real dir (skill subdirs only) is replaced via copy without --force."""
        _make_source(tmp_path, ["skill-a"])
        target = tmp_path / ".claude" / "skills"
        target.mkdir(parents=True)
        _make_skill(target, "old-skill")
        result = self.writer.sync(_data(), tmp_path)
        assert result.action == "created"

    def test_unmanaged_real_dir_blocked_without_force(self, tmp_path: Path) -> None:
        """An unmanaged real dir (non-skill child) blocks sync without --force."""
        _make_source(tmp_path, ["skill-a"])
        target = tmp_path / ".claude" / "skills"
        target.mkdir(parents=True)
        (target / "not_a_skill.txt").write_text("junk", encoding="utf-8")
        result = self.writer.sync(_data(), tmp_path)
        assert result.action == "error"
        assert "--force" in (result.message or "")

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
            (GeminiSkillsWriter, AIToolID.GEMINI, ".gemini/skills"),
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
            (GeminiSkillsWriter, ".gemini/skills"),
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
        update_skills_gitignore(
            data, tmp_path, installed_tools=[AIToolID.CLAUDE, AIToolID.CURSOR]
        )
        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert ".claude/skills" in content
        assert ".cursor/skills" in content
        assert ".github/skills" not in content
        assert ".gemini/skills" not in content

    def test_installed_tools_none_includes_all(self, tmp_path: Path) -> None:
        data = _data()
        update_skills_gitignore(data, tmp_path, installed_tools=None)
        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert ".claude/skills" in content
        assert ".cursor/skills" in content
        assert ".github/skills" in content
        assert ".gemini/skills" in content
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

    def test_writes_skill_md_with_no_lossy_fields_for_claude_target(
        self, tmp_path: Path
    ) -> None:
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

        result = ClaudeSkillsWriter().sync(
            _data(strategy="translate"), tmp_path, dry_run=True
        )
        assert result.action in {"created", "updated"}
        assert "dry-run" in (result.message or "")
        assert not (tmp_path / ".claude" / "skills").exists()

    def test_re_translate_after_source_change(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "my-skill")

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        # Now add allowed-tools to source — manual-fix should appear.
        self._make_skill_with_frontmatter(
            source, "my-skill", allowed_tools=["Read", "Bash"]
        )

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        text = (tmp_path / ".agents" / "skills" / "my-skill" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        # Exactly one manual-fix block, with the latest content.
        assert text.count("<!-- crossby:manual-fix:start -->") == 1
        assert "Bash" in text
