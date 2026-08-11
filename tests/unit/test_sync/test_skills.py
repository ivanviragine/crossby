"""Tests for skills sync writers, gitignore helper, and run_sync integration."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Literal

import pytest

from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern, SyncData
from crossby.sync.file_utils import mirror_tree
from crossby.sync.skills import (
    _GITIGNORE_BLOCK_ID,
    AntigravityCLISkillsWriter,
    ClaudeSkillsWriter,
    CodexSkillsWriter,
    CopilotSkillsWriter,
    CursorSkillsWriter,
    _copy_skills_dir,
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


def test_managed_marker_symlink_is_not_trusted_and_replaced(tmp_path: Path) -> None:
    # Issue #83: a symlinked .crossby-managed marker must not be trusted as an
    # ownership signal, and writing the marker must replace the symlink rather
    # than write through it to an arbitrary external file.
    from crossby.sync.file_utils import (
        MANAGED_MARKER_NAME,
        has_managed_marker,
        write_managed_marker,
    )

    d = tmp_path / "dir"
    d.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("SECRET", encoding="utf-8")
    os.symlink(external, d / MANAGED_MARKER_NAME)

    assert has_managed_marker(d) is False  # a symlinked marker is not trusted
    write_managed_marker(d)  # must replace the symlink, not write through it
    assert not (d / MANAGED_MARKER_NAME).is_symlink()
    assert has_managed_marker(d) is True
    assert external.read_text(encoding="utf-8") == "SECRET"  # external file untouched


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
        """A crossby-marked real dir is refreshed via copy without --force."""
        _make_source(tmp_path, ["skill-a"])
        target = tmp_path / ".claude" / "skills"
        target.mkdir(parents=True)
        _make_skill(target, "old-skill")
        (target / ".crossby-managed").write_text("", encoding="utf-8")
        result = self.writer.sync(_data(), tmp_path)
        # The directory already existed, so the honest action is "updated".
        assert result.action == "updated"
        assert (target / "skill-a" / "SKILL.md").is_file()
        assert not (target / "old-skill").exists()

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


class TestMirrorTree:
    """Direct unit coverage for ``mirror_tree`` change detection edge cases."""

    def test_creating_empty_target_dir_counts_as_change(self, tmp_path: Path) -> None:
        # Issue #83: an empty source dir with the same mode mkdir produces writes
        # no child and needs no chmod, yet the real run still *creates* the target
        # dir. Both the real run and the dry-run must report that as a change so a
        # newly-added empty scripts/ (or --plan) isn't misreported as skipped.
        source = tmp_path / "src"
        source.mkdir()  # empty source dir
        target = tmp_path / "dst"

        assert mirror_tree(source, target, dry_run=True) is True  # would create it
        assert not target.exists()  # dry-run touched nothing

        assert mirror_tree(source, target) is True  # created the empty dir
        assert target.is_dir()

        # Now that it exists and matches, a re-sync is a genuine no-op.
        assert mirror_tree(source, target) is False
        assert mirror_tree(source, target, dry_run=True) is False

    def test_read_only_target_dir_is_updatable(self, tmp_path: Path) -> None:
        # Issue #83: a preserved read-only source mode (0555) is applied to the
        # target on the first sync; a later sync that must write a changed child
        # into that still-read-only dir would raise PermissionError when creating
        # the temp file. mirror_tree must temporarily grant owner-write, then
        # restore the source mode — and an unchanged re-sync must stay skipped.
        source = tmp_path / "src"
        source.mkdir()
        (source / "run.sh").write_text("v1", encoding="utf-8")
        os.chmod(source, 0o555)  # read-only source dir
        target = tmp_path / "dst"

        assert mirror_tree(source, target) is True
        assert stat.S_IMODE(target.stat().st_mode) == 0o555
        assert (target / "run.sh").read_text(encoding="utf-8") == "v1"

        # An unchanged re-sync of the read-only dir reports no change (no churn).
        assert mirror_tree(source, target) is False
        assert stat.S_IMODE(target.stat().st_mode) == 0o555

        # Change source content (relax the source to edit, then restore 0555).
        os.chmod(source, 0o755)
        (source / "run.sh").write_text("v2", encoding="utf-8")
        os.chmod(source, 0o555)

        # The update must not raise PermissionError writing into the 0555 target.
        assert mirror_tree(source, target) is True
        assert (target / "run.sh").read_text(encoding="utf-8") == "v2"
        assert stat.S_IMODE(target.stat().st_mode) == 0o555  # mode restored

        # Restore writable modes so pytest's tmp_path teardown can remove them.
        os.chmod(source, 0o755)
        os.chmod(target, 0o755)

    def test_no_execute_target_dir_is_updatable(self, tmp_path: Path) -> None:
        # Creating a directory entry needs owner write AND execute — granting only
        # owner-write leaves a 0400/0600 target un-traversable, so the temp-file
        # write still raises PermissionError. mirror_tree grants the full owner
        # rwx triple for the mirror, then restores the source mode.
        source = tmp_path / "src"
        source.mkdir()
        (source / "run.sh").write_text("v1", encoding="utf-8")
        target = tmp_path / "dst"
        assert mirror_tree(source, target) is True

        # Change source content, then clamp the target to r-- (no write, no exec).
        (source / "run.sh").write_text("v2", encoding="utf-8")
        os.chmod(target, 0o400)

        assert mirror_tree(source, target) is True  # must not raise PermissionError
        assert (target / "run.sh").read_text(encoding="utf-8") == "v2"

        os.chmod(target, 0o755)  # cleanup safety for tmp_path teardown

    def test_dir_mode_change_under_relax_counts_as_change(self, tmp_path: Path) -> None:
        # A dir-mode-only change must be reported even when the temporary
        # owner-write relax happens to land the target on the source mode: a 0555
        # target under a 0755 source is relaxed to 0755, so _sync_mode sees no
        # diff — the change must be detected from the pre-relax baseline, not that
        # post-relax view, or the run reports skipped despite a real chmod.
        source = tmp_path / "src"
        source.mkdir()  # default (umask) mode, e.g. 0755
        (source / "run.sh").write_text("v1", encoding="utf-8")
        target = tmp_path / "dst"
        assert mirror_tree(source, target) is True  # initial create

        # Clamp only the *target dir* mode to 0555 (read-only); content unchanged.
        os.chmod(target, 0o555)

        # The re-sync restores the target dir to the source mode (0755) — a real
        # change, so it must report True, not a phantom skipped.
        assert mirror_tree(source, target) is True
        assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(source.stat().st_mode)

        # Idempotent now that modes match.
        assert mirror_tree(source, target) is False


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

    def test_unchanged_support_dirs_report_skipped_and_dont_rewrite(self, tmp_path: Path) -> None:
        # Issue #83 (D): a translate re-sync must NOT rewrite unchanged support
        # dirs (scripts/, references/, assets/), and must report skipped.
        source = _make_source(tmp_path, [])
        skill = self._make_skill_with_frontmatter(source, "my-skill")
        (skill / "scripts").mkdir()
        (skill / "scripts" / "run.sh").write_text("#!/bin/sh\necho hi\n", encoding="utf-8")

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        target_script = tmp_path / ".agents" / "skills" / "my-skill" / "scripts" / "run.sh"
        mtime = target_script.stat().st_mtime_ns

        second = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert second.action == "skipped"
        assert target_script.stat().st_mtime_ns == mtime  # not rewritten

    def test_changed_support_file_reports_updated(self, tmp_path: Path) -> None:
        # A support-file change (SKILL.md unchanged) must surface as updated, not
        # skipped — the change flag is threaded through _refresh_skill_support_dirs.
        source = _make_source(tmp_path, [])
        skill = self._make_skill_with_frontmatter(source, "my-skill")
        (skill / "scripts").mkdir()
        script = skill / "scripts" / "run.sh"
        script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        script.write_text("#!/bin/sh\necho CHANGED\n", encoding="utf-8")

        result = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert result.action == "updated"
        target_script = tmp_path / ".agents" / "skills" / "my-skill" / "scripts" / "run.sh"
        assert "CHANGED" in target_script.read_text(encoding="utf-8")

    def test_added_empty_support_dir_reports_change(self, tmp_path: Path) -> None:
        # Issue #83: adding an empty support dir (e.g. scripts/) mutates the tree
        # on a real run (mkdir), so the dry-run/--plan must not report skipped and
        # the real run must report updated — creating the dir is a change.
        source = _make_source(tmp_path, [])
        skill = self._make_skill_with_frontmatter(source, "my-skill")
        assert CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path).action == "created"

        # Add an EMPTY scripts/ dir to the source after the first sync.
        (skill / "scripts").mkdir()

        plan = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path, dry_run=True)
        assert plan.action != "skipped"  # --plan must not hide the pending mkdir

        result = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert result.action == "updated"
        assert (tmp_path / ".agents" / "skills" / "my-skill" / "scripts").is_dir()

        # Now that it exists, a further re-sync is an honest no-op.
        assert CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path).action == "skipped"

    def test_executable_bit_preserved_on_translate(self, tmp_path: Path) -> None:
        # mirror_tree must propagate the executable bit so scripts/ stay runnable
        # (write_bytes would otherwise drop it).
        source = _make_source(tmp_path, [])
        skill = self._make_skill_with_frontmatter(source, "my-skill")
        (skill / "scripts").mkdir()
        script = skill / "scripts" / "run.sh"
        script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        os.chmod(script, 0o755)

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)

        target_script = tmp_path / ".agents" / "skills" / "my-skill" / "scripts" / "run.sh"
        assert target_script.stat().st_mode & 0o111, "executable bit dropped on translate"

    def test_directory_mode_preserved_on_translate(self, tmp_path: Path) -> None:
        # mirror_tree must propagate *directory* permission bits, not just file
        # bits — mkdir creates support dirs under the umask, which would widen a
        # private 0700 tree or drop group access from a 0750 one. Nested dirs too.
        source = _make_source(tmp_path, [])
        skill = self._make_skill_with_frontmatter(source, "my-skill")
        (skill / "assets" / "private").mkdir(parents=True)
        (skill / "assets" / "private" / "secret.txt").write_text("shh", encoding="utf-8")
        os.chmod(skill / "assets", 0o750)
        os.chmod(skill / "assets" / "private", 0o700)

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)

        target_assets = tmp_path / ".agents" / "skills" / "my-skill" / "assets"
        assert target_assets.stat().st_mode & 0o777 == 0o750, "top-level dir mode dropped"
        assert (target_assets / "private").stat().st_mode & 0o777 == 0o700, (
            "nested private dir mode dropped"
        )

        # Idempotent: modes already match, so a re-sync chmods nothing → skipped.
        second = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert second.action == "skipped"

    def test_directory_setgid_bit_preserved_on_translate(self, tmp_path: Path) -> None:
        # mirror_tree must propagate *special* permission bits (setgid/setuid/
        # sticky), not just the rwx triples — masking to 0o777 would both hide a
        # setgid-only difference during comparison (fresh mkdir target has no
        # setgid) and strip it on chmod, regressing copytree's copystat behavior.
        # A source 02770 group-shared scripts/ tree (setgid keeps files created
        # by collaborators group-owned) must land as 02770 at the target.
        source = _make_source(tmp_path, [])
        skill = self._make_skill_with_frontmatter(source, "my-skill")
        (skill / "scripts").mkdir()
        (skill / "scripts" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        os.chmod(skill / "scripts", 0o2770)

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)

        target_scripts = tmp_path / ".agents" / "skills" / "my-skill" / "scripts"
        assert target_scripts.stat().st_mode & stat.S_ISGID, "setgid bit dropped on dir"
        assert stat.S_IMODE(target_scripts.stat().st_mode) == 0o2770, (
            "special + rwx bits not fully preserved"
        )

        # Idempotent: modes already match, so a re-sync chmods nothing → skipped.
        second = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert second.action == "skipped"

    def test_copy_dry_run_reports_skipped_when_unchanged(self, tmp_path: Path) -> None:
        # Issue #83: a normal copy dry-run on an unchanged target reports skipped,
        # not a phantom "updated".
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "my-skill")
        CodexSkillsWriter().sync(_data(strategy="copy"), tmp_path)  # real copy
        result = CodexSkillsWriter().sync(_data(strategy="copy"), tmp_path, dry_run=True)
        assert result.action == "skipped"

    def test_translate_dry_run_reports_skipped_when_unchanged(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "my-skill")
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)  # real translate
        result = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path, dry_run=True)
        assert result.action == "skipped"

    def test_translate_dry_run_detects_symlinked_skill_dir(self, tmp_path: Path) -> None:
        # Issue #83: when the per-skill target dir is a symlink, the real run
        # replaces it — so a dry-run must NOT report skipped.
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "my-skill")
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        target_skill = tmp_path / ".agents" / "skills" / "my-skill"
        shutil.rmtree(target_skill)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_text("# outside\n", encoding="utf-8")
        os.symlink(outside, target_skill)

        result = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path, dry_run=True)
        assert result.action != "skipped"  # the real run would replace the symlink

    def test_symlinked_target_skill_dir_is_replaced_not_followed(self, tmp_path: Path) -> None:
        # A target *skill* dir that is itself a symlink must be replaced, never
        # written through: mkdir(exist_ok=True) silently succeeds on a
        # symlink-to-dir without replacing it, so the direct SKILL.md write would
        # escape the project root (.agents/skills/my-skill -> /outside).
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "my-skill")

        # First sync establishes the managed target skill dir with a real SKILL.md.
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        target_skill = tmp_path / ".agents" / "skills" / "my-skill"
        assert (target_skill / "SKILL.md").is_file()

        # Swap the target skill dir for a symlink pointing outside the project.
        shutil.rmtree(target_skill)
        outside = tmp_path / "outside"
        outside.mkdir()
        os.symlink(outside, target_skill)

        # Re-sync: the symlink is replaced and SKILL.md is written into a real
        # dir under the project, not through the link to `outside`.
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)

        assert not target_skill.is_symlink()
        assert (target_skill / "SKILL.md").is_file()
        assert not (outside / "SKILL.md").exists()

    def test_leaf_skill_md_symlink_is_replaced_not_followed(self, tmp_path: Path) -> None:
        # A *leaf* SKILL.md symlink inside an otherwise real managed skill dir
        # must be replaced, never written through: write_text follows the link to
        # its destination, which may be outside the project root.
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "my-skill")
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        target_md = tmp_path / ".agents" / "skills" / "my-skill" / "SKILL.md"
        assert target_md.is_file()

        # Replace SKILL.md with a symlink pointing outside the project.
        target_md.unlink()
        outside = tmp_path / "outside.md"
        outside.write_text("SHOULD NOT BE OVERWRITTEN", encoding="utf-8")
        os.symlink(outside, target_md)

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)

        assert not target_md.is_symlink()
        assert target_md.is_file()
        assert outside.read_text(encoding="utf-8") == "SHOULD NOT BE OVERWRITTEN"

    def test_stale_skill_dir_symlink_is_unlinked_not_rmtree(self, tmp_path: Path) -> None:
        # A stale skill dir that is a *symlink* must be unlinked during cleanup —
        # shutil.rmtree() raises on a symlink ("Cannot call rmtree on a symbolic
        # link"), which would abort the whole translate sync.
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "keep")
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        target_dir = tmp_path / ".agents" / "skills"

        # A stale skill dir (no matching source) that is a directory symlink.
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keepme.txt").write_text("keep", encoding="utf-8")
        os.symlink(outside, target_dir / "stale")

        # Must not raise; the stale symlink is unlinked, its destination left intact.
        result = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert result.action != "error"
        assert not (target_dir / "stale").exists()  # symlink removed
        assert (outside / "keepme.txt").is_file()  # destination untouched

    def test_stale_broken_skill_symlink_is_removed(self, tmp_path: Path) -> None:
        # A *broken* stale skill symlink (source skill deleted, link dangling)
        # must also be removed — it fails is_dir(), so it previously lingered and
        # the run reported skipped despite a stale link remaining.
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "keep")
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        target_dir = tmp_path / ".agents" / "skills"

        broken = target_dir / "stale-broken"
        os.symlink(tmp_path / "does-not-exist", broken)  # dangling link

        result = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        assert result.action != "error"
        assert result.action != "skipped"  # a stale removal is a change
        assert not broken.is_symlink()  # dangling link removed

    def test_translate_dry_run_detects_broken_stale_symlink(self, tmp_path: Path) -> None:
        # The dry-run stale detector must count a broken stale symlink the real
        # run would remove — otherwise --plan reports skipped while the real sync
        # unlinks it. Mirrors the real cleanup predicate (is_symlink before is_dir).
        source = _make_source(tmp_path, [])
        self._make_skill_with_frontmatter(source, "keep")
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        target_dir = tmp_path / ".agents" / "skills"

        broken = target_dir / "stale-broken"
        os.symlink(tmp_path / "does-not-exist", broken)

        result = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path, dry_run=True)
        assert result.action != "skipped"  # the real run would remove the link
        assert broken.is_symlink()  # dry-run touched nothing

    def test_symlinked_ancestor_dir_is_refused(self, tmp_path: Path) -> None:
        # Issue #83: a symlinked *parent* (.agents -> outside) must be refused for
        # every strategy — mkdir(parents=True)/create_symlink would follow it and
        # write skills outside the project root.
        _make_source(tmp_path, ["my-skill"])
        outside = tmp_path / "outside"
        outside.mkdir()
        os.symlink(os.path.relpath(outside, tmp_path), tmp_path / ".agents")
        w = CodexSkillsWriter()
        for strategy in ("symlink", "copy", "translate"):
            result = w.sync(_data(strategy=strategy), tmp_path)
            assert result.action == "error", strategy
            assert "symlinked directory" in (result.message or ""), strategy
        # Nothing was written through the parent symlink to its destination.
        assert not any(outside.iterdir())

    def test_copy_skills_dir_dry_run_compares_without_writing(self, tmp_path: Path) -> None:
        # Issue #83: _copy_skills_dir(dry_run=True) is a compare-only pass (same
        # logic as the real copy, no drift) — it detects an unchanged tree
        # (→ skipped), a byte change, a mode-only change, a stale skill dir, and a
        # missing target, all without touching disk.
        source = _make_source(tmp_path, ["a", "b"])
        target = tmp_path / "target"
        assert _copy_skills_dir(source, target, dry_run=True) is True  # target absent
        assert not target.exists()  # nothing written in dry-run

        # Real copy, then an unchanged dry-run reports no change.
        assert _copy_skills_dir(source, target) is True
        assert _copy_skills_dir(source, target, dry_run=True) is False

        # A byte change in a SKILL.md → change (dry-run leaves it unwritten).
        (target / "a" / "SKILL.md").write_text("# a changed\n", encoding="utf-8")
        assert _copy_skills_dir(source, target, dry_run=True) is True
        assert (target / "a" / "SKILL.md").read_text(encoding="utf-8") == "# a changed\n"

        # A mode-only difference is detected (this was the reviewer's drift point).
        (target / "a" / "SKILL.md").write_text("# a\n", encoding="utf-8")
        assert _copy_skills_dir(source, target, dry_run=True) is False
        os.chmod(source / "a" / "SKILL.md", 0o600)
        assert _copy_skills_dir(source, target, dry_run=True) is True
        os.chmod(source / "a" / "SKILL.md", 0o644)

        # A stale skill dir in the target → change.
        (target / "stale").mkdir()
        assert _copy_skills_dir(source, target, dry_run=True) is True
        assert (target / "stale").is_dir()  # not removed in dry-run

    def test_copy_skills_dir_removes_stale_symlinks(self, tmp_path: Path) -> None:
        # Issue #83: a stale skill-dir symlink — live (points to a dir) or broken
        # — must be removed by the copy path (matching translate), not left behind
        # while the copy returns skipped. A live dir symlink fails
        # `not is_symlink()`; a broken one fails `is_dir()`; both previously slipped
        # through. A symlink to a real file is left alone (not a stale skill dir).
        source = _make_source(tmp_path, ["a"])
        target = tmp_path / "target"
        assert _copy_skills_dir(source, target) is True
        assert _copy_skills_dir(source, target) is False  # unchanged → no-op

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_text("# outside\n", encoding="utf-8")
        live = target / "stale-live"
        os.symlink(outside, live)  # live directory symlink
        broken = target / "stale-broken"
        os.symlink(tmp_path / "does-not-exist", broken)  # broken symlink

        # dry-run reports the pending removals, touching nothing.
        assert _copy_skills_dir(source, target, dry_run=True) is True
        assert live.is_symlink() and broken.is_symlink()

        # real run unlinks both stale symlinks (no rmtree crash on a link).
        assert _copy_skills_dir(source, target) is True
        assert not live.is_symlink()
        assert not broken.is_symlink()
        # The real skill survives, and the live link's destination is not deleted.
        assert (target / "a" / "SKILL.md").is_file()
        assert (outside / "SKILL.md").is_file()

    def test_symlinked_target_support_dir_is_replaced_not_followed(self, tmp_path: Path) -> None:
        # A target support dir that is a symlink must be replaced, never written
        # through (which would land files outside the project root).
        source = _make_source(tmp_path, [])
        skill = self._make_skill_with_frontmatter(source, "my-skill")
        (skill / "scripts").mkdir()
        (skill / "scripts" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")

        # First sync establishes the managed target with a real scripts/ dir.
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        target_scripts = tmp_path / ".agents" / "skills" / "my-skill" / "scripts"
        assert target_scripts.is_dir()

        # Swap the target scripts/ for a symlink pointing outside the project.
        shutil.rmtree(target_scripts)
        outside = tmp_path / "outside"
        outside.mkdir()
        os.symlink(outside, target_scripts)

        # Re-sync: the refresh replaces the symlink and mirrors into a real dir.
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)

        assert not target_scripts.is_symlink()
        assert (target_scripts / "run.sh").is_file()
        # The symlink target outside the project was not written into.
        assert not (outside / "run.sh").exists()

    def test_nested_symlink_in_support_dir_is_replaced_not_followed(self, tmp_path: Path) -> None:
        # mirror_tree must not follow a *nested* target symlink either — following
        # it would land writes/chmods outside the mirror root.
        source = _make_source(tmp_path, [])
        skill = self._make_skill_with_frontmatter(source, "my-skill")
        (skill / "scripts" / "sub").mkdir(parents=True)
        (skill / "scripts" / "sub" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        target_sub = tmp_path / ".agents" / "skills" / "my-skill" / "scripts" / "sub"
        assert target_sub.is_dir()

        # Swap the nested target dir for a symlink pointing outside the project.
        shutil.rmtree(target_sub)
        outside = tmp_path / "outside"
        outside.mkdir()
        os.symlink(outside, target_sub)

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)

        assert not target_sub.is_symlink()
        assert (target_sub / "run.sh").is_file()
        assert not (outside / "run.sh").exists()

    def test_source_dir_symlink_in_support_dir_is_dereferenced(self, tmp_path: Path) -> None:
        # A *source* support tree may hold a symlink to a directory (e.g.
        # scripts/shared -> ../common). mirror_tree must dereference and copy it
        # as a real directory — matching the old copytree(symlinks=False) — so a
        # translated skill doesn't silently lose those scripts/assets.
        source = _make_source(tmp_path, [])
        skill = self._make_skill_with_frontmatter(source, "my-skill")
        (skill / "scripts").mkdir()
        (skill / "scripts" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        # Shared directory living outside scripts/, reached via a directory symlink.
        shared = skill / "shared"
        (shared / "nested").mkdir(parents=True)
        (shared / "helper.sh").write_text("#!/bin/sh\necho shared\n", encoding="utf-8")
        (shared / "nested" / "deep.txt").write_text("deep", encoding="utf-8")
        os.symlink(shared, skill / "scripts" / "linked")

        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)

        target_linked = tmp_path / ".agents" / "skills" / "my-skill" / "scripts" / "linked"
        assert target_linked.is_dir()
        assert not target_linked.is_symlink()  # dereferenced, not reproduced as a link
        helper = (target_linked / "helper.sh").read_text(encoding="utf-8")
        assert helper == "#!/bin/sh\necho shared\n"
        assert (target_linked / "nested" / "deep.txt").read_text(encoding="utf-8") == "deep"

        # Idempotent: an unchanged dereferenced tree reports skipped on re-sync.
        second = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
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

    def test_command_skill_dry_run_detects_symlinked_dir(self, tmp_path: Path) -> None:
        # Issue #83: when a *command*-skill target dir is a symlink, the real run
        # unlinks it (skipped_all = False) — so a dry-run must NOT report skipped,
        # even when the linked SKILL.md matches the rendered content byte-for-byte.
        # Mirrors the translated-skill dir-symlink guard.
        _make_source(tmp_path, [])
        cmd = tmp_path / ".claude" / "commands" / "review.md"
        cmd.parent.mkdir(parents=True)
        cmd.write_text("Body.", encoding="utf-8")

        # Real sync writes the rendered command skill; move it aside and symlink
        # the target to it so the linked SKILL.md matches byte-for-byte.
        CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path)
        target_cmd = tmp_path / ".agents" / "skills" / "claude-command-review"
        outside = tmp_path / "outside"
        shutil.move(str(target_cmd), str(outside))
        os.symlink(outside, target_cmd)

        result = CodexSkillsWriter().sync(_data(strategy="translate"), tmp_path, dry_run=True)
        assert result.action != "skipped"  # the real run would unlink the symlink

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
