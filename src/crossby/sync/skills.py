"""Skills sync writers — distribute skills from a canonical source to each tool's directory.

Managed directory semantics (differs from agents):
A real target directory is treated as "managed" (safe to replace without --force) iff it is
empty OR every immediate child is a subdirectory containing a SKILL.md file.
Skills are organised as one-directory-per-skill, not as flat .md files — using the
agents rule (all children are .md files) would wrongly reject legitimate skills trees.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import structlog

from crossby.config.linker import create_symlink
from crossby.config.skills import SKILLS_DIR
from crossby.models.ai import AIToolID
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncData, SyncResult
from crossby.sync.file_utils import backup_path
from crossby.sync.gitignore_utils import update_managed_block

logger = structlog.get_logger()

_GITIGNORE_BLOCK_ID = "skills sync"


def update_skills_gitignore(
    data: SyncData,
    project_root: Path,
    *,
    dry_run: bool = False,
    installed_tools: list[AIToolID] | None = None,
) -> SyncResult | None:
    """Write/update the crossby-managed block in .gitignore.

    Returns a SyncResult if a change was made (or would be in dry-run), else None.
    The source directory itself is never gitignored.
    """
    if data.skills_source is None or not data.skills_gitignore:
        return None

    if installed_tools is not None:
        entries = [SKILLS_DIR[t] for t in installed_tools if t in SKILLS_DIR]
    else:
        entries = list(SKILLS_DIR.values())

    if not entries:
        return None

    gitignore_path = project_root / ".gitignore"
    action: Literal["created", "updated"] = "updated" if gitignore_path.is_file() else "created"

    changed = update_managed_block(project_root, _GITIGNORE_BLOCK_ID, entries, dry_run=dry_run)
    if not changed:
        return None

    return SyncResult(
        tool_id=None,
        concern=SyncConcern.SKILLS,
        action=action,
        file_path=gitignore_path,
        message="gitignore",
    )


def _is_managed_skills_dir(directory: Path) -> bool:
    """Return True if directory is empty or every immediate child is a SKILL.md-bearing subdir."""
    contents = list(directory.iterdir())
    if not contents:
        return True
    return all(item.is_dir() and (item / "SKILL.md").is_file() for item in contents)


class _BaseSkillsWriter(AbstractSyncWriter):
    """Common sync logic for all skills writers (directory-level symlinks).

    Each concrete writer sets ``_target_rel`` to the tool-specific skills path from SKILLS_DIR.
    All five tools use directory-level symlinks — there is no per-file variant (unlike agents).
    """

    concern = SyncConcern.SKILLS
    _target_rel: str

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if data.skills_source is None:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no skills source detected",
            )

        source_dir = project_root / data.skills_source
        if not source_dir.exists():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source directory not found: {data.skills_source}",
            )
        if not source_dir.is_dir():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source path is not a directory: {data.skills_source}",
            )

        target_dir = project_root / self._target_rel

        # Circular source/target guard — skip when source and target are literally the same
        # real directory (e.g. syncing claude → claude). Does not fire when target_dir is
        # an existing symlink that points to source_dir; that is the idempotent re-run case
        # handled below by create_symlink.
        try:
            if not target_dir.is_symlink() and source_dir.resolve() == target_dir.resolve():
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="skipped",
                    message="source and target resolve to the same path; nothing to do",
                )
        except OSError:
            pass

        # For copy strategy, guard against following a symlinked target directory —
        # copies would land in the symlink's destination, potentially outside the project.
        if data.skills_strategy == "copy" and target_dir.is_symlink():
            if not force:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    message=(
                        f"{self._target_rel} is a symlinked directory. "
                        "Refusing to copy skills into a symlink target. "
                        "Remove the symlink or re-run with --force to replace it."
                    ),
                )
            if not dry_run:
                target_dir.unlink()
                logger.info("skills.symlink_replaced", target=str(target_dir))

        # Existing real directory — may need to error, proceed, or back up.
        dir_was_cleared = False
        if target_dir.is_dir() and not target_dir.is_symlink():
            if not force:
                if not _is_managed_skills_dir(target_dir):
                    return SyncResult(
                        tool_id=self.tool_id,
                        concern=self.concern,
                        action="error",
                        message=(
                            f"{self._target_rel} exists as a directory. "
                            f"Migrate its contents to {data.skills_source} first, "
                            "or use --force to back it up and replace it."
                        ),
                    )
                # Managed fallback directory: re-sync via copy for idempotent re-runs.
                return self._sync_copy(source_dir, target_dir, dry_run=dry_run)
            dir_was_cleared = True
            if not dry_run:
                bak = backup_path(target_dir)
                shutil.copytree(str(target_dir), str(bak))
                shutil.rmtree(str(target_dir))
                logger.info("skills.dir_backed_up", original=str(target_dir), backup=str(bak))

        if data.skills_strategy == "copy":
            return self._sync_copy(source_dir, target_dir, dry_run=dry_run)

        return self._sync_symlink(
            source_dir, target_dir, dry_run=dry_run, force=force, dir_was_cleared=dir_was_cleared
        )

    def _sync_symlink(
        self,
        source_dir: Path,
        target_dir: Path,
        *,
        dry_run: bool,
        force: bool,
        dir_was_cleared: bool = False,
    ) -> SyncResult:
        if dir_was_cleared and dry_run:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="created",
                file_path=target_dir,
                message="(dry-run: would replace existing directory)",
            )
        try:
            created = create_symlink(source_dir, target_dir, force=force, dry_run=dry_run)
        except OSError as exc:
            logger.warning("skills.symlink_failed", tool=str(self.tool_id), error=str(exc))
            try:
                if not dry_run:
                    _copy_skills_dir(source_dir, target_dir)
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="created",
                    file_path=target_dir,
                    message="copy (symlink failed)",
                )
            except Exception as copy_exc:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    message=str(copy_exc),
                )

        if not created:
            if target_dir.is_symlink() and target_dir.resolve() != source_dir.resolve():
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    file_path=target_dir,
                    message="symlink points to a different location; use --force to replace",
                )
            if target_dir.exists() and not target_dir.is_symlink():
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    file_path=target_dir,
                    message=(
                        f"{self._target_rel} already exists as a regular file or directory; "
                        "use --force to replace with a symlink"
                    ),
                )
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="already linked",
            )
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action="created",
            file_path=target_dir,
        )

    def _sync_copy(
        self, source_dir: Path, target_dir: Path, *, dry_run: bool
    ) -> SyncResult:
        if dry_run:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="created",
                file_path=target_dir,
                message="copy (dry-run)",
            )
        _copy_skills_dir(source_dir, target_dir)
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action="created",
            file_path=target_dir,
        )


def _copy_skills_dir(source_dir: Path, target_dir: Path) -> None:
    """Copy skills directory structure from source to target (one subdir per skill)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(source_dir), str(target_dir), dirs_exist_ok=True)


# ---------------------------------------------------------------------------
# Concrete writers — one per tool, _target_rel sourced directly from SKILLS_DIR
# ---------------------------------------------------------------------------


class ClaudeSkillsWriter(_BaseSkillsWriter):
    """Sync skills → .claude/skills/"""

    tool_id = AIToolID.CLAUDE
    _target_rel = SKILLS_DIR[AIToolID.CLAUDE]


class CursorSkillsWriter(_BaseSkillsWriter):
    """Sync skills → .cursor/skills/"""

    tool_id = AIToolID.CURSOR
    _target_rel = SKILLS_DIR[AIToolID.CURSOR]


class CodexSkillsWriter(_BaseSkillsWriter):
    """Sync skills → .agents/skills/"""

    tool_id = AIToolID.CODEX
    _target_rel = SKILLS_DIR[AIToolID.CODEX]


class GeminiSkillsWriter(_BaseSkillsWriter):
    """Sync skills → .gemini/skills/"""

    tool_id = AIToolID.GEMINI
    _target_rel = SKILLS_DIR[AIToolID.GEMINI]


class CopilotSkillsWriter(_BaseSkillsWriter):
    """Sync skills → .github/skills/"""

    tool_id = AIToolID.COPILOT
    _target_rel = SKILLS_DIR[AIToolID.COPILOT]
