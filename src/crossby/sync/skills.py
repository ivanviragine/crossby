"""Skills sync writers — distribute skills from a canonical source to each tool's directory.

Managed directory semantics (differs from agents):
A real target directory is treated as "managed" (safe to replace without --force) iff it is
empty OR every immediate child is a subdirectory containing a SKILL.md file.
Skills are organised as one-directory-per-skill, not as flat .md files — using the
agents rule (all children are .md files) would wrongly reject legitimate skills trees.

Strategies:
- ``symlink`` (default): the source skills tree is symlinked into the tool's path so
  edits propagate everywhere. Requires every tool to accept the same SKILL.md shape.
- ``copy``: physical copy of the tree. No content rewriting.
- ``translate``: per-skill copy that runs each ``SKILL.md`` through
  :func:`crossby.sync.agent_models.translate_skill_for_target`, appending a
  ``crossby:manual-fix`` block when the source declares fields the target tool
  doesn't natively honour (e.g. Claude ``allowed-tools`` for a Codex target).
  Support directories (``scripts/``, ``references/``, ``assets/``) are copied
  verbatim.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Literal

import structlog

from crossby.config.linker import create_symlink
from crossby.config.skills import SKILLS_DIR
from crossby.models.ai import AIToolID
from crossby.sync.agent_models import (
    parse_markdown_skill,
    render_markdown_skill,
    translate_skill_for_target,
)
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncData, SyncResult
from crossby.sync.file_utils import (
    MANAGED_MARKER_NAME,
    backup_path,
    has_managed_marker,
    write_managed_marker,
)
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

    # Never gitignore the canonical source directory itself.
    source_dir = Path(data.skills_source)
    try:
        source_entry = source_dir.relative_to(project_root).as_posix()
    except ValueError:
        source_entry = source_dir.as_posix()
    entries = [entry for entry in entries if Path(entry).as_posix() != source_entry]

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
    """Return True if the directory is empty or carries the crossby ownership marker.

    The shape check (every child is a ``<name>/SKILL.md``-bearing subdir) is
    indistinguishable from a hand-curated native skills tree, so it can no
    longer stand alone — without the explicit ``.crossby-managed`` marker,
    a user's natively-organized skills directory would be wiped on copy
    strategy. The marker is the only authoritative ownership signal.
    """
    contents = [c for c in directory.iterdir() if c.name != MANAGED_MARKER_NAME]
    if not contents:
        return True
    return has_managed_marker(directory)


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

        # For copy/translate strategies, guard against following a symlinked target
        # directory — writes would land in the symlink's destination, potentially
        # outside the project.
        if data.skills_strategy in {"copy", "translate"} and target_dir.is_symlink():
            if not force:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    message=(
                        f"{self._target_rel} is a symlinked directory. "
                        "Refusing to write skills into a symlink target. "
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
                # Managed fallback directory: re-sync via the configured strategy
                # so subsequent runs preserve translate/copy semantics.
                if data.skills_strategy == "translate":
                    return self._sync_translate(
                        source_dir, target_dir, project_root=project_root, dry_run=dry_run
                    )
                return self._sync_copy(source_dir, target_dir, dry_run=dry_run)
            dir_was_cleared = True
            if not dry_run:
                bak = backup_path(target_dir)
                shutil.copytree(str(target_dir), str(bak))
                shutil.rmtree(str(target_dir))
                logger.info("skills.dir_backed_up", original=str(target_dir), backup=str(bak))

        if data.skills_strategy == "translate":
            return self._sync_translate(
                source_dir, target_dir, project_root=project_root, dry_run=dry_run
            )

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
            # Mark the copy-fallback output so the next sync recognizes its own
            # work and doesn't refuse the dir as "not crossby-managed".
            try:
                if not dry_run:
                    _copy_skills_dir(source_dir, target_dir)
                    write_managed_marker(target_dir)
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

    def _sync_copy(self, source_dir: Path, target_dir: Path, *, dry_run: bool) -> SyncResult:
        if dry_run:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="created",
                file_path=target_dir,
                message="copy (dry-run)",
            )
        _copy_skills_dir(source_dir, target_dir)
        write_managed_marker(target_dir)
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action="created",
            file_path=target_dir,
        )

    def _sync_translate(
        self,
        source_dir: Path,
        target_dir: Path,
        *,
        project_root: Path,
        dry_run: bool,
    ) -> SyncResult:
        """Per-skill copy with target-aware SKILL.md rewriting.

        For each ``<skill>/`` under ``source_dir``: parse SKILL.md, run it
        through :func:`translate_skill_for_target` for ``self.tool_id``, render
        with any manual-fix block appended, and write to
        ``<target_dir>/<skill>/SKILL.md``. Support directories (``scripts/``,
        ``references/``, ``assets/``) are copied verbatim. Hash-based
        idempotency. Stale skill subdirectories whose source disappeared are
        removed.

        When the target is *not* Claude and ``.claude/commands/`` exists
        under ``project_root``, each Claude slash command is also wrapped
        as a single-file skill named ``claude-command-<slug>`` so the
        prompt body survives the migration. See
        :mod:`crossby.sync.slash_commands` for the conversion details.
        """
        from crossby.sync.slash_commands import iter_command_skills

        command_skills: list[tuple[str, str]] = []
        if self.tool_id != AIToolID.CLAUDE:
            for _src_path, definition in iter_command_skills(project_root):
                rendered = render_markdown_skill(definition)
                command_skills.append((definition.name, rendered))

        skill_dirs = [
            child
            for child in sorted(source_dir.iterdir())
            if child.is_dir() and (child / "SKILL.md").is_file()
        ]
        target_existed = target_dir.is_dir()
        # Even when there's nothing to translate, an existing target may have
        # stale entries from a previous run; we still want to walk it once.
        nothing_to_write = not skill_dirs and not command_skills
        if nothing_to_write and not target_existed:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="no skills to translate",
            )

        action: Literal["created", "updated"] = "updated" if target_existed else "created"

        if dry_run:
            # Render in memory so manual-fix blocks are visible to plan
            # summarization (without this, --plan undercounts the manual review
            # surface for translate dry-runs).
            from crossby.sync.manual_fix import has_manual_fix_block

            manual_fix_count = 0
            for skill_dir in skill_dirs:
                definition = parse_markdown_skill(
                    (skill_dir / "SKILL.md").read_text(encoding="utf-8"),
                    fallback_name=skill_dir.name,
                )
                translated = translate_skill_for_target(definition, self.tool_id)
                rendered = render_markdown_skill(translated)
                if has_manual_fix_block(rendered):
                    manual_fix_count += 1
            for _name, rendered in command_skills:
                if has_manual_fix_block(rendered):
                    manual_fix_count += 1
            message = (
                f"translated (dry-run, {manual_fix_count} manual-fix)"
                if manual_fix_count
                else "translated (dry-run)"
            )
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action=action,
                file_path=target_dir,
                message=message,
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        write_managed_marker(target_dir)
        wanted_names = {skill_dir.name for skill_dir in skill_dirs} | {
            name for name, _ in command_skills
        }
        # Stale cleanup
        removed_any = False
        if target_dir.is_dir():
            for child in target_dir.iterdir():
                if child.is_dir() and child.name not in wanted_names:
                    shutil.rmtree(child)
                    logger.info("skills.stale_removed", path=str(child))
                    removed_any = True

        skipped_all = True
        for skill_dir in skill_dirs:
            target_skill = target_dir / skill_dir.name
            target_skill.mkdir(parents=True, exist_ok=True)

            source_skill_md = skill_dir / "SKILL.md"
            target_skill_md = target_skill / "SKILL.md"
            definition = parse_markdown_skill(
                source_skill_md.read_text(encoding="utf-8"),
                fallback_name=skill_dir.name,
            )
            translated = translate_skill_for_target(definition, self.tool_id)
            rendered = render_markdown_skill(translated)

            if target_skill_md.is_file() and (
                hashlib.sha256(
                    target_skill_md.read_text(encoding="utf-8").encode("utf-8")
                ).hexdigest()
                == hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            ):
                # SKILL.md unchanged — but support dirs may still need a refresh.
                _refresh_skill_support_dirs(skill_dir, target_skill)
                continue

            skipped_all = False
            target_skill_md.write_text(rendered, encoding="utf-8")
            _refresh_skill_support_dirs(skill_dir, target_skill)

        for name, rendered in command_skills:
            target_skill = target_dir / name
            target_skill.mkdir(parents=True, exist_ok=True)
            target_skill_md = target_skill / "SKILL.md"
            if target_skill_md.is_file() and (
                hashlib.sha256(
                    target_skill_md.read_text(encoding="utf-8").encode("utf-8")
                ).hexdigest()
                == hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            ):
                continue
            skipped_all = False
            target_skill_md.write_text(rendered, encoding="utf-8")

        if skipped_all and target_existed and not removed_any:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="already translated",
            )

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=target_dir,
            message="translated",
        )


def _copy_skills_dir(source_dir: Path, target_dir: Path) -> None:
    """Copy skills directory structure from source to target (one subdir per skill)."""
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(str(source_dir), str(target_dir))


_SUPPORT_DIRS = ("scripts", "references", "assets")


def _refresh_skill_support_dirs(source_skill: Path, target_skill: Path) -> None:
    """Mirror ``scripts/``, ``references/``, ``assets/`` from source to target.

    Idempotent: replaces the target subdir if it exists. Missing source subdirs
    are also removed from target so deleted support dirs propagate.
    """
    for subdir in _SUPPORT_DIRS:
        source_sub = source_skill / subdir
        target_sub = target_skill / subdir
        if source_sub.is_dir():
            if target_sub.exists():
                shutil.rmtree(target_sub)
            shutil.copytree(str(source_sub), str(target_sub))
        elif target_sub.exists():
            shutil.rmtree(target_sub)


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
