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
    clear_conflicting_type,
    has_managed_marker,
    is_same_path,
    mirror_tree,
    write_if_different,
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
    _owns_whole_file = True  # whole-file overwrite → grouped by target path
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
        # real directory (e.g. syncing claude → claude). See :func:`is_same_path` for why an
        # existing symlinked target isn't a collision.
        if is_same_path(source_dir, target_dir):
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="source and target resolve to the same path; nothing to do",
            )

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
                if dry_run:
                    # Report created-vs-updated by whether the target exists,
                    # rather than always "created". (A fully-unchanged tree isn't
                    # distinguished as "skipped" here — that would need a
                    # dry-run-aware mirror pass; this is a rare symlink-failure
                    # fallback and the real run below reports skipped correctly.)
                    return SyncResult(
                        tool_id=self.tool_id,
                        concern=self.concern,
                        action="updated" if target_dir.is_dir() else "created",
                        file_path=target_dir,
                        message="copy (symlink failed, dry-run)",
                    )
                target_existed = target_dir.is_dir()
                wrote = _copy_skills_dir(source_dir, target_dir)
                write_managed_marker(target_dir)
                if not wrote and target_existed:
                    # A repeated copy-fallback run that changed nothing is an
                    # honest skip, not a phantom "created".
                    return SyncResult(
                        tool_id=self.tool_id,
                        concern=self.concern,
                        action="skipped",
                        file_path=target_dir,
                        message="copy (symlink failed)",
                    )
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="updated" if target_existed else "created",
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
        target_existed = target_dir.is_dir()
        action: Literal["created", "updated"] = "updated" if target_existed else "created"
        if dry_run:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action=action,
                file_path=target_dir,
                message="copy (dry-run)",
            )
        wrote = _copy_skills_dir(source_dir, target_dir)
        write_managed_marker(target_dir)
        if not wrote and target_existed:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="already copied",
            )
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
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

        When a source tool that has a slash-command primitive (Claude /
        Cursor today) lives in the project and ``self.tool_id`` is
        different, each of that tool's commands is wrapped as a single-file
        skill named ``<source>-command-<slug>`` so the prompt body survives
        the migration. See :mod:`crossby.sync.slash_commands` for the
        per-tool runtime caveats.
        """
        from crossby.sync.slash_commands import iter_command_skills

        command_skills: list[tuple[str, str]] = []
        for _src_path, source_tool, definition in iter_command_skills(project_root):
            if source_tool == self.tool_id:
                # Don't wrap a tool's own commands into skills for itself.
                continue
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
                if child.name in wanted_names or not child.is_dir():
                    # Only stale *directories* are cleaned (unchanged scope);
                    # ``is_dir()`` follows symlinks, so a dir symlink lands here.
                    continue
                if child.is_symlink():
                    # A stale directory symlink: ``rmtree`` raises on a symlink
                    # ("Cannot call rmtree on a symbolic link"), so unlink it.
                    child.unlink()
                else:
                    shutil.rmtree(child)
                logger.info("skills.stale_removed", path=str(child))
                removed_any = True

        skipped_all = True
        for skill_dir in skill_dirs:
            target_skill = target_dir / skill_dir.name
            # A symlinked target skill dir must be replaced, never written
            # through: ``mkdir(exist_ok=True)`` silently succeeds on a
            # symlink-to-a-directory (``is_dir()`` follows it) without replacing
            # it, so the direct ``SKILL.md`` write below would land at the
            # symlink's destination — outside the project root for
            # ``.agents/skills/demo -> /outside``. Support dirs guard their own
            # symlinks in ``_refresh_skill_support_dirs``; the skill dir itself
            # is guarded here.
            if target_skill.is_symlink():
                target_skill.unlink()
                skipped_all = False
            target_skill.mkdir(parents=True, exist_ok=True)

            source_skill_md = skill_dir / "SKILL.md"
            target_skill_md = target_skill / "SKILL.md"
            # A *leaf* SKILL.md symlink (inside an otherwise real dir) must be
            # replaced, not read/written through — write_text/read_text follow it
            # to the link's destination, which may be outside the project root.
            if target_skill_md.is_symlink():
                target_skill_md.unlink()
                skipped_all = False
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
                # A support-dir change means this skill was not a no-op, so a
                # re-sync that only touched (or should touch) scripts/references/
                # assets is reported as ``updated`` rather than ``skipped``.
                if _refresh_skill_support_dirs(skill_dir, target_skill):
                    skipped_all = False
                continue

            skipped_all = False
            target_skill_md.write_text(rendered, encoding="utf-8")
            _refresh_skill_support_dirs(skill_dir, target_skill)

        for name, rendered in command_skills:
            target_skill = target_dir / name
            # Same symlink guard as the translated-skill loop above — a
            # symlinked target skill dir would redirect the SKILL.md write
            # outside the project root.
            if target_skill.is_symlink():
                target_skill.unlink()
                skipped_all = False
            target_skill.mkdir(parents=True, exist_ok=True)
            target_skill_md = target_skill / "SKILL.md"
            # Same leaf-symlink guard as the translated-skill loop above.
            if target_skill_md.is_symlink():
                target_skill_md.unlink()
                skipped_all = False
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


def _copy_skills_dir(source_dir: Path, target_dir: Path) -> bool:
    """Copy the skills tree from source to target (one subdir per skill).

    Compare-then-write, not ``rmtree`` + ``copytree``: the previous
    implementation wiped the whole target on every run, so anything a user kept
    alongside the synced skills was destroyed on a re-sync and an interrupted
    copy left the tool with no skills at all.

    Cleanup is targeted the same way :meth:`_BaseSkillsWriter._sync_translate`
    targets it — a skill *directory* whose source disappeared is removed, while
    unrelated top-level files (and the crossby marker) are left alone. Inside a
    skill directory crossby owns everything, so those are mirrored exactly.

    Returns True when any file was written or removed.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    changed = False
    source_names: set[str] = set()

    for child in sorted(source_dir.iterdir()):
        source_names.add(child.name)
        dest = target_dir / child.name
        if child.is_dir() and not child.is_symlink():
            changed |= clear_conflicting_type(dest, want_dir=True)
            if mirror_tree(child, dest):
                changed = True
        elif child.is_file():
            changed |= clear_conflicting_type(dest, want_dir=False)
            if write_if_different(dest, child.read_bytes()):
                changed = True

    for child in target_dir.iterdir():
        if child.name in source_names or child.name == MANAGED_MARKER_NAME:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
            logger.info("skills.stale_removed", path=str(child))
            changed = True

    return changed


_SUPPORT_DIRS = ("scripts", "references", "assets")


def _refresh_skill_support_dirs(source_skill: Path, target_skill: Path) -> bool:
    """Mirror ``scripts/``, ``references/``, ``assets/`` from source to target.

    Compare-then-write via :func:`mirror_tree` instead of ``rmtree`` +
    ``copytree``: an unchanged support tree is left completely untouched, so a
    translate re-sync no longer rewrites those files on disk every run (and the
    lack of churn is reported honestly by the returned flag). ``mirror_tree``
    also preserves the executable bit, so ``scripts/`` stay runnable. A symlinked
    target support dir is replaced outright, never followed. Missing source
    subdirs are removed from the target so deleted support dirs propagate.

    Returns True when any support file was written, chmod'd, or removed.
    """
    changed = False
    for subdir in _SUPPORT_DIRS:
        source_sub = source_skill / subdir
        target_sub = target_skill / subdir
        # Never write through a symlinked target — replace it outright so the
        # mirror lands under the project root, not the symlink's destination.
        if target_sub.is_symlink():
            target_sub.unlink()
            changed = True
        if source_sub.is_dir():
            # A target of the wrong type (a file where a dir belongs) would make
            # mirror_tree's mkdir fail — clear it first.
            if target_sub.exists() and not target_sub.is_dir():
                target_sub.unlink()
                changed = True
            if mirror_tree(source_sub, target_sub):
                changed = True
        elif target_sub.is_dir():
            shutil.rmtree(target_sub)
            changed = True
        elif target_sub.exists():
            target_sub.unlink()
            changed = True
    return changed


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
    """Sync skills → .agents/skills/.

    Codex and Antigravity CLI intentionally share this directory. They are
    **not** idempotent against each other under ``translate`` — each renders
    target-specific ``SKILL.md`` content — so whichever ran second used to
    churn the tree every sync. ``run_sync`` now groups writers by physical
    target path and lets a single winner write ``.agents/skills/``;
    registration order in ``sync/__init__.py`` (Codex before Antigravity CLI)
    is the documented, stable ownership precedence. ``detect_skills_source()``'s
    scan order (`config/skills.py:_SCAN_ORDER`) separately decides which
    `AIToolID` a shared *source* directory is attributed to.
    """

    tool_id = AIToolID.CODEX
    _target_rel = SKILLS_DIR[AIToolID.CODEX]


class AntigravityCLISkillsWriter(_BaseSkillsWriter):
    """See :class:`CodexSkillsWriter` — shares `.agents/skills/` with Codex.

    ``run_sync`` grouping resolves the shared path; Codex wins when both are
    installed (registration order), and this writer runs standalone when Codex
    is absent (e.g. ``--to antigravity-cli``).
    """

    tool_id = AIToolID.ANTIGRAVITY_CLI
    _target_rel = SKILLS_DIR[AIToolID.ANTIGRAVITY_CLI]


class CopilotSkillsWriter(_BaseSkillsWriter):
    """Sync skills → .github/skills/"""

    tool_id = AIToolID.COPILOT
    _target_rel = SKILLS_DIR[AIToolID.COPILOT]
