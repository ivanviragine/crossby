"""Agent sync writers — distribute agents from a canonical source to each tool's directory."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Literal

import structlog
import yaml

from crossby.config.linker import create_symlink
from crossby.models.ai import AIToolID
from crossby.sync.agent_models import (
    parse_markdown_agent,
    parse_toml_agent,
    render_toml_agent,
)
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncData, SyncResult
from crossby.sync.file_utils import backup_path
from crossby.sync.gitignore_utils import update_managed_block

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Gitignore managed-block
# ---------------------------------------------------------------------------

_GITIGNORE_BLOCK_ID = "agents sync"

# Per-tool agent directory paths (relative to project root).
# Codex uses ``.codex/agents`` for custom-agent TOML files (per Codex docs);
# ``.agents/skills/`` is a *skills* directory, handled by sync/skills.py.
_AGENT_TARGET_PATHS: dict[str, str] = {
    "claude": ".claude/agents",
    "copilot": ".github/agents",
    "cursor": ".cursor/agents",
    "gemini": ".gemini/agents",
    "codex": ".codex/agents",
}


def update_agents_gitignore(
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
    if data.agents_source is None or not data.agents_gitignore:
        return None

    # Determine which tool target paths to include in the block
    if installed_tools is not None:
        entries = [
            _AGENT_TARGET_PATHS[str(t)]
            for t in installed_tools
            if str(t) in _AGENT_TARGET_PATHS
        ]
    else:
        entries = list(_AGENT_TARGET_PATHS.values())

    if not entries:
        return None

    gitignore_path = project_root / ".gitignore"
    action: Literal["created", "updated"] = "updated" if gitignore_path.is_file() else "created"

    changed = update_managed_block(
        project_root, _GITIGNORE_BLOCK_ID, entries, dry_run=dry_run
    )
    if not changed:
        return None

    return SyncResult(
        tool_id=None,
        concern=SyncConcern.AGENTS,
        action=action,
        file_path=gitignore_path,
        message="gitignore",
    )


# ---------------------------------------------------------------------------
# Tool name translation (copy strategy)
# ---------------------------------------------------------------------------

_TOOL_NAME_MAP: dict[str, dict[str, str]] = {
    "copilot": {
        "Read": "read",
        "Edit": "edit",
        "Grep": "search",
        "Glob": "glob",
        "Bash": "shell",
        "WebSearch": "web_search",
        "WebFetch": "web_fetch",
    },
    "cursor": {
        "Bash": "Shell",
    },
}


def _translate_tools(tools: list[str], tool_id: str) -> list[str]:
    """Map canonical tool names to tool-specific names."""
    mapping = _TOOL_NAME_MAP.get(tool_id, {})
    return [mapping.get(t, t) for t in tools]


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------


def _parse_frontmatter(content: str) -> tuple[dict[str, object] | None, str]:
    """Split YAML frontmatter from markdown body.

    Returns (fm_dict, body) where fm_dict is None when there is no frontmatter
    or when it could not be parsed (missing closing delimiter or invalid YAML).
    Callers must treat None as "no parseable frontmatter" and copy verbatim.
    """
    if not content.startswith("---\n"):
        return None, content
    end = content.find("\n---\n", 4)
    if end == -1:
        return None, content
    try:
        raw = yaml.safe_load(content[4:end])
        if not isinstance(raw, dict):
            # Non-dict YAML (list, scalar, etc.) — copy verbatim to avoid data loss
            return None, content
        fm: dict[str, object] = raw
    except yaml.YAMLError:
        return None, content
    return fm, content[end + 5:]


def _render_frontmatter(fm: dict[str, object], body: str) -> str:
    """Reassemble frontmatter + body into a markdown string."""
    return f"---\n{yaml.dump(fm, default_flow_style=False, sort_keys=False)}---\n{body}"


def detect_legacy_codex_agents(project_root: Path) -> Path | None:
    """Return the legacy `.agents/` path when it carries old codex-agent content.

    Crossby ≤ 0.2.x synced codex agents to ``<project>/.agents/`` (either as a
    directory symlink to the source or as a markdown copy). The current path
    per upstream Codex docs is ``<project>/.codex/agents/<name>.toml``;
    `.agents/skills/` is the Codex *skills* root and is left alone.

    Detection: the legacy path is reported when ``.agents`` is a symlink
    (the old default), OR when it's a real directory containing one or more
    top-level ``*.md`` files (old copy fallback). A directory containing
    only ``skills/`` (current Codex skills layout) returns ``None``.
    """
    legacy = project_root / ".agents"
    if legacy.is_symlink():
        return legacy
    if not legacy.is_dir():
        return None
    try:
        for child in legacy.iterdir():
            if child.is_file() and child.suffix == ".md":
                return legacy
    except OSError:
        return None
    return None


def _warn_legacy_codex_agents_path(project_root: Path) -> None:
    """Log a one-shot warning when the legacy `.agents/` path is present.

    Non-destructive: never auto-deletes, never blocks the sync. Users can
    remove the directory or symlink at their convenience now that codex
    agents live at ``.codex/agents/``.
    """
    legacy = detect_legacy_codex_agents(project_root)
    if legacy is None:
        return
    logger.warning(
        "agents.legacy_codex_path",
        path=str(legacy),
        hint=(
            f"`.agents/` is no longer the codex agents target — codex agents "
            f"now sync to `.codex/agents/`. The legacy path was left untouched; "
            f"remove `{legacy}` once you've confirmed nothing else relies on it."
        ),
    )


def _copy_agent_file(source: Path, target: Path, tool_id: str) -> bool:
    """Copy one agent file to target, translating tool names.

    Returns True when the target was written or rewritten, False when the
    on-disk content was already byte-identical to the rendered output
    (idempotent re-run).
    """
    content = source.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(content)
    if isinstance(fm, dict):
        raw_tools = fm.get("tools")
        if isinstance(raw_tools, list):
            fm["tools"] = _translate_tools([str(t) for t in raw_tools], tool_id)
        out = _render_frontmatter(fm, body)
    else:
        # Frontmatter could not be parsed — copy verbatim to avoid data loss
        out = content
    if target.is_file():
        try:
            if target.read_text(encoding="utf-8") == out:
                return False
        except OSError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(out, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Base writer
# ---------------------------------------------------------------------------


class _BaseAgentsWriter(AbstractSyncWriter):
    """Common sync logic for non-Copilot agent writers (directory-level symlinks)."""

    concern = SyncConcern.AGENTS
    _target_rel: str  # e.g. ".claude/agents"

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if data.agents_source is None:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no agents source detected",
            )

        source_dir = project_root / data.agents_source
        if not source_dir.exists():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source directory not found: {data.agents_source}",
            )
        if not source_dir.is_dir():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source path is not a directory: {data.agents_source}",
            )

        target_dir = project_root / self._target_rel

        # For copy strategy, explicitly guard against following a symlinked target
        # directory — copies would land in the symlink's destination, potentially
        # outside the project root.
        if data.agents_strategy == "copy" and target_dir.is_symlink():
            if not force:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    message=(
                        f"{self._target_rel} is a symlinked directory. "
                        "Refusing to copy agents into a symlink target. "
                        "Remove the symlink or re-run with --force to replace it."
                    ),
                )
            if not dry_run:
                target_dir.unlink()
                logger.info("agents.symlink_replaced", target=str(target_dir))

        # Existing real directory — may need to error, proceed, or back up
        dir_was_cleared = False
        if target_dir.is_dir() and not target_dir.is_symlink():
            if not force:
                # Check if it's a managed fallback directory
                # (empty, or only .md files = previously synced via copy fallback).
                contents = list(target_dir.iterdir())
                is_managed_fallback = not contents or all(
                    f.suffix == ".md" and not f.is_symlink() for f in contents
                )
                if not is_managed_fallback:
                    return SyncResult(
                        tool_id=self.tool_id,
                        concern=self.concern,
                        action="error",
                        message=(
                            f"{self._target_rel} exists as a directory. "
                            f"Migrate its contents to {data.agents_source} first, "
                            "or use --force to back it up and replace it."
                        ),
                    )
                # Managed fallback directory: proceed with copy for re-entrancy.
                return self._sync_copy(source_dir, target_dir, dry_run=dry_run)
            dir_was_cleared = True
            if not dry_run:
                bak = backup_path(target_dir)
                shutil.copytree(str(target_dir), str(bak))
                shutil.rmtree(str(target_dir))
                logger.info("agents.dir_backed_up", original=str(target_dir), backup=str(bak))

        if data.agents_strategy == "copy":
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
        # When force cleared a real directory (dry_run skips the removal), still
        # report "created" — the symlink would succeed once the directory is gone.
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
            logger.warning("agents.symlink_failed", tool=str(self.tool_id), error=str(exc))
            # Fallback: copy
            try:
                if not dry_run:
                    _copy_all_agents(source_dir, target_dir, str(self.tool_id))
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
        wrote = _copy_all_agents(source_dir, target_dir, str(self.tool_id))
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


def _copy_all_agents(source_dir: Path, target_dir: Path, tool_id: str) -> bool:
    """Copy all .md agent files from source to target, translating tool names.

    Returns True when at least one file was written or rewritten, False
    when every file was already up to date (idempotent re-run).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    wrote_any = False
    for src in source_dir.glob("*.md"):
        if _copy_agent_file(src, target_dir / src.name, tool_id):
            wrote_any = True
    return wrote_any


# ---------------------------------------------------------------------------
# Concrete writers
# ---------------------------------------------------------------------------


class ClaudeAgentsWriter(_BaseAgentsWriter):
    """Sync agents → .claude/agents/"""

    tool_id = AIToolID.CLAUDE
    _target_rel = ".claude/agents"


class CursorAgentsWriter(_BaseAgentsWriter):
    """Sync agents → .cursor/agents/"""

    tool_id = AIToolID.CURSOR
    _target_rel = ".cursor/agents"


class GeminiAgentsWriter(_BaseAgentsWriter):
    """Sync agents → .gemini/agents/"""

    tool_id = AIToolID.GEMINI
    _target_rel = ".gemini/agents"


class CodexAgentsWriter(AbstractSyncWriter):
    """Sync agents → .codex/agents/<name>.toml.

    Codex agents use a TOML schema (``name``, ``description``,
    ``developer_instructions`` plus optional ``model``,
    ``model_reasoning_effort``, ``sandbox_mode``). When the source is a
    different tool (Claude/Cursor/Gemini/Copilot all use markdown +
    YAML frontmatter), we translate per file via
    :mod:`crossby.sync.agent_models`. Lossy fields (``permissionMode:
    plan``, ``allowed-tools``, etc.) become a ``crossby:manual-fix``
    block at the end of the rendered ``developer_instructions``.

    Idempotent: identical rendered TOML produces ``action="skipped"``.
    Stale cleanup: ``.toml`` files whose source ``.md`` is gone are
    removed, matching the behaviour of CopilotAgentsWriter for
    ``.agent.md`` files.
    """

    tool_id = AIToolID.CODEX
    concern = SyncConcern.AGENTS
    _target_rel = ".codex/agents"

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if data.agents_source is None:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no agents source detected",
            )

        source_dir = project_root / data.agents_source
        if not source_dir.exists() or not source_dir.is_dir():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source directory not found: {data.agents_source}",
            )

        _warn_legacy_codex_agents_path(project_root)

        target_dir = project_root / self._target_rel

        if target_dir.is_symlink():
            if not force:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    message=(
                        f"{self._target_rel} exists as a symlink. "
                        "Remove it or rerun with --force to replace it."
                    ),
                )
            if not dry_run:
                target_dir.unlink()
                logger.info("agents.symlink_replaced", path=str(target_dir))

        # When a real directory exists with non-managed content, refuse without --force.
        if target_dir.is_dir():
            unmanaged = [
                f
                for f in target_dir.iterdir()
                if not (f.name.endswith(".toml") and f.is_file())
            ]
            if unmanaged and not force:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    message=(
                        f"{self._target_rel} contains unmanaged content; "
                        f"migrate it first or rerun with --force."
                    ),
                )
            if unmanaged and force and not dry_run:
                bak = backup_path(target_dir)
                shutil.copytree(str(target_dir), str(bak))
                shutil.rmtree(str(target_dir))
                logger.info(
                    "agents.dir_backed_up",
                    original=str(target_dir),
                    backup=str(bak),
                )

        return self._translate_all(
            source_dir, target_dir, dry_run=dry_run
        )

    def _source_files(self, source_dir: Path) -> list[Path]:
        # Source can be either Codex TOML (round-trip) or markdown agents.
        return sorted(
            [p for p in source_dir.glob("*.md") if p.is_file()]
            + [p for p in source_dir.glob("*.toml") if p.is_file()]
        )

    def _render_for_target(self, source: Path) -> str:
        if source.suffix == ".toml":
            definition = parse_toml_agent(
                source.read_text(encoding="utf-8"), fallback_name=source.stem
            )
        else:
            definition = parse_markdown_agent(
                source.read_text(encoding="utf-8"), fallback_name=source.stem
            )
        return render_toml_agent(definition)

    def _translate_all(
        self, source_dir: Path, target_dir: Path, *, dry_run: bool
    ) -> SyncResult:
        sources = self._source_files(source_dir)
        if not sources:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="no agents to translate",
            )

        target_existed = target_dir.is_dir()
        action: Literal["created", "updated"] = "updated" if target_existed else "created"
        wrote_any = False
        skipped_all = True

        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)

        # Stale cleanup — remove .toml outputs whose source is gone.
        if not dry_run and target_dir.is_dir():
            wanted = {f"{src.stem}.toml" for src in sources}
            for existing in target_dir.glob("*.toml"):
                if existing.name not in wanted:
                    existing.unlink()
                    logger.info("agents.stale_removed", path=str(existing))

        for src in sources:
            rendered = self._render_for_target(src)
            dest = target_dir / f"{src.stem}.toml"
            if dest.is_file():
                try:
                    if (
                        hashlib.sha256(dest.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                        == hashlib.sha256(rendered.encode("utf-8")).hexdigest()
                    ):
                        continue
                except OSError:
                    pass
            skipped_all = False
            if not dry_run:
                dest.write_text(rendered, encoding="utf-8")
            wrote_any = True

        if skipped_all and target_existed:
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
            message="translated to TOML"
            + (" (dry-run)" if dry_run and wrote_any else ""),
        )


class CopilotAgentsWriter(AbstractSyncWriter):
    """Sync agents → .github/agents/ using file-level symlinks (.agent.md extension).

    Copilot requires the ``.agent.md`` extension so we create per-file symlinks
    rather than a directory symlink.  Each sync run also removes stale
    ``.agent.md`` symlinks whose source ``.md`` file no longer exists.
    """

    tool_id = AIToolID.COPILOT
    concern = SyncConcern.AGENTS
    _target_rel = ".github/agents"

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if data.agents_source is None:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no agents source detected",
            )

        source_dir = project_root / data.agents_source
        if not source_dir.exists():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source directory not found: {data.agents_source}",
            )
        if not source_dir.is_dir():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source path is not a directory: {data.agents_source}",
            )

        target_dir = project_root / self._target_rel

        # If the target exists as a symlink, error by default to avoid writing into the
        # symlink target (which may be outside the project). With --force, replace the
        # symlink with a real directory under the project root.
        if target_dir.is_symlink():
            if not force:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    message=(
                        f"{self._target_rel} exists as a symlink. Remove it or use --force "
                        "to replace it with a real directory before syncing agents."
                    ),
                )
            if not dry_run:
                target_dir.unlink()
                logger.info("agents.symlink_replaced", path=str(target_dir))
        # For Copilot, the target is always a real directory containing per-file symlinks.
        # Error if an UNMANAGED real directory exists (has non-agent.md content) unless
        # force is set.  Managed content (prior sync) is always re-entrant.
        elif target_dir.is_dir():
            has_unmanaged = any(
                f for f in target_dir.iterdir() if not f.name.endswith(".agent.md")
            )
            if has_unmanaged:
                if not force:
                    return SyncResult(
                        tool_id=self.tool_id,
                        concern=self.concern,
                        action="error",
                        message=(
                            f"{self._target_rel} exists as a directory with unmanaged content. "
                            f"Migrate its contents to {data.agents_source} first, "
                            "or use --force to replace it."
                        ),
                    )
                # force=True: back up and clear the directory before re-syncing
                if not dry_run:
                    bak = backup_path(target_dir)
                    shutil.copytree(str(target_dir), str(bak))
                    shutil.rmtree(str(target_dir))
                    logger.info(
                        "agents.dir_backed_up",
                        original=str(target_dir),
                        backup=str(bak),
                    )

        if data.agents_strategy == "copy":
            return self._sync_copy(source_dir, target_dir, dry_run=dry_run)

        return self._sync_symlinks(source_dir, target_dir, dry_run=dry_run, force=force)

    def _sync_symlinks(
        self, source_dir: Path, target_dir: Path, *, dry_run: bool, force: bool
    ) -> SyncResult:
        """Create/update per-file .agent.md symlinks; clean up stale ones."""
        dir_newly_created = False
        if not dry_run:
            if not target_dir.is_dir():
                target_dir.mkdir(parents=True, exist_ok=True)
                dir_newly_created = True

        source_stems = {f.stem for f in source_dir.glob("*.md")}

        # Stale cleanup: remove managed *.agent.md outputs whose source is gone.
        # The .agent.md extension is crossby-specific; both symlinks and regular
        # files (copy-fallback outputs) are treated as managed and eligible for removal.
        if not dry_run and target_dir.is_dir():
            for link in list(target_dir.glob("*.agent.md")):
                original_stem = link.name.removesuffix(".agent.md")
                if original_stem not in source_stems:
                    os.unlink(link)
                    logger.info("agents.stale_removed", link=str(link))

        # Create/update symlinks for each source file
        created_count = 0
        for src in source_dir.glob("*.md"):
            link = target_dir / f"{src.stem}.agent.md"
            try:
                if create_symlink(src, link, force=force, dry_run=dry_run):
                    created_count += 1
                elif link.is_symlink() and link.resolve() != src.resolve():
                    return SyncResult(
                        tool_id=self.tool_id,
                        concern=self.concern,
                        action="error",
                        file_path=link,
                        message=f"{link.name} symlink points to a different location; use --force to replace",
                    )
                elif link.exists() and not link.is_symlink():
                    # Regular file at the link path — treat as a managed copy-fallback
                    # output (.agent.md is crossby-specific) and keep it up to date.
                    if not dry_run:
                        _copy_agent_file(src, link, "copilot")
                    created_count += 1
            except OSError:
                # Fallback: copy the file
                if not dry_run:
                    _copy_agent_file(src, link, "copilot")
                    created_count += 1

        if created_count == 0 and not dir_newly_created:
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
        target_dir.mkdir(parents=True, exist_ok=True)
        wrote_any = False
        for src in source_dir.glob("*.md"):
            dest = target_dir / f"{src.stem}.agent.md"
            if _copy_agent_file(src, dest, "copilot"):
                wrote_any = True
        if not wrote_any and target_existed:
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
