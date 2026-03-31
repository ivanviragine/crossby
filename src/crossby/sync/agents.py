"""Agent sync writers — distribute agents from a canonical source to each tool's directory."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

import structlog
import yaml

from crossby.config.linker import create_symlink
from crossby.models.ai import AIToolID
from crossby.models.config import AgentsConfig, CrossbyConfig
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncResult
from crossby.sync.file_utils import backup_path
from crossby.sync.gitignore_utils import update_managed_block

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Gitignore managed-block
# ---------------------------------------------------------------------------

_GITIGNORE_BLOCK_ID = "agents sync"

# Per-tool agent directory paths (relative to project root)
_AGENT_TARGET_PATHS: dict[AIToolID, str] = {
    AIToolID.CLAUDE: ".claude/agents",
    AIToolID.COPILOT: ".github/agents",
    AIToolID.CURSOR: ".cursor/agents",
    AIToolID.GEMINI: ".gemini/agents",
    AIToolID.CODEX: ".agents",
}


def update_agents_gitignore(
    config: CrossbyConfig,
    project_root: Path,
    *,
    dry_run: bool = False,
    installed_tools: list[AIToolID] | None = None,
) -> SyncResult | None:
    """Write/update the crossby-managed block in .gitignore.

    Returns a SyncResult if a change was made (or would be in dry-run), else None.
    The source directory itself is never gitignored.
    """
    if not config.agents.enabled or not config.agents.gitignore:
        return None

    # Determine which tool target paths to include in the block
    if config.agents.targets:
        entries = [
            _AGENT_TARGET_PATHS[AIToolID(tid)]
            for tid, enabled in config.agents.targets.items()
            if enabled and AIToolID(tid) in _AGENT_TARGET_PATHS
        ]
    elif installed_tools is not None:
        entries = [
            _AGENT_TARGET_PATHS[t]
            for t in installed_tools
            if t in _AGENT_TARGET_PATHS
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

_TOOL_NAME_MAP: dict[AIToolID, dict[str, str]] = {
    AIToolID.COPILOT: {
        "Read": "read",
        "Edit": "edit",
        "Grep": "search",
        "Glob": "glob",
        "Bash": "shell",
        "WebSearch": "web_search",
        "WebFetch": "web_fetch",
    },
    AIToolID.CURSOR: {
        "Bash": "Shell",
    },
}


def _translate_tools(tools: list[str], tool_id: AIToolID) -> list[str]:
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


def _copy_agent_file(source: Path, target: Path, tool_id: AIToolID) -> None:
    """Copy one agent file to target, translating tool names."""
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
                return
        except OSError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(out, encoding="utf-8")


# ---------------------------------------------------------------------------
# Target-enabled check
# ---------------------------------------------------------------------------


def _is_target_enabled(agents_cfg: AgentsConfig, tool_id: AIToolID) -> bool:
    """Return True if this tool should receive the agent sync."""
    if not agents_cfg.targets:
        return True  # empty dict = all tools
    return agents_cfg.targets.get(str(tool_id), False)


# ---------------------------------------------------------------------------
# Base writer
# ---------------------------------------------------------------------------


class _BaseAgentsWriter(AbstractSyncWriter):
    """Common sync logic for non-Copilot agent writers (directory-level symlinks)."""

    concern = SyncConcern.AGENTS
    _target_rel: str  # e.g. ".claude/agents"

    def sync(
        self,
        config: CrossbyConfig,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        agents_cfg = config.agents

        if not agents_cfg.enabled:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no agents config",
            )

        if not _is_target_enabled(agents_cfg, self.tool_id):
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="not in targets",
            )

        source_dir = project_root / agents_cfg.source
        if not source_dir.exists():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source directory not found: {agents_cfg.source}",
            )
        if not source_dir.is_dir():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source path is not a directory: {agents_cfg.source}",
            )

        target_dir = project_root / self._target_rel

        # For copy strategy, explicitly guard against following a symlinked target
        # directory — copies would land in the symlink's destination, potentially
        # outside the project root.
        if agents_cfg.strategy == "copy" and target_dir.is_symlink():
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
                            f"Migrate its contents to {agents_cfg.source} first, "
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

        if agents_cfg.strategy == "copy":
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
                    _copy_all_agents(source_dir, target_dir, self.tool_id)
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
        _copy_all_agents(source_dir, target_dir, self.tool_id)
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action="created",
            file_path=target_dir,
        )


def _copy_all_agents(source_dir: Path, target_dir: Path, tool_id: AIToolID) -> None:
    """Copy all .md agent files from source to target, translating tool names."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in source_dir.glob("*.md"):
        _copy_agent_file(src, target_dir / src.name, tool_id)


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


class CodexAgentsWriter(_BaseAgentsWriter):
    """Sync agents → .agents/"""

    tool_id = AIToolID.CODEX
    _target_rel = ".agents"


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
        config: CrossbyConfig,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        agents_cfg = config.agents

        if not agents_cfg.enabled:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no agents config",
            )

        if not _is_target_enabled(agents_cfg, self.tool_id):
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="not in targets",
            )

        source_dir = project_root / agents_cfg.source
        if not source_dir.exists():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source directory not found: {agents_cfg.source}",
            )
        if not source_dir.is_dir():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source path is not a directory: {agents_cfg.source}",
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
                            f"Migrate its contents to {agents_cfg.source} first, "
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

        if agents_cfg.strategy == "copy":
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
                        _copy_agent_file(src, link, AIToolID.COPILOT)
                    created_count += 1
            except OSError:
                # Fallback: copy the file
                if not dry_run:
                    _copy_agent_file(src, link, AIToolID.COPILOT)
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
        if dry_run:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="created",
                file_path=target_dir,
                message="copy (dry-run)",
            )
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in source_dir.glob("*.md"):
            dest = target_dir / f"{src.stem}.agent.md"
            _copy_agent_file(src, dest, AIToolID.COPILOT)
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action="created",
            file_path=target_dir,
        )
