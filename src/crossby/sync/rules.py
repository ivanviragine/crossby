"""Rules sync writers — distribute a canonical instruction file to each tool's format."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import structlog

from crossby.config.linker import create_symlink
from crossby.models.ai import AIToolID
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncData, SyncResult
from crossby.sync.file_utils import backup_path
from crossby.sync.gitignore_utils import update_managed_block
from crossby.sync.instruction_markers import (
    is_neutral_for_target,
    manual_fix_notes_for_target,
)
from crossby.sync.manual_fix import append_manual_fix_block

logger = structlog.get_logger()

MANAGED_HEADER = "<!-- managed by crossby — do not edit, changes will be overwritten -->"

# Tool name -> relative target path (from project root)
TOOL_TARGETS: dict[str, str] = {
    "claude": "CLAUDE.md",
    "cursor": ".cursorrules",
    "copilot": ".github/copilot-instructions.md",
    "codex": "AGENTS.md",
    "antigravity-cli": "AGENTS.md",
}

# ---------------------------------------------------------------------------
# Gitignore managed-block
# ---------------------------------------------------------------------------

_GITIGNORE_BLOCK_ID = "rules sync"


def update_rules_gitignore(
    data: SyncData,
    project_root: Path,
    *,
    dry_run: bool = False,
    installed_tools: list[AIToolID] | None = None,
) -> SyncResult | None:
    """Write/update the crossby-managed block in .gitignore for rules targets.

    Entries are computed from installed tools (all non-circular targets),
    so the block only covers tools that actually ran.

    Returns a SyncResult if a change was made (or would be in dry-run), else None.
    """
    if data.rules_source is None or not data.rules_gitignore:
        return None

    source_path = project_root / data.rules_source
    entries: list[str] = []
    for tool_name, rel_path in TOOL_TARGETS.items():
        if installed_tools is not None and AIToolID(tool_name) not in installed_tools:
            continue
        target_path = project_root / rel_path
        # Skip circular (source == target)
        source_canonical = source_path.parent.resolve() / source_path.name
        target_canonical = target_path.parent.resolve() / target_path.name
        if source_canonical == target_canonical:
            continue
        entries.append(rel_path)

    if not entries:
        return None

    gitignore_path = project_root / ".gitignore"
    action: Literal["created", "updated"] = "updated" if gitignore_path.is_file() else "created"

    changed = update_managed_block(
        project_root, _GITIGNORE_BLOCK_ID, sorted(entries), dry_run=dry_run
    )
    if not changed:
        return None

    return SyncResult(
        tool_id=None,
        concern=SyncConcern.RULES,
        action=action,
        file_path=gitignore_path,
        message="gitignore",
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _is_managed(target_path: Path, source_path: Path) -> bool:
    """Check if a target file is managed by crossby."""
    if target_path.is_symlink():
        return target_path.resolve() == source_path.resolve()
    if target_path.is_file():
        try:
            first_line = target_path.read_text(encoding="utf-8").split("\n", 1)[0]
            return first_line.strip() == MANAGED_HEADER
        except (OSError, UnicodeDecodeError):
            return False
    return False


def _is_up_to_date(
    target_path: Path,
    source_path: Path,
    strategy: str,
    target_tool: AIToolID,
) -> bool:
    """Check if the target is already up to date with the configured strategy."""
    if strategy == "symlink":
        if not target_path.is_symlink():
            return False
        try:
            return target_path.resolve() == source_path.resolve()
        except OSError:
            return False

    if strategy == "copy":
        if not target_path.is_file() or target_path.is_symlink():
            return False
        try:
            source_text = source_path.read_text(encoding="utf-8")
            target_content = target_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        if not target_content.startswith(MANAGED_HEADER):
            return False
        after_header = target_content[len(MANAGED_HEADER) :]
        target_body = after_header[1:] if after_header.startswith("\n") else after_header
        expected_body = _render_copy_body(source_text, target_tool)
        expected_hash = hashlib.sha256(expected_body.encode("utf-8")).hexdigest()
        target_hash = hashlib.sha256(target_body.encode("utf-8")).hexdigest()
        return expected_hash == target_hash

    return False


def _render_copy_body(source_text: str, target_tool: AIToolID) -> str:
    """Body that should be written below MANAGED_HEADER for ``target_tool``.

    When source content references surfaces that belong to a different tool
    (e.g. a Claude ``CLAUDE.md`` mentioning ``ExitPlanMode`` being copied to
    ``.cursorrules``), append a manual-fix block enumerating the lossy edges.
    """
    notes = manual_fix_notes_for_target(source_text, target_tool)
    if not notes:
        return source_text
    return append_manual_fix_block(source_text, notes)


def _write_copy(source_path: Path, target_path: Path, target_tool: AIToolID) -> None:
    """Copy source to target with managed header and optional manual-fix block."""
    source_text = source_path.read_text(encoding="utf-8")
    body = _render_copy_body(source_text, target_tool)
    target_path.write_text(MANAGED_HEADER + "\n" + body, encoding="utf-8")


def _backup_file(target_path: Path) -> None:
    """Create a .bak backup of the target file (numbered if .bak exists)."""
    dest = backup_path(target_path)
    if target_path.is_symlink():
        link_target = os.readlink(target_path)
        os.symlink(link_target, dest)
    else:
        shutil.copy2(target_path, dest)


def _warn_if_git_tracked(project_root: Path, rel_path: str) -> None:
    """Log a warning if the target file is already tracked by git."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel_path],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.warning(
                "rules.target_git_tracked",
                path=rel_path,
                hint=f"git rm --cached {rel_path}",
            )
    except (OSError, subprocess.SubprocessError):
        pass


# ---------------------------------------------------------------------------
# Base rules writer
# ---------------------------------------------------------------------------


class _BaseRulesWriter(AbstractSyncWriter):
    """Common sync logic for rules writers."""

    concern = SyncConcern.RULES
    _target_rel: str  # e.g. "CLAUDE.md"

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if data.rules_source is None:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no rules source detected",
            )

        source_path = project_root / data.rules_source
        if not source_path.exists():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source file not found: {data.rules_source}",
            )

        target_path = project_root / self._target_rel

        # Circular symlink guard — compare canonical paths WITHOUT following
        # the target symlink (otherwise all existing symlinks look "circular").
        source_canonical = source_path.parent.resolve() / source_path.name
        target_canonical = target_path.parent.resolve() / target_path.name
        if source_canonical == target_canonical:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="source and target resolve to the same file",
            )

        # Decide effective strategy. Symlink is the configured default, but
        # we force a copy whenever the source mentions surfaces specific to a
        # tool other than ``self.tool_id`` — that way every target gets a
        # locally-edited file with a manual-fix block instead of silently
        # propagating Claude/Cursor/etc-only semantics.
        effective_strategy = data.rules_strategy
        force_copy_reason: str | None = None
        try:
            source_text = source_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            source_text = ""
        if source_text and not is_neutral_for_target(source_text, self.tool_id):
            if effective_strategy == "symlink":
                force_copy_reason = "foreign markers in source"
            effective_strategy = "copy"

        # Check existing target
        target_existed = target_path.exists() or target_path.is_symlink()
        if target_existed:
            managed = _is_managed(target_path, source_path)
            if not managed:
                if not force:
                    return SyncResult(
                        tool_id=self.tool_id,
                        concern=self.concern,
                        action="skipped",
                        file_path=target_path,
                        message="target exists and is not managed by crossby; use --force",
                    )
                if not dry_run:
                    _backup_file(target_path)
            elif _is_up_to_date(target_path, source_path, effective_strategy, self.tool_id):
                _warn_if_git_tracked(project_root, self._target_rel)
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="skipped",
                    file_path=target_path,
                    message="already linked",
                )

        action: Literal["created", "updated"] = "updated" if target_existed else "created"

        if dry_run:
            dry_run_message = f"(dry-run: would sync via {effective_strategy})"
            if force_copy_reason:
                dry_run_message = f"(dry-run: would sync via copy — {force_copy_reason})"
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action=action,
                file_path=target_path,
                message=dry_run_message,
            )

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing file/symlink before writing
        if target_existed:
            target_path.unlink()

        if effective_strategy == "symlink":
            try:
                ok = create_symlink(source_path, target_path)
            except OSError:
                ok = False
            if not ok:
                logger.warning(
                    "rules.symlink_failed",
                    tool=str(self.tool_id),
                    target=self._target_rel,
                )
                _write_copy(source_path, target_path, self.tool_id)
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action=action,
                    file_path=target_path,
                    message="copy (symlink failed)",
                )
        else:
            _write_copy(source_path, target_path, self.tool_id)

        _warn_if_git_tracked(project_root, self._target_rel)

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=target_path,
            message=force_copy_reason,
        )


# ---------------------------------------------------------------------------
# Concrete writers
# ---------------------------------------------------------------------------


class ClaudeRulesWriter(_BaseRulesWriter):
    tool_id = AIToolID.CLAUDE
    _target_rel = "CLAUDE.md"


class CursorRulesWriter(_BaseRulesWriter):
    tool_id = AIToolID.CURSOR
    _target_rel = ".cursorrules"


class CopilotRulesWriter(_BaseRulesWriter):
    tool_id = AIToolID.COPILOT
    _target_rel = ".github/copilot-instructions.md"


class CodexRulesWriter(_BaseRulesWriter):
    tool_id = AIToolID.CODEX
    _target_rel = "AGENTS.md"


class AntigravityCLIRulesWriter(_BaseRulesWriter):
    tool_id = AIToolID.ANTIGRAVITY_CLI
    _target_rel = "AGENTS.md"


# ---------------------------------------------------------------------------
# Detection helpers (used by crossby init)
# ---------------------------------------------------------------------------


def detect_existing_rules(project_root: Path) -> dict[str, Path]:
    """Detect existing instruction files in the project."""
    found: dict[str, Path] = {}
    for tool_name, rel_target in TOOL_TARGETS.items():
        path = project_root / rel_target
        if path.exists() or path.is_symlink():
            found[tool_name] = path
    return found


def suggest_source(existing: dict[str, Path]) -> str:
    """Suggest a source file based on what exists."""
    if "codex" in existing:
        return "AGENTS.md"
    if "claude" in existing:
        return "CLAUDE.md"
    if existing:
        first_tool = next(iter(existing))
        return TOOL_TARGETS[first_tool]
    return "AGENTS.md"
