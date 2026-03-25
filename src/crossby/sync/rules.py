"""Rules/instructions sync — keep tool-specific instruction files in sync."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

from crossby.models.config import RulesConfig, RulesTargetsConfig
from crossby.sync.base import SyncAction, SyncResult
from crossby.sync.gitignore import update_gitignore_block

logger = logging.getLogger(__name__)

MANAGED_HEADER = "<!-- managed by crossby — do not edit, changes will be overwritten -->"

# Tool name -> relative target path (from project root)
TOOL_TARGETS: dict[str, str] = {
    "claude": "CLAUDE.md",
    "cursor": ".cursorrules",
    "copilot": ".github/copilot-instructions.md",
    "gemini": "GEMINI.md",
    "codex": "AGENTS.md",
}


def sync_rules(
    project_root: Path,
    config: RulesConfig,
    *,
    dry_run: bool = False,
    force: bool = False,
    tool_filter: str | None = None,
) -> list[SyncResult]:
    """Sync rules from canonical source to all configured targets.

    Returns a list of SyncResult for each target.
    """
    source_path = project_root / config.source

    if not source_path.exists():
        return [
            SyncResult(
                target=config.source,
                action=SyncAction.ERROR,
                message=f"Source file not found: {config.source}",
            )
        ]

    targets = _active_targets(config.targets, tool_filter)
    results: list[SyncResult] = []

    for tool_name, rel_target in targets.items():
        target_path = project_root / rel_target
        result = _write_rule_target(
            source_path=source_path,
            target_path=target_path,
            rel_target=rel_target,
            strategy=config.strategy,
            dry_run=dry_run,
            force=force,
        )
        results.append(result)

    # Manage .gitignore block
    if config.gitignore and not dry_run:
        gitignore_entries = [
            rel_target
            for _, rel_target in targets.items()
            if any(
                r.target == rel_target
                and r.action in (SyncAction.CREATED, SyncAction.UPDATED, SyncAction.UP_TO_DATE)
                for r in results
            )
        ]
        update_gitignore_block(project_root, gitignore_entries)

        # Warn about already-tracked files by annotating the existing result
        for rel_target in gitignore_entries:
            if _is_git_tracked(project_root, rel_target):
                for r in results:
                    if r.target == rel_target:
                        warning = f"Warning: {rel_target} is tracked by git. Run: git rm --cached {rel_target}"
                        r.message = f"{r.message}; {warning}" if r.message else warning
                        break

    return results


def _active_targets(
    targets: RulesTargetsConfig, tool_filter: str | None
) -> dict[str, str]:
    """Return map of tool_name -> rel_target for enabled targets."""
    result: dict[str, str] = {}
    for tool_name, rel_target in TOOL_TARGETS.items():
        if tool_filter and tool_name != tool_filter:
            continue
        if getattr(targets, tool_name, False):
            result[tool_name] = rel_target
    return result


def _write_rule_target(
    *,
    source_path: Path,
    target_path: Path,
    rel_target: str,
    strategy: str,
    dry_run: bool,
    force: bool,
) -> SyncResult:
    """Write a single rule target (symlink or copy)."""
    # Circular symlink guard — compare canonical paths WITHOUT following
    # the target symlink (otherwise all existing symlinks look "circular").
    source_canonical = source_path.parent.resolve() / source_path.name
    target_canonical = target_path.parent.resolve() / target_path.name
    if source_canonical == target_canonical:
        return SyncResult(
            target=rel_target,
            action=SyncAction.SKIPPED,
            message="Source and target resolve to the same file",
        )

    # Check existing target
    if target_path.exists() or target_path.is_symlink():
        managed = _is_managed(target_path, source_path)
        if not managed:
            if force:
                if not dry_run:
                    _backup_file(target_path)
                # Fall through to create/overwrite
            else:
                return SyncResult(
                    target=rel_target,
                    action=SyncAction.SKIPPED,
                    message=(
                        "Target file exists and is not managed by crossby. "
                        "Use --force to overwrite."
                    ),
                )
        elif _is_up_to_date(target_path, source_path, strategy):
            return SyncResult(
                target=rel_target,
                action=SyncAction.UP_TO_DATE,
                dry_run=dry_run,
            )

    action = SyncAction.CREATED if not (target_path.exists() or target_path.is_symlink()) else SyncAction.UPDATED

    if dry_run:
        return SyncResult(
            target=rel_target,
            action=action,
            message=f"Would {'create' if action == SyncAction.CREATED else 'update'} via {strategy}",
            dry_run=True,
        )

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file/symlink before writing
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()

    if strategy == "symlink":
        try:
            rel_link = os.path.relpath(source_path, target_path.parent)
            target_path.symlink_to(rel_link)
        except OSError:
            logger.warning(
                "Symlink creation failed for %s, falling back to copy", rel_target
            )
            _write_copy(source_path, target_path)
            return SyncResult(
                target=rel_target,
                action=action,
                message="Symlink failed, fell back to copy",
            )
    else:
        _write_copy(source_path, target_path)

    return SyncResult(target=rel_target, action=action)


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


def _is_up_to_date(target_path: Path, source_path: Path, strategy: str) -> bool:
    """Check if the target is already up to date."""
    if strategy == "symlink" and target_path.is_symlink():
        return target_path.resolve() == source_path.resolve()
    if target_path.is_file():
        source_hash = _text_hash(source_path)
        # For copies, strip the managed header before comparing
        target_content = target_path.read_text(encoding="utf-8")
        if target_content.startswith(MANAGED_HEADER):
            target_content = target_content[len(MANAGED_HEADER) :].lstrip("\n")
        target_hash = hashlib.sha256(target_content.encode("utf-8")).hexdigest()
        return source_hash == target_hash
    return False


def _text_hash(path: Path) -> str:
    """Compute SHA-256 hex digest of a file's text content."""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _write_copy(source_path: Path, target_path: Path) -> None:
    """Copy source to target with managed header."""
    content = source_path.read_text(encoding="utf-8")
    target_path.write_text(
        MANAGED_HEADER + "\n" + content,
        encoding="utf-8",
    )


def _backup_file(target_path: Path) -> None:
    """Create a .bak backup of the target file.

    If .bak already exists, tries .bak2, .bak3, etc.
    """
    backup = target_path.with_suffix(target_path.suffix + ".bak")
    counter = 2
    while backup.exists():
        backup = target_path.with_suffix(f"{target_path.suffix}.bak{counter}")
        counter += 1

    if target_path.is_symlink():
        resolved = target_path.resolve()
        if resolved.exists():
            shutil.copy2(resolved, backup)
    else:
        shutil.copy2(target_path, backup)


def _is_git_tracked(project_root: Path, rel_path: str) -> bool:
    """Check if a file is tracked by git."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel_path],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def detect_existing_rules(project_root: Path) -> dict[str, Path]:
    """Detect existing instruction files in the project.

    Returns a map of tool_name -> path for files that exist.
    """
    found: dict[str, Path] = {}
    for tool_name, rel_target in TOOL_TARGETS.items():
        path = project_root / rel_target
        if path.exists():
            found[tool_name] = path
    return found


def suggest_source(existing: dict[str, Path]) -> str:
    """Suggest a source file based on what exists."""
    # Prefer AGENTS.md (universal standard), then CLAUDE.md, then first found
    if "codex" in existing:
        return "AGENTS.md"
    if "claude" in existing:
        return "CLAUDE.md"
    if existing:
        first_tool = next(iter(existing))
        return TOOL_TARGETS[first_tool]
    return "AGENTS.md"
