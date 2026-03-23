"""Sync orchestrator — port configs between AI tools."""

from __future__ import annotations

from pathlib import Path

from crossby.config.instructions import (
    INSTRUCTIONS_FILE,
    UNSUPPORTED_TOOLS,
    get_instructions_source,
    get_instructions_target,
)
from crossby.config.linker import create_symlink
from crossby.config.skills import detect_skills_source, get_skills_target
from crossby.models.ai import AIToolID
from crossby.models.sync import SyncAction, SyncResult, SyncStrategy

# Tools that support persistent allowlist config files.
_ALLOWLIST_TOOLS = frozenset({AIToolID.CLAUDE, AIToolID.CURSOR})

_ALLOWLIST_HINTS: dict[AIToolID, str] = {
    AIToolID.COPILOT: "Copilot uses --allow-tool flags, no persistent config",
    AIToolID.GEMINI: "Gemini uses --allowed-tools flags, no persistent config",
    AIToolID.CODEX: "Codex uses sandbox mode, no allowlist config",
}


def sync_configs(
    from_tool: AIToolID,
    to_tools: list[AIToolID],
    root: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    sync_instructions: bool = True,
    sync_skills: bool = True,
    sync_allowlist: bool = True,
) -> SyncResult:
    """Sync configs from *from_tool* to *to_tools*.

    Args:
        from_tool: Source tool to read configs from.
        to_tools: Target tools to sync configs to.
        root: Project root directory.
        dry_run: Preview without applying changes.
        force: Replace existing files with symlinks.
        sync_instructions: Sync instruction files.
        sync_skills: Sync skills directories.
        sync_allowlist: Sync allowlist patterns.

    Returns:
        SyncResult with actions taken and summary counts.
    """
    result = SyncResult()

    # Pre-detect sources once to avoid duplicate warnings.
    instr_source = get_instructions_source(from_tool, root) if sync_instructions else None
    skills_source = detect_skills_source(root) if sync_skills else None
    source_patterns: list[str] | None = None
    if sync_allowlist and from_tool in _ALLOWLIST_TOOLS:
        source_patterns = _read_source_allowlist(from_tool, root)

    # Emit per-source warnings once.
    if sync_instructions and instr_source is None:
        rel = INSTRUCTIONS_FILE.get(from_tool, "?")
        _add_warn(result, "instructions", f"Source file not found ({rel}), skipping instructions")

    if sync_skills and skills_source is None:
        _add_warn(result, "skills", "No skills directory found, skipping skills")

    if sync_allowlist and from_tool not in _ALLOWLIST_TOOLS:
        _add_warn(result, "allowlist", f"{from_tool.value} has no readable allowlist")
    elif sync_allowlist and source_patterns is not None and not source_patterns:
        _add_warn(result, "allowlist", f"No allowlist patterns found in {from_tool.value} config")

    # Process each target.
    for target in to_tools:
        if target == from_tool:
            continue

        if target in UNSUPPORTED_TOOLS:
            msg = f"{target.value} has no instruction/skills/allowlist config"
            result.actions.append(
                SyncAction(config_type="all", strategy=SyncStrategy.UNSUPPORTED, message=msg)
            )
            result.warnings.append(msg)
            continue

        if sync_instructions and instr_source is not None:
            _link_instructions(instr_source, target, root, result, dry_run=dry_run, force=force)

        if sync_skills and skills_source is not None:
            _link_skills(skills_source, target, root, result, dry_run=dry_run, force=force)

        if sync_allowlist and source_patterns:
            _convert_allowlist(source_patterns, from_tool, target, root, result, dry_run=dry_run)

    return result


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _add_warn(result: SyncResult, config_type: str, message: str) -> None:
    result.actions.append(
        SyncAction(config_type=config_type, strategy=SyncStrategy.WARN, message=message)
    )
    result.warnings.append(message)


def _link_instructions(
    source: Path,
    target: AIToolID,
    root: Path,
    result: SyncResult,
    *,
    dry_run: bool,
    force: bool,
) -> None:
    target_path = get_instructions_target(target, root)
    if target_path is None or target_path == source:
        return

    created = create_symlink(source, target_path, force=force, dry_run=dry_run)
    src_rel = source.relative_to(root)
    tgt_rel = target_path.relative_to(root)

    if created:
        result.linked += 1
        msg = f"{tgt_rel} -> {src_rel}"
        strategy = SyncStrategy.LINK
    elif target_path.is_symlink():
        try:
            if target_path.resolve() == source.resolve():
                msg = f"{tgt_rel} already linked"
                strategy = SyncStrategy.LINK
            else:
                msg = f"{tgt_rel} symlinked elsewhere, use force to overwrite"
                strategy = SyncStrategy.WARN
                result.warnings.append(msg)
        except OSError:
            msg = f"{tgt_rel} broken symlink, use force to overwrite"
            strategy = SyncStrategy.WARN
            result.warnings.append(msg)
    elif target_path.exists():
        msg = f"{tgt_rel} exists, use force to overwrite"
        strategy = SyncStrategy.WARN
        result.warnings.append(msg)
    else:
        msg = f"{tgt_rel} skipped (circular link guard)"
        strategy = SyncStrategy.WARN
        result.warnings.append(msg)

    result.actions.append(
        SyncAction(
            config_type="instructions",
            strategy=strategy,
            source_path=source,
            target_path=target_path,
            message=msg,
        )
    )


def _link_skills(
    source: Path,
    target: AIToolID,
    root: Path,
    result: SyncResult,
    *,
    dry_run: bool,
    force: bool,
) -> None:
    target_path = get_skills_target(target, root)
    if target_path is None:
        return

    # Don't symlink to itself.
    try:
        if target_path.resolve(strict=False) == source.resolve(strict=False):
            return
    except OSError:
        pass

    created = create_symlink(source, target_path, force=force, dry_run=dry_run)
    src_rel = source.relative_to(root)
    tgt_rel = target_path.relative_to(root)

    if created:
        result.linked += 1
        msg = f"{tgt_rel} -> {src_rel}"
        strategy = SyncStrategy.LINK
    elif target_path.is_symlink():
        try:
            if target_path.resolve() == source.resolve():
                msg = f"{tgt_rel} already linked"
                strategy = SyncStrategy.LINK
            else:
                msg = f"{tgt_rel} symlinked elsewhere, use force to overwrite"
                strategy = SyncStrategy.WARN
                result.warnings.append(msg)
        except OSError:
            msg = f"{tgt_rel} broken symlink, use force to overwrite"
            strategy = SyncStrategy.WARN
            result.warnings.append(msg)
    elif target_path.is_dir():
        msg = f"{tgt_rel} is a real directory, skipping"
        strategy = SyncStrategy.WARN
        result.warnings.append(msg)
    elif target_path.is_file():
        msg = f"{tgt_rel} is a regular file, use force to overwrite"
        strategy = SyncStrategy.WARN
        result.warnings.append(msg)
    else:
        msg = f"{tgt_rel} skipped (circular link guard)"
        strategy = SyncStrategy.WARN
        result.warnings.append(msg)

    result.actions.append(
        SyncAction(
            config_type="skills",
            strategy=strategy,
            source_path=source,
            target_path=target_path,
            message=msg,
        )
    )


def _convert_allowlist(
    patterns: list[str],
    from_tool: AIToolID,
    target: AIToolID,
    root: Path,
    result: SyncResult,
    *,
    dry_run: bool,
) -> None:
    if target in _ALLOWLIST_HINTS:
        msg = _ALLOWLIST_HINTS[target]
        result.actions.append(
            SyncAction(config_type="allowlist", strategy=SyncStrategy.WARN, message=msg)
        )
        result.warnings.append(msg)
        return

    if target not in _ALLOWLIST_TOOLS:
        return

    if not dry_run:
        try:
            _write_target_allowlist(target, root, patterns)
        except OSError as exc:
            _add_warn(result, "allowlist", f"Failed to write {target.value} allowlist: {exc}")
            return

    result.converted += 1
    target_file = _allowlist_file(target, root)
    result.actions.append(
        SyncAction(
            config_type="allowlist",
            strategy=SyncStrategy.CONVERT,
            source_path=_allowlist_file(from_tool, root),
            target_path=target_file,
            message=f"{len(patterns)} pattern(s) -> {target_file.relative_to(root)}",
        )
    )


def _read_source_allowlist(tool: AIToolID, root: Path) -> list[str]:
    if tool == AIToolID.CLAUDE:
        from crossby.config.claude_allowlist import read_allowlist

        return read_allowlist(root)
    if tool == AIToolID.CURSOR:
        from crossby.config.cursor_allowlist import read_allowlist

        return read_allowlist(root)
    return []


def _write_target_allowlist(tool: AIToolID, root: Path, patterns: list[str]) -> None:
    if tool == AIToolID.CLAUDE:
        from crossby.config.claude_allowlist import configure_allowlist

        configure_allowlist(root, patterns)
    elif tool == AIToolID.CURSOR:
        from crossby.config.cursor_allowlist import configure_allowlist

        configure_allowlist(root, patterns)


def _allowlist_file(tool: AIToolID, root: Path) -> Path:
    if tool == AIToolID.CLAUDE:
        return root / ".claude" / "settings.json"
    if tool == AIToolID.CURSOR:
        return root / ".cursor" / "cli.json"
    msg = f"No allowlist file mapping for {tool.value}"
    raise NotImplementedError(msg)
