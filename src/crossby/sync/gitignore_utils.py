"""Shared .gitignore managed-block utility.

Provides a single function to update/create managed blocks in .gitignore,
used by both rules and agents sync modules.
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger()


def update_managed_block(
    project_root: Path,
    block_id: str,
    entries: list[str],
    *,
    dry_run: bool = False,
) -> bool:
    """Update or create a managed block in .gitignore.

    The block is delimited by start/end markers derived from ``block_id``:

        # >>> crossby {block_id} (generated — do not edit) >>>
        entry1
        entry2
        # <<< crossby {block_id} <<<

    Args:
        project_root: Project root directory.
        block_id: Identifier for the block (e.g. ``"rules sync"``).
        entries: Lines to place inside the block.
        dry_run: If True, compute the result without writing.

    Returns:
        True if a change was made (or would be in dry-run), False otherwise.
    """
    if not entries:
        return False

    block_start = f"# >>> crossby {block_id} (generated — do not edit) >>>"
    block_end = f"# <<< crossby {block_id} <<<"
    block = "\n".join([block_start, *entries, block_end])

    gitignore_path = project_root / ".gitignore"
    existing = gitignore_path.read_text(encoding="utf-8") if gitignore_path.is_file() else ""

    if block_start in existing:
        lines = existing.splitlines()
        start_idx = lines.index(block_start)
        if block_end not in lines[start_idx + 1 :]:
            # Malformed block: end marker is missing — replace from the orphaned
            # start marker to EOF with a fresh block.
            logger.warning(
                "crossby managed block in .gitignore is missing end marker; "
                "replacing from orphaned start marker",
                block_id=block_id,
            )
            prefix = "\n".join(lines[:start_idx])
            sep = "\n" if prefix else ""
            new_content = prefix + sep + block + "\n"
        else:
            new_lines: list[str] = []
            inside = False
            for line in lines:
                if line == block_start:
                    inside = True
                    new_lines.append(block)
                    continue
                if inside:
                    if line == block_end:
                        inside = False
                    continue
                new_lines.append(line)
            new_content = "\n".join(new_lines)
            if not new_content.endswith("\n"):
                new_content += "\n"
    else:
        sep = "\n" if existing and not existing.endswith("\n") else ""
        new_content = existing + sep + block + "\n"

    if new_content == existing:
        return False

    if not dry_run:
        gitignore_path.write_text(new_content, encoding="utf-8")

    return True
