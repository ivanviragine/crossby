"""Managed .gitignore block for rules sync."""

from __future__ import annotations

from pathlib import Path

_BLOCK_START = "# >>> crossby rules sync (generated — do not edit) >>>"
_BLOCK_END = "# <<< crossby rules sync <<<"


def update_gitignore_block(project_root: Path, entries: list[str]) -> bool:
    """Add or update the managed block in .gitignore.

    Returns True if the file was modified.
    """
    gitignore = project_root / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""

    new_block = _build_block(entries) if entries else ""
    updated = _replace_block(existing, new_block)

    if updated == existing:
        return False

    gitignore.write_text(updated, encoding="utf-8")
    return True


def _build_block(entries: list[str]) -> str:
    lines = [_BLOCK_START]
    for entry in sorted(entries):
        lines.append(entry)
    lines.append(_BLOCK_END)
    return "\n".join(lines)


def _replace_block(content: str, new_block: str) -> str:
    """Replace the managed block in content, or append it."""
    lines = content.splitlines(keepends=True)
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        stripped = line.rstrip("\n\r")
        if stripped == _BLOCK_START:
            start_idx = i
        elif stripped == _BLOCK_END and start_idx is not None:
            end_idx = i
            break

    if start_idx is not None and end_idx is not None:
        if not new_block:
            # Remove the block entirely
            before = lines[:start_idx]
            after = lines[end_idx + 1 :]
            # Remove trailing blank line left by block removal
            if before and before[-1].strip() == "":
                before = before[:-1]
            return "".join(before + after)
        before = lines[:start_idx]
        after = lines[end_idx + 1 :]
        return "".join(before) + new_block + "\n" + "".join(after)

    if not new_block:
        return content

    # Append to end
    result = content
    if result and not result.endswith("\n"):
        result += "\n"
    if result and not result.endswith("\n\n"):
        result += "\n"
    result += new_block + "\n"
    return result
