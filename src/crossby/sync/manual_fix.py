"""Manual-fix block formatting for translated artifacts.

When a writer can't faithfully translate a source field (e.g. Claude
``allowed-tools`` has no Codex permission equivalent, or a Claude
``permissionMode`` value doesn't map cleanly), the lossy edge needs to be
visible in the generated file — not just printed to the terminal where it
scrolls off-screen.

This module owns the format. Blocks are wrapped in stable HTML comments so
they survive markdown rendering, are easy for humans to read, and can be
detected and stripped on the next sync run for idempotency.

Format::

    <!-- crossby:manual-fix:start -->
    ## Manual migration required

    - Note 1.
    - Note 2.
    <!-- crossby:manual-fix:end -->

The same shape works inside a TOML multi-line string (Codex
``developer_instructions``) because the markers are valid markdown comments
and don't interact with TOML escape rules.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


MANUAL_FIX_START = "<!-- crossby:manual-fix:start -->"
MANUAL_FIX_END = "<!-- crossby:manual-fix:end -->"
MANUAL_FIX_HEADING = "## Manual migration required"


_BLOCK_RE = re.compile(
    re.escape(MANUAL_FIX_START) + r".*?" + re.escape(MANUAL_FIX_END),
    re.DOTALL,
)


@dataclass(frozen=True)
class ManualFixNote:
    """A single manual-migration note with optional category metadata.

    ``category`` is a short tag (e.g. ``"permissionMode"``, ``"allowed-tools"``)
    used in reports; the user-facing message is the body.
    """

    message: str
    category: str | None = None

    def render(self) -> str:
        return self.message.strip()


def format_manual_fix_block(notes: Sequence[str | ManualFixNote]) -> str:
    """Render a manual-fix block. Returns ``""`` when there are no notes.

    Items are rendered as a bulleted list under the standard heading.
    Multi-line notes are kept as-is (each note is a single bullet whose body
    can wrap).
    """
    rendered = [_render_note(note) for note in notes if _render_note(note)]
    if not rendered:
        return ""
    bullets = "\n".join(f"- {note}" for note in rendered)
    return f"{MANUAL_FIX_START}\n{MANUAL_FIX_HEADING}\n\n{bullets}\n{MANUAL_FIX_END}"


def append_manual_fix_block(body: str, notes: Sequence[str | ManualFixNote]) -> str:
    """Append a manual-fix block to ``body``, separated by a blank line.

    No-op when ``notes`` is empty. ``body`` is right-stripped to avoid double
    blank lines, then a single blank line is inserted before the block.
    """
    block = format_manual_fix_block(notes)
    if not block:
        return body
    trimmed = body.rstrip()
    if not trimmed:
        return block + "\n"
    return f"{trimmed}\n\n{block}\n"


def strip_manual_fix_blocks(content: str) -> str:
    """Remove every manual-fix block from ``content``.

    Used by writers before re-translating a file so the next run replaces the
    previous run's block instead of stacking them.
    """
    cleaned = _BLOCK_RE.sub("", content)
    # Collapse leftover triple-blank-lines created by the removal.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.rstrip() + ("\n" if content.endswith("\n") else "")


def find_manual_fix_blocks(content: str) -> list[str]:
    """Return the bodies of every manual-fix block in ``content``.

    The returned strings exclude the start/end markers. Useful for surfacing
    pending manual work in a report without re-rendering each artifact.
    """
    return [match.group(0) for match in _BLOCK_RE.finditer(content)]


def has_manual_fix_block(content: str) -> bool:
    """Return True if ``content`` contains at least one manual-fix block."""
    return bool(_BLOCK_RE.search(content))


def _render_note(note: str | ManualFixNote) -> str:
    if isinstance(note, ManualFixNote):
        return note.render()
    return note.strip()


__all__ = [
    "MANUAL_FIX_END",
    "MANUAL_FIX_HEADING",
    "MANUAL_FIX_START",
    "ManualFixNote",
    "append_manual_fix_block",
    "find_manual_fix_blocks",
    "format_manual_fix_block",
    "has_manual_fix_block",
    "strip_manual_fix_blocks",
]
