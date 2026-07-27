"""Shared file utilities for sync writers."""

from __future__ import annotations

import shutil
from pathlib import Path

# Marker file written into managed target directories (agents/skills) so
# subsequent syncs can distinguish a directory crossby owns from one a user
# (or another tool) populated by hand. Without this marker, a native-looking
# layout — e.g. ``.claude/agents/*.md`` or ``.claude/skills/<name>/SKILL.md``
# — is treated as user-owned and refused without ``--force``.
MANAGED_MARKER_NAME = ".crossby-managed"

_MANAGED_MARKER_BODY = (
    "# crossby managed marker\n"
    "#\n"
    "# This directory is managed by crossby (https://github.com/anthropics/crossby).\n"
    "# Edits will be overwritten on the next `crossby sync` run.\n"
    "# Delete this marker to take manual ownership of the directory.\n"
)


def backup_path(target: Path) -> Path:
    """Return the next available numbered backup path for *target*.

    Numbering scheme: ``.bak``, ``.bak2``, ``.bak3``, etc.
    Works for both files and directories.
    """
    candidate = Path(str(target) + ".bak")
    counter = 2
    while candidate.exists() or candidate.is_symlink():
        candidate = Path(str(target) + f".bak{counter}")
        counter += 1
    return candidate


def is_same_path(source: Path, target: Path) -> bool:
    """Return True when *source* and *target* name the same filesystem location.

    Used as a pre-flight guard by every writer that may back up, clear or
    replace its target: syncing a tool to itself (``--from claude --to claude``)
    resolves both sides to one directory, and the ``--force`` path would
    otherwise back up and ``rmtree`` the user's *source*.

    ``target`` is deliberately not followed when it is a symlink — an existing
    symlink pointing at the source is the idempotent re-run case, handled by
    :func:`crossby.config.linker.create_symlink`, not a collision.
    """
    try:
        if target.is_symlink():
            return False
        # Canonicalise the parent only, so a symlinked *source* leaf still
        # compares by name against a same-named target.
        if (source.parent.resolve() / source.name) == (target.parent.resolve() / target.name):
            return True
        # Catches the indirect case: a source that is itself a symlink into
        # the target (e.g. .crossby/agents -> .claude/agents).
        if source.exists() and target.exists():
            return source.samefile(target)
    except OSError:
        return False
    return False


def write_if_different(path: Path, content: bytes) -> bool:
    """Write *content* to *path* only when it differs from what's there.

    Returns True when the file was written. Keeping unchanged files untouched
    is what lets writers report an honest ``skipped`` and keeps re-syncs out of
    ``git status``.
    """
    try:
        if path.is_file() and path.read_bytes() == content:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return True


def mirror_tree(
    source_dir: Path, target_dir: Path, *, preserve: frozenset[str] = frozenset()
) -> bool:
    """Mirror *source_dir* onto *target_dir*, writing only what differs.

    Unlike ``rmtree`` + ``copytree``, an unchanged tree is left completely
    untouched — no file is removed and rewritten just to arrive at the same
    bytes, so an interrupted sync can't leave the target empty. Paths that no
    longer exist under *source_dir* are removed; top-level names in *preserve*
    (the crossby marker, typically) are never removed.

    Returns True when any file was written or removed.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    changed = False
    wanted: set[str] = set()

    for child in sorted(source_dir.iterdir()):
        wanted.add(child.name)
        dest = target_dir / child.name
        if child.is_dir() and not child.is_symlink():
            # A file sitting where a directory should go has to make way.
            if dest.exists() and not dest.is_dir():
                dest.unlink()
                changed = True
            if mirror_tree(child, dest):
                changed = True
        elif child.is_file():
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
                changed = True
            if write_if_different(dest, child.read_bytes()):
                changed = True

    for child in target_dir.iterdir():
        if child.name in wanted or child.name in preserve:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
        changed = True

    return changed


def has_managed_marker(target_dir: Path) -> bool:
    """Return True if ``target_dir`` carries the crossby ownership marker."""
    return (target_dir / MANAGED_MARKER_NAME).is_file()


def write_managed_marker(target_dir: Path) -> None:
    """Idempotently write the crossby ownership marker into ``target_dir``.

    Safe to call repeatedly — the marker content is fixed, so rewriting it
    on every sync is a no-op for git. Callers should invoke this whenever a
    write-bearing sync (copy or translate) produces or refreshes content in
    ``target_dir``.
    """
    if not target_dir.is_dir():
        return
    marker = target_dir / MANAGED_MARKER_NAME
    if marker.is_file() and marker.read_text(encoding="utf-8") == _MANAGED_MARKER_BODY:
        return
    marker.write_text(_MANAGED_MARKER_BODY, encoding="utf-8")
