"""Shared file utilities for sync writers."""

from __future__ import annotations

import shutil
import stat
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


def first_symlinked_ancestor(root: Path, target: Path) -> Path | None:
    """Return the first symlinked directory between *root* and *target*, or None.

    Both endpoints are excluded: *root* is trusted (the project root) and
    *target*'s own final component is the caller's existing final-target guard.
    Every directory *between* them is checked because ``mkdir(parents=True)`` and
    :func:`crossby.config.linker.create_symlink` follow a symlinked *parent* — a
    link like ``.agents -> /outside`` lands writes outside *root* even when the
    final target component is itself guarded. Callers refuse the sync when this
    returns a path.
    """
    try:
        rel = target.relative_to(root)
    except ValueError:
        return None
    current = root
    for part in rel.parts[:-1]:
        current = current / part
        if current.is_symlink():
            return current
    return None


def clear_conflicting_type(dest: Path, *, want_dir: bool, dry_run: bool = False) -> bool:
    """Remove *dest* when it exists as the opposite kind of thing.

    Without this, mirroring a source directory onto a target *file* raises
    ``FileExistsError`` from ``mkdir``, and writing a source file over a target
    *directory* raises ``IsADirectoryError`` — either one aborts the whole sync
    with a traceback instead of a graceful result. Symlinks are left alone;
    the callers decide what to do with those.

    Returns True when something was removed (or, under ``dry_run``, *would* be —
    nothing is touched on disk). ``dry_run`` defaults False, so callers that omit
    it keep the exact prior behavior.
    """
    if dest.is_symlink() or not dest.exists():
        return False
    if want_dir and not dest.is_dir():
        if not dry_run:
            dest.unlink()
        return True
    if not want_dir and dest.is_dir():
        if not dry_run:
            shutil.rmtree(dest)
        return True
    return False


def write_if_different(path: Path, content: bytes, *, dry_run: bool = False) -> bool:
    """Write *content* to *path* only when it differs from what's there.

    Returns True when the file was written (or, under ``dry_run``, *would* be).
    Keeping unchanged files untouched is what lets writers report an honest
    ``skipped`` and keeps re-syncs out of ``git status``. ``dry_run`` defaults
    False, so existing callers keep the exact prior behavior.

    The write itself goes through a temp file plus rename, so an interrupted
    sync leaves the previous content intact rather than a truncated file — the
    same guarantee :func:`crossby.config.json_utils.atomic_write_text` gives
    the config writers.
    """
    try:
        if path.is_file() and path.read_bytes() == content:
            return False
    except OSError:
        pass
    if dry_run:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".crossby-tmp")
    try:
        tmp.write_bytes(content)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return True


def _sync_mode(dest: Path, source: Path, *, dry_run: bool = False) -> bool:
    """Copy *source*'s permission bits onto *dest* when they differ.

    Applies to files *and* directories. ``write_if_different`` writes through a
    temp file and ``mkdir`` creates directories, so a fresh target lands with the
    umask default and loses any non-default bits the source carried — a dropped
    executable bit silently breaks ``scripts/`` mirrored under a skill, and a
    dropped directory mode either widens a private ``0700`` ``assets/`` tree to
    the umask default or strips required group access from a ``0770`` ``scripts/``
    tree. Re-applying the source's full permission mask (``stat.S_IMODE`` — the
    12 low bits, so setgid/setuid/sticky survive alongside the ``rwx`` triples,
    matching the ``copytree`` this replaced) fixes both and is idempotent: once
    matched, the next run makes no change. Masking narrower (e.g. ``& 0o777``)
    would both hide a special-bit-only difference during comparison and clear the
    special bits on ``chmod`` — a source ``02770`` group-shared dir would land as
    a plain ``0770`` at the target.

    Returns True when a ``chmod`` was applied.
    """
    try:
        source_mode = stat.S_IMODE(source.stat().st_mode)
        dest_mode = stat.S_IMODE(dest.stat().st_mode)
    except OSError:
        return False
    if dest_mode != source_mode:
        if not dry_run:
            dest.chmod(source_mode)
        return True
    return False


def mirror_tree(
    source_dir: Path,
    target_dir: Path,
    *,
    preserve: frozenset[str] = frozenset(),
    dry_run: bool = False,
) -> bool:
    """Mirror *source_dir* onto *target_dir*, writing only what differs.

    Unlike ``rmtree`` + ``copytree``, an unchanged tree is left completely
    untouched — no file is removed and rewritten just to arrive at the same
    bytes, so an interrupted sync can't leave the target empty. Paths that no
    longer exist under *source_dir* are removed; top-level names in *preserve*
    (the crossby marker, typically) are never removed. Permission bits are
    propagated too (via :func:`_sync_mode`) — for files *and* for every mirrored
    directory — so an executable ``scripts/`` file stays runnable and a private
    ``0700`` ``assets/`` tree isn't widened to the umask default at the target.

    A symlink anywhere in the target tree — the root or any nested child — is
    **replaced, never followed**: following one would land writes (or a
    :func:`_sync_mode` chmod) on its destination, potentially outside the
    mirror root. Symlinks in *source_dir*, by contrast, are **dereferenced**: a
    symlinked file is copied by its bytes and a symlinked directory is recursed
    into and mirrored as a real directory at the target — matching the
    ``copytree(symlinks=False)`` this replaced, so a skill's support tree can't
    silently lose scripts or assets that live behind a directory symlink. (A
    broken source symlink resolves to neither a file nor a directory and is
    skipped.)

    Returns True when any file was written, chmod'd, or removed (or, under
    ``dry_run``, *would* be — nothing is touched on disk). ``dry_run`` defaults
    False, so every existing caller keeps the exact prior behavior; a dry-run
    caller gets an honest would-change flag for reporting ``skipped``.
    """
    changed = False
    # Never mirror *through* a symlinked target root — replace it with a real
    # directory so nothing lands at the symlink's destination.
    if target_dir.is_symlink():
        if not dry_run:
            target_dir.unlink()
        changed = True
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    wanted: set[str] = set()

    for child in sorted(source_dir.iterdir()):
        wanted.add(child.name)
        dest = target_dir / child.name
        # A symlinked target child is replaced, not written through — otherwise a
        # nested symlink would redirect writes/chmods outside the mirror root.
        if dest.is_symlink():
            if not dry_run:
                dest.unlink()
            changed = True
        # ``is_dir()``/``is_file()`` follow symlinks, so a symlinked *source*
        # entry is dereferenced here — a directory is recursed into (mirrored as
        # a real dir), a file copied by its bytes. The target symlink guard above
        # still prevents writing *through* a link on the destination side.
        if child.is_dir():
            changed |= clear_conflicting_type(dest, want_dir=True, dry_run=dry_run)
            if mirror_tree(child, dest, dry_run=dry_run):
                changed = True
        elif child.is_file():
            changed |= clear_conflicting_type(dest, want_dir=False, dry_run=dry_run)
            if write_if_different(dest, child.read_bytes(), dry_run=dry_run):
                changed = True
            if _sync_mode(dest, child, dry_run=dry_run):
                changed = True

    # In a real run the ``mkdir`` above guarantees a real dir here; under dry-run
    # the target may not exist (not created) or still be a symlink (not
    # replaced), so guard before iterating — any escape was already counted above.
    if target_dir.is_dir() and not target_dir.is_symlink():
        for child in target_dir.iterdir():
            if child.name in wanted or child.name in preserve:
                continue
            if not dry_run:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            changed = True

    # Mirror this directory's own permission bits last — the ``mkdir`` above
    # created it under the umask, which would otherwise widen a private ``0700``
    # tree or strip group access from a ``0770`` one. Deferred to here, after the
    # children are written, so a restrictive source mode can't lock out those
    # writes. Each recursive call syncs its own root, so nested dirs are covered.
    if _sync_mode(target_dir, source_dir, dry_run=dry_run):
        changed = True

    return changed


def has_managed_marker(target_dir: Path) -> bool:
    """Return True if ``target_dir`` carries the crossby ownership marker.

    A *symlinked* marker does not count: trusting it would let a planted
    ``.crossby-managed -> /some/external/file`` symlink mark a directory as
    crossby-owned (so its contents get overwritten) and would make
    :func:`write_managed_marker` write through the link.
    """
    marker = target_dir / MANAGED_MARKER_NAME
    return marker.is_file() and not marker.is_symlink()


def write_managed_marker(target_dir: Path) -> None:
    """Idempotently write the crossby ownership marker into ``target_dir``.

    Safe to call repeatedly — the marker content is fixed, so rewriting it
    on every sync is a no-op for git. Callers should invoke this whenever a
    write-bearing sync (copy or translate) produces or refreshes content in
    ``target_dir``.

    A pre-existing marker *symlink* is replaced outright, never written through —
    otherwise the write would land at the link's destination, potentially an
    arbitrary external file.
    """
    if not target_dir.is_dir():
        return
    marker = target_dir / MANAGED_MARKER_NAME
    if marker.is_symlink():
        marker.unlink()
    elif marker.is_file() and marker.read_text(encoding="utf-8") == _MANAGED_MARKER_BODY:
        return
    marker.write_text(_MANAGED_MARKER_BODY, encoding="utf-8")
