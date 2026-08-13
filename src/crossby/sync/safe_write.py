"""Unified safe filesystem-mutation primitives for the sync layer.

Every mutating filesystem operation in ``sync/`` — write, delete, copy, chmod,
mkdir, symlink — routes through the helpers here so that no sync mutation can
land outside the project root via a symlinked **ancestor** (``.agents ->
/outside``) or a symlinked **leaf** (``.codex/hooks.json -> ~/dotfiles``). This
replaces the four overlapping, non-unified mechanisms that PR #130 patched
per-path without ever converging (see issue #133):

* :func:`crossby.config.json_utils.assert_within` — leaf+ancestor, scenes only.
* :meth:`AbstractSyncWriter.contained_or_error` — ancestor-only, writer entry
  points only.
* ~52 ad-hoc ``if x.is_symlink(): x.unlink()`` leaf guards.
* :func:`crossby.sync.file_utils.write_if_different` — no symlink guard at all.

Design (resolved open decisions, issue #133):

**Scope model.** Every safe op takes an explicit :data:`Scope`:

* :class:`ProjectScope` — the target must resolve under ``project_root``. An
  accidental ``..`` escape, a symlinked ancestor, or a resolved path outside
  root is **refused**. This is the default for essentially every writer.
* :class:`ExternalScope` — the deliberate, opt-in path for writers that write
  outside the project by design. It carries an approved root (not a bare
  ``allow_external=True`` boolean, which would turn an accidental ``..`` into an
  unchecked write) and still refuses symlinked ancestors and unsafe leaves.
  Today the only external writer is Cursor ``global``-scope permissions
  (``~/.cursor``).

**Leaf policy.**

* ``"refuse"`` — a symlink at the leaf raises :class:`SyncContainmentError`
  *before* the file is read or parsed. Recommended for shared merge config
  files (``.claude/settings.json``, ``.codex/config.toml``, hooks/MCP JSON) and
  the ownership ledger.
* ``"replace"`` — a symlink leaf is replaced atomically (``os.replace`` of a
  temp file, **not** unlink-then-write). The link is **never read through** and
  its referent is **never resolved**, so a ``"replace"`` leaf pointing outside
  root is handled safely without contradicting :class:`ProjectScope`. An
  existing symlink leaf counts as **changed even when the resolved bytes would
  match**. Recommended for crossby-owned whole-file artifacts (rules/agents/
  skills).

**Ancestor rule.** A symlinked directory anywhere between the scope root and
the target is refused **unconditionally**, under both scopes and both leaf
policies.

Do **not** reuse :func:`crossby.config.safe_write.write_config_checked` here —
it intentionally *follows* symlinks for ``.crossby.yml`` (a documented
user-controlled trust boundary, the opposite intent).
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Leaf-symlink handling policy — see the module docstring.
LeafPolicy = Literal["replace", "refuse"]


class SyncContainmentError(Exception):
    """A sync mutation target escapes its scope.

    Raised when a target passes through a symlinked ancestor below the scope
    root, resolves outside the approved root, or is a symlinked leaf under a
    ``"refuse"`` policy. :class:`~crossby.sync.base.AbstractSyncWriter` converts
    this into an ``action="error"`` :class:`~crossby.sync.base.SyncResult` so a
    refused write becomes a reported row rather than an escaped exception.
    """


@dataclass(frozen=True)
class ProjectScope:
    """The default scope: the target must resolve under ``project_root``."""

    project_root: Path

    @property
    def root(self) -> Path:
        return self.project_root


@dataclass(frozen=True)
class ExternalScope:
    """Opt-in scope for writers that legitimately write outside the project.

    Carries the exact directory tree the write is allowed to land in. Symlinked
    ancestors and unsafe leaves are still refused; only the containment *root*
    differs from the project root. The single user today is Cursor
    ``global``-scope permissions (``~/.cursor``).
    """

    approved_root: Path

    @property
    def root(self) -> Path:
        return self.approved_root


Scope = ProjectScope | ExternalScope


# ---------------------------------------------------------------------------
# Containment checks (ancestor-only + resolved, layered so a ``"replace"`` leaf
# can be validated lexically without ever following its referent).
# ---------------------------------------------------------------------------


def first_symlinked_ancestor_in_scope(scope: Scope, target: Path) -> Path | None:
    """Return the first symlinked directory between the scope root and *target*.

    Both endpoints are excluded: the scope root is trusted and the target's own
    final component is the caller's leaf, handled separately by leaf policy.
    Returns ``None`` when *target* is not lexically under the scope root — use
    :func:`assert_ancestors` when an off-root target must itself be refused.
    """
    root_abs = Path(os.path.abspath(scope.root))
    target_abs = Path(os.path.abspath(target))
    try:
        relative = target_abs.relative_to(root_abs)
    except ValueError:
        return None
    current = root_abs
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            return current
    return None


def assert_ancestors(scope: Scope, target: Path) -> None:
    """Refuse a target that escapes *scope* through its ancestor chain.

    Two independent refusals, either of which raises :class:`SyncContainmentError`:

    1. *target* is not lexically under the scope root (a ``..`` escape, or a
       target under a different root than the scope authorises).
    2. a directory strictly between the scope root and *target* is a symlink —
       ``mkdir(parents=True)`` and symlink creation would follow it and land the
       write outside the root.

    The scope root itself and *target*'s own leaf are never inspected here; leaf
    handling is layered on top by each safe op via its leaf policy.
    """
    root_abs = Path(os.path.abspath(scope.root))
    target_abs = Path(os.path.abspath(target))
    try:
        relative = target_abs.relative_to(root_abs)
    except ValueError:
        raise SyncContainmentError(
            f"{target} resolves outside the allowed root {scope.root}; refusing to write."
        ) from None
    current = root_abs
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise SyncContainmentError(
                f"{current} is a symlinked directory on the path to {target}; refusing "
                "to write through it (it may point outside the project). Remove the "
                "symlink and re-run."
            )


def assert_resolved_within(scope: Scope, target: Path) -> None:
    """Defense in depth: *target*'s fully-resolved path must stay under the root.

    Only safe to call for a **regular or missing** leaf — resolving a symlink
    leaf would follow its referent, which the ``"replace"`` policy must never do.
    Resolving both sides cancels a symlinked ancestor of the root itself (macOS
    ``/tmp`` → ``/private/tmp``, a pytest ``tmp_path``), so a legitimate project
    under a symlinked root still passes.
    """
    root_resolved = Path(os.path.abspath(scope.root)).resolve()
    target_resolved = Path(os.path.abspath(target)).resolve()
    if not target_resolved.is_relative_to(root_resolved):
        raise SyncContainmentError(
            f"{target} resolves outside the allowed root {scope.root} "
            f"(resolved to {target_resolved}); refusing to write."
        )


def _refuse_symlink_leaf(target: Path) -> None:
    raise SyncContainmentError(
        f"{target} is a symlink; refusing to write through it (it may point outside "
        "the project). Remove the symlink and re-run."
    )


# ---------------------------------------------------------------------------
# Safe mutation ops. Each returns a *would-change* bool so a dry-run caller can
# aggregate every mutation kind (not just file bytes) into an honest report.
# ---------------------------------------------------------------------------


def _atomic_write_bytes(target: Path, content: bytes) -> None:
    """Write *content* to *target* via a unique temp file + atomic ``os.replace``.

    A symlink at *target* is replaced by the rename itself — ``os.replace`` acts
    on the link, never its referent — so the write can never leak through it.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".crossby-tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def safe_write_bytes(
    scope: Scope,
    target: Path,
    content: bytes,
    *,
    dry_run: bool = False,
    leaf_policy: LeafPolicy,
) -> bool:
    """Write *content* to *target*, refusing any symlink escape. Returns would-change.

    * Ancestor symlink between the root and *target* ⇒ :class:`SyncContainmentError`.
    * Symlink at *target*:
        * ``"refuse"`` ⇒ raise (before reading the file);
        * ``"replace"`` ⇒ replace atomically; an existing symlink counts as
          changed and is never read through.
    * Regular/missing leaf ⇒ write-if-different (unchanged bytes write nothing).
    """
    assert_ancestors(scope, target)
    if target.is_symlink():
        if leaf_policy == "refuse":
            _refuse_symlink_leaf(target)
        # "replace": a symlink leaf always counts as changed; never read through.
        if dry_run:
            return True
        _atomic_write_bytes(target, content)
        return True

    assert_resolved_within(scope, target)
    try:
        if target.is_file() and target.read_bytes() == content:
            return False
    except OSError:
        pass
    if dry_run:
        return True
    _atomic_write_bytes(target, content)
    return True


def safe_write_text(
    scope: Scope,
    target: Path,
    text: str,
    *,
    dry_run: bool = False,
    leaf_policy: LeafPolicy,
) -> bool:
    """UTF-8 wrapper around :func:`safe_write_bytes`."""
    return safe_write_bytes(
        scope, target, text.encode("utf-8"), dry_run=dry_run, leaf_policy=leaf_policy
    )


def safe_mkdir(
    scope: Scope,
    target: Path,
    *,
    dry_run: bool = False,
    leaf_policy: LeafPolicy = "replace",
) -> bool:
    """Create *target* as a directory (with parents), refusing symlink escape.

    A symlinked-directory leaf is replaced (``"replace"``) or refused
    (``"refuse"``); a real directory already present is a no-op. A **file** at
    *target* is left untouched — file↔dir replacement is a separately-authorised
    policy (:func:`crossby.sync.file_utils.clear_conflicting_type`), never a
    silent delete here. Returns would-change.
    """
    assert_ancestors(scope, target)
    if target.is_symlink():
        if leaf_policy == "refuse":
            _refuse_symlink_leaf(target)
        if not dry_run:
            target.unlink()
            target.mkdir(parents=True, exist_ok=True)
        return True
    if target.is_dir():
        return False
    assert_resolved_within(scope, target)
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
    return True


def safe_symlink(
    scope: Scope,
    link: Path,
    dest: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """Create a relative symlink at *link* pointing to *dest*, refusing escape.

    Only the **link** side (the write target) is contained — its *dest* referent
    is intentionally free to point anywhere, because a ``symlink``-strategy
    writer's whole job is to point a project path at a canonical source that may
    live elsewhere. Delegates idempotency, force / wrong-link re-point semantics
    and the circular-link guard to :func:`crossby.config.linker.create_symlink`;
    a re-point counts as a change. Returns would-change.
    """
    from crossby.config.linker import create_symlink

    assert_ancestors(scope, link)
    return create_symlink(dest, link, force=force, dry_run=dry_run)


def safe_unlink(
    scope: Scope,
    target: Path,
    *,
    dry_run: bool = False,
    missing_ok: bool = True,
) -> bool:
    """Remove the file or symlink at *target*, refusing a symlinked ancestor.

    ``unlink`` removes a symlink **leaf** itself, never its referent, so no leaf
    policy is needed. Returns would-change (True when something existed to
    remove). Use :func:`safe_rmtree` for directories.
    """
    assert_ancestors(scope, target)
    if not os.path.lexists(target):
        if not missing_ok:
            raise FileNotFoundError(target)
        return False
    if not dry_run:
        target.unlink()
    return True


def safe_rmtree(scope: Scope, target: Path, *, dry_run: bool = False) -> bool:
    """Recursively remove *target*, refusing a symlinked ancestor.

    A symlink **leaf** is unlinked (the link only, never its referent); a real
    directory is ``rmtree``'d; a regular file is unlinked. Returns would-change.
    """
    assert_ancestors(scope, target)
    if not os.path.lexists(target):
        return False
    if target.is_symlink():
        if not dry_run:
            target.unlink()
        return True
    if target.is_dir():
        if not dry_run:
            shutil.rmtree(target)
        return True
    if not dry_run:
        target.unlink()
    return True


def safe_copy(
    scope: Scope,
    dest: Path,
    source: Path,
    *,
    dry_run: bool = False,
    leaf_policy: LeafPolicy = "replace",
) -> bool:
    """Copy *source* onto *dest*, guarding the *dest* (write) side.

    A symlink at *dest* is replaced, never written through. A **symlink**
    *source* is copied as the link itself (``os.symlink`` of its target), never
    dereferenced — the behaviour a backup needs so it captures the link, not
    the file it points at. A directory *source* is copied with ``symlinks=True``
    for the same fidelity; a regular file goes through ``copy2`` (metadata
    preserved). Returns would-change (a copy is always a change).
    """
    assert_ancestors(scope, dest)
    if dest.is_symlink():
        if leaf_policy == "refuse":
            _refuse_symlink_leaf(dest)
        if not dry_run:
            dest.unlink()
    elif os.path.lexists(dest):
        assert_resolved_within(scope, dest)
    if dry_run:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        os.symlink(os.readlink(source), dest)
    elif source.is_dir():
        shutil.copytree(source, dest, symlinks=True)
    else:
        shutil.copy2(source, dest)
    return True


def safe_chmod(scope: Scope, target: Path, mode: int, *, dry_run: bool = False) -> bool:
    """``chmod`` *target* to *mode*, refusing a symlinked ancestor or leaf.

    A symlink at *target* is refused rather than chmod'd through to its referent.
    Callers that mirror permission bits (:func:`crossby.sync.file_utils` mode
    sync) always operate on a real file/dir they just wrote — a target symlink
    is replaced first — so this refusal is defensive. Returns would-change
    (True only when the mode actually differs).
    """
    assert_ancestors(scope, target)
    if target.is_symlink():
        _refuse_symlink_leaf(target)
    try:
        current = stat.S_IMODE(target.stat().st_mode)
    except OSError:
        return False
    if current == mode:
        return False
    if not dry_run:
        target.chmod(mode)
    return True


__all__ = [
    "ExternalScope",
    "LeafPolicy",
    "ProjectScope",
    "Scope",
    "SyncContainmentError",
    "assert_ancestors",
    "assert_resolved_within",
    "first_symlinked_ancestor_in_scope",
    "safe_chmod",
    "safe_copy",
    "safe_mkdir",
    "safe_symlink",
    "safe_unlink",
    "safe_write_bytes",
    "safe_write_text",
]
