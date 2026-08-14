"""Focused unit tests for the sync safe-write primitive (issue #133).

These cover the primitive in isolation — scope resolution, each leaf policy,
dangling symlinks, no-follow deletes/copies/chmods, backup-without-follow, and
the collision contract. The big writer-by-strategy matrix lives separately in
``test_write_safety_matrix.py``.

Every test uses ``root = tmp_path / "project"`` as the project root and
``tmp_path / "outside"`` as a genuinely-external location, so an escape target
is unambiguously outside the scope root.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from crossby.sync.safe_write import (
    ExternalScope,
    ProjectScope,
    SyncContainmentError,
    safe_chmod,
    safe_copy,
    safe_mkdir,
    safe_rmtree,
    safe_symlink,
    safe_unlink,
    safe_write_text,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A project root nested under tmp_path so a sibling can be truly external."""
    r = tmp_path / "project"
    r.mkdir()
    return r


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A directory outside the project root."""
    o = tmp_path / "outside"
    o.mkdir()
    return o


# ---------------------------------------------------------------------------
# safe_write_bytes / safe_write_text
# ---------------------------------------------------------------------------


class TestSafeWriteBytes:
    def test_writes_new_file(self, root: Path) -> None:
        target = root / "sub" / "file.txt"
        changed = safe_write_text(ProjectScope(root), target, "hello", leaf_policy="replace")
        assert changed is True
        assert target.read_text() == "hello"

    def test_write_if_different_skips_unchanged(self, root: Path) -> None:
        target = root / "file.txt"
        safe_write_text(ProjectScope(root), target, "hello", leaf_policy="replace")
        mtime = target.stat().st_mtime_ns
        changed = safe_write_text(ProjectScope(root), target, "hello", leaf_policy="replace")
        assert changed is False
        assert target.stat().st_mtime_ns == mtime  # not rewritten

    def test_dry_run_touches_nothing(self, root: Path) -> None:
        target = root / "file.txt"
        changed = safe_write_text(
            ProjectScope(root), target, "hello", dry_run=True, leaf_policy="replace"
        )
        assert changed is True
        assert not target.exists()

    def test_replace_leaf_never_followed(self, root: Path, outside: Path) -> None:
        """A ``replace`` symlink leaf pointing outside root is replaced in place,
        never written through — the external file is untouched."""
        escape = outside / "escape.txt"
        escape.write_text("original")
        target = root / "link.txt"
        target.symlink_to(escape)

        changed = safe_write_text(ProjectScope(root), target, "new", leaf_policy="replace")
        assert changed is True
        assert not target.is_symlink()
        assert target.read_text() == "new"
        assert escape.read_text() == "original"  # escape target untouched

    def test_replace_leaf_counts_as_changed_even_when_bytes_match(self, root: Path) -> None:
        """An existing symlink leaf is a change even if its resolved bytes match,
        and the referent is never read to decide."""
        real = root / "real.txt"
        real.write_text("same")
        target = root / "link.txt"
        target.symlink_to(real)

        changed = safe_write_text(ProjectScope(root), target, "same", leaf_policy="replace")
        assert changed is True
        assert not target.is_symlink()

    def test_refuse_leaf_raises_before_read(self, root: Path, outside: Path) -> None:
        escape = outside / "escape.txt"
        escape.write_text("original")
        target = root / "link.txt"
        target.symlink_to(escape)

        with pytest.raises(SyncContainmentError):
            safe_write_text(ProjectScope(root), target, "new", leaf_policy="refuse")
        assert target.is_symlink()  # untouched
        assert escape.read_text() == "original"

    def test_ancestor_symlink_outside_refused(self, root: Path, outside: Path) -> None:
        link_dir = root / "sub"
        link_dir.symlink_to(outside)
        target = link_dir / "file.txt"

        with pytest.raises(SyncContainmentError):
            safe_write_text(ProjectScope(root), target, "x", leaf_policy="replace")
        assert not (outside / "file.txt").exists()

    def test_ancestor_symlink_inside_still_refused(self, root: Path) -> None:
        """Ancestor policy is unconditional — even a link resolving inside root."""
        real = root / "real"
        real.mkdir()
        link_dir = root / "sub"
        link_dir.symlink_to(real)
        target = link_dir / "file.txt"

        with pytest.raises(SyncContainmentError):
            safe_write_text(ProjectScope(root), target, "x", leaf_policy="replace")

    def test_dotdot_escape_refused(self, root: Path, outside: Path) -> None:
        target = root / ".." / "outside" / "escape.txt"
        with pytest.raises(SyncContainmentError):
            safe_write_text(ProjectScope(root), target, "x", leaf_policy="replace")
        assert not (outside / "escape.txt").exists()

    def test_new_file_lands_at_umask_default_not_0600(self, root: Path) -> None:
        """``mkstemp`` creates the temp file ``0600`` and ``os.replace`` keeps it;
        the write must instead stamp the ``0o666 & ~umask`` default so a synced
        file is not silently owner-only."""
        old = os.umask(0o022)
        try:
            target = root / "sub" / "file.md"
            safe_write_text(ProjectScope(root), target, "hi", leaf_policy="replace")
            assert (target.stat().st_mode & 0o777) == 0o644
        finally:
            os.umask(old)

    def test_existing_regular_file_mode_preserved(self, root: Path) -> None:
        """Rewriting an existing regular file keeps its mode (not reset to 0600)."""
        target = root / "file.md"
        target.write_text("old")
        target.chmod(0o640)
        safe_write_text(ProjectScope(root), target, "new content", leaf_policy="replace")
        assert (target.stat().st_mode & 0o777) == 0o640

    def test_replaced_symlink_leaf_lands_at_default_mode(self, root: Path, outside: Path) -> None:
        """A replaced symlink leaf must not inherit its referent's mode — the link
        is replaced, not followed — so it lands at the umask default and the
        referent's own mode is left untouched."""
        old = os.umask(0o022)
        try:
            escape = outside / "escape.md"
            escape.write_text("x")
            escape.chmod(0o600)
            target = root / "link.md"
            target.symlink_to(escape)
            safe_write_text(ProjectScope(root), target, "new", leaf_policy="replace")
            assert not target.is_symlink()
            assert (target.stat().st_mode & 0o777) == 0o644
            assert (escape.stat().st_mode & 0o777) == 0o600  # referent untouched
        finally:
            os.umask(old)


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------


class TestScopes:
    def test_external_scope_allows_under_approved_root(self, tmp_path: Path) -> None:
        approved = tmp_path / "home" / ".cursor"
        approved.mkdir(parents=True)
        target = approved / "cli-config.json"
        changed = safe_write_text(ExternalScope(approved), target, "{}", leaf_policy="refuse")
        assert changed is True
        assert target.read_text() == "{}"

    def test_external_scope_refuses_outside_approved_root(self, tmp_path: Path) -> None:
        approved = tmp_path / "home" / ".cursor"
        approved.mkdir(parents=True)
        target = tmp_path / "elsewhere.json"
        with pytest.raises(SyncContainmentError):
            safe_write_text(ExternalScope(approved), target, "{}", leaf_policy="refuse")


# ---------------------------------------------------------------------------
# safe_mkdir
# ---------------------------------------------------------------------------


class TestSafeMkdir:
    def test_creates_dir(self, root: Path) -> None:
        target = root / "a" / "b"
        assert safe_mkdir(ProjectScope(root), target) is True
        assert target.is_dir()

    def test_existing_dir_no_change(self, root: Path) -> None:
        target = root / "a"
        target.mkdir()
        assert safe_mkdir(ProjectScope(root), target) is False

    def test_replace_symlink_dir_leaf(self, root: Path, outside: Path) -> None:
        target = root / "link"
        target.symlink_to(outside)
        assert safe_mkdir(ProjectScope(root), target, leaf_policy="replace") is True
        assert target.is_dir() and not target.is_symlink()

    def test_refuse_symlink_dir_leaf(self, root: Path, outside: Path) -> None:
        target = root / "link"
        target.symlink_to(outside)
        with pytest.raises(SyncContainmentError):
            safe_mkdir(ProjectScope(root), target, leaf_policy="refuse")
        assert target.is_symlink()

    def test_does_not_delete_a_file_at_target(self, root: Path) -> None:
        target = root / "afile"
        target.write_text("keep")
        with pytest.raises(FileExistsError):
            safe_mkdir(ProjectScope(root), target)
        assert target.read_text() == "keep"


# ---------------------------------------------------------------------------
# safe_symlink
# ---------------------------------------------------------------------------


class TestSafeSymlink:
    def test_creates_link(self, root: Path) -> None:
        source = root / "source.txt"
        source.write_text("x")
        link = root / "link.txt"
        assert safe_symlink(ProjectScope(root), link, source) is True
        assert link.is_symlink()
        assert link.resolve() == source.resolve()

    def test_idempotent_relink_no_change(self, root: Path) -> None:
        source = root / "source.txt"
        source.write_text("x")
        link = root / "link.txt"
        safe_symlink(ProjectScope(root), link, source)
        assert safe_symlink(ProjectScope(root), link, source) is False

    def test_ancestor_symlink_refused(self, root: Path, outside: Path) -> None:
        link_parent = root / "sub"
        link_parent.symlink_to(outside)
        source = root / "source.txt"
        source.write_text("x")
        with pytest.raises(SyncContainmentError):
            safe_symlink(ProjectScope(root), link_parent / "link.txt", source)


# ---------------------------------------------------------------------------
# safe_unlink / safe_rmtree — no-follow deletes
# ---------------------------------------------------------------------------


class TestSafeDeletes:
    def test_unlink_removes_symlink_not_referent(self, root: Path, outside: Path) -> None:
        escape = outside / "keep.txt"
        escape.write_text("keep")
        link = root / "link.txt"
        link.symlink_to(escape)
        assert safe_unlink(ProjectScope(root), link) is True
        assert not link.exists()
        assert escape.read_text() == "keep"  # referent untouched

    def test_unlink_missing_returns_false(self, root: Path) -> None:
        assert safe_unlink(ProjectScope(root), root / "nope") is False

    def test_unlink_dry_run(self, root: Path) -> None:
        target = root / "f.txt"
        target.write_text("x")
        assert safe_unlink(ProjectScope(root), target, dry_run=True) is True
        assert target.exists()

    def test_rmtree_removes_symlink_dir_not_referent(self, root: Path, outside: Path) -> None:
        (outside / "keep.txt").write_text("keep")
        link = root / "link"
        link.symlink_to(outside)
        assert safe_rmtree(ProjectScope(root), link) is True
        assert not link.exists()
        assert (outside / "keep.txt").read_text() == "keep"

    def test_rmtree_real_dir(self, root: Path) -> None:
        target = root / "d"
        (target / "nested").mkdir(parents=True)
        assert safe_rmtree(ProjectScope(root), target) is True
        assert not target.exists()

    def test_delete_through_symlinked_ancestor_refused(self, root: Path, outside: Path) -> None:
        (outside / "victim.txt").write_text("keep")
        link_dir = root / "sub"
        link_dir.symlink_to(outside)
        with pytest.raises(SyncContainmentError):
            safe_unlink(ProjectScope(root), link_dir / "victim.txt")
        assert (outside / "victim.txt").read_text() == "keep"


# ---------------------------------------------------------------------------
# safe_copy — backup without following a symlink source
# ---------------------------------------------------------------------------


class TestSafeCopy:
    def test_copy_file(self, root: Path) -> None:
        source = root / "src.txt"
        source.write_text("data")
        dest = root / "bak.txt"
        assert safe_copy(ProjectScope(root), dest, source) is True
        assert dest.read_text() == "data"

    def test_copy_symlink_source_backs_up_link_itself(self, root: Path) -> None:
        real = root / "real.txt"
        real.write_text("data")
        source = root / "src_link.txt"
        source.symlink_to(real)
        dest = root / "bak.txt"
        assert safe_copy(ProjectScope(root), dest, source) is True
        assert dest.is_symlink()  # the link, not the dereferenced file
        assert os.readlink(dest) == os.readlink(source)

    def test_copy_replaces_symlink_dest_not_written_through(
        self, root: Path, outside: Path
    ) -> None:
        escape = outside / "keep.txt"
        escape.write_text("keep")
        dest = root / "dest_link.txt"
        dest.symlink_to(escape)
        source = root / "src.txt"
        source.write_text("new")
        assert safe_copy(ProjectScope(root), dest, source, leaf_policy="replace") is True
        assert not dest.is_symlink()
        assert dest.read_text() == "new"
        assert escape.read_text() == "keep"

    def test_copy_dir_source(self, root: Path) -> None:
        source = root / "srcdir"
        (source / "a.txt").parent.mkdir(parents=True)
        (source / "a.txt").write_text("x")
        dest = root / "bak_dir"
        assert safe_copy(ProjectScope(root), dest, source) is True
        assert (dest / "a.txt").read_text() == "x"


# ---------------------------------------------------------------------------
# safe_chmod
# ---------------------------------------------------------------------------


class TestSafeChmod:
    def test_chmod_changes_mode(self, root: Path) -> None:
        target = root / "f.sh"
        target.write_text("#!/bin/sh")
        target.chmod(0o644)
        assert safe_chmod(ProjectScope(root), target, 0o755) is True
        assert (target.stat().st_mode & 0o777) == 0o755

    def test_chmod_same_mode_no_change(self, root: Path) -> None:
        target = root / "f.sh"
        target.write_text("x")
        target.chmod(0o644)
        assert safe_chmod(ProjectScope(root), target, 0o644) is False

    def test_chmod_refuses_symlink_leaf(self, root: Path) -> None:
        real = root / "real.txt"
        real.write_text("x")
        real.chmod(0o644)
        link = root / "link.txt"
        link.symlink_to(real)
        with pytest.raises(SyncContainmentError):
            safe_chmod(ProjectScope(root), link, 0o600)
        assert (real.stat().st_mode & 0o777) == 0o644  # referent untouched

    def test_chmod_dry_run(self, root: Path) -> None:
        target = root / "f.sh"
        target.write_text("x")
        target.chmod(0o644)
        assert safe_chmod(ProjectScope(root), target, 0o755, dry_run=True) is True
        assert (target.stat().st_mode & 0o777) == 0o644
