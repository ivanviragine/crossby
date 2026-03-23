"""Tests for symlink management."""

from __future__ import annotations

import os
from pathlib import Path

from crossby.config.linker import create_symlink


class TestCreateSymlink:
    """Tests for create_symlink()."""

    def test_creates_symlink(self, tmp_path: Path) -> None:
        source = tmp_path / "source.md"
        source.write_text("hello")
        link = tmp_path / "link.md"

        assert create_symlink(source, link) is True
        assert link.is_symlink()
        assert os.readlink(link) == "source.md"
        assert link.read_text() == "hello"

    def test_relative_path_across_dirs(self, tmp_path: Path) -> None:
        source = tmp_path / "CLAUDE.md"
        source.write_text("hello")
        sub = tmp_path / ".github"
        sub.mkdir()
        link = sub / "copilot-instructions.md"

        assert create_symlink(source, link) is True
        assert os.readlink(link) == "../CLAUDE.md"

    def test_idempotent_noop(self, tmp_path: Path) -> None:
        source = tmp_path / "source.md"
        source.write_text("hello")
        link = tmp_path / "link.md"

        create_symlink(source, link)
        assert create_symlink(source, link) is False  # no-op

    def test_blocked_by_existing_file_without_force(self, tmp_path: Path) -> None:
        source = tmp_path / "source.md"
        source.write_text("hello")
        link = tmp_path / "link.md"
        link.write_text("existing")

        assert create_symlink(source, link, force=False) is False
        assert not link.is_symlink()
        assert link.read_text() == "existing"

    def test_force_replaces_existing_file(self, tmp_path: Path) -> None:
        source = tmp_path / "source.md"
        source.write_text("hello")
        link = tmp_path / "link.md"
        link.write_text("existing")

        assert create_symlink(source, link, force=True) is True
        assert link.is_symlink()
        assert link.read_text() == "hello"

    def test_force_replaces_wrong_symlink(self, tmp_path: Path) -> None:
        source = tmp_path / "source.md"
        source.write_text("hello")
        other = tmp_path / "other.md"
        other.write_text("other")
        link = tmp_path / "link.md"
        os.symlink("other.md", link)

        assert create_symlink(source, link, force=True) is True
        assert os.readlink(link) == "source.md"

    def test_dry_run_does_not_create(self, tmp_path: Path) -> None:
        source = tmp_path / "source.md"
        source.write_text("hello")
        link = tmp_path / "link.md"

        assert create_symlink(source, link, dry_run=True) is True
        assert not link.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        source = tmp_path / "source.md"
        source.write_text("hello")
        link = tmp_path / "sub" / "dir" / "link.md"

        assert create_symlink(source, link) is True
        assert link.is_symlink()

    def test_refuses_to_delete_directory(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        (source / "file.md").write_text("hello")
        link = tmp_path / "link"
        link.mkdir()

        assert create_symlink(source, link, force=True) is False
        assert link.is_dir()
        assert not link.is_symlink()

    def test_blocked_by_different_symlink_without_force(self, tmp_path: Path) -> None:
        source = tmp_path / "source.md"
        source.write_text("hello")
        other = tmp_path / "other.md"
        other.write_text("other")
        link = tmp_path / "link.md"
        os.symlink("other.md", link)

        assert create_symlink(source, link, force=False) is False
        assert os.readlink(link) == "other.md"
