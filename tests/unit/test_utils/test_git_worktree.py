"""Tests for the out-of-root git-metadata resolver.

Uses real ``git init`` + ``git worktree add`` (and ``--detach``) and a real
submodule rather than mocks, so the resolution matches what the sandbox actually
needs. The resolver must never raise: every failure mode returns ``[]``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from crossby.utils.git_worktree import (
    _looks_like_git_metadata,
    outside_root_git_metadata_dirs,
)


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t.co")
    _git(path, "config", "user.name", "t")
    (path / "tracked.txt").write_text("hello\n")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-q", "-m", "init")


def _resolved(*paths: Path) -> set[Path]:
    return {p.resolve() for p in paths}


class TestNormalCheckout:
    def test_repo_root_grants_nothing(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        assert outside_root_git_metadata_dirs(repo) == []

    def test_nested_subdir_grants_nothing(self, tmp_path: Path) -> None:
        # repo/.git is outside repo/src but still inside the worktree ROOT, so a
        # root-relative rule grants nothing — the key false-positive guard.
        repo = tmp_path / "repo"
        _init_repo(repo)
        sub = repo / "src" / "deep"
        sub.mkdir(parents=True)
        assert outside_root_git_metadata_dirs(sub) == []


class TestLinkedWorktree:
    def test_worktree_grants_private_and_common(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        wt = tmp_path / "wt"
        _git(repo, "worktree", "add", "-q", str(wt), "-b", "feature")

        result = set(outside_root_git_metadata_dirs(wt))
        private = repo / ".git" / "worktrees" / "wt"
        common = repo / ".git"
        assert result == _resolved(private, common)

    def test_nested_subdir_of_worktree_grants_metadata(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        wt = tmp_path / "wt"
        _git(repo, "worktree", "add", "-q", str(wt), "-b", "feature")
        sub = wt / "src"
        sub.mkdir()

        result = set(outside_root_git_metadata_dirs(sub))
        assert result == _resolved(repo / ".git" / "worktrees" / "wt", repo / ".git")

    def test_detached_head_worktree_resolves_same_dirs(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        wt = tmp_path / "wt"
        _git(repo, "worktree", "add", "-q", "--detach", str(wt))

        result = set(outside_root_git_metadata_dirs(wt))
        # Worktree dir name is derived from the path, so match by the common dir
        # plus a private dir nested under it.
        common = (repo / ".git").resolve()
        assert common in result
        assert any(p != common and common in p.parents for p in result)

    def test_returned_paths_are_sorted(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        wt = tmp_path / "wt"
        _git(repo, "worktree", "add", "-q", str(wt), "-b", "feature")
        result = outside_root_git_metadata_dirs(wt)
        assert result == sorted(result)


class TestSubmodule:
    def test_submodule_grants_outside_root_metadata(self, tmp_path: Path) -> None:
        # A submodule's working tree also uses a gitlink: its .git file points at
        # <super>/.git/modules/<name>, outside the submodule root.
        sub_origin = tmp_path / "sub_origin"
        _init_repo(sub_origin)
        super_repo = tmp_path / "super"
        _init_repo(super_repo)
        _git(
            super_repo,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(sub_origin),
            "sub",
        )
        _git(super_repo, "commit", "-q", "-m", "add submodule")

        sub_wd = super_repo / "sub"
        result = outside_root_git_metadata_dirs(sub_wd)
        # The submodule's metadata lives under the superproject, outside the
        # submodule root — so at least one dir is granted, deduped and sorted.
        assert result
        assert result == sorted(set(result))
        modules_dir = (super_repo / ".git" / "modules" / "sub").resolve()
        assert any(modules_dir == p or modules_dir in p.parents for p in result)


class TestFailureModes:
    def test_non_repo_returns_empty(self, tmp_path: Path) -> None:
        assert outside_root_git_metadata_dirs(tmp_path) == []

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        assert outside_root_git_metadata_dirs(tmp_path / "nope") == []

    def test_git_absent_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        wt = tmp_path / "wt"
        _git(repo, "worktree", "add", "-q", str(wt), "-b", "feature")
        # Empty PATH → the ``git`` binary cannot be found → FileNotFoundError,
        # which the resolver swallows into [].
        monkeypatch.setenv("PATH", "")
        assert outside_root_git_metadata_dirs(wt) == []

    def test_env_contamination_does_not_skew_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An ambient GIT_DIR/GIT_COMMON_DIR (e.g. from a parent tool) must not
        # steer resolution of the launch target — the resolver scrubs them.
        repo = tmp_path / "repo"
        _init_repo(repo)
        wt = tmp_path / "wt"
        _git(repo, "worktree", "add", "-q", str(wt), "-b", "feature")

        other = tmp_path / "other"
        _init_repo(other)
        monkeypatch.setenv("GIT_DIR", str(other / ".git"))
        monkeypatch.setenv("GIT_COMMON_DIR", str(other / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(other))

        result = set(outside_root_git_metadata_dirs(wt))
        assert result == _resolved(repo / ".git" / "worktrees" / "wt", repo / ".git")


class TestHostileMetadataGuard:
    """A crafted gitlink must not expand the sandbox writable roots."""

    def test_looks_like_git_metadata_requires_head(self, tmp_path: Path) -> None:
        d = tmp_path / "notgit"
        d.mkdir()
        assert _looks_like_git_metadata(d) is False
        (d / "HEAD").write_text("ref: refs/heads/x\n")
        assert _looks_like_git_metadata(d) is True

    def test_crafted_gitlink_to_non_git_dir_grants_nothing(self, tmp_path: Path) -> None:
        # A hostile working tree whose .git points at a sensitive, non-git dir
        # (no HEAD). The resolver must grant nothing — otherwise crossby would add
        # that dir to the sandbox writable roots.
        sensitive = tmp_path / "sensitive"
        sensitive.mkdir()
        (sensitive / "secret").write_text("x\n")
        work = tmp_path / "work"
        work.mkdir()
        (work / ".git").write_text(f"gitdir: {sensitive}\n")
        assert outside_root_git_metadata_dirs(work) == []


class TestSymlinks:
    def test_symlinked_worktree_path_resolves_consistently(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        wt = tmp_path / "wt"
        _git(repo, "worktree", "add", "-q", str(wt), "-b", "feature")

        link = tmp_path / "wt_link"
        link.symlink_to(wt)
        # Reached via a symlinked path, resolution still yields the canonical
        # metadata dirs (no duplicates from the symlink component).
        result = set(outside_root_git_metadata_dirs(link))
        assert result == _resolved(repo / ".git" / "worktrees" / "wt", repo / ".git")
