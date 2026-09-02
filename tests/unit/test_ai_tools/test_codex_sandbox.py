"""Codex sandbox composition: writable roots (--add-dir), single sandbox flag, pin.

Argv matrix over a **real** linked worktree (git init + commit + worktree add,
plus a detached-HEAD variant) crossed with autonomy tiers, trusted dirs, and
network on/off — plus the non-worktree byte-identity guarantee.

The git-metadata dirs are granted via **additive** ``--add-dir`` (so a user's
configured ``sandbox_workspace_write.writable_roots`` is preserved), while the
network flag is deliberately the replacing ``-c`` pin.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from crossby.ai_tools.codex import CodexAdapter


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


@dataclass
class Worktree:
    path: Path
    private: Path
    common: Path

    @property
    def metadata_add_dir_args(self) -> list[str]:
        # Resolver returns the out-of-root metadata dirs canonicalised + sorted;
        # common (``.git``) sorts before the nested private dir.
        dirs = sorted({self.common.resolve(), self.private.resolve()})
        out: list[str] = []
        for d in dirs:
            out += ["--add-dir", str(d)]
        return out


def _make_worktree(tmp_path: Path, *, detach: bool = False) -> Worktree:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.co")
    _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("x\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "init")
    wt = tmp_path / "wt"
    if detach:
        _git(repo, "worktree", "add", "-q", "--detach", str(wt))
    else:
        _git(repo, "worktree", "add", "-q", str(wt), "-b", "feature")
    return Worktree(
        path=wt,
        private=(repo / ".git" / "worktrees" / "wt"),
        common=(repo / ".git"),
    )


@pytest.fixture
def wt(tmp_path: Path) -> Worktree:
    return _make_worktree(tmp_path)


class TestNonWorktreeByteIdentity:
    """Where crossby emits no sandbox flag, the command is byte-identical."""

    def test_plain_launch_is_bare(self) -> None:
        assert CodexAdapter().build_launch_command() == ["codex"]

    def test_plain_launch_in_non_repo_is_bare(self, tmp_path: Path) -> None:
        assert CodexAdapter().build_launch_command(working_dir=tmp_path) == ["codex"]

    def test_yolo_only_non_worktree_is_bare(self) -> None:
        assert CodexAdapter().build_launch_command(yolo=True) == ["codex", "-a", "never"]


class TestWorktreeLaunch:
    def test_plain_launch_in_worktree_grants_roots(self, wt: Worktree) -> None:
        cmd = CodexAdapter().build_launch_command(working_dir=wt.path)
        assert cmd == [
            "codex",
            "--sandbox",
            "workspace-write",
            *wt.metadata_add_dir_args,
            "-c",
            "sandbox_workspace_write.network_access=false",
        ]

    def test_single_sandbox_before_add_dir(self, wt: Worktree) -> None:
        cmd = CodexAdapter().build_launch_command(working_dir=wt.path, trusted_dirs=["/tmp/plan"])
        assert cmd.count("--sandbox") == 1
        assert cmd.count("workspace-write") == 1
        assert cmd.index("--sandbox") < cmd.index("--add-dir")
        assert "/tmp/plan" in cmd  # trusted dir also via --add-dir

    def test_no_writable_roots_config_key(self, wt: Worktree) -> None:
        # Metadata dirs are additive --add-dir, never a replacing -c writable_roots.
        # (Match the config-key form, not a loose substring — pytest's tmp_path is
        # named after the test, so paths contain "writable_roots" incidentally.)
        cmd = CodexAdapter().build_launch_command(working_dir=wt.path)
        assert not any(a.startswith("sandbox_workspace_write.writable_roots") for a in cmd)

    def test_detached_head_worktree_grants_roots(self, tmp_path: Path) -> None:
        wt = _make_worktree(tmp_path, detach=True)
        cmd = CodexAdapter().build_launch_command(working_dir=wt.path)
        assert "--sandbox" in cmd
        assert "--add-dir" in cmd
        assert str(wt.common.resolve()) in cmd

    @pytest.mark.parametrize("network", [True, False])
    def test_network_pin_reflects_flag(self, wt: Worktree, network: bool) -> None:
        cmd = CodexAdapter().build_launch_command(working_dir=wt.path, network_access=network)
        pin = f"sandbox_workspace_write.network_access={'true' if network else 'false'}"
        assert pin in cmd

    def test_yolo_in_worktree_keeps_approval_and_grants_roots(self, wt: Worktree) -> None:
        cmd = CodexAdapter().build_launch_command(yolo=True, working_dir=wt.path)
        assert cmd[:3] == ["codex", "-a", "never"]
        assert "--sandbox" in cmd
        assert str(wt.common.resolve()) in cmd

    def test_accept_edits_in_worktree(self, wt: Worktree) -> None:
        cmd = CodexAdapter().build_launch_command(accept_edits=True, working_dir=wt.path)
        # Modern accept-edits approval policy (Codex CLI 0.152 rejects ``untrusted``).
        assert cmd[1:3] == ["-a", "on-request"]
        assert "untrusted" not in cmd
        assert cmd.count("--sandbox") == 1
        assert cmd.index("on-request") < cmd.index("--sandbox")
        assert wt.metadata_add_dir_args[1] in cmd  # a granted metadata path


class TestNetworkForcesWorkspaceWrite:
    def test_network_non_worktree_forces_sandbox(self) -> None:
        cmd = CodexAdapter().build_launch_command(network_access=True)
        assert cmd == [
            "codex",
            "--sandbox",
            "workspace-write",
            "-c",
            "sandbox_workspace_write.network_access=true",
        ]

    def test_trusted_dir_non_worktree_pins_network_off(self) -> None:
        cmd = CodexAdapter().build_launch_command(trusted_dirs=["/tmp/plan"])
        assert "sandbox_workspace_write.network_access=false" in cmd


class TestResume:
    def test_non_worktree_resume_byte_identical(self) -> None:
        assert CodexAdapter().build_resume_command("sid") == ["codex", "resume", "sid"]

    def test_non_repo_resume_byte_identical(self, tmp_path: Path) -> None:
        assert CodexAdapter().build_resume_command("sid", working_dir=tmp_path) == [
            "codex",
            "resume",
            "sid",
        ]

    def test_worktree_resume_grants_roots_approval_neutral(self, wt: Worktree) -> None:
        cmd = CodexAdapter().build_resume_command("sid", working_dir=wt.path)
        assert cmd is not None
        # Sandbox + --add-dir metadata + network pin, but NO injected approval flag.
        assert cmd[:3] == ["codex", "resume", "sid"]
        assert "--sandbox" in cmd
        assert "--add-dir" in cmd
        assert str(wt.common.resolve()) in cmd
        assert "sandbox_workspace_write.network_access=false" in cmd
        assert "-a" not in cmd
        assert "never" not in cmd

    def test_worktree_resume_with_network(self, wt: Worktree) -> None:
        cmd = CodexAdapter().build_resume_command("sid", working_dir=wt.path, network_access=True)
        assert cmd is not None
        assert "sandbox_workspace_write.network_access=true" in cmd
        assert "-a" not in cmd
