"""Opt-in sandbox smoke gate: proves the writable-roots fix on a real Codex.

This is the empirical feasibility check the plan mandates — argv-shape tests
cannot prove that ``sandbox_workspace_write.writable_roots`` actually overrides
the ``.git``/gitdir read-only default under the OS sandbox. It shells out to a
real ``codex`` and is therefore **opt-in**: skipped unless
``CROSSBY_CODEX_SANDBOX_SMOKE=1`` and ``codex`` is on PATH.

Validated matrix (recorded for the record):

- codex-cli **0.147.0**, macOS Seatbelt, git 2.50.1 — both cases below pass.
- ``--sandbox workspace-write`` **without** writable roots blocks ``index.lock``
  in a linked worktree (the bug); adding the private+common git dirs to
  ``writable_roots`` (or the equivalent ``--add-dir``) lets ``git add`` succeed,
  network on or off.

Critical constraint baked into the fixture: workspace-write **always** grants
``$TMPDIR`` and ``/tmp``, so a repo created there would let git writes succeed
regardless of writable_roots — proving nothing. The throwaway repo is therefore
created under ``~`` (a non-temp, sandbox-restricted location) and removed after.

The deterministic control (``codex sandbox --permission-profile :workspace``)
needs no auth; the writable-roots fix and the networking case use ``codex exec``,
which needs a logged-in codex and one model turn (hence opt-in).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("CROSSBY_CODEX_SANDBOX_SMOKE") != "1",
        reason="opt-in sandbox smoke test (set CROSSBY_CODEX_SANDBOX_SMOKE=1)",
    ),
    pytest.mark.skipif(shutil.which("codex") is None, reason="codex not installed"),
    pytest.mark.skipif(
        not sys.platform.startswith(("darwin", "linux")),
        reason="OS sandbox smoke test is macOS/Linux only",
    ),
]


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def worktree() -> Iterator[tuple[Path, Path, Path]]:
    # Outside $TMPDIR / /tmp on purpose (both are always writable under
    # workspace-write). ``~`` is sandbox-restricted, so it exercises the real
    # gitdir carve-out.
    base = Path.home() / f".crossby-sandbox-smoke-{os.getpid()}"
    repo = base / "repo"
    repo.mkdir(parents=True)
    try:
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@t.co")
        _git(repo, "config", "user.name", "t")
        (repo / "f.txt").write_text("x\n")
        _git(repo, "add", "f.txt")
        _git(repo, "commit", "-q", "-m", "init")
        wt = base / "wt"
        _git(repo, "worktree", "add", "-q", str(wt), "-b", "feature")
        private = Path(
            subprocess.run(
                ["git", "-C", str(wt), "rev-parse", "--absolute-git-dir"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        common = (repo / ".git").resolve()
        yield wt, private, common
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_sandbox_blocks_git_metadata_write_without_roots(
    worktree: tuple[Path, Path, Path],
) -> None:
    """Deterministic control (no auth): workspace-write blocks the external gitdir."""
    wt, _private, _common = worktree
    (wt / "new.txt").write_text("y\n")
    cmd = ["codex", "sandbox", "--permission-profile", ":workspace", "-C", str(wt)]
    cmd += ["--", "git", "add", "-A"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0, "expected the sandbox to block the git-metadata write"
    assert "index.lock" in proc.stderr or "permitted" in proc.stderr.lower()


@pytest.mark.skipif(
    os.environ.get("CROSSBY_CODEX_SANDBOX_SMOKE_EXEC") != "1",
    reason="writable-roots fix uses `codex exec` (auth + a model turn); "
    "set CROSSBY_CODEX_SANDBOX_SMOKE_EXEC=1",
)
@pytest.mark.parametrize("network", [False, True])
def test_writable_roots_enable_git_write(
    worktree: tuple[Path, Path, Path], network: bool
) -> None:
    """Fix: with the metadata dirs in writable_roots, sandboxed git add succeeds."""
    wt, private, common = worktree
    (wt / "new.txt").write_text("y\n")
    roots = f'["{private}","{common}"]'
    proc = subprocess.run(
        [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "-C",
            str(wt),
            "-c",
            f"sandbox_workspace_write.writable_roots={roots}",
            "-c",
            f"sandbox_workspace_write.network_access={'true' if network else 'false'}",
            "-c",
            'model_reasoning_effort="low"',
            "Run exactly this one shell command, then stop: git add -A && echo STAGED_OK",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    # The model turn ran the command inside the sandbox; verify the write landed
    # by checking the (unsandboxed) index shows the staged file.
    status = subprocess.run(
        ["git", "-C", str(wt), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "new.txt" in status.stdout, f"expected staged file; exec stdout=\n{proc.stdout}"
    assert status.stdout.lstrip().startswith("A"), status.stdout
