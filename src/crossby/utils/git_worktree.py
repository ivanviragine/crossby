"""Resolve git-metadata directories that live *outside* a working tree's root.

Linked Git worktrees (and, by the same mechanism, submodules) use a *gitlink*:
the working tree's ``.git`` is a **file** pointing at metadata that lives outside
the working directory. A sandbox that confines writes to the working tree — most
notably Codex's ``--sandbox workspace-write`` (Seatbelt on macOS, Landlock on
Linux) — therefore blocks every git write (``index``/``index.lock``, ref
updates, objects) that targets those external directories, so ``git add``,
``commit``, ``checkout`` and friends fail.

This resolver returns the absolute metadata dirs that fall **outside** the
worktree root so a caller can grant them as sandbox *writable roots*. It is
best-effort by design: any failure (git absent, non-zero exit, timeout,
malformed output, not a repo) yields an empty list rather than raising, because
a launch path must never crash on it.

Feasibility validated empirically (codex-cli 0.147.0, macOS Seatbelt, git
2.50.1): a ``codex exec --sandbox workspace-write`` process cannot create
``index.lock`` in a linked worktree, but adding the private + common git dirs to
``sandbox_workspace_write.writable_roots`` (equivalently ``--add-dir``) lets the
write succeed — the ``.git``/gitdir read-only default *is* overridden by an
explicit writable root. See ``CodexAdapter.sandbox_config_args``.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Bound the probe so a hung git binary can never stall a launch.
_GIT_TIMEOUT_S = 5.0
_GIT_POLL_SECONDS = 0.05
_GIT_TERMINATE_GRACE_SECONDS = 0.2

# Env vars that redirect git's notion of where metadata lives. Left in place they
# would let an ambient value (e.g. wade's own worktree, a parent repo) skew the
# resolution of the *launch target*, so they are scrubbed for the probe.
_CONTAMINATING_GIT_ENV = ("GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")


def outside_root_git_metadata_dirs(
    working_dir: Path,
    *,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> list[Path]:
    """Return absolute git-metadata dirs that live outside ``working_dir``'s root.

    Resolution is **root-relative**: the worktree root comes from ``git
    rev-parse --show-toplevel`` and a metadata dir is included only when it is
    *not contained within that root*. This is what distinguishes a linked
    worktree/submodule (metadata outside the root → granted) from a plain
    checkout launched from a nested subdirectory (``repo/.git`` is outside
    ``repo/src`` but still inside the root → granted nothing).

    Returns the private per-worktree git dir and the common git dir when they
    fall outside the root, canonicalised, de-duplicated and sorted. For a normal
    checkout — or on any error — returns ``[]``.
    """
    # One ``git rev-parse`` emits the three values as three ordered lines, so a
    # single spawn (not three) keeps the launch path cheap.
    lines = _run_git(
        working_dir,
        "--show-toplevel",
        "--absolute-git-dir",
        "--git-common-dir",
        deadline=deadline,
        cancel_event=cancel_event,
    )
    if lines is None or len(lines) != 3:
        return []
    toplevel, private, common = lines
    if not toplevel or not private or not common:
        return []

    try:
        root = Path(toplevel).resolve()
        # ``--git-common-dir`` may be relative (a plain checkout returns ``.git``);
        # resolve it against the working dir before comparing. ``--absolute-git-dir``
        # is already absolute. Canonicalise both so symlinked path components
        # (e.g. macOS ``/var`` → ``/private/var``) compare consistently.
        private_path = Path(private)
        common_path = Path(common)
        if not common_path.is_absolute():
            common_path = Path(working_dir) / common_path
        candidates = {private_path.resolve(), common_path.resolve()}
    except (OSError, ValueError, RuntimeError) as exc:  # RuntimeError: symlink loop
        logger.debug("git_worktree.resolve_failed", working_dir=str(working_dir), error=str(exc))
        return []

    outside = [c for c in candidates if not _is_within(c, root) and _looks_like_git_metadata(c)]
    return sorted(outside)


def _is_within(path: Path, root: Path) -> bool:
    """True when *path* is *root* itself or nested under it."""
    try:
        return path.is_relative_to(root)
    except ValueError:  # pragma: no cover - different drives (Windows)
        return False


def _looks_like_git_metadata(path: Path) -> bool:
    """Defense-in-depth check that *path* is a real git-metadata dir.

    Both a common git dir and a per-worktree/submodule private git dir contain a
    ``HEAD`` file. Requiring it rejects a hostile repo whose ``.git`` gitlink (or
    ``commondir``) points at an arbitrary non-git location — otherwise crossby
    would add e.g. ``~/.ssh`` to the sandbox writable roots. This is not a full
    trust boundary (the working dir is still the user's chosen repo), but it
    blocks pointing writable roots at a sensitive directory.
    """
    try:
        return (path / "HEAD").is_file()
    except OSError:  # pragma: no cover - defensive
        return False


def _run_git(
    working_dir: Path,
    *args: str,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> list[str] | None:
    """Run ``git -C <working_dir> rev-parse <args...>`` and return its lines.

    Returns ``None`` on git-absent, non-zero exit, or timeout. Never raises. The
    git-mutating env vars in :data:`_CONTAMINATING_GIT_ENV` are scrubbed so they
    cannot steer the result.
    """
    env = {k: v for k, v in os.environ.items() if k not in _CONTAMINATING_GIT_ENV}
    git_deadline = time.monotonic() + _GIT_TIMEOUT_S
    if deadline is not None:
        git_deadline = min(git_deadline, deadline)
    if cancel_event is not None and cancel_event.is_set():
        return None
    if _git_remaining_seconds(git_deadline) <= 0:
        return None
    try:
        proc = subprocess.Popen(
            ["git", "-C", str(working_dir), "rev-parse", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # UnicodeError: text=True can hit a path with non-UTF-8 bytes.
        logger.debug("git_worktree.git_failed", args=args, error=str(exc))
        return None
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                _stop_git(proc)
                return None
            remaining = _git_remaining_seconds(git_deadline)
            if remaining <= 0:
                _stop_git(proc)
                return None
            try:
                stdout, _ = proc.communicate(timeout=min(_GIT_POLL_SECONDS, remaining))
            except subprocess.TimeoutExpired:
                continue
            break
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        _stop_git(proc)
        logger.debug("git_worktree.git_failed", args=args, error=str(exc))
        return None
    if proc.returncode != 0:
        return None
    return stdout.splitlines()


def _git_remaining_seconds(deadline: float) -> float:
    """Return the remaining time in the probe's fixed absolute budget."""
    return max(0.0, deadline - time.monotonic())


def _stop_git(proc: subprocess.Popen[str]) -> None:
    """Terminate a cancelled or expired metadata probe before returning."""
    with suppress(OSError):
        proc.terminate()
    try:
        proc.wait(timeout=_GIT_TERMINATE_GRACE_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        with suppress(OSError):
            proc.kill()
        with suppress(OSError, subprocess.TimeoutExpired):
            proc.wait(timeout=_GIT_TERMINATE_GRACE_SECONDS)
