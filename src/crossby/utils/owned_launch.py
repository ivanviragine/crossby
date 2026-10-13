"""Isolated launch process ownership, including terminal and transcript transport."""

from __future__ import annotations

import errno
import os
import select
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from crossby.ai_tools.session_process import (
    kill_process_group,
    own_process_tree,
    process_tree_popen_kwargs,
)


class ProcessTreeCleanupError(RuntimeError):
    """The caller must retain the checkout because process teardown failed."""


def _foreground(fd: int, group: int) -> None:
    previous = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
    try:
        os.tcsetpgrp(fd, group)
    finally:
        signal.signal(signal.SIGTTOU, previous)


def run_owned(
    command: list[str],
    cwd: Path | str | None,
    env: dict[str, str] | None,
    transcript_path: Path | None = None,
) -> int:
    """Kill descendants before returning or unwinding; retain native terminal input."""
    options: dict[str, Any] = process_tree_popen_kwargs()
    terminal: int | None = None
    foreground: int | None = None
    watcher: Any = None
    master: int | None = None
    slave: int | None = None
    saved_tty: Any = None
    transcript: Any = None
    if os.name == "posix":
        if not hasattr(os, "waitid") and not hasattr(select, "kqueue"):
            raise RuntimeError("This platform cannot observe an owned launch without reaping it.")
        options = {"process_group": 0}
        try:
            terminal = os.open("/dev/tty", os.O_RDWR)
            foreground = os.tcgetpgrp(terminal)
        except OSError:
            if terminal is not None:
                os.close(terminal)
            terminal = None
    proc: subprocess.Popen[bytes] | None = None
    stopped = False
    try:
        if transcript_path is not None:
            if os.name != "posix":
                raise RuntimeError("Isolated transcript capture currently requires POSIX.")
            import fcntl
            import pty
            import termios
            import tty

            transcript = transcript_path.open("ab")
            master, slave = pty.openpty()

            def controlling_terminal() -> None:
                fcntl.ioctl(0, termios.TIOCSCTTY, 0)

            options = {
                "start_new_session": True,
                "preexec_fn": controlling_terminal,
                "stdin": slave,
                "stdout": slave,
                "stderr": slave,
            }
            if terminal is not None:
                saved_tty = termios.tcgetattr(terminal)
                size = os.get_terminal_size(terminal)
                termios.tcsetwinsize(slave, (size.lines, size.columns))
                tty.setraw(terminal)
        proc = subprocess.Popen(command, cwd=cwd, env=env, **options)
        own_process_tree(proc)
        if slave is not None:
            os.close(slave)
            slave = None
        if terminal is not None and master is None:
            _foreground(terminal, proc.pid)
            os.killpg(proc.pid, signal.SIGCONT)
        if os.name != "posix":
            return proc.wait()
        if not hasattr(os, "waitid"):
            watcher = select.kqueue()
            event = select.kevent(
                proc.pid,
                filter=select.KQ_FILTER_PROC,
                flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                fflags=select.KQ_NOTE_EXIT,
            )
            with suppress(ProcessLookupError):
                watcher.control([event], 0, 0)

        def exited(block: bool) -> bool:
            if watcher is not None:
                return bool(watcher.control([], 1, None if block else 0))
            flags = os.WEXITED | os.WNOWAIT | (0 if block else os.WNOHANG)
            waitid: Any = getattr(os, "waitid", None)
            return waitid(os.P_PID, proc.pid, flags) is not None

        # Observe without reaping: the leader reserves the process-group ID until
        # our final signal. A transcript owns the actual CLI's PTY directly;
        # BSD script would move it to a separate, unowned process group.
        if master is not None:
            while True:
                readable, _, _ = select.select(
                    [master, *([terminal] if terminal is not None else [])], [], [], 0.05
                )
                if terminal is not None and terminal in readable:
                    data = os.read(terminal, 65536)
                    if data:
                        os.write(master, data)
                if master in readable:
                    try:
                        data = os.read(master, 65536)
                    except OSError as exc:
                        if exc.errno != errno.EIO:
                            raise
                        data = b""
                    if data:
                        os.write(sys.stdout.fileno(), data)
                        transcript.write(data)
                        transcript.flush()
                    else:
                        break
                if exited(False):
                    break
        else:
            exited(True)
        kill_process_group(proc)
        stopped = True
        if master is not None:
            # The exit event can arrive before the final output has been drained.
            drain_deadline = time.monotonic() + 2
            while time.monotonic() < drain_deadline:
                readable, _, _ = select.select([master], [], [], 0.05)
                if not readable:
                    continue
                try:
                    data = os.read(master, 65536)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    break
                if not data:
                    break
                os.write(sys.stdout.fileno(), data)
                transcript.write(data)
            transcript.flush()
        try:
            return proc.wait(timeout=2)
        except subprocess.TimeoutExpired as exc:
            raise ProcessTreeCleanupError("Owned tool did not stop; retain its worktree.") from exc
    finally:
        try:
            if proc is not None:
                if not stopped:
                    kill_process_group(proc)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired as exc:
                    raise ProcessTreeCleanupError(
                        "Owned tool teardown timed out; retain its worktree."
                    ) from exc
        finally:
            if watcher is not None:
                watcher.close()
            if transcript is not None:
                transcript.close()
            for fd in (master, slave):
                if fd is not None:
                    os.close(fd)
            if terminal is not None:
                try:
                    if saved_tty is not None:
                        termios.tcsetattr(terminal, termios.TCSADRAIN, saved_tty)
                    elif foreground is not None:
                        with suppress(OSError):
                            _foreground(terminal, foreground)
                finally:
                    os.close(terminal)
