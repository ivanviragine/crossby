"""PTY-backed process sessions for embedded terminals.

``utils/process.py`` runs a child on *this* process's stdio: fine for the CLI,
useless for a UI that has no terminal of its own. This module is the other half
— it allocates a pseudo-terminal, hands the child the slave side, and keeps the
master so a caller (the local web UI, a future TUI) can pump bytes both ways.

The child sees a genuine TTY, so ``claude``/``codex``/``cursor-agent`` render
their full-screen interface exactly as they would in a real terminal. The
calling process does **not** need a TTY itself.

Three details make the difference between "bytes appear" and "the TUI actually
works":

- **Controlling terminal.** ``start_new_session=True`` makes the child a session
  leader (CPython performs that ``setsid()`` in C, before ``preexec_fn``), but a
  session leader does not acquire a controlling terminal merely by inheriting
  the slave as fd 0/1/2. Without one, ``Ctrl-C`` generates no ``SIGINT`` and
  ``SIGWINCH`` reaches nobody. :func:`_acquire_controlling_tty` issues the
  ``TIOCSCTTY`` ioctl in the forked child to close that gap. ``preexec_fn`` is
  documented as unsafe in threaded programs; the body here is deliberately a
  single ioctl on an already-open descriptor, allocating nothing.
- **Window size.** ``TIOCSWINSZ`` on the master both records the size and makes
  the kernel deliver ``SIGWINCH`` to the foreground process group. A session
  that never sets it inherits 0x0 and every TUI wraps wrong.
- **``TERM``.** Callers pass the terminal type their emulator implements. Unset
  or ``dumb`` and the tools fall back to degraded output or refuse to start.

Output is exposed as **bytes**, never decoded text: a chunk boundary routinely
splits a UTF-8 sequence or an ANSI escape, so decoding per chunk corrupts the
stream. Consumers either forward the bytes verbatim or run an incremental
decoder of their own.
"""

from __future__ import annotations

import errno
import fcntl
import os
import pty
import queue
import re
import signal
import struct
import subprocess
import termios
import threading
import uuid
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import structlog

logger = structlog.get_logger()

# One read syscall's worth of terminal output.
READ_CHUNK_SIZE = 64 * 1024

# Bytes of recent output retained so a reconnecting viewer can repaint.
# Deliberately generous: a tool with an idle animation (Codex emits ~10.8 KB/s
# doing nothing) otherwise pushes every real frame out within seconds, and a
# reattaching viewer replays pure animation. This is a cushion, not the
# mechanism — a reattaching viewer should also force a redraw, which makes the
# tool repaint from its own state.
SCROLLBACK_LIMIT = 1024 * 1024

# How far past the trim point to look for an escape-sequence boundary.
SCROLLBACK_RESYNC_WINDOW = 8 * 1024

# Pending chunks a subscriber may fall behind before it is cut loose (see
# _offer: bytes are never dropped, because a gap mid-escape-sequence corrupts
# the viewer's screen silently). The bound exists so one stalled browser tab
# cannot block the reader thread and thereby the child's stdout.
SUBSCRIBER_QUEUE_LIMIT = 2048

# Grace period between SIGHUP and SIGKILL when closing a session.
TERMINATE_GRACE_SECONDS = 3.0

DEFAULT_TERM = "xterm-256color"

# Reports a terminal emulator generates by itself, never a keystroke: focus
# in/out (CSI I / CSI O), primary and secondary device attributes (CSI ? … c,
# CSI > … c), and the cursor position report (CSI … R).
_TERMINAL_REPORT = re.compile(rb"^(?:\x1b\[[IO]|\x1b\[\?[0-9;]*c|\x1b\[>[0-9;]*c|\x1b\[[0-9;]*R)+$")


class PtyUnsupportedError(RuntimeError):
    """Raised on platforms without POSIX pseudo-terminal support."""


class SubscriberDesyncError(RuntimeError):
    """Raised in :meth:`PtySession.subscribe` when a viewer fell too far behind.

    The stream is cut rather than silently losing bytes. Callers should resubscribe;
    the fresh scrollback plus the next repaint restores a correct screen.
    """


class _Signal(Enum):
    """Non-data markers multiplexed into a subscriber's queue."""

    END = "end"
    DESYNC = "desync"
    DETACH = "detach"


@dataclass(frozen=True)
class WindowSize:
    """Terminal dimensions in character cells."""

    cols: int
    rows: int

    def __post_init__(self) -> None:
        if not (1 <= self.cols <= 10_000) or not (1 <= self.rows <= 10_000):
            raise ValueError("terminal dimensions must be between 1 and 10000 cells")


def pty_supported() -> bool:
    """Whether this platform can allocate a pseudo-terminal.

    ``pty`` is POSIX-only. Windows would need ConPTY (``pywinpty``), which this
    module does not implement — the same gap ``script`` already leaves in
    :func:`crossby.utils.process.run_with_transcript`.
    """
    return os.name == "posix"


def _acquire_controlling_tty() -> None:  # pragma: no cover - runs post-fork
    """Make the inherited slave this session's controlling terminal."""
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


@dataclass(frozen=True)
class Subscription:
    """A registered viewer's private channel into one session's output."""

    channel: queue.Queue[bytes | _Signal]


def drain(subscription: Subscription) -> Iterator[bytes]:
    """Yield a subscription's live chunks until it ends, detaches, or desyncs."""
    while True:
        chunk = subscription.channel.get()
        if chunk is _Signal.END or chunk is _Signal.DETACH:
            return
        if chunk is _Signal.DESYNC:
            raise SubscriberDesyncError(
                "terminal output outpaced this viewer; resubscribe to resync"
            )
        assert isinstance(chunk, bytes)
        yield chunk


class PtySession:
    """One child process attached to a pseudo-terminal.

    A reader thread drains the master descriptor into a scrollback ring and
    every live subscriber queue. Callers write keystrokes with :meth:`write`,
    push window changes with :meth:`resize`, and consume output by iterating
    :meth:`subscribe`.

    Instances are safe to use from multiple threads.
    """

    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        size: WindowSize = WindowSize(cols=80, rows=24),
        term: str = DEFAULT_TERM,
        session_id: str | None = None,
    ) -> None:
        if not pty_supported():
            raise PtyUnsupportedError(
                "embedded terminals require POSIX pseudo-terminal support; "
                "Windows would need a ConPTY backend"
            )
        if not command:
            raise ValueError("command must not be empty")

        self.id = session_id or uuid.uuid4().hex
        self.command = list(command)
        self.cwd = cwd

        self._lock = threading.Lock()
        self._scrollback = bytearray()
        self._subscribers: list[queue.Queue[bytes | _Signal]] = []
        self._exited = threading.Event()
        self._exit_code: int | None = None
        self._size = size
        self._closed = False

        self._master_fd, slave_fd = pty.openpty()
        try:
            _set_window_size(self._master_fd, size)
            child_env = {**os.environ, **(env or {}), "TERM": term}
            self._proc = subprocess.Popen(
                self.command,
                cwd=cwd,
                env=child_env,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                start_new_session=True,
                preexec_fn=_acquire_controlling_tty,
            )
        except BaseException:
            os.close(self._master_fd)
            raise
        finally:
            # The child holds its own duplicate; keeping ours open would stop
            # the master ever reporting EOF when the child exits.
            os.close(slave_fd)

        logger.info(
            "pty.session.start",
            session=self.id,
            command=self.command[0],
            cwd=str(cwd),
            cols=size.cols,
            rows=size.rows,
        )

        self._reader = threading.Thread(
            target=self._pump_output,
            name=f"pty-reader-{self.id}",
            daemon=True,
        )
        self._reader.start()

    # -- lifecycle --------------------------------------------------------
    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def exit_code(self) -> int | None:
        """Exit status, or ``None`` while the child is still running."""
        return self._exit_code

    @property
    def running(self) -> bool:
        return not self._exited.is_set()

    @property
    def stopped(self) -> bool:
        """Whether this session ended because :meth:`close` was called."""
        return self._closed

    @property
    def exit_signal(self) -> str | None:
        """Name of the signal that killed the child, if one did.

        ``Popen.returncode`` encodes this as ``-N``, which surfaces as a
        baffling "exit -1" unless it is translated.
        """
        code = self._exit_code
        if code is None or code >= 0:
            return None
        try:
            return signal.Signals(-code).name
        except ValueError:  # pragma: no cover - unknown signal number
            return f"signal {-code}"

    @property
    def size(self) -> WindowSize:
        with self._lock:
            return self._size

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until the child exits, returning its status (``None`` on timeout)."""
        self._exited.wait(timeout)
        return self._exit_code

    def close(self) -> None:
        """Terminate the child's process group and release the master descriptor."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        if self._proc.poll() is None:
            _signal_group(self._proc.pid, signal.SIGHUP)
            try:
                self._proc.wait(timeout=TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _signal_group(self._proc.pid, signal.SIGKILL)
                with suppress(subprocess.TimeoutExpired):
                    self._proc.wait(timeout=TERMINATE_GRACE_SECONDS)

        self._reader.join(timeout=TERMINATE_GRACE_SECONDS)
        with suppress(OSError):
            os.close(self._master_fd)
        self._finish(self._proc.poll())
        logger.info("pty.session.close", session=self.id, exit_code=self._exit_code)

    # -- input ------------------------------------------------------------
    def write(self, data: bytes) -> None:
        """Send keystrokes to the child. A no-op once the child has exited.

        Emulator-generated reports are dropped while the tty still echoes. A
        tool enables focus reporting, the browser answers, and a millisecond
        later the tool turns the mode off again — which re-enables echo. The
        reply then arrives over the network *after* that switch and the kernel
        prints it as literal ``^[[I`` / ``^[[?1;2c``. A local terminal answers in
        microseconds and wins that race; a browser cannot. The tool is not
        reading raw replies in that window anyway, and re-queries once it is back
        in raw mode.
        """
        if self._exited.is_set():
            return
        if _TERMINAL_REPORT.match(data) and self._echo_enabled() is True:
            logger.debug("pty.write.dropped_report", session=self.id, data=data[:32])
            return
        try:
            os.write(self._master_fd, data)
        except OSError as exc:
            if exc.errno not in (errno.EIO, errno.EBADF, errno.EPIPE):
                raise
            logger.debug("pty.write.after_exit", session=self.id, errno=exc.errno)

    def _echo_enabled(self) -> bool | None:
        """Whether the line discipline is echoing, or ``None`` if unknowable.

        A raw-mode tool has ECHO off, so ECHO on means the tool is not consuming
        input raw. Reading termios through the master works on Linux; anywhere it
        does not, this returns ``None`` and no filtering happens — degrading to
        the previous behaviour rather than dropping input blindly.
        """
        try:
            attributes = termios.tcgetattr(self._master_fd)
        except (termios.error, OSError, ValueError):
            return None
        local_flags = attributes[3]
        return bool(local_flags & termios.ECHO)

    def resize(self, size: WindowSize) -> None:
        """Apply a new window size.

        Setting ``TIOCSWINSZ`` on the master makes the kernel deliver
        ``SIGWINCH`` to the terminal's foreground process group, which is what
        prompts a TUI to repaint at the new dimensions.
        """
        if self._exited.is_set():
            return
        with self._lock:
            self._size = size
        try:
            _set_window_size(self._master_fd, size)
        except OSError as exc:
            if exc.errno not in (errno.EIO, errno.EBADF):
                raise

    # -- output -----------------------------------------------------------
    def attach(self) -> tuple[bytes, Subscription | None]:
        """Register a viewer, returning ``(scrollback, subscription)``.

        The snapshot and the registration happen under one lock, so no chunk can
        slip between them. ``subscription`` is ``None`` when the child has
        already exited — the scrollback is then the whole story.

        Callers must pass the subscription to :meth:`detach` when done.
        :meth:`subscribe` wraps this pair for the simple single-session case;
        the multiplexer uses them directly so it can release a viewer that is
        parked in ``Queue.get()`` on a quiet session.
        """
        channel: queue.Queue[bytes | _Signal] = queue.Queue(maxsize=SUBSCRIBER_QUEUE_LIMIT)
        with self._lock:
            backlog = bytes(self._scrollback)
            if self._exited.is_set():
                return backlog, None
            self._subscribers.append(channel)
        return backlog, Subscription(channel)

    def detach(self, subscription: Subscription) -> None:
        """Release a viewer and wake it if it is parked waiting for output."""
        with self._lock:
            if subscription.channel in self._subscribers:
                self._subscribers.remove(subscription.channel)
        _force(subscription.channel, _Signal.DETACH)

    def subscribe(self) -> Iterator[bytes]:
        """Yield output chunks, starting with the current scrollback.

        The iterator ends when the child exits and its output is drained. A
        subscriber that cannot keep up raises :class:`SubscriberDesyncError`
        rather than stalling the reader thread or silently losing bytes.
        """
        backlog, subscription = self.attach()
        try:
            if backlog:
                yield backlog
            if subscription is None:
                return
            yield from drain(subscription)
        finally:
            if subscription is not None:
                self.detach(subscription)

    def _pump_output(self) -> None:
        """Drain the master descriptor until the child closes it."""
        try:
            while True:
                try:
                    chunk = os.read(self._master_fd, READ_CHUNK_SIZE)
                except OSError as exc:
                    # Linux signals "last slave closed" with EIO; macOS returns
                    # b"". EBADF means close() won the race.
                    if exc.errno in (errno.EIO, errno.EBADF):
                        break
                    raise
                if not chunk:
                    break
                self._broadcast(chunk)
        except Exception:  # pragma: no cover - defensive
            logger.exception("pty.reader.failed", session=self.id)
        finally:
            with suppress(Exception):
                self._proc.wait(timeout=TERMINATE_GRACE_SECONDS)
            self._finish(self._proc.poll())

    def _broadcast(self, chunk: bytes) -> None:
        with self._lock:
            self._scrollback.extend(chunk)
            self._trim_scrollback()
            targets = list(self._subscribers)
        for channel in targets:
            if not _offer(channel, chunk):
                self._cut(channel)

    def _cut(self, channel: queue.Queue[bytes | _Signal]) -> None:
        """Drop a subscriber that fell behind, telling it to resync."""
        with self._lock:
            if channel in self._subscribers:
                self._subscribers.remove(channel)
        _force(channel, _Signal.DESYNC)
        logger.warning("pty.subscriber.desync", session=self.id)

    def _trim_scrollback(self) -> None:
        """Bound the scrollback, cutting at an escape-sequence boundary.

        Cutting at an arbitrary byte can land mid-escape-sequence, so the replay
        opens with a fragment the emulator renders as garbage (a viewer saw a
        stray ``;72;48;2;43;45;49m``). Resuming at the next ESC guarantees the
        replay starts where a sequence starts.
        """
        excess = len(self._scrollback) - SCROLLBACK_LIMIT
        if excess <= 0:
            return
        boundary = self._scrollback.find(b"\x1b", excess, excess + SCROLLBACK_RESYNC_WINDOW)
        del self._scrollback[: boundary if boundary != -1 else excess]

    def _finish(self, code: int | None) -> None:
        """Record the exit status once and release every waiting subscriber."""
        with self._lock:
            if self._exited.is_set():
                return
            self._exit_code = code
            targets = list(self._subscribers)
            self._subscribers.clear()
        self._exited.set()
        for channel in targets:
            _force(channel, _Signal.END)


def _force(channel: queue.Queue[bytes | _Signal], marker: _Signal) -> None:
    """Enqueue a control marker, evicting queued data to make room if needed.

    Data may be dropped here; a marker may not. END and DETACH are what release a
    consumer parked in ``Queue.get()``, so losing one to a full queue leaks that
    thread forever. The session is ending or the viewer is leaving either way, so
    trailing bytes are the cheaper thing to lose. Bounded so a producer racing to
    refill cannot spin this forever.
    """
    for _ in range(SUBSCRIBER_QUEUE_LIMIT + 8):
        try:
            channel.put_nowait(marker)
            return
        except queue.Full:
            with suppress(queue.Empty):
                channel.get_nowait()
    logger.error("pty.marker.undeliverable", marker=marker.value)


def _offer(channel: queue.Queue[bytes | _Signal], item: bytes | _Signal) -> bool:
    """Enqueue *item*. Returns ``False`` when the subscriber has fallen behind.

    Dropping the oldest chunk would be the usual answer, and it is the wrong one
    here: terminal output is a stateful escape-sequence stream, so discarding
    bytes mid-sequence desynchronizes the emulator's screen permanently and
    silently. A subscriber that cannot keep up is cut instead.
    """
    try:
        channel.put_nowait(item)
        return True
    except queue.Full:
        return False


def _set_window_size(fd: int, size: WindowSize) -> None:
    packed = struct.pack("HHHH", size.rows, size.cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)


def _signal_group(pid: int, sig: int) -> None:
    """Signal the child's whole process group, falling back to the leader."""
    with suppress(OSError, ProcessLookupError):
        os.killpg(pid, sig)
        return
    with suppress(OSError, ProcessLookupError):
        os.kill(pid, sig)
