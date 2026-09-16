"""Tests for :mod:`crossby.utils.pty_runner` — the embedded-terminal primitive.

These drive a real child on a real pseudo-terminal: the behaviours that matter
(controlling terminal, SIGWINCH delivery, signal-generated exits) cannot be
observed through mocks.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from crossby.utils.pty_runner import (
    SCROLLBACK_LIMIT,
    SCROLLBACK_RESYNC_WINDOW,
    SUBSCRIBER_QUEUE_LIMIT,
    PtySession,
    SubscriberDesyncError,
    WindowSize,
    _Signal,
    pty_supported,
)

pytestmark = pytest.mark.skipif(not pty_supported(), reason="requires POSIX pty support")

# Reports its terminal view, then echoes stdin until told to quit.
_PROBE = r"""
import os, shutil, signal, sys
print("isatty", sys.stdin.isatty(), flush=True)
try:
    os.close(os.open("/dev/tty", os.O_RDWR)); print("ctty yes", flush=True)
except OSError:
    print("ctty no", flush=True)
size = shutil.get_terminal_size()
print("size %sx%s" % (size.columns, size.lines), flush=True)
signal.signal(signal.SIGWINCH, lambda *a: print("winch %sx%s" % (
    shutil.get_terminal_size().columns, shutil.get_terminal_size().lines), flush=True))
for line in sys.stdin:
    line = line.strip()
    if line == "quit":
        sys.exit(5)
    print("got", line, flush=True)
"""


def _collect(session: PtySession) -> list[bytes]:
    """Drain a session's output on a background thread."""
    chunks: list[bytes] = []
    thread = threading.Thread(target=lambda: chunks.extend(session.subscribe()), daemon=True)
    thread.start()
    return chunks


def _fill(channel: Any) -> None:
    """Saturate a subscriber queue so the next broadcast finds it full."""
    while channel.qsize() < SUBSCRIBER_QUEUE_LIMIT:
        channel.put_nowait(b"x")


def _await(chunks: list[bytes], needle: str, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if needle in b"".join(chunks).decode(errors="replace"):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def probe(tmp_path: Path) -> PtySession:
    session = PtySession(
        [sys.executable, "-u", "-c", _PROBE],
        cwd=tmp_path,
        size=WindowSize(cols=100, rows=30),
    )
    yield session
    session.close()


class TestTerminalFidelity:
    def test_child_sees_a_tty(self, probe: PtySession) -> None:
        assert _await(_collect(probe), "isatty True")

    def test_child_acquires_a_controlling_terminal(self, probe: PtySession) -> None:
        """Without TIOCSCTTY there is no ctty, so Ctrl-C and SIGWINCH go nowhere."""
        assert _await(_collect(probe), "ctty yes")

    def test_initial_window_size_is_applied(self, probe: PtySession) -> None:
        assert _await(_collect(probe), "size 100x30")

    def test_term_defaults_to_a_capable_terminal(self, tmp_path: Path) -> None:
        session = PtySession(
            [sys.executable, "-u", "-c", "import os,sys; print(os.environ['TERM'], flush=True)"],
            cwd=tmp_path,
        )
        try:
            assert _await(_collect(session), "xterm-256color")
        finally:
            session.close()


class TestInteraction:
    def test_write_reaches_the_child(self, probe: PtySession) -> None:
        chunks = _collect(probe)
        assert _await(chunks, "isatty")
        probe.write(b"ping\n")
        assert _await(chunks, "got ping")

    def test_resize_delivers_sigwinch_with_new_size(self, probe: PtySession) -> None:
        chunks = _collect(probe)
        assert _await(chunks, "size 100x30")
        probe.resize(WindowSize(cols=132, rows=44))
        assert _await(chunks, "winch 132x44")
        assert probe.size == WindowSize(cols=132, rows=44)

    def test_exit_code_is_reported(self, probe: PtySession) -> None:
        chunks = _collect(probe)
        assert _await(chunks, "isatty")
        probe.write(b"quit\n")
        assert probe.wait(timeout=5) == 5
        assert not probe.running

    def test_write_after_exit_is_a_noop(self, probe: PtySession) -> None:
        probe.write(b"quit\n")
        probe.wait(timeout=5)
        probe.write(b"ignored\n")  # must not raise

    def test_resize_after_exit_is_a_noop(self, probe: PtySession) -> None:
        probe.write(b"quit\n")
        probe.wait(timeout=5)
        probe.resize(WindowSize(cols=80, rows=24))  # must not raise


class TestSubscribers:
    def test_late_subscriber_receives_scrollback(self, probe: PtySession) -> None:
        assert _await(_collect(probe), "isatty")
        late = _collect(probe)
        assert _await(late, "isatty"), "a viewer joining later must be able to repaint"

    def test_subscribing_after_exit_terminates(self, probe: PtySession) -> None:
        """A viewer arriving after exit gets the scrollback, then a clean stop."""
        chunks = _collect(probe)
        assert _await(chunks, "isatty")
        probe.write(b"quit\n")
        assert probe.wait(timeout=5) == 5
        replayed = b"".join(probe.subscribe())  # must not hang
        assert b"isatty" in replayed

    def test_slow_subscriber_is_cut_not_silently_truncated(self, probe: PtySession) -> None:
        """Terminal bytes are a stateful escape stream.

        Dropping chunks to keep a slow viewer alive would desynchronize its
        emulator permanently and invisibly, so an overrun subscriber is cut and
        told to resync instead.
        """
        stream = probe.subscribe()
        next(stream, None)  # register, then never drain again
        channel = probe._subscribers[0]
        _fill(channel)

        probe._broadcast(b"overflow")  # the chunk that finds the queue full

        assert channel not in probe._subscribers, "overrun subscriber must be dropped"
        assert _Signal.DESYNC in list(channel.queue)

    def test_desync_surfaces_to_the_consumer(self, probe: PtySession) -> None:
        stream = probe.subscribe()
        next(stream, None)
        channel = probe._subscribers[0]
        _fill(channel)
        probe._broadcast(b"overflow")

        with pytest.raises(SubscriberDesyncError, match="resubscribe"):
            for _ in stream:
                pass

    def test_scrollback_is_bounded(self, tmp_path: Path) -> None:
        noisy = PtySession(
            [sys.executable, "-u", "-c", f"print('x' * {SCROLLBACK_LIMIT * 2}, flush=True)"],
            cwd=tmp_path,
        )
        try:
            noisy.wait(timeout=10)
            time.sleep(0.3)
            replayed = b"".join(noisy.subscribe())
            assert len(replayed) <= SCROLLBACK_LIMIT
        finally:
            noisy.close()


class TestLifecycle:
    def test_close_terminates_a_live_child(self, tmp_path: Path) -> None:
        sleeper = PtySession(
            [sys.executable, "-u", "-c", "import time; time.sleep(300)"], cwd=tmp_path
        )
        assert sleeper.running
        sleeper.close()
        assert not sleeper.running

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        session = PtySession([sys.executable, "-c", "pass"], cwd=tmp_path)
        session.close()
        session.close()

    def test_empty_command_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            PtySession([], cwd=tmp_path)

    def test_missing_binary_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PtySession(["crossby-no-such-binary-xyz"], cwd=tmp_path)


class TestWindowSize:
    @pytest.mark.parametrize("cols,rows", [(0, 24), (80, 0), (-1, 24), (80, 20_000)])
    def test_rejects_implausible_dimensions(self, cols: int, rows: int) -> None:
        with pytest.raises(ValueError, match="between 1 and 10000"):
            WindowSize(cols=cols, rows=rows)


# Enters raw mode, then drops back to cooked — the window in which a late
# terminal reply is echoed as visible junk.
_MODE_FLIP = r"""
import sys, termios, time, tty
fd = sys.stdin.fileno()
saved = termios.tcgetattr(fd)
tty.setraw(fd)
sys.stdout.write("RAW\r\n"); sys.stdout.flush()
time.sleep(0.05)
termios.tcsetattr(fd, termios.TCSANOW, saved)
sys.stdout.write("COOKED\n"); sys.stdout.flush()
for line in sys.stdin:
    sys.stdout.write("got %s" % line); sys.stdout.flush()
"""


class TestTerminalReportFiltering:
    """A tool enables focus reporting, then turns it off ~1ms later, which
    re-enables echo. The browser's reply arrives after that flip and the kernel
    prints it as literal ``^[[I`` / ``^[[?1;2c``. A local terminal answers in
    microseconds and wins the race; a browser cannot.
    """

    @pytest.fixture
    def cooked(self, tmp_path: Path) -> PtySession:
        session = PtySession([sys.executable, "-u", "-c", _MODE_FLIP], cwd=tmp_path)
        yield session
        session.close()

    def _settled(self, session: PtySession) -> list[bytes]:
        chunks = _collect(session)
        assert _await(chunks, "COOKED"), "child never left raw mode"
        return chunks

    @pytest.mark.parametrize(
        "report",
        [b"\x1b[I", b"\x1b[O", b"\x1b[?1;2c", b"\x1b[>0;276;0c", b"\x1b[24;80R"],
    )
    def test_reports_are_dropped_while_echo_is_on(self, cooked: PtySession, report: bytes) -> None:
        chunks = self._settled(cooked)
        cooked.write(report)
        time.sleep(0.4)
        rendered = b"".join(chunks)
        assert report.lstrip(b"\x1b") not in rendered

    def test_real_keystrokes_are_never_dropped(self, cooked: PtySession) -> None:
        """The filter must not swallow input that merely contains an escape."""
        chunks = self._settled(cooked)
        cooked.write(b"hello\n")
        assert _await(chunks, "got hello")

    def test_report_with_trailing_input_is_delivered(self, cooked: PtySession) -> None:
        """Only a pure report is dropped; a report glued to real input is not."""
        chunks = self._settled(cooked)
        cooked.write(b"\x1b[Ityped\n")
        assert _await(chunks, "got")

    def test_echo_state_is_readable_through_the_master(self, cooked: PtySession) -> None:
        assert cooked._echo_enabled() is not None, "termios unreadable; filter would no-op"


class TestScrollbackTrimming:
    def test_trim_resumes_at_an_escape_boundary(self, tmp_path: Path) -> None:
        """Cutting at an arbitrary byte can land mid-sequence, and the replay
        then opens with a fragment the emulator renders as garbage."""
        session = PtySession([sys.executable, "-c", "pass"], cwd=tmp_path)
        try:
            session.wait(timeout=5)
            # Fill past the limit with well-formed sequences.
            unit = b"\x1b[38;2;43;45;49mX"
            session._scrollback = bytearray(unit * (SCROLLBACK_LIMIT // len(unit) + 64))
            session._trim_scrollback()
            assert len(session._scrollback) <= SCROLLBACK_LIMIT
            assert session._scrollback.startswith(b"\x1b"), (
                "replay must begin where a sequence begins"
            )
        finally:
            session.close()

    def test_trim_falls_back_when_no_boundary_is_near(self, tmp_path: Path) -> None:
        """Plain output with no escapes still gets bounded."""
        session = PtySession([sys.executable, "-c", "pass"], cwd=tmp_path)
        try:
            session.wait(timeout=5)
            session._scrollback = bytearray(
                b"x" * (SCROLLBACK_LIMIT + SCROLLBACK_RESYNC_WINDOW * 2)
            )
            session._trim_scrollback()
            assert len(session._scrollback) == SCROLLBACK_LIMIT
        finally:
            session.close()
