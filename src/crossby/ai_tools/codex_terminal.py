"""Temporary Codex 0.154 terminal Plan activation.

Only startup is automated. After activation/input acknowledgement the same PTY
relays the user's terminal unchanged. No model answers or artifacts are parsed.
"""

from __future__ import annotations

import argparse
import errno
import math
import os
import re
import select
import signal
import sys
import threading
import time
import unicodedata
from collections.abc import Callable
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import BinaryIO

from crossby.ai_tools.interactive import InteractiveLaunchHandler
from crossby.models.ai import AIToolID, InteractiveLaunchEvent, InteractiveLaunchEventKind

MAX_MESSAGE_BYTES = 100_000
STARTUP_TIMEOUT = 120.0


class TerminalStartupError(RuntimeError):
    """Startup could not be confirmed; the initial task is never retried."""


def validate_message(message: str) -> None:
    """Keep terminal controls and native slash/shell commands out of task input."""
    if not message.strip():
        raise ValueError("The initial task message must not be blank")
    if message.lstrip().startswith(("/", "!")):
        raise ValueError("The initial task must be ordinary text, not a slash or shell command")
    if any(unicodedata.category(char) in {"Cc", "Cs"} and char not in "\n\t" for char in message):
        raise ValueError("The initial task contains unsupported terminal control characters")
    if len(message.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ValueError(f"The initial task exceeds {MAX_MESSAGE_BYTES} UTF-8 bytes")


class _InputPort:
    def __init__(self) -> None:
        self.message: str | None = None
        self.open = False

    def send_message(self, message: str) -> None:
        if not self.open or self.message is not None:
            raise ValueError("Send exactly one initial task during PLAN_READY")
        validate_message(message)
        self.message = message


class _Phase(StrEnum):
    READY = "waiting for Codex's ready composer"
    COMMAND = "waiting for /plan to appear"
    PLAN = "waiting for native Plan activation"
    PASTE = "waiting for the initial task paste"
    START = "waiting for the first native Plan turn"
    ACTIVE = "interactive"


class _Startup:
    """Bounded state machine driven by reconstructed terminal screen updates."""

    def __init__(self, prompt: str | None, on_event: InteractiveLaunchHandler | None) -> None:
        if prompt is not None:
            validate_message(prompt)
        if prompt is not None and on_event is not None:
            raise ValueError("Choose prompt or an event handler for initial input, not both")
        self.phase = _Phase.READY
        self.prompt = prompt
        self.on_event = on_event
        self.port = _InputPort()
        self.user_screen = False

    def _emit(self, kind: InteractiveLaunchEventKind) -> None:
        if self.on_event is not None:
            self.on_event(InteractiveLaunchEvent(kind=kind, tool_id=AIToolID.CODEX), self.port)

    def advance(self, rendered: str) -> bytes:
        lines = [line.rstrip() for line in rendered.splitlines() if line.strip()]
        footer = lines[-1] if lines else ""
        empty = any(line.lstrip() == "\u203a Ask Codex to do anything" for line in lines)
        composer = next(
            (line.lstrip()[2:] for line in lines if line.lstrip().startswith("\u203a ")), ""
        )
        self.user_screen = self.phase is _Phase.READY and (
            "Do you trust the contents of this directory?" in " ".join(rendered.split())
            or "sign in" in rendered.lower()
        )
        if self.phase is _Phase.READY:
            loaded = re.search(r"model:\s+(?!loading\b)\S", rendered)
            if loaded and empty and not self.user_screen:
                self.phase = _Phase.COMMAND
                return b"/plan"
        elif self.phase is _Phase.COMMAND and composer == "/plan":
            self.phase = _Phase.PLAN
            return b"\r"
        elif self.phase is _Phase.PLAN and footer.endswith("Plan mode") and empty:
            self.port.open = True
            try:
                self._emit(InteractiveLaunchEventKind.PLAN_READY)
                if self.prompt is not None:
                    self.port.send_message(self.prompt)
            finally:
                self.port.open = False
            if self.port.message is None:
                self.phase = _Phase.ACTIVE
                return b""
            self.phase = _Phase.PASTE
            return b"\x1b[200~" + self.port.message.encode("utf-8") + b"\x1b[201~"
        elif self.phase is _Phase.PASTE:
            assert self.port.message is not None
            # Codex expands large-paste placeholders only on submission. Small
            # pastes render directly; the leading line proves paste completion.
            placeholder = f"[Pasted Content {len(self.port.message)} chars]"
            first_line = next(
                line for line in self.port.message.splitlines() if line.strip()
            ).expandtabs(4)
            if composer == placeholder or (
                len(self.port.message) <= 1000
                and composer.startswith(first_line[:40])
                and bool(first_line[:40].strip())
            ):
                self.phase = _Phase.START
                return b"\r"
        elif (
            self.phase is _Phase.START
            and "esc to interrupt" in rendered
            and footer.endswith("Plan mode")
        ):
            self.phase = _Phase.ACTIVE
            self._emit(InteractiveLaunchEventKind.MESSAGE_SUBMITTED)
        return b""


class _TerminalReplies:
    """Relay terminal query responses while rejecting keystrokes during startup.

    Replies may arrive split across reads. Keep incomplete sequences bounded;
    never forward a partial escape that could consume the injected task text.
    """

    _reply = re.compile(
        rb"\x1b\[(?:[0-9]+;[0-9]+R|[?>][0-9;]*c|\?[0-9]+u|[IO])"
        rb"|\x1b\](?:10|11);rgb:[0-9a-fA-F/]+(?:\x07|\x1b\\)"
    )

    def __init__(self) -> None:
        self.pending = b""

    def feed(self, data: bytes) -> bytes:
        self.pending += data
        complete = bytearray()
        while self.pending:
            match = self._reply.match(self.pending)
            if match:
                complete.extend(match.group())
                self.pending = self.pending[match.end() :]
                continue
            # A possible split CSI/OSC response, with no final byte yet.
            partial = re.fullmatch(
                rb"\x1b(?:\[[?>0-9;]*|\][0-9a-fA-Frgb/:;]*\x1b?)?",
                self.pending,
            )
            if partial and len(self.pending) < 256:
                break
            raise TerminalStartupError("Input interrupted automatic Plan startup; task not retried")
        return bytes(complete)


class _SignalExit(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signum


def run_terminal_plan(
    command: list[str],
    working_dir: Path,
    *,
    prompt: str | None = None,
    transcript_path: Path | None = None,
    env: dict[str, str] | None = None,
    on_event: InteractiveLaunchHandler | None = None,
    startup_timeout: float = STARTUP_TIMEOUT,
) -> int:
    """Run the native terminal with a one-shot, observed startup handshake."""
    if os.name != "posix" or not sys.stdin.isatty() or not sys.stdout.isatty():
        raise TerminalStartupError("Codex Plan startup requires a POSIX input/output terminal")
    if threading.current_thread() is not threading.main_thread():
        raise TerminalStartupError("Interactive Plan launch must run on the main thread")
    if not math.isfinite(startup_timeout) or startup_timeout <= 0 or not command:
        raise ValueError("A command and positive startup timeout are required")

    # Keep POSIX and optional terminal dependencies off other adapters' import paths.
    import termios
    import tty

    import pexpect
    import pyte

    startup = _Startup(prompt, on_event)
    input_fd = sys.stdin.fileno()
    output_fd = sys.stdout.fileno()
    size = os.get_terminal_size(output_fd)
    rows, columns = size.lines or 24, size.columns or 80
    saved_tty = termios.tcgetattr(input_fd)
    child = pexpect.spawn(
        command[0],
        command[1:],
        cwd=str(working_dir),
        env=env,
        encoding=None,
        dimensions=(rows, columns),
        echo=False,
    )
    screen = pyte.Screen(columns, rows)
    stream = pyte.ByteStream(screen)
    child_fd = child.child_fd
    os.set_blocking(child_fd, False)
    outgoing = bytearray()
    replies = _TerminalReplies()
    deadline = time.monotonic() + startup_timeout
    saved_handlers: dict[
        signal.Signals, Callable[[int, FrameType | None], object] | int | None
    ] = {}
    transcript: BinaryIO | None = None

    def resize(_signum: int, _frame: object) -> None:
        dimensions = os.get_terminal_size(output_fd)
        if dimensions.lines > 0 and dimensions.columns > 0:
            screen.resize(dimensions.lines, dimensions.columns)
            child.setwinsize(dimensions.lines, dimensions.columns)

    def terminate(signum: int, _frame: object) -> None:
        raise _SignalExit(signum)

    try:
        if transcript_path is not None:
            transcript = transcript_path.open("ab")
        tty.setraw(input_fd)
        for signum in (signal.SIGWINCH, signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            saved_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, resize if signum == signal.SIGWINCH else terminate)
        while True:
            active = startup.phase is _Phase.ACTIVE
            remaining = deadline - time.monotonic()
            if not active and remaining <= 0:
                raise TerminalStartupError(
                    f"Codex startup timed out {startup.phase.value}. The task was not retried."
                )
            readable, writable, _ = select.select(
                [child_fd, input_fd],
                [child_fd] if outgoing else [],
                [],
                1.0 if active else min(1.0, remaining),
            )
            if input_fd in readable:
                user_input = os.read(input_fd, 65536)
                if not user_input:
                    return 130
                if not active and not startup.user_screen:
                    user_input = replies.feed(user_input)
                if len(outgoing) + len(user_input) > MAX_MESSAGE_BYTES + 65536:
                    raise TerminalStartupError("Terminal input exceeded the bounded relay buffer")
                outgoing.extend(user_input)
            if child_fd in writable:
                try:
                    count = os.write(child_fd, outgoing[:65536])
                    del outgoing[:count]
                except BlockingIOError:
                    pass
            if child_fd not in readable:
                if not child.isalive():
                    break
                continue
            try:
                data = os.read(child_fd, 65536)
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                data = b""
            if not data:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            if transcript is not None:
                transcript.write(data)
                transcript.flush()
            if not active:
                stream.feed(data)
                if not outgoing:
                    outgoing.extend(startup.advance("\n".join(screen.display)))
        if startup.phase is not _Phase.ACTIVE:
            raise TerminalStartupError(f"Codex exited {startup.phase.value}; task not retried")
        # EOF can precede waitpid visibility (notably on macOS). Allow bounded
        # process reaping without treating an ordinary native exit as failure.
        exit_deadline = time.monotonic() + 2.0
        while child.isalive():
            if time.monotonic() >= exit_deadline:
                raise TerminalStartupError("Codex closed its terminal but did not exit")
            time.sleep(0.01)
        if child.exitstatus is not None:
            return int(child.exitstatus)
        return 128 + int(child.signalstatus or 1)
    except _SignalExit as exc:
        return 128 + exc.signum
    finally:
        try:
            # PTY children lead their own process group. Only clean up this run.
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(child.pid, signal.SIGTERM)
            child.close(force=True)
        finally:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(child.pid, signal.SIGKILL)
            for signum, handler in saved_handlers.items():
                signal.signal(signum, handler)
            termios.tcsetattr(input_fd, termios.TCSADRAIN, saved_tty)
            if transcript is not None:
                transcript.close()


def parse_wrapper_args(argv: list[str]) -> tuple[str | None, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-message")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a native Codex command is required after --")
    return args.initial_message, command


def main() -> int:
    from crossby.ai_tools.codex import CodexAdapter
    from crossby.ai_tools.plan_mode import PlanModeLaunchError

    prompt, command = parse_wrapper_args(sys.argv[1:])
    try:
        CodexAdapter().validate_plan_mode_request(plan_mode=True, initial_message=prompt)
        return run_terminal_plan(command, Path.cwd(), prompt=prompt)
    except (TerminalStartupError, PlanModeLaunchError, ValueError) as exc:
        print(f"Crossby: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
