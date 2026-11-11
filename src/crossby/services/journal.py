"""The event journal of a consultation: one ordered stream every view is built from.

Every consultation emits :class:`ConversationEvent` values with a sequence
number, appended to ``events.jsonl`` as they happen. The transcript,
``--json`` and ``--continue`` are all rebuilt from that stream, and a UI can
follow it live, by reading the file or through an :data:`EventSink`.

Sinks never run inside a session: the journal hands each event to a
dispatcher thread, so a slow or failing sink can delay only itself, never a
turn, and a sink that raises is dropped and noted while the consultation goes
on. Events carry messages as written but never native output: progress events
name only the kind of step a tool took.

One process writes a journal at a time: a :class:`Journal` holds an exclusive
lock on its file from opening until :meth:`Journal.close`, and a second
writer is refused with :class:`JournalBusyError`. Readers never take it.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import stat
import sys
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEMA_VERSION = 1
JOURNAL_FILE = "events.jsonl"


class EventType(StrEnum):
    """What happened. See the README for each type's ``data`` fields."""

    CONVERSATION_STARTED = "conversation.started"
    CONVERSATION_CONTINUED = "conversation.continued"
    PARTICIPANT_READY = "participant.ready"
    TURN_STARTED = "turn.started"
    TURN_PROGRESS = "turn.progress"
    TURN_FINISHED = "turn.finished"
    MESSAGE_COMMITTED = "message.committed"
    INPUT_REQUESTED = "input.requested"
    INPUT_RESOLVED = "input.resolved"
    CONVERSATION_FINISHED = "conversation.finished"


class ConversationEvent(BaseModel):
    """One journal entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    run_id: str
    seq: int = Field(ge=1)
    ts: datetime
    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)


EventSink = Callable[[ConversationEvent], None]


class JournalError(Exception):
    """A journal file is not a readable consultation journal."""


class JournalBusyError(JournalError):
    """Another process is writing this journal."""


class JournalExistsError(JournalError):
    """A new journal was asked for where one already holds events."""


_STOP = object()
# Windows byte-range locks are mandatory, so the lock sits on a byte no event
# reaches: readers in other processes are never blocked.
_WINDOWS_LOCK_OFFSET = 1 << 30
_OPEN_FLAGS = os.O_RDWR | os.O_APPEND | getattr(os, "O_BINARY", 0)


class Journal:
    """Sequences events, appends them to a file, and delivers them to sinks.

    ``emit`` is safe from any thread and does only bounded work: the file
    append and a queue put. Sinks run in order on one dispatcher thread.

    With ``path``, the journal is new: an existing file that holds anything
    raises :class:`JournalExistsError`, and one another process is writing
    raises :class:`JournalBusyError`. A file it created but never wrote to is
    removed on :meth:`close`, so a run that failed to start leaves nothing to
    refuse or continue. Any other failure to open or append is kept in
    ``store_error`` and the consultation goes on. :meth:`reopen` continues an
    existing journal.
    """

    def __init__(
        self,
        run_id: str,
        *,
        path: Path | None = None,
        sinks: Sequence[EventSink] = (),
        start_seq: int = 1,
        clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    ) -> None:
        self.run_id = run_id
        self.path = path
        self.events: list[ConversationEvent] = []
        self.store_error: str | None = None
        self.sink_errors: list[str] = []
        self._seq = start_seq - 1
        self._clock = clock
        self._lock = threading.Lock()
        self._fd: int | None = None
        self._created = False
        self._written = 0
        if path is not None:
            try:
                self._fd, self._created = _create(path)
            except OSError as exc:
                self.store_error = f"could not open {path}: {exc}"
        self._sinks = list(sinks)
        self._queue: queue.Queue[object] = queue.Queue()
        self._dispatcher: threading.Thread | None = None
        if self._sinks:
            self._dispatcher = threading.Thread(
                target=self._dispatch, daemon=True, name="crossby-journal"
            )
            self._dispatcher.start()

    @classmethod
    def reopen(
        cls,
        path: Path,
        *,
        sinks: Sequence[EventSink] = (),
        clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    ) -> tuple[Journal, list[ConversationEvent]]:
        """Lock an existing journal, then return it with its events, ready to append.

        The lock is taken before reading, so the events are the whole journal
        and no other process appends until :meth:`close`. An unfinished last
        line is dropped and a complete one terminated. Raises
        :class:`JournalBusyError` if another process is writing it, and
        :class:`JournalError` if it cannot be read or holds no events.
        """
        try:
            fd = os.open(path, _OPEN_FLAGS)
        except OSError as exc:
            raise JournalError(f"Cannot open {path}: {exc}") from None
        try:
            _claim(fd, path)
            raw = _repair_tail(fd)
            events = _parse(raw.decode("utf-8", errors="surrogateescape"), path)
            if not events:
                raise JournalError(f"{path} holds no events.")
        except OSError as exc:
            os.close(fd)
            raise JournalError(f"Cannot open {path}: {exc}") from None
        except BaseException:
            os.close(fd)
            raise
        journal = cls(events[-1].run_id, sinks=sinks, start_seq=events[-1].seq + 1, clock=clock)
        journal.path = path
        journal._fd = fd
        return journal, events

    def emit(self, type_: EventType, **data: Any) -> ConversationEvent:
        with self._lock:
            self._seq += 1
            event = ConversationEvent(
                run_id=self.run_id, seq=self._seq, ts=self._clock(), type=type_, data=data
            )
            self.events.append(event)
            if self._fd is not None:
                try:
                    _write_all(self._fd, (event.model_dump_json() + "\n").encode())
                    self._written += 1
                except OSError as exc:
                    # Keep the consultation going; the caller can report it.
                    self.store_error = f"could not append to {self.path}: {exc}"
            if self._dispatcher is not None:
                self._queue.put(event)
        return event

    def _dispatch(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            assert isinstance(item, ConversationEvent)
            for sink in list(self._sinks):
                try:
                    sink(item)
                except Exception as exc:
                    self._sinks.remove(sink)
                    self.sink_errors.append(f"an event sink failed and was dropped: {exc!r}")

    def close(self, timeout: float = 5.0) -> None:
        """Deliver what is queued (up to ``timeout``), close the file and release it."""
        if self._dispatcher is not None:
            self._queue.put(_STOP)
            self._dispatcher.join(timeout)
            self._dispatcher = None
        with self._lock:
            fd, self._fd = self._fd, None
            if fd is None:
                return
            if not (self._created and self._written == 0 and self.path is not None):
                os.close(fd)
                return
            # Nothing was recorded: drop the empty file this journal created.
            # Unlink while still holding the lock where the platform allows it.
            if sys.platform == "win32":
                os.close(fd)
                with contextlib.suppress(OSError):
                    if os.stat(self.path).st_size == 0:
                        self.path.unlink()
                return
            with contextlib.suppress(OSError):
                if _same_file(fd, self.path) and os.fstat(fd).st_size == 0:
                    self.path.unlink()
            os.close(fd)


def _create(path: Path) -> tuple[int, bool]:
    """Open a new journal at ``path``, locked; ``(fd, created)``.

    An existing empty regular file is taken over (not ``created``); one that
    holds anything raises :class:`JournalExistsError`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, _OPEN_FLAGS | os.O_CREAT | os.O_EXCL, 0o666)
        created = True
    except FileExistsError:
        fd = os.open(path, _OPEN_FLAGS)
        created = False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise JournalError(f"{path} is not a regular file.")
        _claim(fd, path)
        if not created and os.fstat(fd).st_size:
            raise JournalExistsError(f"{path} already holds a journal.")
    except BaseException:
        os.close(fd)
        raise
    return fd, created


def _claim(fd: int, path: Path) -> None:
    """Take this process's exclusive lock on an open journal, without waiting.

    The lock goes with the descriptor: closing it, or the process ending,
    releases it, so a crash never leaves a journal claimed.
    """
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if sys.platform == "win32" or isinstance(exc, BlockingIOError):
            raise JournalBusyError(
                f"Another process is writing {path}; wait for it to finish."
            ) from None
        raise


def _same_file(fd: int, path: Path) -> bool:
    held, named = os.fstat(fd), os.stat(path, follow_symlinks=False)
    return (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _read_all(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(fd, 1 << 16):
        chunks.append(chunk)
    return b"".join(chunks)


def _repair_tail(fd: int) -> bytes:
    """Make an open, locked journal safe to append to; return its bytes as kept.

    A last line without its newline is either an append that did not finish
    (dropped, as :func:`read_journal` ignores it) or a complete event that
    only lacks its newline (kept, and terminated).
    """
    raw = _read_all(fd)
    if not raw or raw.endswith(b"\n"):
        return raw
    cut = raw.rfind(b"\n") + 1
    try:
        ConversationEvent.model_validate(json.loads(raw[cut:]))
    except (ValueError, ValidationError):
        os.ftruncate(fd, cut)
        return raw[:cut]
    _write_all(fd, b"\n")
    return raw + b"\n"


def read_journal(path: Path) -> list[ConversationEvent]:
    """Every event in a journal file, in order.

    A last line without its newline is an append the process did not finish
    and is ignored; any other unreadable line raises :class:`JournalError`.
    Reading takes no lock, so it is safe while a consultation is running.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="surrogateescape")
    except OSError as exc:
        raise JournalError(f"Cannot read {path}: {exc}") from None
    return _parse(raw, path)


def _parse(raw: str, path: Path) -> list[ConversationEvent]:
    lines = raw.split("\n")
    complete, last = lines[:-1], lines[-1]
    events: list[ConversationEvent] = []
    for number, line in enumerate(complete, start=1):
        if not line.strip():
            continue
        try:
            events.append(ConversationEvent.model_validate(json.loads(line)))
        except (ValueError, ValidationError) as exc:
            raise JournalError(f"{path} line {number} is not a journal event: {exc}") from None
    if last.strip():
        # A last line without its newline is an append that did not finish.
        with contextlib.suppress(ValueError, ValidationError):
            events.append(ConversationEvent.model_validate(json.loads(last)))
    for expected, event in enumerate(events, start=1):
        if event.seq != expected:
            raise JournalError(f"{path} skips from event {expected - 1} to {event.seq}.")
        if event.schema_version != SCHEMA_VERSION:
            raise JournalError(f"{path} uses journal schema {event.schema_version}.")
    return events


def finished(events: Sequence[ConversationEvent]) -> bool:
    """Whether the last run recorded in ``events`` reached its end."""
    for event in reversed(events):
        if event.type is EventType.CONVERSATION_FINISHED:
            return True
        if event.type in {EventType.CONVERSATION_STARTED, EventType.CONVERSATION_CONTINUED}:
            return False
    return False


__all__ = [
    "JOURNAL_FILE",
    "SCHEMA_VERSION",
    "ConversationEvent",
    "EventSink",
    "EventType",
    "Journal",
    "JournalBusyError",
    "JournalError",
    "JournalExistsError",
    "finished",
    "read_journal",
]
