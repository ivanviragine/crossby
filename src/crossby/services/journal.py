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
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import IO, Any

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


_STOP = object()


class Journal:
    """Sequences events, appends them to a file, and delivers them to sinks.

    ``emit`` is safe from any thread and does only bounded work: the file
    append and a queue put. Sinks run in order on one dispatcher thread.
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
        self._file: IO[str] | None = None
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._file = path.open("a", encoding="utf-8", errors="surrogateescape")
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

    def emit(self, type_: EventType, **data: Any) -> ConversationEvent:
        with self._lock:
            self._seq += 1
            event = ConversationEvent(
                run_id=self.run_id, seq=self._seq, ts=self._clock(), type=type_, data=data
            )
            self.events.append(event)
            if self._file is not None:
                try:
                    self._file.write(event.model_dump_json() + "\n")
                    self._file.flush()
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
        """Deliver what is queued (up to ``timeout``) and close the file."""
        if self._dispatcher is not None:
            self._queue.put(_STOP)
            self._dispatcher.join(timeout)
            self._dispatcher = None
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


def read_journal(path: Path) -> list[ConversationEvent]:
    """Every event in a journal file, in order.

    A last line without its newline is an append the process did not finish
    and is ignored; any other unreadable line raises :class:`JournalError`.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="surrogateescape")
    except OSError as exc:
        raise JournalError(f"Cannot read {path}: {exc}") from None
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
    "JournalError",
    "finished",
    "read_journal",
]
