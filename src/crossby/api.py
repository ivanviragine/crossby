"""crossby's public Python API.

Everything exported here is the supported contract for programs that use
crossby as a library; every other module is internal and may change in any
release. The existing ``crossby.ai_tools`` exports stay supported as well.

Before 1.0, patch releases keep this module compatible. A minor release may
change it only with a migration note in its release notes, so pin a minor
series (``crossby>=0.46,<0.47``). Serialized results carry their own
``schema_version``, versioned independently of the package.

Asking one agent one question::

    from pathlib import Path
    from crossby.api import AskRequest, CliBackend, ProjectWorkspace, RestrictionChannel, ask

    result = ask(
        AskRequest(
            backend=CliBackend(tool="claude"),
            prompt="What would make you reject this pricing page?",
            workspace=ProjectWorkspace(path=Path("personas/cto"), instructions=persona_text),
            require_blocked=frozenset({RestrictionChannel.FILE_EDITS, RestrictionChannel.HOOKS}),
        )
    )
    if result.succeeded:
        print(result.text)

Each ask is a fresh, unattended session running under the tool's verified
read-only profile; the agent answers from ``workspace.path`` and never changes
it. A persona folder (:func:`provision`) holds one tool's own instruction and
skill files instead of an instructions envelope::

    folder = provision(
        OwnedWorkspaceSpec(path=Path("personas/cto-claude"), tool="claude",
                           instructions=persona_text, skills=(Path("skills/voice"),))
    )
    result = ask(AskRequest(backend=CliBackend(tool="claude"), prompt=q, workspace=folder))
    cleanup(folder)
 Calls are independent and safe to run in parallel threads: each session
gets its own process, environment and cancellation.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from crossby.models.ai import (
    AIToolID,
    EffortLevel,
    Restriction,
    RestrictionChannel,
    RestrictionState,
)
from crossby.models.ai import (
    HeadlessFailureReason as FailureReason,
)
from crossby.models.config import MCPServerConfig
from crossby.services.ask import (
    ENVELOPE_VERSION,
    SCHEMA_VERSION,
    AskError,
    AskRequest,
    AskRequestError,
    AskResult,
    AskStatus,
    AskUnsupportedError,
    CliBackend,
    CliProvenance,
    JsonOutput,
    NestedAskError,
    Output,
    PreflightReport,
    ProjectHooksError,
    ProjectWorkspace,
    RequirementNotMetError,
    TextOutput,
    UsageReport,
    WorkspaceChangedError,
    render_envelope,
    run_ask,
    run_preflight,
)
from crossby.services.converse import (
    ConversationResult,
    ConverseError,
    ConverseTurn,
    StopReason,
)
from crossby.services.converse import (
    Participant as ConverseParticipant,
)
from crossby.services.converse import (
    Role as ConverseRole,
)
from crossby.services.journal import ConversationEvent, EventSink, EventType
from crossby.services.panel import (
    HumanBackend,
    HumanInput,
    HumanReply,
    InputRequest,
    Lead,
    Member,
    PanelError,
    PanelResult,
    PanelSpec,
    PanelTurn,
    ProtocolError,
    render_panel_transcript,
)
from crossby.services.workspaces import (
    OwnedWorkspace,
    OwnedWorkspaceSpec,
    WorkspaceConflictError,
    WorkspaceDriftError,
    WorkspaceError,
    cleanup,
    provision,
)
from crossby.services.workspaces import verify as verify_workspace

__all__ = [
    "ENVELOPE_VERSION",
    "SCHEMA_VERSION",
    "AIToolID",
    "AskError",
    "AskRequest",
    "AskRequestError",
    "AskResult",
    "AskStatus",
    "AskUnsupportedError",
    "CliBackend",
    "CliProvenance",
    "ConversationEvent",
    "ConversationResult",
    "ConverseError",
    "ConverseParticipant",
    "ConverseRole",
    "ConverseTurn",
    "EffortLevel",
    "EventSink",
    "EventType",
    "FailureReason",
    "HumanBackend",
    "HumanInput",
    "HumanReply",
    "InputRequest",
    "JsonOutput",
    "Lead",
    "MCPServerConfig",
    "Member",
    "NestedAskError",
    "Output",
    "OwnedWorkspace",
    "OwnedWorkspaceSpec",
    "PanelError",
    "PanelResult",
    "PanelSpec",
    "PanelTurn",
    "PreflightReport",
    "ProjectHooksError",
    "ProjectWorkspace",
    "ProtocolError",
    "RequirementNotMetError",
    "Restriction",
    "RestrictionChannel",
    "RestrictionState",
    "StopReason",
    "TextOutput",
    "UsageReport",
    "WorkspaceChangedError",
    "WorkspaceConflictError",
    "WorkspaceDriftError",
    "WorkspaceError",
    "ask",
    "cleanup",
    "continue_panel",
    "panel",
    "preflight",
    "provision",
    "read_conversation",
    "read_panel",
    "render_envelope",
    "render_panel_transcript",
    "verify_workspace",
]


def ask(request: AskRequest, *, cancel: threading.Event | None = None) -> AskResult:
    """Ask one agent one question in a fresh read-only session.

    Raises :class:`AskError` (a subclass of it) when the ask is refused before
    any agent starts: an invalid request, a tool or build that cannot run it,
    an unmet ``require_blocked`` channel, or a call from inside an agent that
    crossby started. Once the agent has started, the outcome is always an
    :class:`AskResult`, including timeouts, cancellation and failures, with
    whatever usage and native IDs the run produced.

    Setting ``cancel`` stops the session and tears down the tool's processes;
    the result's status is then ``CANCELLED``.
    """
    return run_ask(request, cancel=cancel)


def preflight(request: AskRequest, *, resumed: bool = False) -> PreflightReport:
    """Report what ``ask(request)`` would run with, without starting the agent.

    It probes the installed CLI build and resolves every channel's
    restriction state for the read-only profile that build would use, and
    lists the required channels that would make the ask be refused. Checks
    only a real run can make, such as authentication, are listed as deferred.

    With ``resumed=True`` the seat is checked as a session that resumes an
    earlier one, as every panel turn after a seat's first does: a build whose
    unattended resume was not verified raises :class:`AskUnsupportedError`,
    as the panel's own preflight would refuse it.
    """
    return run_preflight(request, resumed=resumed)


def panel(
    spec: PanelSpec,
    *,
    on_event: EventSink | None = None,
    events_path: Path | None = None,
    inputs: HumanInput | None = None,
    cancel: threading.Event | None = None,
) -> PanelResult:
    """Run a panel: a lead consults read-only members by id until it answers.

    ``on_event`` receives every journal event, in order, on a dispatcher
    thread (it can never slow a turn, and an exception drops it). With
    ``events_path`` the events are also appended to that JSONL file as they
    happen. Agent members are always read-only, and so is the lead with
    ``Lead(read_only=True)``; see :class:`PanelSpec`. Human seats
    (``HumanBackend``) get their replies from ``inputs``: call
    ``inputs.submit(request_id, text)`` for each ``input.requested`` event.

    Setting ``cancel`` stops the running sessions and starts no new one; the
    result's ``stop_reason`` is then ``interrupted``, with every reply already
    committed, and :func:`continue_panel` can carry on from its journal.

    A journal problem never stops the panel: the result's ``journal_error``
    says why the events file is missing or incomplete, and ``sink_errors``
    lists each ``on_event`` failure that dropped the sink.
    """
    import uuid

    from crossby.services.journal import Journal, JournalError, JournalExistsError
    from crossby.services.panel import run_panel

    try:
        # Created exclusively and locked: a panel never appends to another
        # run's journal, and a run refused before it starts leaves none behind.
        journal = Journal(
            uuid.uuid4().hex, path=events_path, sinks=[on_event] if on_event is not None else []
        )
    except JournalExistsError:
        raise PanelError(
            f"{events_path} already holds a journal; choose another events_path."
        ) from None
    except JournalError as exc:
        raise PanelError(str(exc)) from None
    try:
        result = run_panel(spec, journal=journal, inputs=inputs, cancel=cancel)
    finally:
        journal.close()
    return _with_journal(result, journal)


def continue_panel(
    spec: PanelSpec,
    events_path: Path,
    *,
    on_event: EventSink | None = None,
    inputs: HumanInput | None = None,
    cancel: threading.Event | None = None,
) -> PanelResult:
    """Continue a panel that stopped without a final answer, from its journal.

    ``spec`` is the panel as first run: the same seats, each with the same
    tool, model and effort, or :class:`PanelError` is raised; its limits apply
    afresh, so the lead may send up to ``spec.max_rounds`` more messages.
    Every seat resumes its own session, members that had not answered the
    lead's last message are asked it again, and the events are appended to
    ``events_path``, which no other process may be writing. The result holds
    the whole panel, turns from before included.
    """
    from crossby.services.journal import Journal, JournalError
    from crossby.services.panel import result_from_journal, run_panel

    try:
        journal, events = Journal.reopen(
            events_path, sinks=[on_event] if on_event is not None else []
        )
    except JournalError as exc:
        raise PanelError(str(exc)) from None
    try:
        prior = result_from_journal(spec, events)
        result = run_panel(spec, journal=journal, inputs=inputs, cancel=cancel, resume=prior)
    finally:
        journal.close()
    return _with_journal(result, journal)


def read_panel(spec: PanelSpec, events_path: Path) -> PanelResult:
    """A panel's result as its journal records it, without running anything.

    ``spec`` names the panel's seats, as for :func:`continue_panel`. A panel
    that never recorded its end reads as ``interrupted``.
    """
    from crossby.services.journal import JournalError, read_journal
    from crossby.services.panel import result_from_journal

    try:
        events = read_journal(events_path)
    except JournalError as exc:
        raise PanelError(str(exc)) from None
    return result_from_journal(spec, events)


def read_conversation(events_path: Path) -> ConversationResult:
    """A ``crossby converse`` consultation as its journal records it.

    The goal, both sides with their tools, models and efforts, and every
    committed message in order; a consultation that never recorded its end
    reads as ``interrupted``. Raises :class:`ConverseError` when the file is
    not a consultation's journal.
    """
    from crossby.services.converse import result_from_journal
    from crossby.services.journal import JournalError, read_journal

    try:
        events = read_journal(events_path)
    except JournalError as exc:
        raise ConverseError(str(exc)) from None
    return result_from_journal(events)


def _with_journal(result: PanelResult, journal: Any) -> PanelResult:
    import dataclasses

    return dataclasses.replace(
        result, journal_error=journal.store_error, sink_errors=tuple(journal.sink_errors)
    )
