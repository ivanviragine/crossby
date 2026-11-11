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
from crossby.services.journal import ConversationEvent, EventSink, EventType
from crossby.services.panel import (
    Lead,
    Member,
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
    "EffortLevel",
    "EventSink",
    "EventType",
    "FailureReason",
    "JsonOutput",
    "Lead",
    "Member",
    "NestedAskError",
    "Output",
    "OwnedWorkspace",
    "OwnedWorkspaceSpec",
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
    "TextOutput",
    "UsageReport",
    "WorkspaceChangedError",
    "WorkspaceConflictError",
    "WorkspaceDriftError",
    "WorkspaceError",
    "ask",
    "cleanup",
    "panel",
    "preflight",
    "provision",
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


def preflight(request: AskRequest) -> PreflightReport:
    """Report what ``ask(request)`` would run with, without starting the agent.

    It probes the installed CLI build and resolves every channel's
    restriction state for the read-only profile that build would use, and
    lists the required channels that would make the ask be refused. Checks
    only a real run can make, such as authentication, are listed as deferred.
    """
    return run_preflight(request)


def panel(
    spec: PanelSpec,
    *,
    on_event: EventSink | None = None,
    events_path: Path | None = None,
) -> PanelResult:
    """Run a panel: a lead consults read-only members by id until it answers.

    ``on_event`` receives every journal event, in order, on a dispatcher
    thread (it can never slow a turn, and an exception drops it). With
    ``events_path`` the events are also appended to that JSONL file as they
    happen. Members are always read-only; see :class:`PanelSpec`.

    A journal problem never stops the panel: the result's ``journal_error``
    says why the events file is missing or incomplete, and ``sink_errors``
    lists each ``on_event`` failure that dropped the sink.
    """
    import dataclasses
    import uuid

    from crossby.services.journal import Journal, JournalError, JournalExistsError
    from crossby.services.panel import PanelError, run_panel

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
        result = run_panel(spec, journal=journal)
    finally:
        journal.close()
    return dataclasses.replace(
        result, journal_error=journal.store_error, sink_errors=tuple(journal.sink_errors)
    )
