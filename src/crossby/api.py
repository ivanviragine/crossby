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
    ProjectWorkspace,
    RequirementNotMetError,
    TextOutput,
    UsageReport,
    WorkspaceChangedError,
    render_envelope,
    run_ask,
    run_preflight,
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
    "EffortLevel",
    "FailureReason",
    "JsonOutput",
    "NestedAskError",
    "Output",
    "OwnedWorkspace",
    "OwnedWorkspaceSpec",
    "PreflightReport",
    "ProjectWorkspace",
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
    "preflight",
    "provision",
    "render_envelope",
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
