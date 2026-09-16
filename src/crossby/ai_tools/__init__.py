"""AI tool adapters — ABC with self-registering concrete implementations.

Import all adapters here to trigger __init_subclass__ registration.
"""

# Import adapters to trigger registration
import crossby.ai_tools.antigravity
import crossby.ai_tools.antigravity_cli
import crossby.ai_tools.claude
import crossby.ai_tools.codex
import crossby.ai_tools.copilot
import crossby.ai_tools.cursor
import crossby.ai_tools.opencode
import crossby.ai_tools.vscode  # noqa: F401
from crossby.ai_tools.base import AbstractAITool, pick_best_model
from crossby.ai_tools.interactive import InteractiveLaunchHandler, InteractiveSession
from crossby.ai_tools.plan_mode import (
    PlanArtifactAmbiguousError,
    PlanArtifactLocationError,
    PlanArtifactMalformedError,
    PlanArtifactMissingError,
    PlanBindingMismatchError,
    PlanCommandPolicyUnsupportedError,
    PlanInteractionHandler,
    PlanInteractionRequiredError,
    PlanModeAdapterContractError,
    PlanModeConflictError,
    PlanModeLaunchError,
    PlanModeUnsupportedError,
    PlanSessionError,
    PlanSessionUnsupportedError,
    PlanTransportError,
    terminal_interaction_handler,
)
from crossby.models.ai import (
    AIToolID,
    InteractiveLaunchEvent,
    InteractiveLaunchEventKind,
    PlanApprovalPolicy,
    PlanArtifactSource,
    PlanCommandPolicy,
    PlanCommandPolicySupport,
    PlanInteraction,
    PlanInteractionKind,
    PlanInteractionOutcome,
    PlanInteractionResponse,
    PlanNativeBindingID,
    PlanOperation,
    PlanOperationKind,
    PlanPermissionTarget,
    PlanPermissionTargetKind,
    PlanPreflightCheck,
    PlanPreflightDeferredCheck,
    PlanQuestionOption,
    PlanSessionBinding,
    PlanSessionPreflight,
    PlanSessionRequest,
    PlanSessionResult,
)


def preflight_plan_session(
    tool: AIToolID | str,
    request: PlanSessionRequest,
    *,
    timeout_seconds: float = 5.0,
) -> PlanSessionPreflight:
    """Preflight a complete collected session without creating workspace artifacts."""
    return AbstractAITool.get(tool).preflight_plan_session(
        request,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "AbstractAITool",
    "InteractiveLaunchEvent",
    "InteractiveLaunchEventKind",
    "InteractiveLaunchHandler",
    "InteractiveSession",
    "PlanApprovalPolicy",
    "PlanArtifactAmbiguousError",
    "PlanArtifactLocationError",
    "PlanArtifactMalformedError",
    "PlanArtifactMissingError",
    "PlanArtifactSource",
    "PlanBindingMismatchError",
    "PlanCommandPolicy",
    "PlanCommandPolicySupport",
    "PlanCommandPolicyUnsupportedError",
    "PlanInteraction",
    "PlanInteractionHandler",
    "PlanInteractionKind",
    "PlanInteractionOutcome",
    "PlanInteractionRequiredError",
    "PlanInteractionResponse",
    "PlanModeAdapterContractError",
    "PlanModeConflictError",
    "PlanModeLaunchError",
    "PlanModeUnsupportedError",
    "PlanNativeBindingID",
    "PlanOperation",
    "PlanOperationKind",
    "PlanPermissionTarget",
    "PlanPermissionTargetKind",
    "PlanPreflightCheck",
    "PlanPreflightDeferredCheck",
    "PlanQuestionOption",
    "PlanSessionBinding",
    "PlanSessionError",
    "PlanSessionPreflight",
    "PlanSessionRequest",
    "PlanSessionResult",
    "PlanSessionUnsupportedError",
    "PlanTransportError",
    "pick_best_model",
    "preflight_plan_session",
    "terminal_interaction_handler",
]
