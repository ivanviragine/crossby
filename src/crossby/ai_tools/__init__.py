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
from crossby.ai_tools.plan_mode import (
    PlanArtifactAmbiguousError,
    PlanArtifactLocationError,
    PlanArtifactMalformedError,
    PlanArtifactMissingError,
    PlanBindingMismatchError,
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
    PlanApprovalPolicy,
    PlanArtifactSource,
    PlanInteraction,
    PlanInteractionKind,
    PlanInteractionOutcome,
    PlanInteractionResponse,
    PlanQuestionOption,
    PlanSessionBinding,
    PlanSessionRequest,
    PlanSessionResult,
)

__all__ = [
    "AbstractAITool",
    "PlanApprovalPolicy",
    "PlanArtifactAmbiguousError",
    "PlanArtifactLocationError",
    "PlanArtifactMalformedError",
    "PlanArtifactMissingError",
    "PlanArtifactSource",
    "PlanBindingMismatchError",
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
    "PlanQuestionOption",
    "PlanSessionBinding",
    "PlanSessionError",
    "PlanSessionRequest",
    "PlanSessionResult",
    "PlanSessionUnsupportedError",
    "PlanTransportError",
    "pick_best_model",
    "terminal_interaction_handler",
]
