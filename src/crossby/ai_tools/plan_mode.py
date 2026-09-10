"""Typed failures and interactions for native plan-mode sessions."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from crossby.models.ai import (
    AIToolID,
    PlanInteraction,
    PlanInteractionOutcome,
    PlanInteractionResponse,
    PlanModeCapability,
)

PlanInteractionHandler = Callable[[PlanInteraction], PlanInteractionResponse]

_SECRET_RE = re.compile(
    r"(?i)\b(token|password|secret|api[_ -]?key|authorization|bearer)\b\s*[:=]?\s*\S+"
)


def safe_error_excerpt(text: str | None, *, limit: int = 500) -> str | None:
    """Return a short single-line diagnostic with obvious secret values redacted."""
    if not text:
        return None
    compact = " ".join(text.split())
    redacted = _SECRET_RE.sub(lambda match: f"{match.group(1)}=<redacted>", compact)
    return redacted[:limit] or None


class PlanModeLaunchError(RuntimeError):
    """Base class for requests Crossby must reject before tool launch."""

    def __init__(
        self,
        message: str,
        *,
        tool_id: AIToolID,
        capability: PlanModeCapability,
    ) -> None:
        super().__init__(message)
        self.tool_id = tool_id
        self.capability = capability


class PlanModeUnsupportedError(PlanModeLaunchError):
    """The adapter cannot guarantee native plan-mode activation."""

    @classmethod
    def for_tool(
        cls,
        *,
        tool_id: AIToolID,
        display_name: str,
        capability: PlanModeCapability,
    ) -> PlanModeUnsupportedError:
        remediation = capability.remediation or "Use a natively supported planning harness."
        return cls(
            f"{display_name} cannot guarantee native plan mode: "
            f"{capability.activation_detail} Remediation: {remediation}",
            tool_id=tool_id,
            capability=capability,
        )

    @classmethod
    def for_installed_version(
        cls,
        *,
        tool_id: AIToolID,
        display_name: str,
        capability: PlanModeCapability,
        installed_version: tuple[int, int, int] | None,
    ) -> PlanModeUnsupportedError:
        """Report an installed CLI that is older than, or cannot satisfy, the contract."""
        detected = (
            ".".join(str(part) for part in installed_version)
            if installed_version is not None
            else "unknown"
        )
        verified = capability.verified_version or "an adapter-verified release"
        return cls(
            f"{display_name} cannot guarantee native plan mode for installed version "
            f"{detected}. Crossby requires {capability.version_requirement} "
            f"The oldest release verified by this adapter is {verified}. "
            f"Remediation: upgrade {display_name} to {verified} or newer, then retry.",
            tool_id=tool_id,
            capability=capability,
        )


class PlanModeConflictError(PlanModeLaunchError):
    """Plan mode was combined with a contradictory autonomy request."""

    @classmethod
    def for_tool(
        cls,
        *,
        tool_id: AIToolID,
        display_name: str,
        capability: PlanModeCapability,
        conflicts: tuple[str, ...],
    ) -> PlanModeConflictError:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in conflicts)
        return cls(
            f"{display_name} plan mode cannot be combined with {flags}. "
            "Remove the autonomy flag(s); native plan mode must remain the effective mode.",
            tool_id=tool_id,
            capability=capability,
        )


class PlanArtifactLocationError(PlanModeLaunchError):
    """The requested filesystem plan output conflicts with harness storage."""

    @classmethod
    def for_tool(
        cls,
        *,
        tool_id: AIToolID,
        display_name: str,
        capability: PlanModeCapability,
        requested_dir: Path,
    ) -> PlanArtifactLocationError:
        remediation = capability.remediation or "Use a harness that supports the requested path."
        return cls(
            f"{display_name} cannot guarantee plan artifacts in {requested_dir}: "
            f"{capability.artifact_location_detail} Remediation: {remediation}",
            tool_id=tool_id,
            capability=capability,
        )


class PlanModeAdapterContractError(PlanModeLaunchError):
    """An adapter's typed declaration and activation implementation disagree."""


class PlanSessionError(RuntimeError):
    """Base failure for a complete collected planning lifecycle."""

    def __init__(
        self,
        message: str,
        *,
        tool_id: AIToolID,
        capability: PlanModeCapability,
        exit_code: int | None = None,
        session_id: str | None = None,
        thread_id: str | None = None,
        turn_id: str | None = None,
        artifact_id: str | None = None,
        paths: tuple[Path, ...] = (),
        stderr: str | None = None,
    ) -> None:
        excerpt = safe_error_excerpt(stderr)
        if excerpt:
            message = f"{message} Diagnostic: {excerpt}"
        super().__init__(message)
        self.tool_id = tool_id
        self.capability = capability
        self.exit_code = exit_code
        self.session_id = session_id
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.artifact_id = artifact_id
        self.paths = paths
        self.stderr_excerpt = excerpt


class PlanSessionUnsupportedError(PlanSessionError):
    """The adapter/version/request cannot provide a collected plan session."""

    @classmethod
    def for_tool(
        cls,
        *,
        tool_id: AIToolID,
        display_name: str,
        capability: PlanModeCapability,
    ) -> PlanSessionUnsupportedError:
        remediation = capability.remediation or "Use an adapter with a complete collector."
        return cls(
            f"{display_name} does not support collected native plan sessions. "
            f"Remediation: {remediation}",
            tool_id=tool_id,
            capability=capability,
        )

    @classmethod
    def for_installed_version(
        cls,
        *,
        tool_id: AIToolID,
        display_name: str,
        capability: PlanModeCapability,
        installed_version: str | None,
    ) -> PlanSessionUnsupportedError:
        detected = installed_version or "unknown"
        verified = capability.verified_version or "an adapter-verified release"
        return cls(
            f"{display_name} cannot collect a native plan for installed version {detected}. "
            f"Crossby requires {capability.version_requirement} The oldest release verified by "
            f"this collector is {verified}. Remediation: upgrade {display_name} to {verified} "
            "or newer, then retry.",
            tool_id=tool_id,
            capability=capability,
        )


class PlanInteractionRequiredError(PlanSessionError):
    """A native question cannot continue without an explicit caller response."""

    def __init__(
        self,
        message: str,
        *,
        interaction: PlanInteraction,
        tool_id: AIToolID,
        capability: PlanModeCapability,
    ) -> None:
        super().__init__(
            message,
            tool_id=tool_id,
            capability=capability,
            session_id=interaction.session_id,
            thread_id=interaction.thread_id,
            turn_id=interaction.turn_id,
            artifact_id=interaction.artifact_id,
        )
        self.interaction = interaction


class PlanTransportError(PlanSessionError):
    """The harness process or protocol failed before a valid artifact arrived."""


class PlanArtifactMissingError(PlanSessionError):
    """A successful harness run produced no session-bound artifact."""


class PlanArtifactAmbiguousError(PlanSessionError):
    """More than one candidate artifact matched the launched session."""


class PlanArtifactMalformedError(PlanSessionError):
    """The authoritative artifact was present but violated its contract."""


class PlanBindingMismatchError(PlanSessionError):
    """An artifact or event belongs to a different session/turn/run."""


def terminal_interaction_handler(interaction: PlanInteraction) -> PlanInteractionResponse:
    """Minimal terminal-backed handler used only when stdin is interactive."""
    print(interaction.prompt)
    for option in interaction.options:
        description = f" — {option.description}" if option.description else ""
        print(f"  {option.option_id}: {option.label}{description}")
    answer = input("> ").strip()
    if not answer:
        return PlanInteractionResponse(outcome=PlanInteractionOutcome.SKIPPED)
    option_ids = {option.option_id for option in interaction.options}
    selected = tuple(part.strip() for part in answer.split(",") if part.strip())
    if interaction.allow_multiple and selected and all(part in option_ids for part in selected):
        return PlanInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED,
            option_ids=selected,
        )
    if answer in option_ids:
        return PlanInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED,
            option_id=answer,
        )
    return PlanInteractionResponse(outcome=PlanInteractionOutcome.ANSWERED, answer=answer)
