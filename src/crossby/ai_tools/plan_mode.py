"""Typed failures for the native plan-mode launch contract."""

from __future__ import annotations

from pathlib import Path

from crossby.models.ai import AIToolID, PlanModeCapability


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
