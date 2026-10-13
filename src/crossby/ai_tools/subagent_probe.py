"""Bounded, content-free failures shared by runtime-readiness probes."""

from __future__ import annotations

import subprocess
from time import monotonic
from typing import TYPE_CHECKING

from crossby.ai_tools.session_process import run_captured
from crossby.models.ai import HeadlessInteractionMode, SubagentSpawnReason
from crossby.utils.versioning import parse_semver

if TYPE_CHECKING:
    from crossby.ai_tools.base import AbstractAITool
    from crossby.ai_tools.subagent_context import SubagentLaunchContext
    from crossby.models.ai import HeadlessSessionRequest

PROBE_SECONDS = 5.0


class ReadinessProbeError(ValueError):
    """Safe failure classification; never retain a native message or payload."""

    def __init__(self, reason: SubagentSpawnReason) -> None:
        super().__init__("Runtime subagent readiness could not be established.")
        self.reason = reason


def remaining(deadline: float) -> float:
    value = deadline - monotonic()
    if value <= 0:
        raise TimeoutError("readiness probe deadline expired")
    return value


def probe_version(
    context: SubagentLaunchContext, binary: str, minimum: tuple[int, int, int], deadline: float
) -> str:
    result = run_captured(
        [binary, "--version"],
        cwd=context.working_dir,
        env=context.environment,
        timeout=remaining(deadline),
        deadline=deadline,
    )
    if result.returncode != 0:
        raise ReadinessProbeError(SubagentSpawnReason.PROBE_FAILED)
    version = parse_semver(result.stdout) or parse_semver(result.stderr)
    if version is None or version < minimum:
        raise ReadinessProbeError(SubagentSpawnReason.VERSION_UNVERIFIED)
    return ".".join(str(part) for part in version)


def ordinary_request(
    adapter: AbstractAITool, context: SubagentLaunchContext, deadline: float
) -> HeadlessSessionRequest | None:
    request = context.request
    if request is None:
        return None
    if request.interaction_mode is not HeadlessInteractionMode.UNATTENDED or request.resume_id:
        raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
    from crossby.ai_tools.headless import HeadlessSessionError

    try:
        normalized = adapter._validate_headless_request(
            request,
            interaction_handler=None,
            require_existing_working_dir=True,
            deadline=deadline,
        )
        adapter._validate_headless_requirements(normalized)
    except (HeadlessSessionError, ValueError):
        raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED) from None
    return normalized


PROBE_FAILURES = (OSError, EOFError, subprocess.SubprocessError, ValueError)
