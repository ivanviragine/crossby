"""Conservative Claude readiness without a partial managed-policy resolver."""

from __future__ import annotations

import re
import subprocess
from time import monotonic
from typing import TYPE_CHECKING

from crossby.ai_tools.subagent_probe import (
    PROBE_FAILURES,
    PROBE_SECONDS,
    ReadinessProbeError,
    ordinary_request,
    probe_version,
)
from crossby.models.ai import (
    AIToolID,
    SubagentSpawnEvidence,
    SubagentSpawnReason,
    SubagentSpawnState,
    SubagentSpawnStatus,
)

if TYPE_CHECKING:
    from crossby.ai_tools.base import AbstractAITool
    from crossby.ai_tools.subagent_context import SubagentLaunchContext

_MINIMUM_VERSION = (2, 1, 263)
_SPAWN_TOOLS = {"Agent", "Task"}


def _explicit_exclusion(args: tuple[str, ...]) -> bool:
    """Interpret direct native tool-pool selectors, never permission-rule files."""
    index = 0
    seen: set[str] = set()
    excluded = False
    values = {
        "--model",
        "--effort",
        "--permission-mode",
        "--permission-prompts",
        "--output-format",
        "--setting-sources",
        "--add-dir",
    }
    variadic = {"--disallowedTools", "--tools", "--allowedTools"}
    switches = {"--print", "-p", "--verbose"}
    while index < len(args):
        flag = args[index]
        if flag not in values | variadic | switches:
            raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
        if flag in seen and flag != "--add-dir":
            raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
        seen.add(flag)
        index += 1
        if flag in switches:
            continue
        start = index
        if flag in variadic:
            while index < len(args) and not args[index].startswith("-"):
                index += 1
        else:
            index += 1
        if start == index or index > len(args):
            raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
        entries = args[start:index]
        if flag in values and (
            entries[0].startswith("-") or (not entries[0] and flag != "--setting-sources")
        ):
            raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
        if flag in {"--tools", "--disallowedTools"}:
            tokens = [token for value in entries for token in re.split(r"[\s,]+", value) if token]
            if any("*" in token for token in tokens):
                raise ReadinessProbeError(SubagentSpawnReason.EFFECTIVE_CONTEXT_UNRESOLVED)
            if flag == "--disallowedTools":
                excluded |= bool(_SPAWN_TOOLS.intersection(tokens))
            else:
                # A literal scoped Agent(name) still exposes spawning for that type.
                has_spawn = any(
                    token in _SPAWN_TOOLS or token.startswith(("Agent(", "Task("))
                    for token in tokens
                )
                excluded |= not has_spawn and "default" not in tokens
        elif (
            (
                flag == "--permission-mode"
                and entries[0]
                not in {"default", "acceptEdits", "auto", "dontAsk", "bypassPermissions"}
            )
            or (flag == "--effort" and entries[0] not in {"low", "medium", "high", "xhigh", "max"})
            or (flag == "--permission-prompts" and entries[0] != "none")
            or (flag == "--output-format" and entries[0] not in {"text", "json", "stream-json"})
            or (
                flag == "--setting-sources"
                and any(
                    source not in {"user", "project", "local"}
                    for source in entries[0].split(",")
                    if source
                )
            )
        ):
            raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
    return excluded


def claude_readiness(
    adapter: AbstractAITool, context: SubagentLaunchContext
) -> SubagentSpawnStatus:
    deadline = monotonic() + PROBE_SECONDS
    version: str | None = None
    evidence = SubagentSpawnEvidence.EFFECTIVE_CONFIG
    try:
        if not context.headless and any(flag in {"--print", "-p"} for flag in context.launch_args):
            raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
        ordinary_request(adapter, context, deadline)
        excluded = _explicit_exclusion(context.launch_args)
        binary = context.binary_path("claude")
        if binary is None:
            raise ReadinessProbeError(SubagentSpawnReason.TOOL_NOT_INSTALLED)
        evidence = SubagentSpawnEvidence.VERSION_GATE
        version = probe_version(context, binary, _MINIMUM_VERSION, deadline)
        evidence = SubagentSpawnEvidence.EFFECTIVE_CONFIG
        if excluded:
            return context.status(
                AIToolID.CLAUDE,
                SubagentSpawnState.DISABLED,
                SubagentSpawnReason.CONFIG_DISABLED,
                "The native CLI tool pool explicitly excludes runtime spawning.",
                evidence,
                tool_version=version,
            )
        reason = SubagentSpawnReason.EFFECTIVE_CONTEXT_UNRESOLVED
    except (TimeoutError, subprocess.TimeoutExpired):
        reason = SubagentSpawnReason.PROBE_TIMEOUT
    except ReadinessProbeError as exc:
        reason = exc.reason
    except PROBE_FAILURES as exc:
        reason = (
            SubagentSpawnReason.PROBE_UNPARSEABLE
            if isinstance(exc, ValueError)
            else SubagentSpawnReason.PROBE_FAILED
        )
    return context.status(
        AIToolID.CLAUDE,
        SubagentSpawnState.UNKNOWN,
        reason,
        "Claude's effective managed policy and spawning permissions are unresolved.",
        evidence,
        tool_version=version,
    )
