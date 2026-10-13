"""Codex runtime readiness from effective native reads, never an agent turn.

Native startup may refresh catalogs/caches using the installed credentials.
Full native config and error payloads must remain in memory and never be logged.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any

from crossby.ai_tools.session_process import HeaderlessJsonRpcProcess
from crossby.ai_tools.subagent_probe import (
    PROBE_FAILURES,
    PROBE_SECONDS,
    ReadinessProbeError,
    ordinary_request,
    probe_version,
    remaining,
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
    from crossby.models.ai import HeadlessSessionRequest

_MINIMUM_VERSION = (0, 162, 1)
_MAX_PAGES = 8
_MAX_ROWS = 800
_STAGES = {"stable", "beta", "underDevelopment", "deprecated", "removed"}
_CONFIG_KEYS = {
    "model",
    "model_reasoning_effort",
    "sandbox_mode",
    "approval_policy",
    "sandbox_workspace_write.network_access",
    "features.multi_agent",
    "features.multi_agent_v2",
    "agents.enabled",
    "agents.max_depth",
    "agents.max_concurrent_threads_per_session",
    "features.multi_agent_v2.max_concurrent_threads_per_session",
    "features.multi_agent_v2.non_code_mode_only",
    "features.code_mode",
    "features.code_mode.excluded_tool_namespaces",
}


@dataclass(frozen=True)
class NativeReadiness:
    config: dict[str, Any]
    requirements: dict[str, Any] | None
    features: list[dict[str, Any]]
    models: list[dict[str, Any]]


def _unknown() -> ReadinessProbeError:
    return ReadinessProbeError(SubagentSpawnReason.PROBE_UNPARSEABLE)


def _options(
    adapter: AbstractAITool,
    context: SubagentLaunchContext,
    request: HeadlessSessionRequest | None,
) -> list[str]:
    """Keep native semantics narrow; never reorder mixed feature/config forms."""
    options: list[str] = []
    selectors: dict[str, str] = {}
    args = context.launch_args
    seen: set[str] = set()
    config_seen = toggle_seen = False
    index = 0
    value_flags = {
        "-m": "model",
        "--model": "model",
        "-s": "sandbox_mode",
        "--sandbox": "sandbox_mode",
        "-a": "approval_policy",
        "--ask-for-approval": "approval_policy",
        "--color": "color",
        "--add-dir": "add_dir",
    }
    single_flags = {"--json", "--ephemeral", "--skip-git-repo-check"}
    while index < len(args):
        token = args[index]
        if not context.headless and token in single_flags | {"--color"}:
            raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
        if context.headless and token in {"-a", "--ask-for-approval"}:
            # Exec has no native approval channel; root-only argv placement is
            # not represented by this ordinary subcommand option sequence.
            raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
        if token in {"-c", "--config", "--enable", "--disable"}:
            if index + 1 >= len(args):
                raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
            value = args[index + 1]
            if token in {"-c", "--config"}:
                key, separator, _ = value.partition("=")
                if not separator or key.strip() not in _CONFIG_KEYS:
                    raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
                config_seen = True
            else:
                if value not in {"multi_agent", "multi_agent_v2"}:
                    raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
                toggle_seen = True
            options.extend((token, value))
            index += 2
            continue
        if token in value_flags:
            key = value_flags[token]
            if index + 1 >= len(args) or (key in seen and key != "add_dir"):
                raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
            value = args[index + 1]
            if not value or value.startswith("-"):
                raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
            seen.add(key)
            if key == "color":
                if value not in {"never", "always", "auto"}:
                    raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
            elif key != "add_dir":
                selectors[key] = value
            index += 2
            continue
        if token in single_flags and token not in seen:
            seen.add(token)
            index += 1
            continue
        raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
    if config_seen and toggle_seen:
        raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
    if request is not None:
        if request.model is not None:
            selectors["model"] = request.model
        if request.effort is not None:
            options.extend(adapter.effort_args(request.effort))
        selectors["sandbox_mode"] = "workspace-write" if request.sandbox else "danger-full-access"
        if request.sandbox:
            options.extend(
                (
                    "-c",
                    "sandbox_workspace_write.network_access="
                    + ("true" if request.network_access else "false"),
                )
            )
    if selectors.get("sandbox_mode", "workspace-write") not in {
        "read-only",
        "workspace-write",
        "danger-full-access",
    }:
        raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
    if selectors.get("approval_policy", "on-request") not in {"never", "on-request", "untrusted"}:
        raise ReadinessProbeError(SubagentSpawnReason.CONTEXT_UNSUPPORTED)
    # Typed native selectors override raw config, regardless of argv position.
    for key, value in selectors.items():
        options.extend(("-c", f"{key}={json.dumps(value)}"))
    return options


def _read_native(
    context: SubagentLaunchContext, binary: str, options: list[str], deadline: float
) -> NativeReadiness:
    rpc = HeaderlessJsonRpcProcess(
        [binary, *options, "app-server", "--stdio"],
        cwd=context.working_dir,
        env=context.environment,
        timeout=remaining(deadline),
        deadline=deadline,
    )
    next_id = 0

    def read(method: str, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal next_id
        next_id += 1
        identifier = next_id
        rpc.request(identifier, method, params, timeout=remaining(deadline))
        while True:
            frame = rpc.read(timeout=remaining(deadline))
            if "method" in frame:
                if "id" in frame:
                    # Read-only introspection cannot approve an unsolicited native request.
                    raise ReadinessProbeError(SubagentSpawnReason.EFFECTIVE_CONTEXT_UNRESOLVED)
                continue
            if type(frame.get("id")) is not int or frame["id"] != identifier:
                raise _unknown()
            if "error" in frame:
                reason = (
                    SubagentSpawnReason.CONFIG_UNREADABLE
                    if method.startswith("config")
                    else SubagentSpawnReason.PROBE_FAILED
                )
                raise ReadinessProbeError(reason)
            result = frame.get("result")
            if not isinstance(result, dict):
                raise _unknown()
            return result

    def pages(method: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _ in range(_MAX_PAGES):
            params: dict[str, Any] = {"limit": 100}
            if method == "model/list":
                params["includeHidden"] = True
            if cursor is not None:
                params["cursor"] = cursor
            page = read(method, params)
            data = page.get("data")
            if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
                raise _unknown()
            rows.extend(data)
            if len(rows) > _MAX_ROWS or "nextCursor" not in page:
                raise _unknown()
            cursor = page["nextCursor"]
            if cursor is None:
                return rows
            if not isinstance(cursor, str) or not cursor or cursor in seen_cursors:
                raise _unknown()
            seen_cursors.add(cursor)
        raise _unknown()

    try:
        read(
            "initialize",
            {
                "clientInfo": {"name": "crossby", "title": "Crossby", "version": "readiness-v1"},
                "capabilities": {"experimentalApi": True},
            },
        )
        rpc.notify("initialized", {}, timeout=remaining(deadline))
        config = read("config/read", {"cwd": str(context.working_dir), "includeLayers": False})
        requirements = read("configRequirements/read", {})
        if not isinstance(config.get("config"), dict) or "requirements" not in requirements:
            raise _unknown()
        required = requirements["requirements"]
        if required is not None and not isinstance(required, dict):
            raise _unknown()
        return NativeReadiness(
            config["config"], required, pages("experimentalFeature/list"), pages("model/list")
        )
    finally:
        rpc.close(timeout=0.2)


def _optional_table(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _unknown()
    return value


def _optional_bool(parent: dict[str, Any], key: str, default: bool) -> bool:
    value = parent.get(key)
    if value is None:
        return default
    if type(value) is not bool:
        raise _unknown()
    return value


def _optional_int(parent: dict[str, Any], key: str, default: int) -> int:
    value = parent.get(key)
    if value is None:
        return default
    if type(value) is not int or value < 0:
        raise _unknown()
    return value


def resolve_codex_readiness(
    context: SubagentLaunchContext, native: NativeReadiness, version: str
) -> SubagentSpawnStatus:
    """Follow the audited native resolver, with conservative exposure guards."""

    def status(
        state: SubagentSpawnState,
        reason: SubagentSpawnReason,
        text: str,
        evidence: SubagentSpawnEvidence = SubagentSpawnEvidence.EFFECTIVE_CONFIG,
    ) -> SubagentSpawnStatus:
        return context.status(AIToolID.CODEX, state, reason, text, evidence, tool_version=version)

    config = native.config
    # Exec first loads with `never` approvals, then may rebuild for AutoReview.
    # Standalone config/read does neither. Managed permission constraints can
    # make that intermediate load fail even when these native reads succeed.
    # Do not approximate native permission-profile/reviewer fallback semantics.
    if (
        context.headless
        and native.requirements is not None
        and any(
            native.requirements.get(key) is not None
            for key in (
                "allowedApprovalPolicies",
                "allowedApprovalsReviewers",
                "allowedSandboxModes",
            )
        )
    ):
        return status(
            SubagentSpawnState.UNKNOWN,
            SubagentSpawnReason.EFFECTIVE_CONTEXT_UNRESOLVED,
            "Managed approval/sandbox requirements leave exec configuration loading unresolved.",
        )
    if any(key not in config for key in ("agents", "features", "model")):
        raise _unknown()
    agents = _optional_table(config, "agents")
    agents_enabled = _optional_bool(agents, "enabled", True)
    if _optional_int(agents, "max_concurrent_threads_per_session", 6) == 0:
        raise _unknown()
    raw_features = _optional_table(config, "features")
    # The introspection API does not expose every model's effective tool routing.
    exposure_unresolved = any(key.startswith("code_mode") for key in raw_features)
    raw_v2 = raw_features.get("multi_agent_v2")
    if raw_v2 is not None and type(raw_v2) is not bool and not isinstance(raw_v2, dict):
        raise _unknown()
    v2_config = raw_v2 if isinstance(raw_v2, dict) else {}
    exposure_unresolved |= not _optional_bool(v2_config, "non_code_mode_only", True)
    _optional_bool(v2_config, "enabled", False)
    _optional_int(agents, "max_depth", 1)
    features: dict[str, dict[str, Any]] = {}
    for row in native.features:
        name = row.get("name")
        if not isinstance(name, str) or name in features:
            raise _unknown()
        features[name] = row

    def enabled(name: str) -> bool:
        row = features.get(name)
        if (
            row is None
            or not isinstance(row.get("stage"), str)
            or row["stage"] not in _STAGES
            or type(row.get("enabled")) is not bool
        ):
            raise _unknown()
        if row["stage"] == "removed" and row["enabled"]:
            raise _unknown()
        return bool(row["enabled"])

    if native.requirements is not None:
        if "featureRequirements" not in native.requirements:
            raise _unknown()
        pins = native.requirements["featureRequirements"]
        if pins is not None:
            if not isinstance(pins, dict):
                raise _unknown()
            for name in ("multi_agent", "multi_agent_v2"):
                if name in pins and (type(pins[name]) is not bool or enabled(name) != pins[name]):
                    return status(
                        SubagentSpawnState.UNKNOWN,
                        SubagentSpawnReason.CONFIG_AMBIGUOUS,
                        "Managed feature requirements and native enablement disagree.",
                    )
    v2_enabled = enabled("multi_agent_v2")
    if not v2_enabled and not agents_enabled:
        return status(
            SubagentSpawnState.DISABLED,
            SubagentSpawnReason.CONFIG_DISABLED,
            "Effective agents.enabled disables spawning while v2 is off.",
        )
    selected = config["model"]
    if selected is not None and (not isinstance(selected, str) or not selected):
        raise _unknown()
    matches = [
        row
        for row in native.models
        if (
            (row.get("model") == selected or row.get("id") == selected)
            if selected is not None
            else row.get("isDefault") is True
        )
    ]
    if len(matches) != 1 or "multiAgentVersion" not in matches[0]:
        return status(
            SubagentSpawnState.UNKNOWN,
            SubagentSpawnReason.EFFECTIVE_CONTEXT_UNRESOLVED,
            "The selected model's native delegation metadata is unresolved.",
        )
    if any(
        not isinstance(matches[0].get(key), str) or not matches[0][key] for key in ("id", "model")
    ):
        raise _unknown()
    model_runtime = matches[0]["multiAgentVersion"]
    if model_runtime is not None and (
        not isinstance(model_runtime, str) or model_runtime not in {"v1", "v2", "disabled"}
    ):
        raise _unknown()
    runtime = "v2" if v2_enabled else model_runtime
    if runtime is None:
        runtime = "v1" if enabled("multi_agent") else "disabled"
    if runtime == "disabled":
        return status(
            SubagentSpawnState.DISABLED,
            SubagentSpawnReason.MODEL_DISABLED
            if model_runtime is not None
            else SubagentSpawnReason.FEATURE_DISABLED,
            "The native model/feature resolver disables runtime spawning.",
        )
    if runtime == "v1":
        if _optional_int(agents, "max_depth", 1) == 0:
            return status(
                SubagentSpawnState.DISABLED,
                SubagentSpawnReason.CONFIG_DISABLED,
                "The effective v1 depth limit prevents a fresh root from spawning.",
            )
        if _optional_int(agents, "max_concurrent_threads_per_session", 6) == 0:
            raise _unknown()  # Native v1 rejects zero, rather than accepting a disabled launch.
    else:
        threads = _optional_int(agents, "max_concurrent_threads_per_session", 3)
        slots = _optional_int(v2_config, "max_concurrent_threads_per_session", threads + 1)
        if slots == 0:
            raise _unknown()
        if slots == 1:
            return status(
                SubagentSpawnState.DISABLED,
                SubagentSpawnReason.CONFIG_DISABLED,
                "The effective v2 concurrency cap leaves only the parent slot.",
            )
    if exposure_unresolved:
        return status(
            SubagentSpawnState.UNKNOWN,
            SubagentSpawnReason.EFFECTIVE_CONTEXT_UNRESOLVED,
            "Native Code Mode routing leaves the spawning surface unresolved.",
        )
    return status(
        SubagentSpawnState.AVAILABLE,
        SubagentSpawnReason.MODEL_ENABLED
        if not v2_enabled and model_runtime is not None
        else SubagentSpawnReason.FEATURE_ENABLED,
        "Native effective controls expose spawning for this fresh ordinary launch.",
    )


def codex_readiness(adapter: AbstractAITool, context: SubagentLaunchContext) -> SubagentSpawnStatus:
    deadline = monotonic() + PROBE_SECONDS
    version: str | None = None
    evidence = SubagentSpawnEvidence.EFFECTIVE_CONFIG
    try:
        if "CODEX_API_KEY" in context.environment:
            return context.status(
                AIToolID.CODEX,
                SubagentSpawnState.UNKNOWN,
                SubagentSpawnReason.EFFECTIVE_CONTEXT_UNRESOLVED,
                "Authentication selection is unverified for this native read probe.",
                evidence,
            )
        request = ordinary_request(adapter, context, deadline)
        options = _options(adapter, context, request)
        binary = context.binary_path("codex")
        if binary is None:
            raise ReadinessProbeError(SubagentSpawnReason.TOOL_NOT_INSTALLED)
        evidence = SubagentSpawnEvidence.VERSION_GATE
        version = probe_version(context, binary, _MINIMUM_VERSION, deadline)
        evidence = SubagentSpawnEvidence.EFFECTIVE_CONFIG
        native = _read_native(context, binary, options, deadline)
        remaining(deadline)
        return resolve_codex_readiness(context, native, version)
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
        AIToolID.CODEX,
        SubagentSpawnState.UNKNOWN,
        reason,
        "Exact Codex readiness could not be established by the bounded native read probe.",
        evidence,
        tool_version=version,
    )
