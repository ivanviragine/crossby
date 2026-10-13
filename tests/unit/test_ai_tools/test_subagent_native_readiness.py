"""Native read fixtures exercise effective context, not real model execution."""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools import (
    AbstractAITool,
    HeadlessInteractionMode,
    HeadlessNativeOutput,
    HeadlessSessionRequest,
    SubagentSpawnReason,
    SubagentSpawnState,
    codex_readiness,
    subagent_probe,
)
from crossby.ai_tools.session_process import CapturedProcess
from crossby.ai_tools.subagent_context import SubagentLaunchContext


@pytest.fixture
def native(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-home"))
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.setattr(
        SubagentLaunchContext, "binary_path", lambda _self, name: f"/fixture/{name}"
    )
    monkeypatch.setattr(
        "crossby.ai_tools.codex.outside_root_git_metadata_dirs", lambda *_a, **_k: []
    )
    fixture: dict[str, Any] = {
        "version": {"codex": "codex-cli 0.162.1", "claude": "2.1.263 (Claude Code)"},
        "config/read": {"config": {"agents": None, "features": {}, "model": None}},
        "configRequirements/read": {"requirements": None},
        "experimentalFeature/list": {
            "data": [
                {"name": "multi_agent", "stage": "stable", "enabled": True},
                {"name": "multi_agent_v2", "stage": "stable", "enabled": False},
            ],
            "nextCursor": None,
        },
        "model/list": {
            "data": [
                {
                    "id": "fixture-model",
                    "model": "fixture-model",
                    "isDefault": True,
                    "multiAgentVersion": "v2",
                }
            ],
            "nextCursor": None,
        },
        "calls": [],
        "launches": [],
        "versions": [],
        "closed": 0,
    }

    def version(command: list[str], **kwargs: Any) -> CapturedProcess:
        fixture["versions"].append((command, kwargs))
        return CapturedProcess(0, fixture["version"][Path(command[0]).name], "")

    class NativeRPC:
        def __init__(self, command: list[str], **kwargs: Any) -> None:
            fixture["launches"].append((command, kwargs))
            self.queue: list[dict[str, Any]] = []

        def request(self, identifier: int, method: str, params: dict[str, Any], **_kw: Any) -> None:
            fixture["calls"].append((method, copy.deepcopy(params)))
            result = {} if method == "initialize" else fixture[method]
            if callable(result):
                result = result(params)
            frame = {"id": identifier, "result": copy.deepcopy(result)}
            if fixture.get("rpc_error") == method:
                frame = {"id": identifier, "error": {"code": 1, "message": "private-native-value"}}
            if fixture.get("bad_id"):
                frame["id"] = float(identifier)
            self.queue.append(frame)

        def notify(self, method: str, params: dict[str, Any], **_kw: Any) -> None:
            fixture["calls"].append((method, params))

        def read(self, **_kw: Any) -> dict[str, Any]:
            if fixture.get("timeout"):
                raise TimeoutError("private-native-value")
            return self.queue.pop(0)

        def close(self, **_kw: Any) -> int:
            fixture["closed"] += 1
            return 0

    monkeypatch.setattr(subagent_probe, "run_captured", version)
    monkeypatch.setattr(codex_readiness, "HeaderlessJsonRpcProcess", NativeRPC)
    return fixture


def check_codex(tmp_path: Path, **kwargs: Any) -> Any:
    return AbstractAITool.get("codex").subagent_spawning_status(working_dir=tmp_path, **kwargs)


def test_default_model_metadata_and_exact_scope(native: dict[str, Any], tmp_path: Path) -> None:
    environment = {"PATH": "child-bin", "CODEX_HOME": "relative-home", "SECRET": "fixture-secret"}
    result = check_codex(tmp_path, environment=environment)
    assert result.state is SubagentSpawnState.AVAILABLE
    assert result.tool_version == "0.162.1"
    assert result.reason_code is SubagentSpawnReason.MODEL_ENABLED
    methods = [method for method, _ in native["calls"]]
    assert methods == [
        "initialize",
        "initialized",
        "config/read",
        "configRequirements/read",
        "experimentalFeature/list",
        "model/list",
    ]
    assert all(
        not method.startswith(("thread/", "turn/", "config/value", "config/batch"))
        for method in methods
    )
    assert native["calls"][2][1] == {"cwd": str(tmp_path), "includeLayers": False}
    assert native["launches"][0][1]["env"] == environment
    assert native["launches"][0][1]["cwd"] == tmp_path
    assert native["versions"][0][1]["env"] == environment
    assert native["closed"] == 1
    assert "fixture-secret" not in result.model_dump_json()
    assert "--enable" not in native["launches"][0][0]


@pytest.mark.parametrize(
    ("v2", "agents_enabled", "model_runtime", "legacy", "expected"),
    [
        (True, False, "disabled", False, SubagentSpawnState.AVAILABLE),
        (False, False, "v2", True, SubagentSpawnState.DISABLED),
        (False, True, "v2", False, SubagentSpawnState.AVAILABLE),
        (False, True, "v1", False, SubagentSpawnState.AVAILABLE),
        (False, True, "disabled", True, SubagentSpawnState.DISABLED),
        (False, True, None, True, SubagentSpawnState.AVAILABLE),
        (False, True, None, False, SubagentSpawnState.DISABLED),
    ],
)
def test_native_precedence(
    native: dict[str, Any],
    tmp_path: Path,
    v2: bool,
    agents_enabled: bool,
    model_runtime: str | None,
    legacy: bool,
    expected: SubagentSpawnState,
) -> None:
    native["experimentalFeature/list"]["data"][0]["enabled"] = legacy
    native["experimentalFeature/list"]["data"][1]["enabled"] = v2
    native["config/read"]["config"]["agents"] = {"enabled": agents_enabled}
    native["model/list"]["data"][0]["multiAgentVersion"] = model_runtime
    assert check_codex(tmp_path).state is expected


@pytest.mark.parametrize(
    ("runtime", "agents", "v2_config", "expected"),
    [
        ("v1", {"max_depth": 0}, {}, SubagentSpawnState.DISABLED),
        ("v2", {"max_depth": 0}, {}, SubagentSpawnState.AVAILABLE),
        ("v2", {}, {"max_concurrent_threads_per_session": 1}, SubagentSpawnState.DISABLED),
        ("v2", {"max_concurrent_threads_per_session": 1}, {}, SubagentSpawnState.AVAILABLE),
        ("v2", {}, {"max_concurrent_threads_per_session": 2}, SubagentSpawnState.AVAILABLE),
        ("v2", {"max_concurrent_threads_per_session": 0}, {}, SubagentSpawnState.UNKNOWN),
        ("v2", {}, {"max_concurrent_threads_per_session": 0}, SubagentSpawnState.UNKNOWN),
        ("v2", {"max_depth": True}, {}, SubagentSpawnState.UNKNOWN),
    ],
)
def test_fresh_runtime_limits(
    native: dict[str, Any],
    tmp_path: Path,
    runtime: str,
    agents: dict[str, Any],
    v2_config: dict[str, Any],
    expected: SubagentSpawnState,
) -> None:
    native["model/list"]["data"][0]["multiAgentVersion"] = runtime
    native["config/read"]["config"]["agents"] = agents
    native["config/read"]["config"]["features"] = {"multi_agent_v2": v2_config}
    assert check_codex(tmp_path).state is expected


def test_system_or_managed_agents_denial_is_effective(
    native: dict[str, Any], tmp_path: Path
) -> None:
    # These controls are supplied by native config/read, not a three-file local scan.
    native["config/read"]["config"]["agents"] = {"enabled": False}
    native["configRequirements/read"]["requirements"] = {
        "featureRequirements": {"multi_agent_v2": False}
    }
    result = check_codex(tmp_path)
    assert result.state is SubagentSpawnState.DISABLED
    assert result.reason_code is SubagentSpawnReason.CONFIG_DISABLED


def test_managed_feature_disagreement_fails_closed(native: dict[str, Any], tmp_path: Path) -> None:
    native["configRequirements/read"]["requirements"] = {
        "featureRequirements": {"multi_agent_v2": True}
    }
    result = check_codex(tmp_path)
    assert result.state is SubagentSpawnState.UNKNOWN
    assert result.reason_code is SubagentSpawnReason.CONFIG_AMBIGUOUS


def test_authentication_equivalence_guard_never_copies_credentials(
    native: dict[str, Any],
    tmp_path: Path,
) -> None:
    result = check_codex(tmp_path, environment={"CODEX_API_KEY": "fixture-credential"})
    assert not result
    assert result.reason_code is SubagentSpawnReason.EFFECTIVE_CONTEXT_UNRESOLVED
    assert native["calls"] == native["versions"] == []
    assert "fixture-credential" not in result.model_dump_json()
    assert "authentication" in result.reason.lower()


def test_reviewed_code_mode_exposure_counterexample(native: dict[str, Any], tmp_path: Path) -> None:
    native["experimentalFeature/list"]["data"][1]["enabled"] = True
    native["config/read"]["config"]["features"] = {
        "multi_agent_v2": {"enabled": True, "non_code_mode_only": False},
        "code_mode": {"excluded_tool_namespaces": ["collaboration"]},
    }
    assert check_codex(tmp_path).state is SubagentSpawnState.UNKNOWN
    # A known runtime denial remains definitive even when the surface is hidden.
    native["experimentalFeature/list"]["data"][1]["enabled"] = False
    native["config/read"]["config"]["agents"] = {"enabled": False}
    assert check_codex(tmp_path).state is SubagentSpawnState.DISABLED


@pytest.mark.parametrize(
    "args",
    [
        ("--disable", "multi_agent", "-c", "features.multi_agent=true"),
        ("--profile", "review"),
        ("--ignore-user-config",),
        ("--worktree",),
        ("--oss",),
        ("--cd", "/elsewhere"),
        ("--thread-source", "subAgent"),
        ("--effort", "high"),
        ("-m", "model-one", "--model", "model-two"),
        ("-c", "model_provider=other"),
        ("-c",),
        ("--sandbox", "bogus"),
        ("-a", "never"),
    ],
)
def test_unmodeled_options_do_not_probe(
    native: dict[str, Any],
    tmp_path: Path,
    args: tuple[str, ...],
) -> None:
    result = check_codex(tmp_path, launch_args=args)
    assert result.state is SubagentSpawnState.UNKNOWN
    assert result.reason_code is SubagentSpawnReason.CONTEXT_UNSUPPORTED
    assert native["versions"] == native["calls"] == []


def test_native_feature_toggle_is_preserved(native: dict[str, Any], tmp_path: Path) -> None:
    check_codex(tmp_path, launch_args=("--disable", "multi_agent"))
    assert native["launches"][0][0][:3] == ["/fixture/codex", "--disable", "multi_agent"]
    assert "features.multi_agent=false" not in native["launches"][0][0]


def test_typed_model_selector_overrides_raw_config(native: dict[str, Any], tmp_path: Path) -> None:
    check_codex(tmp_path, launch_args=("--model", "fixture-model", "-c", 'model="other-model"'))
    assert native["launches"][0][0][1:5] == [
        "-c",
        'model="other-model"',
        "-c",
        'model="fixture-model"',
    ]


@pytest.mark.parametrize("mode", ["brokered", "resume", "output"])
def test_managed_request_compatibility(
    native: dict[str, Any],
    tmp_path: Path,
    mode: str,
) -> None:
    request = HeadlessSessionRequest(
        prompt="private-prompt",
        working_dir=tmp_path,
        interaction_mode=HeadlessInteractionMode.BROKERED
        if mode == "brokered"
        else HeadlessInteractionMode.UNATTENDED,
        resume_id="session-id" if mode == "resume" else None,
        native_output=HeadlessNativeOutput.JSON if mode == "output" else HeadlessNativeOutput.TEXT,
    )
    result = AbstractAITool.get("codex").subagent_spawning_status(request=request)
    assert result.reason_code is SubagentSpawnReason.CONTEXT_UNSUPPORTED
    assert native["calls"] == native["versions"] == []
    assert "private-prompt" not in result.model_dump_json()


def test_managed_context_preserves_model_effort_and_network(
    native: dict[str, Any],
    tmp_path: Path,
) -> None:
    request = HeadlessSessionRequest(
        prompt="private-prompt",
        working_dir=tmp_path,
        model="fixture-model",
        effort="high",
        network_access=False,
    )
    result = AbstractAITool.get("codex").subagent_spawning_status(request=request)
    assert result.state is SubagentSpawnState.AVAILABLE
    command = native["launches"][0][0]
    assert 'model="fixture-model"' in command
    assert 'model_reasoning_effort="high"' in command
    assert 'sandbox_mode="workspace-write"' in command
    assert "sandbox_workspace_write.network_access=false" in command
    assert "private-prompt" not in repr(native["calls"]) + repr(command)


@pytest.mark.parametrize(
    "mutation",
    [
        "agents-shape",
        "agents-bool",
        "features-shape",
        "stage",
        "stage-shape",
        "feature-bool",
        "duplicate",
        "missing-v2",
        "model-version",
        "model-shape",
        "model-missing-field",
        "model-absent",
        "model-duplicate",
        "requirements-shape",
        "requirements-missing-field",
    ],
)
def test_changed_or_malformed_native_evidence_is_unknown(
    native: dict[str, Any],
    tmp_path: Path,
    mutation: str,
) -> None:
    config = native["config/read"]["config"]
    rows = native["experimentalFeature/list"]["data"]
    models = native["model/list"]["data"]
    if mutation == "agents-shape":
        config["agents"] = []
    elif mutation == "agents-bool":
        config["agents"] = {"enabled": "false"}
    elif mutation == "features-shape":
        config["features"] = []
    elif mutation == "stage":
        rows[1]["stage"] = "futureStage"
    elif mutation == "stage-shape":
        rows[1]["stage"] = []
    elif mutation == "feature-bool":
        rows[1]["enabled"] = "false"
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[1]))
    elif mutation == "missing-v2":
        rows.pop()
    elif mutation == "model-version":
        models[0]["multiAgentVersion"] = "v3"
    elif mutation == "model-shape":
        models[0]["multiAgentVersion"] = []
    elif mutation == "model-missing-field":
        del models[0]["multiAgentVersion"]
    elif mutation == "model-absent":
        config["model"] = "uncatalogued-model"
    elif mutation == "model-duplicate":
        models.append(copy.deepcopy(models[0]))
    elif mutation == "requirements-shape":
        native["configRequirements/read"]["requirements"] = []
    elif mutation == "requirements-missing-field":
        native["configRequirements/read"]["requirements"] = {}
    result = check_codex(tmp_path)
    assert result.state is SubagentSpawnState.UNKNOWN
    assert not result
    assert native["closed"] == 1


def test_bounded_pagination_and_cursor_loop(native: dict[str, Any], tmp_path: Path) -> None:
    model = native["model/list"]["data"][0]
    native["model/list"] = lambda params: {
        "data": [model] if "cursor" in params else [],
        "nextCursor": None if "cursor" in params else "next",
    }
    assert check_codex(tmp_path).state is SubagentSpawnState.AVAILABLE
    native["model/list"] = {"data": [], "nextCursor": "loop"}
    assert check_codex(tmp_path).state is SubagentSpawnState.UNKNOWN
    model_calls = [params for method, params in native["calls"] if method == "model/list"]
    assert len(model_calls) == 4


@pytest.mark.parametrize(
    "failure", ["old-version", "bad-version", "timeout", "rpc-error", "bad-id"]
)
def test_probe_failures_are_unknown_without_native_content(
    native: dict[str, Any],
    tmp_path: Path,
    failure: str,
) -> None:
    if failure == "old-version":
        native["version"]["codex"] = "0.160.0"
    elif failure == "bad-version":
        native["version"]["codex"] = "unknown"
    elif failure == "timeout":
        native["timeout"] = True
    elif failure == "rpc-error":
        native["rpc_error"] = "config/read"
    elif failure == "bad-id":
        native["bad_id"] = True
    result = check_codex(tmp_path)
    assert result.state is SubagentSpawnState.UNKNOWN
    assert "private-native-value" not in result.model_dump_json()
    if failure in {"old-version", "bad-version"}:
        assert result.reason_code is SubagentSpawnReason.VERSION_UNVERIFIED
        assert native["calls"] == []
    else:
        assert native["closed"] == 1


@pytest.mark.parametrize(
    "error",
    [
        OSError("private-native-value"),
        subprocess.TimeoutExpired("private-native-value", 5),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"),
    ],
)
def test_version_probe_exception_never_escapes(
    native: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(subagent_probe, "run_captured", fail)
    assert check_codex(tmp_path).state is SubagentSpawnState.UNKNOWN
    assert native["calls"] == []


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((), SubagentSpawnState.UNKNOWN),
        (("--disallowedTools", "Agent"), SubagentSpawnState.DISABLED),
        (("--disallowedTools", "Task", "Bash"), SubagentSpawnState.DISABLED),
        (("--disallowedTools", "Agent(Explore)"), SubagentSpawnState.UNKNOWN),
        (("--disallowedTools", "Agent(*)"), SubagentSpawnState.UNKNOWN),
        (("--tools", ""), SubagentSpawnState.DISABLED),
        (("--tools", "Read,Bash"), SubagentSpawnState.DISABLED),
        (("--tools", "Agent(worker)", "Read"), SubagentSpawnState.UNKNOWN),
        (("--tools", "default"), SubagentSpawnState.UNKNOWN),
        (("--disallowedTools", "Agent", "--settings", "{}"), SubagentSpawnState.UNKNOWN),
        (("--disallowedTools", "Agent", "--disallowedTools", "Bash"), SubagentSpawnState.UNKNOWN),
        (("--disallowedTools", "Agent", "--model", "--effort", "high"), SubagentSpawnState.UNKNOWN),
    ],
)
def test_claude_direct_exclusions_and_managed_uncertainty(
    native: dict[str, Any],
    tmp_path: Path,
    args: tuple[str, ...],
    expected: SubagentSpawnState,
) -> None:
    result = AbstractAITool.get("claude").subagent_spawning_status(
        working_dir=tmp_path, launch_args=args
    )
    assert result.state is expected
    assert not result
    assert native["calls"] == []


def test_unaudited_adapters_do_not_probe(native: dict[str, Any], tmp_path: Path) -> None:
    for tool in ("cursor", "copilot", "opencode", "antigravity", "antigravity-cli", "vscode"):
        result = AbstractAITool.get(tool).subagent_spawning_status(working_dir=tmp_path)
        assert result.reason_code is SubagentSpawnReason.ADAPTER_NOT_AUDITED
    assert native["calls"] == native["versions"] == []


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("codex", ("--json",)),
        ("claude", ("--print", "--disallowedTools", "Agent")),
    ],
)
def test_native_mode_overrides_cannot_mislabel_interactive_context(
    native: dict[str, Any],
    tmp_path: Path,
    tool: str,
    args: tuple[str, ...],
) -> None:
    result = AbstractAITool.get(tool).subagent_spawning_status(
        working_dir=tmp_path,
        headless=False,
        launch_args=args,
    )
    assert result.state is SubagentSpawnState.UNKNOWN
    assert result.reason_code is SubagentSpawnReason.CONTEXT_UNSUPPORTED
    assert native["calls"] == native["versions"] == []
