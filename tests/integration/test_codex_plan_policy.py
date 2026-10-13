"""Opt-in real app-server policy verification; stops before model inference."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools import AbstractAITool, PlanCommandPolicy, PlanSessionRequest, plan_process
from crossby.models.ai import AIToolID

pytestmark = pytest.mark.skipif(
    os.environ.get("CROSSBY_CODEX_LOCAL_SMOKE") != "1" or shutil.which("codex") is None,
    reason="opt-in native Codex policy test (no credentials or paid model calls)",
)


@pytest.mark.parametrize("extra_root", [False, True])
def test_real_codex_replaces_ambient_writable_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_root: bool
) -> None:
    config_dir = tmp_path / "codex-config"
    config_dir.mkdir()
    ambient_root = tmp_path / "ambient-root"
    ambient_root.mkdir()
    requested_root = tmp_path / "requested-root"
    requested_root.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (config_dir / "config.toml").write_text(
        "[sandbox_workspace_write]\nnetwork_access = true\nwritable_roots = ["
        + json.dumps(str(ambient_root))
        + "]\n",
        encoding="utf-8",
    )
    thread_result: dict[str, Any] = {}

    class StopBeforeInferenceError(RuntimeError):
        pass

    class CapturePolicy(plan_process.HeaderlessJsonRpcProcess):
        def __init__(self, command: list[str], *, cwd: Path, timeout: float) -> None:
            super().__init__(
                command,
                cwd=cwd,
                timeout=timeout,
                env={**os.environ, "CODEX_HOME": str(config_dir)},
            )

        def request(self, request_id: int, method: str, params: dict[str, Any]) -> None:
            if method == "turn/start":
                raise StopBeforeInferenceError
            super().request(request_id, method, params)

        def read(self, *, timeout: float) -> dict[str, Any]:
            response = super().read(timeout=timeout)
            if response.get("id") == 3:
                thread_result.update(response["result"])
            return response

    monkeypatch.setattr(plan_process, "HeaderlessJsonRpcProcess", CapturePolicy)
    with pytest.raises(StopBeforeInferenceError):
        AbstractAITool.get(AIToolID.CODEX).run_plan_session(
            PlanSessionRequest(
                prompt="No inference should be reached.",
                working_dir=workspace,
                timeout_seconds=15,
                trusted_dirs=(requested_root,) if extra_root else (),
                command_policy=PlanCommandPolicy(allowed_commands=("git status",)),
            )
        )

    policy = thread_result["sandbox"]
    assert policy["type"] == "workspaceWrite"
    assert policy["networkAccess"] is False
    assert policy["writableRoots"] == ([str(requested_root)] if extra_root else [])
