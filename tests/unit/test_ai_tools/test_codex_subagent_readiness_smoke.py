"""Opt-in native read audit; no thread, model turn, tool or config write RPC.

Native startup may contact its usual service and refresh fixture-local caches.
This tests readiness controls, never proof of actual delegation.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from crossby.ai_tools import AbstractAITool, SubagentSpawnReason, SubagentSpawnState
from crossby.utils.versioning import detect_binary_version

pytestmark = pytest.mark.skipif(
    os.environ.get("CROSSBY_SUBAGENT_READINESS_SMOKE") != "1" or shutil.which("codex") is None,
    reason="opt-in native read smoke (CROSSBY_SUBAGENT_READINESS_SMOKE=1, codex installed)",
)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("", SubagentSpawnState.AVAILABLE),
        (
            "[features]\nmulti_agent_v2=false\n[agents]\nenabled=false\n",
            SubagentSpawnState.DISABLED,
        ),
        ('model="gpt-6.1-sol"\n[features]\nmulti_agent=false\n', SubagentSpawnState.AVAILABLE),
    ],
)
def test_native_effective_controls_without_model_turn(
    tmp_path: Path, body: str, expected: SubagentSpawnState
) -> None:
    installed = detect_binary_version("codex")
    if installed is None or installed < (0, 162, 1):
        pytest.skip("native readiness read contract requires Codex 0.162.1 or newer")
    if "CODEX_API_KEY" in os.environ:
        pytest.skip("exec/app-server authentication equivalence is unresolved for CODEX_API_KEY")
    home = tmp_path / "codex-home"
    home.mkdir()
    config = home / "config.toml"
    config.write_text(body)
    snapshot = config.read_bytes()
    # This is a complete new fixture context, never a partial copy of user config.
    environment = {**os.environ, "CODEX_HOME": str(home)}
    result = AbstractAITool.get("codex").subagent_spawning_status(
        working_dir=tmp_path,
        environment=environment,
    )
    assert result.state is expected, result.model_dump(mode="json")
    if expected is SubagentSpawnState.DISABLED:
        assert result.reason_code is SubagentSpawnReason.CONFIG_DISABLED
    assert config.read_bytes() == snapshot
