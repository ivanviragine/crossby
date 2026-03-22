"""Opt-in live smoke tests for real AI tool headless execution.

Policy:
  - Run each tool in an isolated temporary working directory.
  - Require an explicit per-tool model via ``CROSSBY_LIVE_MODEL_<TOOL>``.
  - Skip cleanly for missing binaries or missing auth/session prerequisites.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest
from tests.live_support import is_prerequisite_failure, parse_selected_tools

import crossby.ai_tools  # noqa: F401 - register adapters
from crossby.ai_tools.base import AbstractAITool

_SCRIPTABLE_TOOLS = ("claude", "copilot", "gemini", "codex", "cursor", "opencode")
_DEFAULT_TIMEOUT = 60
_SENTINEL = "CROSSBY_LIVE_OK_7F2D"

pytestmark = [
    pytest.mark.live_ai,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_AI_TESTS") != "1",
        reason="Live AI tests disabled (set RUN_LIVE_AI_TESTS=1)",
    ),
]


def _selected_tools() -> set[str]:
    return parse_selected_tools(
        os.environ.get("CROSSBY_LIVE_AI_TOOLS"),
        _SCRIPTABLE_TOOLS,
        fallback=(tool for tool in _SCRIPTABLE_TOOLS if _configured_model(tool)),
    )


def _timeout() -> int:
    raw = os.environ.get("CROSSBY_LIVE_AI_TIMEOUT")
    if raw is None:
        return _DEFAULT_TIMEOUT
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT
    return parsed if parsed > 0 else _DEFAULT_TIMEOUT


def _configured_model(tool: str) -> str | None:
    env_key = f"CROSSBY_LIVE_MODEL_{tool.upper()}"
    value = os.environ.get(env_key, "").strip()
    return value or None


@pytest.mark.parametrize("tool", _SCRIPTABLE_TOOLS)
def test_headless_smoke(tool: str, tmp_path) -> None:
    try:
        selected_tools = _selected_tools()
    except ValueError as err:
        pytest.fail(str(err))

    if not selected_tools:
        pytest.skip(
            "No live AI tools configured. Set CROSSBY_LIVE_MODEL_<TOOL> or CROSSBY_LIVE_AI_TOOLS."
        )
    if tool not in selected_tools:
        pytest.skip(f"{tool} not selected by CROSSBY_LIVE_AI_TOOLS")

    adapter = AbstractAITool.get(tool)
    caps = adapter.capabilities()
    if not caps.supports_headless:
        pytest.skip(f"{tool} does not support headless execution")
    if not shutil.which(caps.binary):
        pytest.skip(f"{caps.binary} not found in PATH")
    if not shutil.which("git"):
        pytest.skip("git not found in PATH")

    model = _configured_model(tool)
    if model is None:
        pytest.fail(
            "Missing "
            f"CROSSBY_LIVE_MODEL_{tool.upper()} "
            f"for explicitly selected live AI tool {tool!r}."
        )
    prompt = (
        "You are running a smoke test.\n"
        f"Reply with exactly this token and nothing else: {_SENTINEL}"
    )
    command = adapter.build_launch_command(model=model, prompt=prompt)
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    work_dir = tmp_path / tool
    work_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=work_dir, check=False, capture_output=True)

    result = subprocess.run(
        command,
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=_timeout(),
        env=env,
    )

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0 and is_prerequisite_failure(combined):
        pytest.skip(f"{tool} auth/session/workspace prerequisites missing")

    assert result.returncode == 0, (
        f"{tool} headless execution failed.\n"
        f"Command: {command!r}\n"
        f"Stdout:\n{result.stdout}\n"
        f"Stderr:\n{result.stderr}"
    )
    assert _SENTINEL in combined, (
        f"{tool} output did not contain sentinel {_SENTINEL!r}.\n"
        f"Stdout:\n{result.stdout}\n"
        f"Stderr:\n{result.stderr}"
    )
