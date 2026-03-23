"""Opt-in live probe tests for installed AI CLIs and the bundled registry."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.live_support import parse_selected_tools

_SUPPORTED_TOOLS = ("claude", "copilot", "gemini", "codex", "cursor", "opencode")
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "probe_models.py"
_DEFAULT_TIMEOUT = 120

pytestmark = [
    pytest.mark.live_probe,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_PROBE_TESTS") != "1",
        reason="Live probe tests disabled (set RUN_LIVE_PROBE_TESTS=1)",
    ),
]


def _selected_tools() -> set[str]:
    return parse_selected_tools(
        os.environ.get("CROSSBY_LIVE_PROBE_TOOLS"),
        _SUPPORTED_TOOLS,
        fallback=_SUPPORTED_TOOLS,
    )


def _timeout() -> int:
    raw = os.environ.get("CROSSBY_LIVE_PROBE_TIMEOUT")
    if raw is None:
        return _DEFAULT_TIMEOUT
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT
    return parsed if parsed > 0 else _DEFAULT_TIMEOUT


@pytest.mark.parametrize("tool", _SUPPORTED_TOOLS)
def test_registry_and_help_probe(tool: str) -> None:
    try:
        selected_tools = _selected_tools()
    except ValueError as err:
        pytest.fail(str(err))

    if tool not in selected_tools:
        pytest.skip(f"{tool} not selected by CROSSBY_LIVE_PROBE_TOOLS")

    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--tool", tool, "--json"],
        capture_output=True,
        text=True,
        timeout=_timeout(),
        env=os.environ.copy(),
    )

    assert result.stdout.strip(), f"{tool} probe returned no JSON output:\n{result.stderr}"
    payload = json.loads(result.stdout)
    assert len(payload["tools"]) == 1
    report = payload["tools"][0]

    if not report["installed"]:
        pytest.skip(f"{report['binary'] or tool} not found in PATH")

    assert result.returncode == 0, (
        f"{tool} probe drift detected.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )
    assert payload["has_diff"] is False
    assert report["probe_failures"] == [], f"{tool} probe failures: {report['probe_failures']}"
    assert report["help_missing"] == [], f"{tool} missing flags: {report['help_missing']}"
    assert report["models_missing"] == [], f"{tool} registry missing: {report['models_missing']}"
    assert report["models_new"] == [], f"{tool} registry drift: {report['models_new']}"
