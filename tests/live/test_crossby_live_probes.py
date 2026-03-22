"""Opt-in live probe tests for installed AI CLIs and the bundled registry."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from tests.live_support import parse_selected_tools

_SUPPORTED_TOOLS = ("claude", "copilot", "gemini", "codex", "cursor", "opencode")
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "probe_models.py"
_MODULE_NAME = "crossby_live_probe_models"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
probe_models = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_MODULE_NAME, probe_models)
_SPEC.loader.exec_module(probe_models)

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


@pytest.mark.parametrize("tool", _SUPPORTED_TOOLS)
def test_registry_and_help_probe(tool: str) -> None:
    try:
        selected_tools = _selected_tools()
    except ValueError as err:
        pytest.fail(str(err))

    if tool not in selected_tools:
        pytest.skip(f"{tool} not selected by CROSSBY_LIVE_PROBE_TOOLS")

    report = probe_models._build_report(tool, probe_models._read_registry())
    if not report.installed:
        pytest.skip(f"{report.binary or tool} not found in PATH")

    assert report.probe_failures == [], f"{tool} probe failures: {report.probe_failures}"
    assert report.help_missing == [], f"{tool} missing flags: {report.help_missing}"
    assert report.models_missing == set(), (
        f"{tool} registry missing: {sorted(report.models_missing)}"
    )
    assert report.models_new == set(), f"{tool} registry drift: {sorted(report.models_new)}"
