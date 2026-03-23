"""Tests for the developer model probe utility."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
PROBE_MODELS_PATH = SCRIPTS_DIR / "probe_models.py"
PROBE_MODELS_SPEC = importlib.util.spec_from_file_location("probe_models", PROBE_MODELS_PATH)
assert PROBE_MODELS_SPEC is not None
assert PROBE_MODELS_SPEC.loader is not None
probe_models = importlib.util.module_from_spec(PROBE_MODELS_SPEC)
sys.modules.setdefault("probe_models", probe_models)
PROBE_MODELS_SPEC.loader.exec_module(probe_models)


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> CompletedProcess[str]:
    return CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _caps(binary: str) -> SimpleNamespace:
    return SimpleNamespace(binary=binary)


def test_probe_claude_models_falls_back_to_docs_on_timeout() -> None:
    with (
        patch(
            "probe_models._run",
            side_effect=TimeoutExpired(cmd=["claude", "models"], timeout=30),
        ),
        patch("probe_models._scrape_models", return_value={"claude-sonnet-4.6"}) as mock_scrape,
    ):
        assert probe_models._probe_claude_models() == {"claude-sonnet-4.6"}

    mock_scrape.assert_called_once_with("claude")


def test_probe_codex_models_falls_back_to_docs_when_cache_missing(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()

    with (
        patch("probe_models.Path.home", return_value=home),
        patch("probe_models._scrape_models", return_value={"gpt-5.4"}) as mock_scrape,
    ):
        assert probe_models._probe_codex_models() == {"gpt-5.4"}

    mock_scrape.assert_called_once_with("codex")


def test_probe_cli_flags_uses_subcommand_help_for_codex() -> None:
    def fake_run(command: list[str], timeout: int = 20, extra_env=None) -> CompletedProcess[str]:
        if command == ["codex", "--help"]:
            return _completed(stdout="exec\nresume\n--sandbox\n--cd\n--add-dir\n--model")
        if command == ["codex", "exec", "--help"]:
            return _completed(
                stdout="model_reasoning_effort\n--dangerously-bypass-approvals-and-sandbox"
            )
        raise AssertionError(f"Unexpected command: {command!r}")

    with (
        patch(
            "probe_models._get_adapter",
            return_value=SimpleNamespace(capabilities=lambda: _caps("codex")),
        ),
        patch("probe_models._run", side_effect=fake_run),
    ):
        matched, missing = probe_models._probe_cli_flags("codex")

    assert "effort (model_reasoning_effort)" in matched
    assert "yolo (--dangerously-bypass-approvals-and-sandbox)" in matched
    assert missing == []


def test_build_report_marks_probe_failures_as_diff() -> None:
    with (
        patch(
            "probe_models._get_adapter",
            return_value=SimpleNamespace(capabilities=lambda: _caps("claude")),
        ),
        patch("probe_models.shutil.which", return_value="/usr/bin/claude"),
        patch("probe_models._probe_models", side_effect=probe_models.ProbeFailure("timed out")),
        patch(
            "probe_models._probe_cli_flags",
            side_effect=probe_models.ProbeFailure("no help output"),
        ),
    ):
        report = probe_models._build_report("claude", {"claude": {"claude-sonnet-4.6"}})

    assert report.has_diff is True
    assert report.probe_failures == [
        "model probe failed: timed out",
        "help probe failed: no help output",
    ]


def test_json_ready_includes_probe_failures() -> None:
    report = probe_models.ToolReport(
        tool="claude",
        installed=True,
        binary="claude",
        probe_failures=["model probe failed: timed out"],
    )

    payload = probe_models._json_ready(report)

    assert payload["probe_failures"] == ["model probe failed: timed out"]


def test_main_json_emits_public_report_contract(capsys) -> None:
    report = probe_models.ToolReport(tool="claude", installed=True, binary="claude")

    with (
        patch("probe_models._read_registry", return_value={"claude": set()}),
        patch("probe_models._build_report", return_value=report),
    ):
        exit_code = probe_models.main(["--tool", "claude", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["has_diff"] is False
    assert payload["tools"] == [
        {
            "binary": "claude",
            "help_matched": [],
            "help_missing": [],
            "installed": True,
            "models_found": [],
            "models_missing": [],
            "models_new": [],
            "notes": [],
            "probe_failures": [],
            "tool": "claude",
        }
    ]
