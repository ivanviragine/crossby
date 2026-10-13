"""Cursor command validation uses its native dialect without running commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID
from crossby.sync.validate import validate_mcp_command_paths

runner = CliRunner()


def _write(root: Path, command: str) -> Path:
    path = root / ".cursor/mcp.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"mcpServers": {"local": {"command": command}}}))
    return path


def _executable(path: Path) -> Path:
    # A PATH/access check is safe; executing this file would create a marker.
    path.write_text('#!/bin/sh\n: > "' + str(path.parent / "executed") + '"\n')
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("surrounding", [False, True])
def test_cursor_native_reference_finds_executable_without_disclosing_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surrounding: bool
) -> None:
    executable = _executable(tmp_path / "private-executable-value")
    if surrounding:
        monkeypatch.setenv("CROSSBY_TEST_EXE", "executable")
        command = str(tmp_path / "private-${env:CROSSBY_TEST_EXE}-value")
    else:
        monkeypatch.setenv("CROSSBY_TEST_EXE", str(executable))
        command = "${env:CROSSBY_TEST_EXE}"
    path = _write(tmp_path, command)
    before = path.read_bytes()
    findings = validate_mcp_command_paths(tmp_path)
    assert len(findings) == 1
    assert findings[0].tool_id == AIToolID.CURSOR
    assert findings[0].level == "ok"
    assert command in findings[0].detail
    assert str(executable) not in findings[0].detail
    assert path.read_bytes() == before
    assert not (tmp_path / "executed").exists()


@pytest.mark.parametrize(
    "replacement", ["$CROSSBY_TEST_OTHER", "${CROSSBY_TEST_OTHER}", "${env:CROSSBY_TEST_OTHER}"]
)
def test_cursor_replacement_is_not_recursively_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    monkeypatch.setenv("CROSSBY_TEST_EXE", replacement)
    monkeypatch.setenv("CROSSBY_TEST_OTHER", sys.executable)
    _write(tmp_path, "${env:CROSSBY_TEST_EXE}")
    finding = validate_mcp_command_paths(tmp_path)[0]
    assert finding.level == "warning"
    assert "${env:CROSSBY_TEST_EXE}" in finding.detail
    assert sys.executable not in finding.detail
    assert replacement not in finding.detail


@pytest.mark.parametrize("literal", ["$CROSSBY_TEST_EXE", "${CROSSBY_TEST_EXE}"])
def test_cursor_literal_dollar_forms_can_name_real_executables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, literal: str
) -> None:
    monkeypatch.setenv("CROSSBY_TEST_EXE", "different-name")
    executable = _executable(tmp_path / literal)
    _write(tmp_path, str(executable))
    assert validate_mcp_command_paths(tmp_path)[0].level == "ok"
    assert not (tmp_path / "executed").exists()


@pytest.mark.parametrize(
    "command", ["${env:CROSSBY_TEST_UNSET}", "prefix-${env:CROSSBY_TEST_UNSET}-suffix"]
)
def test_unset_cursor_reference_is_unverifiable_even_with_literal_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    monkeypatch.delenv("CROSSBY_TEST_UNSET", raising=False)
    _executable(tmp_path / command)
    monkeypatch.setenv("PATH", str(tmp_path))
    _write(tmp_path, command)
    finding = validate_mcp_command_paths(tmp_path)[0]
    assert finding.level == "warning"
    assert command in finding.detail
    assert "unset" in finding.detail
    assert not (tmp_path / "executed").exists()


def test_cursor_ordinary_absolute_executable(tmp_path: Path) -> None:
    _write(tmp_path, sys.executable)
    assert validate_mcp_command_paths(tmp_path)[0].level == "ok"


@pytest.mark.parametrize(
    "command,expected",
    [
        ("${env:CROSSBY_TEST_EXE}", "ok"),
        ("${env:CROSSBY_TEST_DIR}/${env:CROSSBY_TEST_FILE}", "ok"),
        ("${env:CROSSBY_TEST_UNSET}", "warning"),
        ("$CROSSBY_TEST_EXE", "warning"),
        ("${CROSSBY_TEST_EXE}", "warning"),
        ("ordinary-executable", "ok"),
    ],
)
def test_actual_cli_validate_target_preserves_native_files_and_reference_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, expected: str
) -> None:
    executable = _executable(tmp_path / "private-executable-value")
    _executable(tmp_path / "ordinary-executable")
    monkeypatch.setenv("CROSSBY_TEST_EXE", str(executable))
    monkeypatch.setenv("CROSSBY_TEST_DIR", str(tmp_path))
    monkeypatch.setenv("CROSSBY_TEST_FILE", executable.name)
    monkeypatch.delenv("CROSSBY_TEST_UNSET", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "240")
    source = _write(tmp_path, command)
    before = source.read_bytes()
    result = runner.invoke(app, ["sync", "--validate-target", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    command_row = next(line for line in result.output.splitlines() if "MCP `local`" in line)
    assert expected in command_row
    assert command in result.output
    assert "private-executable-value" not in result.output
    assert source.read_bytes() == before
    assert not (tmp_path / ".crossby").exists()
    assert not (tmp_path / "executed").exists()


@pytest.mark.parametrize(
    "tool,relative,key",
    [
        (AIToolID.CLAUDE, ".mcp.json", "mcpServers"),
        (AIToolID.COPILOT, ".vscode/mcp.json", "servers"),
        (AIToolID.ANTIGRAVITY_CLI, ".agents/mcp_config.json", "mcpServers"),
    ],
)
def test_other_json_targets_retain_existing_command_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: AIToolID, relative: str, key: str
) -> None:
    monkeypatch.setenv("CROSSBY_TEST_EXE", sys.executable)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: {"local": {"command": "${CROSSBY_TEST_EXE}"}}}))
    finding = validate_mcp_command_paths(tmp_path)[0]
    assert finding.tool_id == tool
    assert finding.level == "ok"
    assert "${CROSSBY_TEST_EXE}" in finding.detail
    assert sys.executable not in finding.detail
