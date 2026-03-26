"""Deterministic contract tests for `crossby init`."""

from __future__ import annotations

import json

import pytest
import yaml
from tests.e2e._support import install_mock_binary, run_crossby

pytestmark = [pytest.mark.contract, pytest.mark.e2e_deterministic]


def test_init_creates_config_from_detected_tools(e2e_context) -> None:
    install_mock_binary(e2e_context.bin_dir, "claude")
    install_mock_binary(e2e_context.bin_dir, "codex")

    result = run_crossby(
        ["init", str(e2e_context.project)],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 0, result.stderr
    config_path = e2e_context.project / ".crossby.yml"
    assert config_path.exists()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["ai"]["default_tool"] == "claude"
    assert "claude" in config["models"]
    assert "codex" in config["models"]
    assert config["permissions"]["allowed_commands"] == []
    assert "Created" in result.stdout


def test_init_fails_when_config_already_exists(e2e_context) -> None:
    config_path = e2e_context.project / ".crossby.yml"
    config_path.write_text("version: 1\n", encoding="utf-8")

    result = run_crossby(
        ["init", str(e2e_context.project)],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 1
    assert "already exists" in result.stderr
    assert config_path.read_text(encoding="utf-8") == "version: 1\n"


def test_init_fails_when_no_tools_are_installed(e2e_context) -> None:
    result = run_crossby(
        ["init", str(e2e_context.project)],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 1
    assert "No AI tools found in PATH" in result.stderr


def test_init_discovers_existing_mcp_servers(e2e_context) -> None:
    install_mock_binary(e2e_context.bin_dir, "claude")
    settings_path = e2e_context.project / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "context7": {
                        "command": "npx",
                        "args": ["-y", "@upstash/context7-mcp"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_crossby(
        ["init", str(e2e_context.project)],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 0, result.stderr
    config = yaml.safe_load((e2e_context.project / ".crossby.yml").read_text(encoding="utf-8"))
    assert config["mcp_servers"]["context7"]["command"] == "npx"
    assert config["mcp_servers"]["context7"]["args"] == ["-y", "@upstash/context7-mcp"]
    assert "Discovered 1 MCP server(s)" in result.stdout
    assert "Run 'crossby sync mcp'" in result.stdout


def test_init_reports_mcp_conflicts_and_keeps_first_seen_definition(e2e_context) -> None:
    install_mock_binary(e2e_context.bin_dir, "claude")
    install_mock_binary(e2e_context.bin_dir, "agent")

    claude_path = e2e_context.project / ".claude" / "settings.json"
    claude_path.parent.mkdir(parents=True)
    claude_path.write_text(
        json.dumps({"mcpServers": {"shared": {"command": "claude-version"}}}),
        encoding="utf-8",
    )

    cursor_path = e2e_context.project / ".cursor" / "mcp.json"
    cursor_path.parent.mkdir(parents=True)
    cursor_path.write_text(
        json.dumps({"mcpServers": {"shared": {"command": "cursor-version"}}}),
        encoding="utf-8",
    )

    result = run_crossby(
        ["init", str(e2e_context.project)],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 0, result.stderr
    config = yaml.safe_load((e2e_context.project / ".crossby.yml").read_text(encoding="utf-8"))
    assert config["mcp_servers"]["shared"]["command"] == "claude-version"
    combined = result.stdout + result.stderr
    assert "kept claude definition" in combined
