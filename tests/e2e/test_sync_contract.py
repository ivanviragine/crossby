"""Deterministic contract tests for the current `crossby sync` workflow."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tests.e2e._support import install_mock_binary, run_crossby

pytestmark = [pytest.mark.contract, pytest.mark.e2e_deterministic]

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


def _setup_sync_project(project: Path, *, sync_tools: list[str] | None = None) -> None:
    config = {
        "version": 1,
        "permissions": {"allowed_commands": ["myapp:*"]},
        "sync": {
            "auto": True,
            "tools": sync_tools or [],
        },
        "rules": {
            "source": "AGENTS.md",
            "strategy": "symlink",
            "gitignore": True,
            "targets": {
                "claude": True,
                "cursor": True,
                "copilot": False,
                "gemini": False,
                "codex": False,
            },
        },
        "agents": {
            "source": ".crossby/agents",
            "strategy": "symlink",
            "gitignore": True,
            "targets": {
                "claude": True,
                "cursor": True,
                "copilot": False,
                "gemini": False,
                "codex": False,
            },
        },
    }
    (project / ".crossby.yml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (project / "AGENTS.md").write_text("# Shared rules\n", encoding="utf-8")
    agents_dir = project / ".crossby" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "reviewer.md").write_text("# Reviewer\n", encoding="utf-8")


def _setup_mcp_project(project: Path, *, sync_tools: list[str] | None = None) -> None:
    config = {
        "version": 1,
        "sync": {
            "auto": True,
            "tools": sync_tools or [],
        },
        "mcp_servers": {
            "context7": {
                "command": "npx",
                "args": ["-y", "@upstash/context7-mcp"],
            },
            "api": {
                "transport": "http",
                "url": "http://localhost:8080/mcp",
            },
        },
    }
    (project / ".crossby.yml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def test_sync_dry_run_reports_changes_without_writing(e2e_context) -> None:
    install_mock_binary(e2e_context.bin_dir, "claude")
    install_mock_binary(e2e_context.bin_dir, "agent")
    _setup_sync_project(e2e_context.project)

    result = run_crossby(
        ["sync", "--dry-run", "--path", str(e2e_context.project)],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "Dry-run mode" in combined
    assert ".claude/settings.json" in combined
    assert ".cursor/cli.json" in combined
    assert "CLAUDE.md" in combined
    assert ".cursorrules" in combined
    assert ".claude/agents" in combined
    assert ".cursor/agents" in combined
    assert ".gitignore" in combined
    assert not (e2e_context.project / ".claude" / "settings.json").exists()
    assert not (e2e_context.project / ".cursor" / "cli.json").exists()
    assert not (e2e_context.project / "CLAUDE.md").exists()
    assert not (e2e_context.project / ".claude" / "agents").exists()
    assert not (e2e_context.project / ".gitignore").exists()


def test_sync_creates_permissions_rules_agents_and_gitignore(e2e_context) -> None:
    install_mock_binary(e2e_context.bin_dir, "claude")
    install_mock_binary(e2e_context.bin_dir, "agent")
    _setup_sync_project(e2e_context.project)

    result = run_crossby(
        ["sync", "--path", str(e2e_context.project)],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 0, result.stderr

    claude_settings = e2e_context.project / ".claude" / "settings.json"
    cursor_config = e2e_context.project / ".cursor" / "cli.json"
    claude_rules = e2e_context.project / "CLAUDE.md"
    cursor_rules = e2e_context.project / ".cursorrules"
    claude_agents = e2e_context.project / ".claude" / "agents"
    cursor_agents = e2e_context.project / ".cursor" / "agents"
    gitignore = e2e_context.project / ".gitignore"

    assert claude_settings.is_file()
    assert cursor_config.is_file()
    assert claude_rules.is_symlink()
    assert claude_rules.resolve() == e2e_context.project / "AGENTS.md"
    assert cursor_rules.is_symlink()
    assert cursor_rules.resolve() == e2e_context.project / "AGENTS.md"
    assert claude_agents.is_symlink()
    assert claude_agents.resolve() == e2e_context.project / ".crossby" / "agents"
    assert cursor_agents.is_symlink()
    assert cursor_agents.resolve() == e2e_context.project / ".crossby" / "agents"

    gitignore_text = gitignore.read_text(encoding="utf-8")
    assert "# >>> crossby rules sync (generated — do not edit) >>>" in gitignore_text
    assert "CLAUDE.md" in gitignore_text
    assert ".cursorrules" in gitignore_text
    assert "# >>> crossby agents sync (generated — do not edit) >>>" in gitignore_text
    assert ".claude/agents" in gitignore_text
    assert ".cursor/agents" in gitignore_text


def test_sync_respects_configured_tool_filter(e2e_context) -> None:
    install_mock_binary(e2e_context.bin_dir, "claude")
    install_mock_binary(e2e_context.bin_dir, "agent")
    _setup_sync_project(e2e_context.project, sync_tools=["claude"])

    result = run_crossby(
        ["sync", "permissions", "--path", str(e2e_context.project)],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert ".claude/settings.json" in combined
    assert ".cursor/cli.json" not in combined
    assert (e2e_context.project / ".claude" / "settings.json").is_file()
    assert not (e2e_context.project / ".cursor" / "cli.json").exists()


def test_sync_mcp_writes_json_and_toml_configs(e2e_context) -> None:
    install_mock_binary(e2e_context.bin_dir, "claude")
    install_mock_binary(e2e_context.bin_dir, "codex")
    _setup_mcp_project(e2e_context.project)

    result = run_crossby(
        ["sync", "mcp", "--path", str(e2e_context.project)],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert ".claude/settings.json" in combined
    assert ".codex/config.toml" in combined

    claude_settings = e2e_context.project / ".claude" / "settings.json"
    codex_config = e2e_context.project / ".codex" / "config.toml"
    assert claude_settings.is_file()
    assert codex_config.is_file()

    claude_data = yaml.safe_load(claude_settings.read_text(encoding="utf-8"))
    assert claude_data["mcpServers"]["context7"]["command"] == "npx"
    assert claude_data["mcpServers"]["api"]["transport"] == "http"

    codex_data = tomllib.loads(codex_config.read_text(encoding="utf-8"))
    assert codex_data["mcp_servers"]["context7"]["args"] == ["-y", "@upstash/context7-mcp"]
    assert codex_data["mcp_servers"]["api"]["url"] == "http://localhost:8080/mcp"
