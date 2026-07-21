"""Tests for post-sync target validators."""

from __future__ import annotations

import json
from pathlib import Path

import tomli_w

from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern
from crossby.sync.validate import (
    INSTRUCTION_SIZE_LIMIT_BYTES,
    has_errors,
    validate_codex_agents,
    validate_codex_config,
    validate_instruction_sizes,
    validate_json_configs,
    validate_skill_frontmatter,
    validate_target,
)


def _write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(data), encoding="utf-8")


class TestCodexConfig:
    def test_no_file_no_findings(self, tmp_path: Path) -> None:
        assert validate_codex_config(tmp_path) == []

    def test_valid_toml(self, tmp_path: Path) -> None:
        _write_toml(tmp_path / ".codex" / "config.toml", {"model": "gpt-5.4"})
        findings = validate_codex_config(tmp_path)
        assert any(f.level == "ok" and "valid TOML" in f.detail for f in findings)

    def test_invalid_toml(self, tmp_path: Path) -> None:
        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir()
        path.write_text("not = valid toml [[", encoding="utf-8")
        findings = validate_codex_config(tmp_path)
        assert any(f.level == "error" for f in findings)

    def test_mcp_command_warning_when_not_on_path(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path / ".codex" / "config.toml",
            {
                "model": "gpt-5.4",
                "mcp_servers": {"fake": {"command": "definitely-not-a-real-binary-xyz123"}},
            },
        )
        findings = validate_codex_config(tmp_path)
        assert any(f.level == "warning" and "not on PATH" in f.detail for f in findings)

    def test_mcp_command_ok_when_on_path(self, tmp_path: Path) -> None:
        # Use a binary that's almost certainly on PATH.
        _write_toml(
            tmp_path / ".codex" / "config.toml",
            {"mcp_servers": {"sh": {"command": "sh"}}},
        )
        findings = validate_codex_config(tmp_path)
        assert any(f.level == "ok" and "on PATH" in f.detail for f in findings)


class TestCodexAgents:
    def test_no_dir_no_findings(self, tmp_path: Path) -> None:
        assert validate_codex_agents(tmp_path) == []

    def test_valid_agent(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path / ".codex" / "agents" / "x.toml",
            {
                "name": "x",
                "description": "y",
                "developer_instructions": "Body.",
            },
        )
        findings = validate_codex_agents(tmp_path)
        assert findings
        assert all(f.level == "ok" for f in findings)

    def test_missing_required_field(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path / ".codex" / "agents" / "x.toml",
            {"name": "x", "description": "y"},  # missing developer_instructions
        )
        findings = validate_codex_agents(tmp_path)
        assert any(f.level == "error" and "developer_instructions" in f.detail for f in findings)

    def test_invalid_toml_in_agent(self, tmp_path: Path) -> None:
        agent = tmp_path / ".codex" / "agents" / "x.toml"
        agent.parent.mkdir(parents=True)
        agent.write_text("[[ broken", encoding="utf-8")
        findings = validate_codex_agents(tmp_path)
        assert any(f.level == "error" for f in findings)


class TestSkillFrontmatter:
    def _make_skill(self, root: Path, name: str, content: str) -> None:
        path = root / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_valid_skill(self, tmp_path: Path) -> None:
        self._make_skill(
            tmp_path / ".claude" / "skills",
            "my-skill",
            "---\nname: my-skill\ndescription: desc\n---\nBody.",
        )
        findings = validate_skill_frontmatter(tmp_path)
        assert any(f.level == "ok" for f in findings)

    def test_missing_frontmatter(self, tmp_path: Path) -> None:
        self._make_skill(tmp_path / ".claude" / "skills", "x", "no frontmatter at all")
        findings = validate_skill_frontmatter(tmp_path)
        assert any(f.level == "error" for f in findings)

    def test_missing_field(self, tmp_path: Path) -> None:
        self._make_skill(
            tmp_path / ".claude" / "skills",
            "x",
            "---\nname: x\n---\nBody.",
        )
        findings = validate_skill_frontmatter(tmp_path)
        assert any(f.level == "error" and "description" in f.detail for f in findings)

    def test_walks_each_tool_dir(self, tmp_path: Path) -> None:
        self._make_skill(
            tmp_path / ".claude" / "skills",
            "claude-skill",
            "---\nname: claude-skill\ndescription: x\n---\n",
        )
        self._make_skill(
            tmp_path / ".agents" / "skills",
            "codex-skill",
            "---\nname: codex-skill\ndescription: x\n---\n",
        )
        findings = validate_skill_frontmatter(tmp_path)
        tool_ids = {f.tool_id for f in findings}
        assert AIToolID.CLAUDE in tool_ids
        assert AIToolID.CODEX in tool_ids


class TestInstructionSizes:
    def test_under_threshold_ok(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("small body", encoding="utf-8")
        findings = validate_instruction_sizes(tmp_path)
        assert findings
        assert all(f.level == "ok" for f in findings)

    def test_over_threshold_warning(self, tmp_path: Path) -> None:
        # Write 33KB of content.
        (tmp_path / "AGENTS.md").write_text("x" * (33 * 1024), encoding="utf-8")
        findings = validate_instruction_sizes(tmp_path)
        assert any(f.level == "warning" for f in findings)

    def test_threshold_constant_correct(self) -> None:
        assert INSTRUCTION_SIZE_LIMIT_BYTES == 32 * 1024


class TestJsonConfigs:
    def test_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"x": 1}), encoding="utf-8")
        findings = validate_json_configs(tmp_path)
        assert any(f.level == "ok" for f in findings)

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir(parents=True)
        path.write_text("{ broken", encoding="utf-8")
        findings = validate_json_configs(tmp_path)
        assert any(f.level == "error" for f in findings)

    def test_invalid_cursor_hooks_json(self, tmp_path: Path) -> None:
        from crossby.sync.base import SyncConcern

        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir(parents=True)
        path.write_text("{ broken", encoding="utf-8")
        findings = validate_json_configs(tmp_path)
        hook_errors = [f for f in findings if f.level == "error" and f.concern == SyncConcern.HOOKS]
        assert len(hook_errors) == 1
        assert "hooks.json" in str(hook_errors[0].path)

    def test_invalid_copilot_hooks_json(self, tmp_path: Path) -> None:
        from crossby.sync.base import SyncConcern

        path = tmp_path / ".github" / "hooks" / "hooks.json"
        path.parent.mkdir(parents=True)
        path.write_text("{ broken", encoding="utf-8")
        findings = validate_json_configs(tmp_path)
        hook_errors = [f for f in findings if f.level == "error" and f.concern == SyncConcern.HOOKS]
        assert len(hook_errors) == 1
        assert "hooks.json" in str(hook_errors[0].path)

    def test_invalid_root_claude_json_surfaces_error(self, tmp_path: Path) -> None:
        """Regression: malformed `.claude.json` must produce a parse-error finding.

        Previously the MCP PATH walker swallowed JSON errors and the JSON
        validator didn't cover `.claude.json` / `.mcp.json`, so invalid JSON
        in those files went entirely unreported.
        """
        path = tmp_path / ".claude.json"
        path.write_text("{ broken", encoding="utf-8")
        findings = validate_json_configs(tmp_path)
        errors = [
            f
            for f in findings
            if f.level == "error" and f.tool_id == AIToolID.CLAUDE and "claude.json" in str(f.path)
        ]
        assert errors, "malformed .claude.json must surface a JSON parse error"

    def test_invalid_root_mcp_json_surfaces_error(self, tmp_path: Path) -> None:
        path = tmp_path / ".mcp.json"
        path.write_text("{ broken", encoding="utf-8")
        findings = validate_json_configs(tmp_path)
        errors = [
            f
            for f in findings
            if f.level == "error" and f.tool_id == AIToolID.CLAUDE and ".mcp.json" in str(f.path)
        ]
        assert errors, "malformed .mcp.json must surface a JSON parse error"


class TestValidateTargetTopLevel:
    def test_empty_project_returns_empty(self, tmp_path: Path) -> None:
        assert validate_target(tmp_path) == []

    def test_finds_issues_across_validators(self, tmp_path: Path) -> None:
        # Inject one error per validator.
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / "settings.json").write_text("{ broken", encoding="utf-8")
        _write_toml(
            tmp_path / ".codex" / "agents" / "x.toml",
            {"name": "x", "description": "y"},
        )
        findings = validate_target(tmp_path)
        assert has_errors(findings)
        # Concerns are populated for every finding.
        assert all(f.concern is not None for f in findings)

    def test_clean_project_no_errors(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("# Project\n", encoding="utf-8")
        _write_toml(
            tmp_path / ".codex" / "config.toml",
            {"model": "gpt-5.4"},
        )
        findings = validate_target(tmp_path)
        assert not has_errors(findings)


class TestHasErrors:
    def test_true_with_error(self) -> None:
        from crossby.sync.validate import ValidationFinding

        f = ValidationFinding(
            tool_id=AIToolID.CLAUDE,
            concern=SyncConcern.MCP,
            level="error",
            path=Path("x"),
            detail="bad",
        )
        assert has_errors([f])

    def test_false_without_error(self) -> None:
        from crossby.sync.validate import ValidationFinding

        f = ValidationFinding(
            tool_id=None,
            concern=None,
            level="ok",
            path=Path("x"),
            detail="fine",
        )
        assert not has_errors([f])


# ---------------------------------------------------------------------------
# Multi-tool MCP PATH validation
# ---------------------------------------------------------------------------


from crossby.sync.validate import validate_mcp_command_paths  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class TestMCPCommandPaths:
    @staticmethod
    def _absent() -> str:
        return "definitely-not-a-real-binary-xyz123"

    @staticmethod
    def _present() -> str:
        # `sh` is on PATH on every supported development host.
        return "sh"

    def test_claude_settings_warns_when_absent(self, tmp_path: Path) -> None:
        _write_json(
            tmp_path / ".claude" / "settings.json",
            {"mcpServers": {"fake": {"command": self._absent()}}},
        )
        findings = validate_mcp_command_paths(tmp_path)
        assert any(
            f.tool_id == AIToolID.CLAUDE and f.concern == SyncConcern.MCP and f.level == "warning"
            for f in findings
        )

    def test_dot_mcp_json_warns_when_absent(self, tmp_path: Path) -> None:
        _write_json(
            tmp_path / ".mcp.json",
            {"mcpServers": {"fake": {"command": self._absent()}}},
        )
        findings = validate_mcp_command_paths(tmp_path)
        assert any(f.tool_id == AIToolID.CLAUDE and f.level == "warning" for f in findings)

    def test_dot_claude_json_warns_when_absent(self, tmp_path: Path) -> None:
        _write_json(
            tmp_path / ".claude.json",
            {"mcpServers": {"fake": {"command": self._absent()}}},
        )
        findings = validate_mcp_command_paths(tmp_path)
        assert any(f.tool_id == AIToolID.CLAUDE and f.level == "warning" for f in findings)

    def test_cursor_mcp_json_warns_when_absent(self, tmp_path: Path) -> None:
        _write_json(
            tmp_path / ".cursor" / "mcp.json",
            {"mcpServers": {"fake": {"command": self._absent()}}},
        )
        findings = validate_mcp_command_paths(tmp_path)
        assert any(f.tool_id == AIToolID.CURSOR and f.level == "warning" for f in findings)

    def test_vscode_mcp_json_uses_servers_key(self, tmp_path: Path) -> None:
        """Copilot uses `servers`, not `mcpServers` — verify the key dispatch."""
        # Wrong key under .vscode/mcp.json → no findings.
        _write_json(
            tmp_path / ".vscode" / "mcp.json",
            {"mcpServers": {"fake": {"command": self._absent()}}},
        )
        findings_wrong_key = validate_mcp_command_paths(tmp_path)
        assert not any(f.tool_id == AIToolID.COPILOT for f in findings_wrong_key)
        # Right key produces a warning.
        _write_json(
            tmp_path / ".vscode" / "mcp.json",
            {"servers": {"fake": {"command": self._absent()}}},
        )
        findings_right_key = validate_mcp_command_paths(tmp_path)
        assert any(
            f.tool_id == AIToolID.COPILOT and f.level == "warning" for f in findings_right_key
        )

    def test_antigravity_cli_config_warns_when_absent(self, tmp_path: Path) -> None:
        _write_json(
            tmp_path / ".agents" / "mcp_config.json",
            {"mcpServers": {"fake": {"command": self._absent()}}},
        )
        findings = validate_mcp_command_paths(tmp_path)
        assert any(
            f.tool_id == AIToolID.ANTIGRAVITY_CLI and f.level == "warning" for f in findings
        )

    def test_ok_when_present(self, tmp_path: Path) -> None:
        _write_json(
            tmp_path / ".cursor" / "mcp.json",
            {"mcpServers": {"shell": {"command": self._present()}}},
        )
        findings = validate_mcp_command_paths(tmp_path)
        assert any(f.tool_id == AIToolID.CURSOR and f.level == "ok" for f in findings)

    def test_skips_entries_without_command(self, tmp_path: Path) -> None:
        """HTTP/SSE-only entries have no `command`; should be silently skipped."""
        _write_json(
            tmp_path / ".cursor" / "mcp.json",
            {"mcpServers": {"remote": {"url": "https://example.com/mcp"}}},
        )
        findings = validate_mcp_command_paths(tmp_path)
        assert not any(f.tool_id == AIToolID.CURSOR for f in findings)

    def test_silently_skips_malformed_json(self, tmp_path: Path) -> None:
        """Malformed JSON is handled by validate_json_configs; PATH walker stays quiet."""
        path = tmp_path / ".cursor" / "mcp.json"
        path.parent.mkdir()
        path.write_text("{not json", encoding="utf-8")
        # No exception, no findings from this validator.
        findings = validate_mcp_command_paths(tmp_path)
        assert not any(f.tool_id == AIToolID.CURSOR for f in findings)

    def test_env_var_command_expanded(self, tmp_path: Path, monkeypatch) -> None:
        """${VAR} in command is expanded via os.path.expandvars before lookup."""
        monkeypatch.setenv("CROSSBY_TEST_SHELL", "sh")
        _write_json(
            tmp_path / ".cursor" / "mcp.json",
            {"mcpServers": {"shell": {"command": "${CROSSBY_TEST_SHELL}"}}},
        )
        findings = validate_mcp_command_paths(tmp_path)
        assert any(f.tool_id == AIToolID.CURSOR and f.level == "ok" for f in findings)
