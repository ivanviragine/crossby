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
                "mcp_servers": {
                    "fake": {"command": "definitely-not-a-real-binary-xyz123"}
                },
            },
        )
        findings = validate_codex_config(tmp_path)
        assert any(
            f.level == "warning" and "not on PATH" in f.detail for f in findings
        )

    def test_mcp_command_ok_when_on_path(self, tmp_path: Path) -> None:
        # Use a binary that's almost certainly on PATH.
        _write_toml(
            tmp_path / ".codex" / "config.toml",
            {"mcp_servers": {"sh": {"command": "sh"}}},
        )
        findings = validate_codex_config(tmp_path)
        assert any(
            f.level == "ok" and "on PATH" in f.detail for f in findings
        )


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
        assert any(
            f.level == "error" and "developer_instructions" in f.detail
            for f in findings
        )

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
        self._make_skill(
            tmp_path / ".claude" / "skills", "x", "no frontmatter at all"
        )
        findings = validate_skill_frontmatter(tmp_path)
        assert any(f.level == "error" for f in findings)

    def test_missing_field(self, tmp_path: Path) -> None:
        self._make_skill(
            tmp_path / ".claude" / "skills",
            "x",
            "---\nname: x\n---\nBody.",
        )
        findings = validate_skill_frontmatter(tmp_path)
        assert any(
            f.level == "error" and "description" in f.detail for f in findings
        )

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
