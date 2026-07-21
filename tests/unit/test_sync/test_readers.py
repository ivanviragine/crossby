"""Tests for sync readers — detect_skills, suggest_skills_source, build_sync_data, scan_project."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from crossby.models.ai import AIToolID
from crossby.sync.readers import (
    build_sync_data,
    detect_skills,
    discover_mcp,
    scan_project,
    suggest_skills_source,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skills_dir(root: Path, tool_rel: str, skills: list[str] | None = None) -> Path:
    """Create a skills directory for a tool with optional skill subdirectories."""
    d = root / tool_rel
    d.mkdir(parents=True, exist_ok=True)
    for name in skills or []:
        skill = d / name
        skill.mkdir()
        (skill / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# detect_skills
# ---------------------------------------------------------------------------


class TestDetectSkills:
    def test_returns_empty_when_no_skills(self, tmp_path: Path) -> None:
        result = detect_skills(tmp_path)
        assert result == {}

    def test_detects_claude_skills(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".claude/skills", ["skill-a"])
        result = detect_skills(tmp_path)
        assert AIToolID.CLAUDE in result
        assert result[AIToolID.CLAUDE] == ".claude/skills"

    def test_detects_cursor_skills(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".cursor/skills")
        result = detect_skills(tmp_path)
        assert AIToolID.CURSOR in result

    def test_detects_codex_skills(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".agents/skills")
        result = detect_skills(tmp_path)
        assert AIToolID.CODEX in result
        assert result[AIToolID.CODEX] == ".agents/skills"

    def test_detects_antigravity_cli_skills(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".agents/skills")
        result = detect_skills(tmp_path)
        assert AIToolID.ANTIGRAVITY_CLI in result

    def test_detects_copilot_skills(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".github/skills")
        result = detect_skills(tmp_path)
        assert AIToolID.COPILOT in result
        assert result[AIToolID.COPILOT] == ".github/skills"

    def test_detects_multiple_tools(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".claude/skills")
        _make_skills_dir(tmp_path, ".cursor/skills")
        result = detect_skills(tmp_path)
        assert AIToolID.CLAUDE in result
        assert AIToolID.CURSOR in result

    def test_returns_relative_paths(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".claude/skills")
        result = detect_skills(tmp_path)
        assert result[AIToolID.CLAUDE] == ".claude/skills"
        assert not Path(result[AIToolID.CLAUDE]).is_absolute()

    def test_includes_symlinked_dir(self, tmp_path: Path) -> None:
        """detect_skills includes symlinked dirs (unlike detect_skills_source)."""
        import os

        source = tmp_path / "real-skills"
        source.mkdir()
        target = tmp_path / ".claude" / "skills"
        target.parent.mkdir(parents=True)
        os.symlink(source, target)
        result = detect_skills(tmp_path)
        assert AIToolID.CLAUDE in result


# ---------------------------------------------------------------------------
# suggest_skills_source
# ---------------------------------------------------------------------------


class TestSuggestSkillsSource:
    def test_returns_none_for_empty(self) -> None:
        assert suggest_skills_source({}) is None

    def test_prefers_claude_over_cursor(self) -> None:
        found = {AIToolID.CURSOR: ".cursor/skills", AIToolID.CLAUDE: ".claude/skills"}
        assert suggest_skills_source(found) == AIToolID.CLAUDE

    def test_prefers_antigravity_cli_over_cursor(self) -> None:
        found = {AIToolID.CURSOR: ".cursor/skills", AIToolID.ANTIGRAVITY_CLI: ".agents/skills"}
        assert suggest_skills_source(found) == AIToolID.ANTIGRAVITY_CLI

    def test_prefers_codex_over_cursor(self) -> None:
        found = {AIToolID.CURSOR: ".cursor/skills", AIToolID.CODEX: ".agents/skills"}
        assert suggest_skills_source(found) == AIToolID.CODEX

    def test_falls_back_to_cursor_when_only_option(self) -> None:
        found = {AIToolID.CURSOR: ".cursor/skills"}
        assert suggest_skills_source(found) == AIToolID.CURSOR

    def test_returns_copilot_when_only_option(self) -> None:
        found = {AIToolID.COPILOT: ".github/skills"}
        result = suggest_skills_source(found)
        assert result == AIToolID.COPILOT

    def test_single_claude_entry(self) -> None:
        found = {AIToolID.CLAUDE: ".claude/skills"}
        assert suggest_skills_source(found) == AIToolID.CLAUDE


# ---------------------------------------------------------------------------
# build_sync_data — skills population
# ---------------------------------------------------------------------------


class TestBuildSyncDataSkills:
    def test_populates_skills_source_auto(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".claude/skills", ["skill-a"])
        data = build_sync_data(tmp_path)
        assert data.skills_source == ".claude/skills"

    def test_populates_skills_source_from_tool(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".claude/skills")
        _make_skills_dir(tmp_path, ".cursor/skills")
        data = build_sync_data(tmp_path, from_tool=AIToolID.CURSOR)
        assert data.skills_source == ".cursor/skills"

    def test_skills_source_none_when_no_dirs(self, tmp_path: Path) -> None:
        data = build_sync_data(tmp_path)
        assert data.skills_source is None

    def test_skills_source_none_when_from_tool_missing(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".cursor/skills")
        data = build_sync_data(tmp_path, from_tool=AIToolID.CLAUDE)
        assert data.skills_source is None

    def test_prefers_claude_source_over_cursor(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".claude/skills")
        _make_skills_dir(tmp_path, ".cursor/skills")
        data = build_sync_data(tmp_path)
        assert data.skills_source == ".claude/skills"

    def test_skills_strategy_defaults_to_symlink(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".claude/skills")
        data = build_sync_data(tmp_path)
        assert data.skills_strategy == "symlink"

    def test_skills_gitignore_defaults_to_true(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".claude/skills")
        data = build_sync_data(tmp_path)
        assert data.skills_gitignore is True


# ---------------------------------------------------------------------------
# scan_project — skills branch
# ---------------------------------------------------------------------------


class TestScanProjectSkills:
    def test_skills_none_found(self, tmp_path: Path) -> None:
        scan = scan_project(tmp_path, [AIToolID.CLAUDE])
        assert scan.skills.found == {}
        assert "none found" in scan.skills.summary

    def test_skills_detected_in_summary(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".claude/skills", ["skill-a", "skill-b"])
        scan = scan_project(tmp_path, [AIToolID.CLAUDE])
        assert AIToolID.CLAUDE in scan.skills.found
        assert "2 skills" in scan.skills.summary

    def test_skills_singular_label(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".claude/skills", ["skill-a"])
        scan = scan_project(tmp_path, [AIToolID.CLAUDE])
        assert "1 skill" in scan.skills.summary
        assert "1 skills" not in scan.skills.summary

    def test_skills_multiple_tools_in_summary(self, tmp_path: Path) -> None:
        _make_skills_dir(tmp_path, ".claude/skills", ["skill-a"])
        _make_skills_dir(tmp_path, ".cursor/skills", ["skill-b"])
        scan = scan_project(tmp_path, [AIToolID.CLAUDE, AIToolID.CURSOR])
        assert AIToolID.CLAUDE in scan.skills.found
        assert AIToolID.CURSOR in scan.skills.found


# ---------------------------------------------------------------------------
# discover_mcp — conflict surfacing
# ---------------------------------------------------------------------------


class TestDiscoverMCPConflicts:
    def test_same_name_in_two_tools_warns_and_keeps_first_seen(self, tmp_path: Path) -> None:
        claude_path = tmp_path / ".claude" / "settings.json"
        claude_path.parent.mkdir()
        claude_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "ctx": {"command": "npx", "args": ["claude-ctx"]},
                    }
                }
            ),
            encoding="utf-8",
        )
        cursor_path = tmp_path / ".cursor" / "mcp.json"
        cursor_path.parent.mkdir()
        cursor_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "ctx": {"command": "npx", "args": ["cursor-ctx"]},
                    }
                }
            ),
            encoding="utf-8",
        )

        with structlog.testing.capture_logs() as logs:
            servers = discover_mcp(tmp_path)

        assert "ctx" in servers
        assert servers["ctx"].args == ["claude-ctx"]
        conflict_logs = [e for e in logs if e.get("event") == "mcp.conflict"]
        assert len(conflict_logs) == 1
        assert conflict_logs[0]["name"] == "ctx"
        assert conflict_logs[0]["kept_from"] == "claude"
        assert conflict_logs[0]["ignored_from"] == "cursor"

    def test_no_conflict_no_warning(self, tmp_path: Path) -> None:
        claude_path = tmp_path / ".claude" / "settings.json"
        claude_path.parent.mkdir()
        claude_path.write_text(
            json.dumps({"mcpServers": {"a": {"command": "x"}}}),
            encoding="utf-8",
        )

        with structlog.testing.capture_logs() as logs:
            servers = discover_mcp(tmp_path)

        assert "a" in servers
        assert [e for e in logs if e.get("event") == "mcp.conflict"] == []
