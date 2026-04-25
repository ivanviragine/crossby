"""Tests for project config detection."""

from __future__ import annotations

import json
from pathlib import Path

from crossby.config.detection import DetectedConfig, detect_source_configs
from crossby.models.ai import AIToolID


class TestDetectInstructions:
    def test_detects_claude_instructions(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("# Instructions")
        items = detect_source_configs(AIToolID.CLAUDE, tmp_path)
        instr = [i for i in items if i.config_type == "instructions"]
        assert len(instr) == 1
        assert instr[0].portable is True
        assert "CLAUDE.md" in instr[0].detail

    def test_detects_cursor_instructions(self, tmp_path: Path) -> None:
        (tmp_path / ".cursorrules").write_text("rules")
        items = detect_source_configs(AIToolID.CURSOR, tmp_path)
        instr = [i for i in items if i.config_type == "instructions"]
        assert len(instr) == 1
        assert ".cursorrules" in instr[0].detail

    def test_no_instructions_returns_empty(self, tmp_path: Path) -> None:
        items = detect_source_configs(AIToolID.CLAUDE, tmp_path)
        instr = [i for i in items if i.config_type == "instructions"]
        assert len(instr) == 0


class TestDetectSkills:
    def test_detects_skills_with_count(self, tmp_path: Path) -> None:
        skills = tmp_path / ".claude" / "skills"
        (skills / "task").mkdir(parents=True)
        (skills / "task" / "SKILL.md").write_text("# Task")
        (skills / "review").mkdir()
        (skills / "review" / "SKILL.md").write_text("# Review")

        items = detect_source_configs(AIToolID.CLAUDE, tmp_path)
        sk = [i for i in items if i.config_type == "skills"]
        assert len(sk) == 1
        assert sk[0].portable is True
        assert "2 skills" in sk[0].detail

    def test_no_skills_returns_empty(self, tmp_path: Path) -> None:
        items = detect_source_configs(AIToolID.CLAUDE, tmp_path)
        sk = [i for i in items if i.config_type == "skills"]
        assert len(sk) == 0


class TestDetectAllowlist:
    def test_detects_claude_allowlist(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {"permissions": {"allow": ["Bash(myapp:*)", "Bash(npm:*)", "Read(**)"]}}
        (claude_dir / "settings.json").write_text(json.dumps(settings))

        items = detect_source_configs(AIToolID.CLAUDE, tmp_path)
        al = [i for i in items if i.config_type == "allowlist"]
        assert len(al) == 1
        assert al[0].portable is True
        assert "2 patterns" in al[0].detail  # Read(**) is not Bash(), so filtered

    def test_detects_cursor_allowlist(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir()
        settings = {"permissions": {"allow": ["Shell(myapp:*)"]}}
        (cursor_dir / "cli.json").write_text(json.dumps(settings))

        items = detect_source_configs(AIToolID.CURSOR, tmp_path)
        al = [i for i in items if i.config_type == "allowlist"]
        assert len(al) == 1
        assert "1 pattern" in al[0].detail

    def test_no_allowlist_returns_empty(self, tmp_path: Path) -> None:
        items = detect_source_configs(AIToolID.CLAUDE, tmp_path)
        al = [i for i in items if i.config_type == "allowlist"]
        assert len(al) == 0


class TestDetectHooks:
    def test_detects_claude_hooks(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {
            "hooks": {
                "PreToolUse": [{"matcher": "Edit", "hooks": [{"type": "command", "command": "echo"}]}]
            }
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings))

        items = detect_source_configs(AIToolID.CLAUDE, tmp_path)
        hooks = [i for i in items if i.config_type == "hooks"]
        assert len(hooks) == 1
        assert hooks[0].portable is True
        assert "1 hook" in hooks[0].detail
        assert hooks[0].reason == ""

    def test_no_hooks_returns_empty(self, tmp_path: Path) -> None:
        items = detect_source_configs(AIToolID.CLAUDE, tmp_path)
        hooks = [i for i in items if i.config_type == "hooks"]
        assert len(hooks) == 0


class TestDetectMcpServers:
    def test_detects_claude_mcp_servers(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {
            "mcpServers": {
                "memory": {"command": "npx", "args": ["-y", "@mcp/memory"]},
                "search": {"command": "npx", "args": ["-y", "@mcp/search"]},
            }
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings))

        items = detect_source_configs(AIToolID.CLAUDE, tmp_path)
        mcp = [i for i in items if i.config_type == "mcp_servers"]
        assert len(mcp) == 1
        assert mcp[0].portable is True
        assert "2 MCP servers" in mcp[0].detail

    def test_detects_cursor_mcp_servers(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir()
        (cursor_dir / "mcp.json").write_text(
            json.dumps({"mcpServers": {"ctx": {"command": "npx"}}})
        )
        items = detect_source_configs(AIToolID.CURSOR, tmp_path)
        mcp = [i for i in items if i.config_type == "mcp_servers"]
        assert len(mcp) == 1
        assert mcp[0].portable is True

    def test_detects_copilot_mcp_servers(self, tmp_path: Path) -> None:
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        (vscode_dir / "mcp.json").write_text(
            json.dumps({"servers": {"ctx": {"type": "stdio", "command": "npx"}}})
        )
        items = detect_source_configs(AIToolID.COPILOT, tmp_path)
        mcp = [i for i in items if i.config_type == "mcp_servers"]
        assert len(mcp) == 1

    def test_detects_gemini_mcp_servers(self, tmp_path: Path) -> None:
        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        (gemini_dir / "settings.json").write_text(
            json.dumps({"mcpServers": {"ctx": {"command": "npx"}}})
        )
        items = detect_source_configs(AIToolID.GEMINI, tmp_path)
        mcp = [i for i in items if i.config_type == "mcp_servers"]
        assert len(mcp) == 1

    def test_detects_codex_mcp_servers(self, tmp_path: Path) -> None:
        import tomli_w

        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "config.toml").write_text(
            tomli_w.dumps({"mcp_servers": {"ctx": {"command": "npx"}}})
        )
        items = detect_source_configs(AIToolID.CODEX, tmp_path)
        mcp = [i for i in items if i.config_type == "mcp_servers"]
        assert len(mcp) == 1

    def test_no_mcp_for_tool_without_config(self, tmp_path: Path) -> None:
        items = detect_source_configs(AIToolID.CURSOR, tmp_path)
        mcp = [i for i in items if i.config_type == "mcp_servers"]
        assert len(mcp) == 0


class TestDetectGeminiHooksAndPermissions:
    def test_detects_gemini_hooks_dict_shape(self, tmp_path: Path) -> None:
        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        (gemini_dir / "settings.json").write_text(
            json.dumps({
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Edit", "hooks": [{"type": "command", "command": "x"}]},
                        {"matcher": "Write", "hooks": [{"type": "command", "command": "y"}]},
                    ]
                }
            })
        )
        items = detect_source_configs(AIToolID.GEMINI, tmp_path)
        hooks = [i for i in items if i.config_type == "hooks"]
        assert len(hooks) == 1
        assert "2 hooks" in hooks[0].detail

    def test_detects_gemini_hooks_legacy_list(self, tmp_path: Path) -> None:
        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        (gemini_dir / "settings.json").write_text(
            json.dumps({"hooks": [{"event": "preToolUse", "command": "x"}]})
        )
        items = detect_source_configs(AIToolID.GEMINI, tmp_path)
        hooks = [i for i in items if i.config_type == "hooks"]
        assert len(hooks) == 1
        assert "1 hook" in hooks[0].detail

    def test_detects_gemini_permissions_toml(self, tmp_path: Path) -> None:
        policies = tmp_path / ".gemini" / "policies"
        policies.mkdir(parents=True)
        (policies / "crossby.toml").write_text(
            '[[rule]]\n'
            'toolName = "run_shell_command"\n'
            'decision = "allow"\n'
            'commandPrefix = "npm"\n'
            '[[rule]]\n'
            'toolName = "run_shell_command"\n'
            'decision = "allow"\n'
            'commandPrefix = "git"\n'
        )
        items = detect_source_configs(AIToolID.GEMINI, tmp_path)
        al = [i for i in items if i.config_type == "allowlist"]
        assert len(al) == 1
        assert "2 patterns" in al[0].detail


class TestDetectCustomCommands:
    def test_detects_custom_commands(self, tmp_path: Path) -> None:
        cmds = tmp_path / ".claude" / "commands"
        cmds.mkdir(parents=True)
        (cmds / "deploy.md").write_text("Deploy command")
        (cmds / "test.md").write_text("Test command")

        items = detect_source_configs(AIToolID.CLAUDE, tmp_path)
        cc = [i for i in items if i.config_type == "custom_commands"]
        assert len(cc) == 1
        assert cc[0].portable is False
        assert "2 custom commands" in cc[0].detail

    def test_no_commands_dir_returns_empty(self, tmp_path: Path) -> None:
        items = detect_source_configs(AIToolID.CLAUDE, tmp_path)
        cc = [i for i in items if i.config_type == "custom_commands"]
        assert len(cc) == 0

    def test_only_claude_has_commands(self, tmp_path: Path) -> None:
        items = detect_source_configs(AIToolID.CURSOR, tmp_path)
        cc = [i for i in items if i.config_type == "custom_commands"]
        assert len(cc) == 0


class TestDetectFull:
    def test_full_claude_detection(self, tmp_path: Path) -> None:
        """Detects all config types for a fully configured Claude project."""
        (tmp_path / "CLAUDE.md").write_text("# Instructions")
        skills = tmp_path / ".claude" / "skills" / "task"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text("# Task")
        settings = {
            "permissions": {"allow": ["Bash(myapp:*)"]},
            "hooks": {"PreToolUse": [{"matcher": "Edit"}]},
            "mcpServers": {"memory": {"command": "npx"}},
        }
        (tmp_path / ".claude" / "settings.json").write_text(json.dumps(settings))
        cmds = tmp_path / ".claude" / "commands"
        cmds.mkdir()
        (cmds / "deploy.md").write_text("Deploy")

        items = detect_source_configs(AIToolID.CLAUDE, tmp_path)
        types = {i.config_type for i in items}
        assert types == {"instructions", "skills", "allowlist", "hooks", "mcp_servers", "custom_commands"}

        portable = [i for i in items if i.portable]
        not_portable = [i for i in items if not i.portable]
        assert len(portable) == 5  # instructions, skills, allowlist, hooks, mcp_servers
        assert len(not_portable) == 1  # custom_commands
