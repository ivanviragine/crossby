"""Per-tool DECLARE activators — provenance, reverts, trust, malformed refusal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossby.models.ai import AIToolID
from crossby.scenes import declare
from crossby.sync.ownership import OwnershipLedger, SceneDeclareKey
from tests.unit.test_scenes.conftest import read_json, write_json

CLAUDE_SETTINGS = Path(".claude") / "settings.json"
CODEX_CONFIG = Path(".codex") / "config.toml"
ANTIGRAVITY_MCP = Path(".agents") / "mcp_config.json"


class TestClaudeSkillOverrides:
    def test_writes_and_records(self, tmp_path: Path) -> None:
        ledger = OwnershipLedger()
        declare.apply_claude_skill_overrides(tmp_path, {"a", "b"}, ledger, version=(2, 1, 218))
        assert read_json(tmp_path / CLAUDE_SETTINGS)["skillOverrides"] == {"a": "off", "b": "off"}
        assert ledger.scene_declare(AIToolID.CLAUDE, SceneDeclareKey.SKILL_OVERRIDES) == frozenset(
            {"a", "b"}
        )

    def test_old_version_blocks_adds_but_not_removes(self, tmp_path: Path) -> None:
        ledger = OwnershipLedger()
        # First disable on a new build.
        declare.apply_claude_skill_overrides(tmp_path, {"a"}, ledger, version=(2, 1, 218))
        # Now an old build: adding "b" is blocked, but removing owned "a" still works.
        result = declare.apply_claude_skill_overrides(tmp_path, {"b"}, ledger, version=(2, 1, 100))
        settings = read_json(tmp_path / CLAUDE_SETTINGS)
        assert "b" not in settings.get("skillOverrides", {})
        assert "a" not in settings.get("skillOverrides", {})  # owned removal honoured
        assert result.message and "not filtered" in result.message

    def test_never_overwrites_a_user_non_off_override(self, tmp_path: Path) -> None:
        # A user pinned "review-skill" to a non-"off" value; a scene deselecting it
        # must not clobber that value or claim ownership of it.
        write_json(tmp_path / CLAUDE_SETTINGS, {"skillOverrides": {"review-skill": "on"}})
        ledger = OwnershipLedger()
        result = declare.apply_claude_skill_overrides(
            tmp_path, {"review-skill"}, ledger, version=(2, 1, 218)
        )
        assert read_json(tmp_path / CLAUDE_SETTINGS)["skillOverrides"] == {"review-skill": "on"}
        assert ledger.scene_declare(AIToolID.CLAUDE, SceneDeclareKey.SKILL_OVERRIDES) == frozenset()
        assert result.message and "user override" in result.message

    def test_revert_leaves_user_entry(self, tmp_path: Path) -> None:
        write_json(tmp_path / CLAUDE_SETTINGS, {"skillOverrides": {"user": "off"}})
        ledger = OwnershipLedger()
        declare.apply_claude_skill_overrides(tmp_path, {"a"}, ledger, version=(2, 1, 218))
        declare.apply_claude_skill_overrides(tmp_path, set(), ledger, version=(2, 1, 218))
        assert read_json(tmp_path / CLAUDE_SETTINGS)["skillOverrides"] == {"user": "off"}


class TestClaudeDenyAgents:
    def test_writes_agent_rule(self, tmp_path: Path) -> None:
        ledger = OwnershipLedger()
        declare.apply_claude_deny_agents(tmp_path, {"deployer"}, ledger)
        assert read_json(tmp_path / CLAUDE_SETTINGS)["permissions"]["deny"] == ["Agent(deployer)"]

    def test_preserves_user_deny_and_allow(self, tmp_path: Path) -> None:
        write_json(
            tmp_path / CLAUDE_SETTINGS,
            {"permissions": {"allow": ["Bash(git diff:*)"], "deny": ["Agent(user)"]}},
        )
        ledger = OwnershipLedger()
        declare.apply_claude_deny_agents(tmp_path, {"deployer"}, ledger)
        declare.apply_claude_deny_agents(tmp_path, set(), ledger)  # revert crossby's
        perms = read_json(tmp_path / CLAUDE_SETTINGS)["permissions"]
        assert perms["allow"] == ["Bash(git diff:*)"]
        assert perms["deny"] == ["Agent(user)"]

    def test_malformed_settings_refused(self, tmp_path: Path) -> None:
        (tmp_path / ".claude").mkdir()
        (tmp_path / CLAUDE_SETTINGS).write_text("{ not json", encoding="utf-8")
        result = declare.apply_claude_deny_agents(tmp_path, {"x"}, OwnershipLedger())
        assert result.action == "error"
        assert (tmp_path / CLAUDE_SETTINGS).read_text(encoding="utf-8") == "{ not json"


class TestClaudeDisabledMcp:
    def test_writes_disabled_list(self, tmp_path: Path) -> None:
        ledger = OwnershipLedger()
        declare.apply_claude_disabled_mcp(tmp_path, {"linear"}, ledger)
        assert read_json(tmp_path / CLAUDE_SETTINGS)["disabledMcpjsonServers"] == ["linear"]


class TestCodexDisabledMcp:
    def _config(self, tmp_path: Path) -> None:
        (tmp_path / ".codex").mkdir()
        (tmp_path / CODEX_CONFIG).write_text(
            '[mcp_servers.github]\ncommand = "gh-mcp"\n\n'
            '[mcp_servers.linear]\ncommand = "lin-mcp"\n',
            encoding="utf-8",
        )

    def test_sets_enabled_false_and_reverts_cleanly(self, tmp_path: Path) -> None:
        self._config(tmp_path)
        original = (tmp_path / CODEX_CONFIG).read_text(encoding="utf-8")
        ledger = OwnershipLedger()
        declare.apply_codex_disabled_mcp(tmp_path, {"linear"}, ledger, trusted=True)
        text = (tmp_path / CODEX_CONFIG).read_text(encoding="utf-8")
        assert "[mcp_servers.linear]" in text and "enabled = false" in text
        # Revert restores byte-identity (the added enabled line is removed).
        declare.apply_codex_disabled_mcp(tmp_path, set(), ledger, trusted=True)
        assert (tmp_path / CODEX_CONFIG).read_text(encoding="utf-8") == original

    def test_untrusted_project_reports_caveat(self, tmp_path: Path) -> None:
        self._config(tmp_path)
        result = declare.apply_codex_disabled_mcp(
            tmp_path, {"linear"}, OwnershipLedger(), trusted=False
        )
        assert result.message and "does not trust" in result.message

    def test_read_back_confirms_disabled(self, tmp_path: Path) -> None:
        self._config(tmp_path)
        declare.apply_codex_disabled_mcp(tmp_path, {"linear"}, OwnershipLedger(), trusted=True)
        import tomllib

        data = tomllib.loads((tmp_path / CODEX_CONFIG).read_text(encoding="utf-8"))
        assert data["mcp_servers"]["linear"]["enabled"] is False
        assert "enabled" not in data["mcp_servers"]["github"]

    def test_clear_preserves_user_flipped_enabled_true(self, tmp_path: Path) -> None:
        # Crossby disables linear (enabled=false); the user then flips it to true.
        # Clear must release ownership WITHOUT deleting the user's setting.
        self._config(tmp_path)
        ledger = OwnershipLedger()
        declare.apply_codex_disabled_mcp(tmp_path, {"linear"}, ledger, trusted=True)
        path = tmp_path / CODEX_CONFIG
        path.write_text(
            path.read_text(encoding="utf-8").replace("enabled = false", "enabled = true"),
            encoding="utf-8",
        )
        declare.apply_codex_disabled_mcp(tmp_path, set(), ledger, trusted=True)  # clear
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8"))
        assert data["mcp_servers"]["linear"]["enabled"] is True  # user value kept
        assert (
            ledger.scene_declare(AIToolID.CODEX, SceneDeclareKey.CODEX_MCP_DISABLED) == frozenset()
        )

    def test_all_splices_fail_errors_and_owns_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every enabled=false splice failing must surface an error, not an
        # "already applied" no-op, and claim no ledger ownership.
        self._config(tmp_path)
        monkeypatch.setattr("crossby.scenes.declare.set_scalar", lambda *_a, **_k: None)
        ledger = OwnershipLedger()
        result = declare.apply_codex_disabled_mcp(tmp_path, {"linear"}, ledger, trusted=True)
        assert result.action == "error"
        assert (
            ledger.scene_declare(AIToolID.CODEX, SceneDeclareKey.CODEX_MCP_DISABLED) == frozenset()
        )


class TestAntigravityDisabledMcp:
    def test_sets_disabled_true_and_reverts(self, tmp_path: Path) -> None:
        write_json(
            tmp_path / ANTIGRAVITY_MCP,
            {"mcpServers": {"github": {"command": "gh"}, "linear": {"command": "lin"}}},
        )
        ledger = OwnershipLedger()
        declare.apply_antigravity_disabled_mcp(tmp_path, {"linear"}, ledger)
        data = read_json(tmp_path / ANTIGRAVITY_MCP)
        assert data["mcpServers"]["linear"]["disabled"] is True
        assert "disabled" not in data["mcpServers"]["github"]
        declare.apply_antigravity_disabled_mcp(tmp_path, set(), ledger)
        assert "disabled" not in read_json(tmp_path / ANTIGRAVITY_MCP)["mcpServers"]["linear"]

    def test_absent_server_is_noop(self, tmp_path: Path) -> None:
        write_json(tmp_path / ANTIGRAVITY_MCP, {"mcpServers": {"github": {"command": "gh"}}})
        result = declare.apply_antigravity_disabled_mcp(tmp_path, {"ghost"}, OwnershipLedger())
        assert result.action == "skipped"
        # No stray key added.
        assert json.loads((tmp_path / ANTIGRAVITY_MCP).read_text())["mcpServers"] == {
            "github": {"command": "gh"}
        }

    def test_non_dict_entry_excluded(self, tmp_path: Path) -> None:
        # A malformed non-dict server entry must not be disabled, claimed, or
        # counted — only the valid dict entry is toggled.
        write_json(
            tmp_path / ANTIGRAVITY_MCP,
            {"mcpServers": {"github": {"command": "gh"}, "broken": "not-a-dict"}},
        )
        ledger = OwnershipLedger()
        result = declare.apply_antigravity_disabled_mcp(tmp_path, {"broken", "github"}, ledger)
        data = read_json(tmp_path / ANTIGRAVITY_MCP)
        assert data["mcpServers"]["github"]["disabled"] is True
        assert data["mcpServers"]["broken"] == "not-a-dict"  # left untouched
        assert result.added == 1
        assert ledger.scene_declare(
            AIToolID.ANTIGRAVITY_CLI, SceneDeclareKey.ANTIGRAVITY_MCP_DISABLED
        ) == frozenset({"github"})
