"""Per-tool DECLARE activators — provenance, reverts, trust, malformed refusal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossby.models.ai import AIToolID
from crossby.scenes import declare
from crossby.sync.ownership import (
    OwnershipLedger,
    SceneDeclareKey,
    load_ledger,
    save_ledger,
)
from tests.unit.test_scenes.conftest import read_json, write_json

CLAUDE_SETTINGS = Path(".claude") / "settings.json"
CODEX_CONFIG = Path(".codex") / "config.toml"
ANTIGRAVITY_MCP = Path(".agents") / "mcp_config.json"


def _revert_with_failed_write(
    root: Path, clear_fn, monkeypatch: pytest.MonkeyPatch
) -> dict[str, bool]:
    """Run *clear_fn* with the JSON write raising, persisting the ledger in a
    ``finally`` exactly like ``scenes.engine.clear_scene``.

    Returns a mutable ``{"fail": True}`` flag whose value can be flipped to
    ``False`` to restore the write for a retry.
    """
    real = declare.write_json_file
    flag = {"fail": True}

    def _maybe(path: Path, data: object) -> None:
        if flag["fail"]:
            raise OSError("disk full")
        real(path, data)  # type: ignore[arg-type]

    monkeypatch.setattr("crossby.scenes.declare.write_json_file", _maybe)
    ledger = load_ledger(root)
    with pytest.raises(OSError):
        try:
            clear_fn(ledger)
        finally:
            save_ledger(root, ledger)
    return flag


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


class TestTransactionalLedger:
    """A JSON write failure during clear must never drop ownership of a setting
    still on disk: ``record_scene_declare`` runs only AFTER the write succeeds, so
    the ledger the engine persists in its ``finally`` stays consistent with disk
    and a retried clear reverts it.
    """

    def test_skill_overrides_write_failure_retains_ownership(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ledger = OwnershipLedger()
        declare.apply_claude_skill_overrides(tmp_path, {"deploy-prod"}, ledger, version=(2, 1, 218))
        save_ledger(tmp_path, ledger)

        flag = _revert_with_failed_write(
            tmp_path,
            lambda lg: declare.apply_claude_skill_overrides(
                tmp_path, set(), lg, version=(2, 1, 218)
            ),
            monkeypatch,
        )
        # (a) setting still on disk, (b) ledger still records prior ownership.
        assert read_json(tmp_path / CLAUDE_SETTINGS)["skillOverrides"] == {"deploy-prod": "off"}
        assert load_ledger(tmp_path).scene_declare(
            AIToolID.CLAUDE, SceneDeclareKey.SKILL_OVERRIDES
        ) == frozenset({"deploy-prod"})

        # (c) a retry with the write restored fully reverts.
        flag["fail"] = False
        retry = load_ledger(tmp_path)
        declare.apply_claude_skill_overrides(tmp_path, set(), retry, version=(2, 1, 218))
        save_ledger(tmp_path, retry)
        assert "skillOverrides" not in read_json(tmp_path / CLAUDE_SETTINGS)
        assert (
            load_ledger(tmp_path).scene_declare(AIToolID.CLAUDE, SceneDeclareKey.SKILL_OVERRIDES)
            == frozenset()
        )

    def test_deny_agents_write_failure_retains_ownership(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ledger = OwnershipLedger()
        declare.apply_claude_deny_agents(tmp_path, {"deployer"}, ledger)
        save_ledger(tmp_path, ledger)

        flag = _revert_with_failed_write(
            tmp_path,
            lambda lg: declare.apply_claude_deny_agents(tmp_path, set(), lg),
            monkeypatch,
        )
        assert read_json(tmp_path / CLAUDE_SETTINGS)["permissions"]["deny"] == ["Agent(deployer)"]
        assert load_ledger(tmp_path).scene_declare(
            AIToolID.CLAUDE, SceneDeclareKey.DENY_AGENTS
        ) == frozenset({"Agent(deployer)"})

        flag["fail"] = False
        retry = load_ledger(tmp_path)
        declare.apply_claude_deny_agents(tmp_path, set(), retry)
        save_ledger(tmp_path, retry)
        assert read_json(tmp_path / CLAUDE_SETTINGS).get("permissions", {}).get("deny", []) == []
        assert (
            load_ledger(tmp_path).scene_declare(AIToolID.CLAUDE, SceneDeclareKey.DENY_AGENTS)
            == frozenset()
        )

    def test_disabled_mcp_write_failure_retains_ownership(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ledger = OwnershipLedger()
        declare.apply_claude_disabled_mcp(tmp_path, {"linear"}, ledger)
        save_ledger(tmp_path, ledger)

        flag = _revert_with_failed_write(
            tmp_path,
            lambda lg: declare.apply_claude_disabled_mcp(tmp_path, set(), lg),
            monkeypatch,
        )
        assert read_json(tmp_path / CLAUDE_SETTINGS)["disabledMcpjsonServers"] == ["linear"]
        assert load_ledger(tmp_path).scene_declare(
            AIToolID.CLAUDE, SceneDeclareKey.DISABLED_MCP
        ) == frozenset({"linear"})

        flag["fail"] = False
        retry = load_ledger(tmp_path)
        declare.apply_claude_disabled_mcp(tmp_path, set(), retry)
        save_ledger(tmp_path, retry)
        assert "disabledMcpjsonServers" not in read_json(tmp_path / CLAUDE_SETTINGS)
        assert (
            load_ledger(tmp_path).scene_declare(AIToolID.CLAUDE, SceneDeclareKey.DISABLED_MCP)
            == frozenset()
        )

    def test_antigravity_write_failure_retains_ownership(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_json(tmp_path / ANTIGRAVITY_MCP, {"mcpServers": {"linear": {"command": "lin"}}})
        ledger = OwnershipLedger()
        declare.apply_antigravity_disabled_mcp(tmp_path, {"linear"}, ledger)
        save_ledger(tmp_path, ledger)

        flag = _revert_with_failed_write(
            tmp_path,
            lambda lg: declare.apply_antigravity_disabled_mcp(tmp_path, set(), lg),
            monkeypatch,
        )
        assert read_json(tmp_path / ANTIGRAVITY_MCP)["mcpServers"]["linear"]["disabled"] is True
        assert load_ledger(tmp_path).scene_declare(
            AIToolID.ANTIGRAVITY_CLI, SceneDeclareKey.ANTIGRAVITY_MCP_DISABLED
        ) == frozenset({"linear"})

        flag["fail"] = False
        retry = load_ledger(tmp_path)
        declare.apply_antigravity_disabled_mcp(tmp_path, set(), retry)
        save_ledger(tmp_path, retry)
        assert "disabled" not in read_json(tmp_path / ANTIGRAVITY_MCP)["mcpServers"]["linear"]
        assert (
            load_ledger(tmp_path).scene_declare(
                AIToolID.ANTIGRAVITY_CLI, SceneDeclareKey.ANTIGRAVITY_MCP_DISABLED
            )
            == frozenset()
        )


class TestCodexPartialFailure:
    """A Codex ``unset_scalar`` splice that returns ``None`` during clear must keep
    the server owned (so a retry reverts it) and emit an ``error`` row — while any
    splice that *did* land is still written and its ownership released.
    """

    def _config(self, tmp_path: Path) -> None:
        (tmp_path / ".codex").mkdir()
        (tmp_path / CODEX_CONFIG).write_text(
            '[mcp_servers.github]\ncommand = "gh-mcp"\n\n'
            '[mcp_servers.linear]\ncommand = "lin-mcp"\n',
            encoding="utf-8",
        )

    def _disable_both(self, tmp_path: Path) -> None:
        ledger = OwnershipLedger()
        declare.apply_codex_disabled_mcp(tmp_path, {"github", "linear"}, ledger, trusted=True)
        save_ledger(tmp_path, ledger)

    def _codex_data(self, tmp_path: Path) -> dict:
        import tomllib

        return tomllib.loads((tmp_path / CODEX_CONFIG).read_text(encoding="utf-8"))

    @pytest.mark.parametrize(
        "config_text",
        [
            'mcp_servers.linear.command = "lin-mcp"\nmcp_servers.linear.enabled = true\n',
            'mcp_servers.linear = { command = "lin-mcp", enabled = true }\n',
        ],
    )
    def test_implicit_enabled_value_reverts_cleanly(self, tmp_path: Path, config_text: str) -> None:
        # set_scalar supports both dotted assignments and inline-table entries;
        # clearing the scene must be able to undo either representation too.
        (tmp_path / ".codex").mkdir()
        (tmp_path / CODEX_CONFIG).write_text(config_text, encoding="utf-8")
        ledger = OwnershipLedger()

        applied = declare.apply_codex_disabled_mcp(tmp_path, {"linear"}, ledger, trusted=True)
        assert applied.action == "updated"
        assert self._codex_data(tmp_path)["mcp_servers"]["linear"]["enabled"] is False
        result = declare.apply_codex_disabled_mcp(tmp_path, set(), ledger, trusted=True)
        assert result.action == "updated"
        assert result.revoked == 1
        assert "enabled" not in self._codex_data(tmp_path)["mcp_servers"]["linear"]
        assert (
            ledger.scene_declare(AIToolID.CODEX, SceneDeclareKey.CODEX_MCP_DISABLED) == frozenset()
        )

    def test_all_removals_fail_retain_all_ownership(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._config(tmp_path)
        self._disable_both(tmp_path)
        monkeypatch.setattr("crossby.scenes.declare.unset_scalar", lambda *_a, **_k: None)

        ledger = load_ledger(tmp_path)
        result = declare.apply_codex_disabled_mcp(tmp_path, set(), ledger, trusted=True)
        save_ledger(tmp_path, ledger)  # engine's finally

        assert result.action == "error"
        # Nothing was written — both stay disabled on disk and stay owned.
        data = self._codex_data(tmp_path)
        assert data["mcp_servers"]["github"]["enabled"] is False
        assert data["mcp_servers"]["linear"]["enabled"] is False
        assert load_ledger(tmp_path).scene_declare(
            AIToolID.CODEX, SceneDeclareKey.CODEX_MCP_DISABLED
        ) == frozenset({"github", "linear"})

    def test_mixed_writes_success_and_retains_failed_owner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._config(tmp_path)
        self._disable_both(tmp_path)

        real = declare.unset_scalar

        def _fail_linear(
            text: str,
            path: tuple[str, ...],
            key: str,
            *,
            include_implicit: bool = False,
        ) -> str | None:
            return (
                None
                if path[-1] == "linear"
                else real(text, path, key, include_implicit=include_implicit)
            )

        monkeypatch.setattr("crossby.scenes.declare.unset_scalar", _fail_linear)

        ledger = load_ledger(tmp_path)
        result = declare.apply_codex_disabled_mcp(tmp_path, set(), ledger, trusted=True)
        save_ledger(tmp_path, ledger)  # engine's finally

        assert result.action == "error"
        # The successful revert (github) is written; the failed one (linear) is not.
        data = self._codex_data(tmp_path)
        assert "enabled" not in data["mcp_servers"]["github"]  # reverted + written
        assert data["mcp_servers"]["linear"]["enabled"] is False  # still disabled
        # revoked counts only the successful removal.
        assert result.revoked == 1
        # Only the failed server stays owned, so a retry re-attempts exactly it.
        assert load_ledger(tmp_path).scene_declare(
            AIToolID.CODEX, SceneDeclareKey.CODEX_MCP_DISABLED
        ) == frozenset({"linear"})

        # Retry with the splice restored fully reverts linear and drops ownership.
        monkeypatch.setattr("crossby.scenes.declare.unset_scalar", real)
        retry = load_ledger(tmp_path)
        retry_result = declare.apply_codex_disabled_mcp(tmp_path, set(), retry, trusted=True)
        save_ledger(tmp_path, retry)
        assert retry_result.action == "updated"
        assert "enabled" not in self._codex_data(tmp_path)["mcp_servers"]["linear"]
        assert (
            load_ledger(tmp_path).scene_declare(AIToolID.CODEX, SceneDeclareKey.CODEX_MCP_DISABLED)
            == frozenset()
        )
