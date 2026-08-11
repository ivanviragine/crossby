"""Integration test: full crossby sync mcp flow."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from crossby.models.ai import AIToolID
from crossby.models.config import HookEntry, MCPServerConfig
from crossby.sync import run_sync
from crossby.sync.base import SyncConcern, SyncData, SyncResult


def _build_sync_data(servers_yaml: dict[str, Any]) -> SyncData:
    """Build SyncData with MCP servers from a dict (matching YAML structure)."""
    mcp_servers = {name: MCPServerConfig(**entry) for name, entry in servers_yaml.items()}
    return SyncData(mcp_servers=mcp_servers)


def _sync_mcp(project_root: Path, servers_yaml: dict[str, Any]) -> None:
    """Helper: build SyncData and run all MCP writers."""
    data = _build_sync_data(servers_yaml)
    all_tools = list(AIToolID)
    run_sync(data, project_root, concern=SyncConcern.MCP, installed_tools=all_tools)


class TestFullSyncMCP:
    def test_syncs_to_all_five_tools(self, tmp_path: Path) -> None:
        servers = {
            "context7": {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]},
        }
        _sync_mcp(tmp_path, servers)

        # Claude — servers land in .mcp.json (what Claude Code reads), not settings.json
        data = json.loads((tmp_path / ".mcp.json").read_text())
        assert "context7" in data["mcpServers"]

        # Cursor
        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
        assert "context7" in data["mcpServers"]

        # Copilot
        data = json.loads((tmp_path / ".vscode" / "mcp.json").read_text())
        assert "context7" in data["servers"]
        assert data["servers"]["context7"]["type"] == "stdio"

        # Antigravity CLI
        data = json.loads((tmp_path / ".agents" / "mcp_config.json").read_text())
        assert "context7" in data["mcpServers"]

        # Codex
        data = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text())
        assert "context7" in data["mcp_servers"]

    def test_second_sync_is_idempotent(self, tmp_path: Path) -> None:
        servers = {"ctx": {"command": "npx", "args": ["-y", "mcp"]}}
        _sync_mcp(tmp_path, servers)

        # Second sync should produce all "skipped"
        data = _build_sync_data(servers)
        results = run_sync(data, tmp_path, concern=SyncConcern.MCP, installed_tools=list(AIToolID))
        for result in results:
            assert result.action == "skipped", (
                f"{result.tool_id}: expected skipped, got {result.action}"
            )

    def test_enabled_false_removes_from_all_tools(self, tmp_path: Path) -> None:
        # First: add the server
        _sync_mcp(
            tmp_path,
            {"old": {"command": "npx", "args": ["-y", "old-mcp"]}},
        )

        # Verify it's there
        data = json.loads((tmp_path / ".mcp.json").read_text())
        assert "old" in data["mcpServers"]

        # Now disable it
        _sync_mcp(
            tmp_path,
            {"old": {"command": "npx", "enabled": False}},
        )

        # Should be removed
        data = json.loads((tmp_path / ".mcp.json").read_text())
        assert "old" not in data["mcpServers"]

        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
        assert "old" not in data["mcpServers"]

    def test_preserves_unmanaged_servers_in_all_tools(self, tmp_path: Path) -> None:
        # A user server left in the legacy .claude/settings.json location must
        # be preserved untouched — crossby writes .mcp.json and never edits the
        # settings.json server table it no longer owns.
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / "settings.json").write_text(
            json.dumps({"mcpServers": {"user-srv": {"command": "node"}}}),
            encoding="utf-8",
        )

        _sync_mcp(
            tmp_path,
            {"crossby-srv": {"command": "npx", "args": ["-y", "mcp"]}},
        )

        settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert "user-srv" in settings["mcpServers"]  # untouched
        assert json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"].keys() == {
            "crossby-srv"
        }

    def test_env_vars_preserved_across_all_formats(self, tmp_path: Path) -> None:
        servers = {
            "github": {
                "command": "npx",
                "args": ["-y", "server-github"],
                "env": {"TOKEN": "${GITHUB_TOKEN}"},
            }
        }
        _sync_mcp(tmp_path, servers)

        for path, key, entry_key in [
            (tmp_path / ".mcp.json", "mcpServers", "github"),
            (tmp_path / ".cursor" / "mcp.json", "mcpServers", "github"),
            (tmp_path / ".agents" / "mcp_config.json", "mcpServers", "github"),
            (tmp_path / ".vscode" / "mcp.json", "servers", "github"),
        ]:
            data = json.loads(path.read_text())
            assert data[key][entry_key]["env"]["TOKEN"] == "${GITHUB_TOKEN}", f"Failed for {path}"

        toml_data = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text())
        assert toml_data["mcp_servers"]["github"]["env"]["TOKEN"] == "${GITHUB_TOKEN}"


class TestMCPScopeAndRoundTrip:
    def test_claude_mcp_round_trips_through_mcp_json(self, tmp_path: Path) -> None:
        # A server Claude already has in .mcp.json is read from and written back
        # to that one file; settings.json only ever carries the approval list.
        from crossby.sync.readers import build_sync_data

        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"ctx": {"command": "npx", "args": ["-y", "ctx"]}}}),
            encoding="utf-8",
        )
        data = build_sync_data(tmp_path, from_tool=AIToolID.CLAUDE)
        assert "ctx" in data.mcp_servers

        run_sync(
            data,
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            concern=SyncConcern.MCP,
            installed_tools=[AIToolID.CLAUDE],
        )
        assert "ctx" in json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]
        settings = tmp_path / ".claude" / "settings.json"
        if settings.exists():
            assert "mcpServers" not in json.loads(settings.read_text())

    def test_user_scope_servers_stay_out_of_project_files_by_default(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        from crossby.sync import mcp_discovery
        from crossby.sync.readers import build_sync_data

        home = tmp_path / "home" / ".claude.json"
        home.parent.mkdir()
        home.write_text(
            json.dumps({"mcpServers": {"personal": {"command": "npx", "env": {"S": "secret"}}}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(mcp_discovery, "_GLOBAL_CLAUDE_JSON_PATH", home)

        # Default build: user-scope servers are never discovered...
        data = build_sync_data(tmp_path, from_tool=AIToolID.CLAUDE)
        assert data.mcp_servers == {}

        # ...and a full sync writes nothing into any committed project file.
        run_sync(data, tmp_path, concern=SyncConcern.MCP, installed_tools=list(AIToolID))
        for rel in (
            ".mcp.json",
            ".cursor/mcp.json",
            ".vscode/mcp.json",
            ".agents/mcp_config.json",
            ".codex/config.toml",
            ".claude/settings.json",
        ):
            assert not (tmp_path / rel).exists(), f"{rel} must not be created from user scope"

        # Opting in surfaces them.
        opted = build_sync_data(tmp_path, from_tool=AIToolID.CLAUDE, include_user_scope=True)
        assert "personal" in opted.mcp_servers


def _snapshot_tree(root: Path) -> dict[str, tuple[bytes, int]]:
    """Map every file under *root* to its (bytes, mtime_ns), for churn checks."""
    out: dict[str, tuple[bytes, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            st = path.stat()
            out[path.relative_to(root).as_posix()] = (path.read_bytes(), st.st_mtime_ns)
    return out


class TestConvergenceSharedTargets:
    """Issue #83: with codex + antigravity-cli both installed, a second sync of
    the shared ``AGENTS.md`` / ``.agents/skills`` targets must be a no-op —
    every writer ``skipped``, nothing rewritten, content deterministic."""

    def _setup(self, tmp_path: Path) -> SyncData:
        # Rules source carries a Codex-specific marker: rendered per-target it
        # diverges (agy would append a foreign-marker manual-fix block Codex
        # doesn't), which is exactly the collision that used to churn AGENTS.md.
        (tmp_path / ".crossby").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".crossby" / "rules.md").write_text(
            "# Rules\n\nConfigure sandbox_mode in .codex/config.toml.\n",
            encoding="utf-8",
        )
        # Skills source with a Claude ``allowed-tools`` field: translate renders
        # target-specific SKILL.md content (a manual-fix block for non-Claude
        # targets), so codex and agy would otherwise produce different bytes.
        skill = tmp_path / ".crossby" / "skills" / "reviewer"
        skill.mkdir(parents=True, exist_ok=True)
        (skill / "SKILL.md").write_text(
            "---\nname: reviewer\ndescription: Review code.\n"
            "allowed-tools:\n  - Read\n---\nBody.\n",
            encoding="utf-8",
        )
        (skill / "scripts").mkdir()
        (skill / "scripts" / "run.sh").write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        return SyncData(
            rules_source=".crossby/rules.md",
            rules_strategy="copy",
            skills_source=".crossby/skills",
            skills_strategy="translate",
        )

    def test_three_consecutive_syncs_converge_and_dont_churn(self, tmp_path: Path) -> None:
        data = self._setup(tmp_path)
        installed = [AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI]

        run1 = run_sync(data, tmp_path, installed_tools=installed)

        # Codex (registered before agy) is the winner on both shared targets;
        # agy's rows are covered-by, not a second write.
        def _shared_rows(results: list[SyncResult], concern: SyncConcern) -> list[SyncResult]:
            return [
                r
                for r in results
                if r.concern == concern and r.tool_id in {AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI}
            ]

        for concern in (SyncConcern.RULES, SyncConcern.SKILLS):
            rows = _shared_rows(run1, concern)
            winners = [r for r in rows if r.action != "skipped"]
            covered = [r for r in rows if r.action == "skipped"]
            assert [r.tool_id for r in winners] == [AIToolID.CODEX], concern
            assert [r.tool_id for r in covered] == [AIToolID.ANTIGRAVITY_CLI], concern
            assert all(r.message == "covered by codex" for r in covered), concern

        agents_md = tmp_path / "AGENTS.md"
        skills_dir = tmp_path / ".agents" / "skills"
        assert agents_md.is_file()
        assert (skills_dir / "reviewer" / "SKILL.md").is_file()

        content_after_1 = agents_md.read_bytes()
        skills_after_1 = _snapshot_tree(skills_dir)
        agents_mtime_1 = agents_md.stat().st_mtime_ns

        # Run 2: every writer must report skipped (no sync target rewritten).
        run2 = run_sync(data, tmp_path, installed_tools=installed)
        assert all(r.action == "skipped" for r in run2), [
            (r.tool_id, r.concern, r.action, r.message) for r in run2 if r.action != "skipped"
        ]

        # Run 3: still skipped, and nothing changed on disk since run 1.
        run3 = run_sync(data, tmp_path, installed_tools=installed)
        assert all(r.action == "skipped" for r in run3)

        # Byte-identical + no rewrite (mtime stable) across re-syncs.
        assert agents_md.read_bytes() == content_after_1
        assert agents_md.stat().st_mtime_ns == agents_mtime_1
        assert _snapshot_tree(skills_dir) == skills_after_1

    def test_executable_bit_preserved_on_skill_scripts(self, tmp_path: Path) -> None:
        # Regression guard for the mode-preserving mirror: a translate sync must
        # keep scripts/ executable rather than dropping the bit via write_bytes.
        data = self._setup(tmp_path)
        os.chmod(tmp_path / ".crossby" / "skills" / "reviewer" / "scripts" / "run.sh", 0o755)

        run_sync(data, tmp_path, installed_tools=[AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI])

        target_script = tmp_path / ".agents" / "skills" / "reviewer" / "scripts" / "run.sh"
        assert target_script.is_file()
        assert target_script.stat().st_mode & 0o111, "executable bit was dropped"

    def test_agy_syncs_shared_targets_alone_when_codex_absent(self, tmp_path: Path) -> None:
        # --to antigravity-cli equivalent: Codex not installed, so agy is the sole
        # group member and still writes AGENTS.md / .agents/skills.
        data = self._setup(tmp_path)

        results = run_sync(data, tmp_path, installed_tools=[AIToolID.ANTIGRAVITY_CLI])

        rules_rows = [r for r in results if r.concern == SyncConcern.RULES and r.tool_id]
        skills_rows = [r for r in results if r.concern == SyncConcern.SKILLS and r.tool_id]
        assert [r.tool_id for r in rules_rows] == [AIToolID.ANTIGRAVITY_CLI]
        assert [r.tool_id for r in skills_rows] == [AIToolID.ANTIGRAVITY_CLI]
        assert rules_rows[0].action != "skipped"
        assert skills_rows[0].action != "skipped"
        assert (tmp_path / "AGENTS.md").is_file()
        assert (tmp_path / ".agents" / "skills" / "reviewer" / "SKILL.md").is_file()


class TestMergeWriterCoexistence:
    """Merge writers that share a file must NOT be grouped — they co-write by
    key and preserve each other's content, in either execution order."""

    def test_claude_hooks_and_permissions_coexist_both_orders(self, tmp_path: Path) -> None:
        from crossby.sync.base import SyncRegistry
        from crossby.sync.hooks import ClaudeHooksWriter
        from crossby.sync.permissions import ClaudePermissionWriter

        data = SyncData(
            allowed_commands=["Bash(ls:*)"],
            hooks=[HookEntry(event="pre_tool_use", command="guard", tools=["Edit"])],
        )

        orders = {
            "perms-first": [ClaudePermissionWriter(), ClaudeHooksWriter()],
            "hooks-first": [ClaudeHooksWriter(), ClaudePermissionWriter()],
        }
        for label, order in orders.items():
            root = tmp_path / f"proj-{label}"
            root.mkdir()
            reg = SyncRegistry()
            for w in order:
                reg.register(w)
            run_sync(data, root, tool_id=AIToolID.CLAUDE, registry=reg)

            settings = json.loads((root / ".claude" / "settings.json").read_text(encoding="utf-8"))
            # Both concerns landed in the one shared file — neither writer wiped
            # the other's section (they merge by key, they are not grouped).
            assert "PreToolUse" in settings.get("hooks", {})
            perms = settings.get("permissions", {}).get("allow", [])
            assert perms, "permission writer's allow-list was lost"
            assert any("ls" in p for p in perms)

    def test_codex_hooks_and_mcp_coexist_both_orders(self, tmp_path: Path) -> None:
        from crossby.sync.base import SyncRegistry
        from crossby.sync.hooks import CodexHooksWriter
        from crossby.sync.mcp import CodexMCPWriter

        data = SyncData(
            mcp_servers={"ctx": MCPServerConfig(command="npx", args=["-y", "ctx"])},
            hooks=[HookEntry(event="pre_tool_use", command="guard", tools=["Edit"])],
        )

        for hooks_first in (True, False):
            root = tmp_path / f"proj-{'hooks' if hooks_first else 'mcp'}-first"
            root.mkdir()
            reg = SyncRegistry()
            order = (
                [CodexHooksWriter(), CodexMCPWriter()]
                if hooks_first
                else [CodexMCPWriter(), CodexHooksWriter()]
            )
            for w in order:
                reg.register(w)
            run_sync(data, root, tool_id=AIToolID.CODEX, registry=reg)

            config = tomllib.loads((root / ".codex" / "config.toml").read_text(encoding="utf-8"))
            # MCP server table and the hooks feature flag both survive.
            assert "ctx" in config.get("mcp_servers", {})
            assert config.get("features", {}).get("hooks") is True
            assert (root / ".codex" / "hooks.json").is_file()
