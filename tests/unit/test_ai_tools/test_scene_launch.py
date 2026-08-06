"""Tests for session-scoped scene launch — ``crossby launch --scene``.

Covers each adapter's ``scene_launch_args`` (argv + env), the environment
plumbing through ``launch()`` / ``run_with_transcript``, and the Codex profile
ownership + pruning behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools.claude import ClaudeAdapter
from crossby.ai_tools.codex import CodexAdapter
from crossby.ai_tools.copilot import CopilotAdapter
from crossby.ai_tools.cursor import CursorAdapter
from crossby.ai_tools.opencode import OpenCodeAdapter
from crossby.models.ai import AIToolID
from crossby.models.config import MCPServerConfig
from crossby.scenes import launch as scene_launch
from crossby.scenes.launch import SceneLaunchContext
from crossby.services.scene_resolution import ResolvedGroup, ResolvedScene
from crossby.sync.base import SyncData

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _resolved(**selected: tuple[str, ...]) -> ResolvedScene:
    """A ResolvedScene whose ``names(concern)`` equals ``selected[concern]``."""
    groups = tuple(
        ResolvedGroup(
            concern=concern,
            target_path=None,
            tools=(AIToolID.CLAUDE,),
            names=tuple(sorted(names)),
        )
        for concern, names in selected.items()
    )
    return ResolvedScene(tool_id=None, groups=groups, warnings=())


def _context(
    project_root: Path,
    *,
    name: str = "pr-review",
    all_mcp: tuple[str, ...] = (),
    selected_mcp: tuple[str, ...] = (),
    selected_skills: tuple[str, ...] = (),
    selected_agents: tuple[str, ...] = (),
    allow_tools: tuple[str, ...] = (),
) -> SceneLaunchContext:
    servers = {n: MCPServerConfig(command="run", args=[n]) for n in all_mcp}
    sync_data = SyncData(
        mcp_servers=servers,
        skills_source=".claude/skills",
        agents_source=".claude/agents",
    )
    resolved = _resolved(mcp=selected_mcp, skills=selected_skills, agents=selected_agents)
    return SceneLaunchContext(
        name=name,
        resolved=resolved,
        project_root=project_root,
        sync_data=sync_data,
        allow_tools=allow_tools,
    )


def _make_skills(root: Path, names: tuple[str, ...]) -> None:
    for skill in names:
        d = root / ".claude" / "skills" / skill
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(f"# {skill}\n")


def _make_agents(root: Path, names: tuple[str, ...]) -> None:
    d = root / ".claude" / "agents"
    d.mkdir(parents=True, exist_ok=True)
    for agent in names:
        (d / f"{agent}.md").write_text(f"# {agent}\n")


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class TestClaudeSceneLaunch:
    def test_mcp_narrowing_emits_strict_config(self, tmp_path: Path) -> None:
        ctx = _context(tmp_path, all_mcp=("github", "linear"), selected_mcp=("github",))
        result = ClaudeAdapter().scene_launch_args(ctx)

        assert "--mcp-config" in result.args
        assert "--strict-mcp-config" in result.args
        cfg_path = Path(result.args[result.args.index("--mcp-config") + 1])
        assert cfg_path == tmp_path / ".crossby" / "scene" / "pr-review" / "launch" / "mcp.json"
        body = json.loads(cfg_path.read_text())
        # Only the selected server survives in the strict config.
        assert set(body["mcpServers"]) == {"github"}
        assert dict(result.env) == {}

    def test_no_mcp_narrowing_omits_config(self, tmp_path: Path) -> None:
        ctx = _context(tmp_path, all_mcp=("github",), selected_mcp=("github",))
        result = ClaudeAdapter().scene_launch_args(ctx)
        assert "--mcp-config" not in result.args

    def test_writes_nothing_tracked(self, tmp_path: Path) -> None:
        ctx = _context(tmp_path, all_mcp=("github", "linear"), selected_mcp=("github",))
        ClaudeAdapter().scene_launch_args(ctx)
        # The acceptance guarantee: nothing lands in .claude/ or .mcp.json.
        assert not (tmp_path / ".mcp.json").exists()
        assert not (tmp_path / ".claude" / "settings.json").exists()

    def test_skill_overrides_settings_file_when_new_enough(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_skills(tmp_path, ("review", "deploy"))
        monkeypatch.setattr(
            "crossby.scenes.versioning.detect_tool_version", lambda _tool: (2, 1, 200)
        )
        ctx = _context(tmp_path, selected_skills=("review",))
        result = ClaudeAdapter().scene_launch_args(ctx)

        assert "--settings" in result.args
        settings_path = Path(result.args[result.args.index("--settings") + 1])
        body = json.loads(settings_path.read_text())
        assert body == {"skillOverrides": {"deploy": "off"}}

    def test_skill_overrides_gated_on_old_claude(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_skills(tmp_path, ("review", "deploy"))
        monkeypatch.setattr(
            "crossby.scenes.versioning.detect_tool_version", lambda _tool: (2, 1, 0)
        )
        ctx = _context(tmp_path, selected_skills=("review",))
        with pytest.warns(UserWarning, match="skillOverrides needs claude"):
            result = ClaudeAdapter().scene_launch_args(ctx)
        assert "--settings" not in result.args

    def test_deselected_agents_disallowed(self, tmp_path: Path) -> None:
        _make_agents(tmp_path, ("reviewer", "deployer"))
        ctx = _context(tmp_path, selected_agents=("reviewer",))
        result = ClaudeAdapter().scene_launch_args(ctx)
        assert "--disallowedTools" in result.args
        assert "Agent(deployer)" in result.args
        assert "Agent(reviewer)" not in result.args

    def test_omitted_skills_agents_leave_args_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scene that omits skills/agents disables neither.

        ``resolve_scene`` selects the whole universe for an omitted selector, so
        the adapter's disable set is empty and no ``--settings`` /
        ``--disallowedTools`` are emitted — the guard is the narrowing gate. Uses
        the real resolver (not a hand-built ResolvedScene) to exercise the path.
        """
        from crossby.models.config import SceneConfig
        from crossby.services.scene_resolution import resolve_scene
        from crossby.sync.readers import build_sync_data, scan_project

        _make_skills(tmp_path, ("review", "deploy"))
        _make_agents(tmp_path, ("reviewer", "deployer"))
        monkeypatch.setattr(
            "crossby.scenes.versioning.detect_tool_version", lambda _tool: (9, 9, 9)
        )
        scene = SceneConfig()  # every concern omitted
        scan = scan_project(tmp_path, [AIToolID.CLAUDE])
        resolved = resolve_scene(scene, scan, tmp_path, tool_id=None)
        ctx = SceneLaunchContext(
            name="pr-review",
            resolved=resolved,
            project_root=tmp_path,
            sync_data=build_sync_data(tmp_path),
        )
        result = ClaudeAdapter().scene_launch_args(ctx)
        assert "--settings" not in result.args
        assert "--disallowedTools" not in result.args


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------


class TestCodexSceneLaunch:
    def test_scene_launch_ready_version_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "crossby.scenes.versioning.detect_tool_version", lambda _tool: (0, 133, 9)
        )
        assert CodexAdapter().scene_launch_ready() is False
        monkeypatch.setattr(
            "crossby.scenes.versioning.detect_tool_version", lambda _tool: (0, 134, 0)
        )
        assert CodexAdapter().scene_launch_ready() is True

    def test_profile_written_and_namespaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "codex_home"
        monkeypatch.setenv("CODEX_HOME", str(home))
        ctx = _context(tmp_path, all_mcp=("github", "linear"), selected_mcp=("github",))
        result = CodexAdapter().scene_launch_args(ctx)

        assert result.args[0] == "--profile"
        profile_name = result.args[1]
        assert profile_name.startswith("crossby-")
        assert profile_name.endswith("-pr-review")
        profile_file = home / f"{profile_name}.config.toml"
        text = profile_file.read_text()
        assert text.startswith(scene_launch.CODEX_PROFILE_MARKER)
        assert "[mcp_servers.linear]" in text
        assert "enabled = false" in text

    def test_no_mcp_narrowing_no_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex_home"))
        ctx = _context(tmp_path, all_mcp=("github",), selected_mcp=("github",))
        result = CodexAdapter().scene_launch_args(ctx)
        assert result.args == ()

    def test_pruning_leaves_handwritten_profile_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "codex_home"
        home.mkdir()
        monkeypatch.setenv("CODEX_HOME", str(home))
        slug = scene_launch.project_slug(tmp_path)

        # A hand-written profile that matches the crossby naming pattern exactly
        # but has no ownership header.
        handwritten = home / f"crossby-{slug}-legacy.config.toml"
        handwritten.write_text("model = 'gpt-5'\n")
        # A genuine crossby-owned profile for a scene that no longer exists.
        owned = scene_launch.write_codex_profile(tmp_path, "gone", {"linear"})

        pruned = scene_launch.prune_stale_artifacts(tmp_path, defined_scenes=set())

        assert handwritten.exists(), "must never delete a non-crossby profile"
        assert not owned.exists(), "must prune the crossby-owned stale profile"
        assert str(owned) in pruned

    def test_write_refuses_to_clobber_handwritten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "codex_home"
        home.mkdir()
        monkeypatch.setenv("CODEX_HOME", str(home))
        path = scene_launch.codex_profile_path(tmp_path, "pr-review")
        path.write_text("model = 'gpt-5'\n")  # no header
        with pytest.raises(FileExistsError):
            scene_launch.write_codex_profile(tmp_path, "pr-review", {"linear"})

    def test_profile_collision_falls_back_to_persistent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A namespaced-profile collision degrades to persistent activation.

        Rather than let ``FileExistsError`` abort ``crossby launch --scene``, the
        adapter warns, applies the scene persistently for Codex, and emits no
        ``--profile`` — leaving the hand-written profile untouched.
        """
        import crossby.scenes.engine as engine

        home = tmp_path / "codex_home"
        home.mkdir()
        monkeypatch.setenv("CODEX_HOME", str(home))
        path = scene_launch.codex_profile_path(tmp_path, "pr-review")
        path.write_text("model = 'gpt-5'\n")  # hand-written, no crossby header

        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            engine, "apply_scene", lambda _resolved, _root, **kw: calls.append(kw) or []
        )

        ctx = _context(tmp_path, all_mcp=("github", "linear"), selected_mcp=("github",))
        with pytest.warns(UserWarning, match="Falling back to persistent"):
            result = CodexAdapter().scene_launch_args(ctx)

        assert result.args == ()  # no --profile emitted
        assert path.read_text() == "model = 'gpt-5'\n"  # collision left intact
        assert calls and calls[0]["tools"] == (AIToolID.CODEX,)


# ---------------------------------------------------------------------------
# Copilot
# ---------------------------------------------------------------------------


class TestCopilotSceneLaunch:
    def test_deselected_servers_disabled(self, tmp_path: Path) -> None:
        ctx = _context(tmp_path, all_mcp=("github", "linear", "sentry"), selected_mcp=("github",))
        result = CopilotAdapter().scene_launch_args(ctx)
        # Repeated --disable-mcp-server per deselected server.
        assert result.args.count("--disable-mcp-server") == 2
        assert "linear" in result.args
        assert "sentry" in result.args

    def test_profile_allow_of_excluded_tool_is_dropped(self, tmp_path: Path) -> None:
        # The scene excludes `github`; the profile allows both `github` and an
        # unrelated shell tool. The emitted argv must resolve this unambiguously:
        # github is disabled and NOT re-allowed, git shell survives.
        ctx = _context(
            tmp_path,
            all_mcp=("github", "linear"),
            selected_mcp=("linear",),
            allow_tools=(
                "github",
                "github__create_issue",
                "github(create_issue)",
                "shell(git:*)",
            ),
        )
        result = CopilotAdapter().scene_launch_args(ctx)

        assert list(result.args[:2]) == ["--disable-mcp-server", "github"]
        assert "--allow-tool" in result.args
        # The unrelated tool is still allowed; neither the excluded server nor
        # either per-tool spelling (documented `(...)` or `__` namespacing) is.
        allow_values = [
            result.args[i + 1] for i, a in enumerate(result.args) if a == "--allow-tool"
        ]
        assert allow_values == ["shell(git:*)"]


# ---------------------------------------------------------------------------
# Cursor / OpenCode — config-dir env vars
# ---------------------------------------------------------------------------


class TestCursorSceneLaunch:
    """Cursor has no session-scoped lever.

    ``CURSOR_CONFIG_DIR`` relocates Cursor's whole config base (auth +
    cli-config.json, not just mcp.json), so scoping MCP through it would launch
    Cursor unauthenticated. crossby therefore leaves Cursor to the persistent
    ``scene use`` fallback instead of emitting a launch-time lever.
    """

    def test_no_session_lever(self) -> None:
        adapter = CursorAdapter()
        assert adapter.capabilities().supports_scene_launch is False
        assert adapter.capabilities().scene_config_dir_env is None
        assert adapter.scene_launch_ready() is False
        assert adapter.scene_launch_concerns() == set()

    def test_scene_launch_args_is_noop(self, tmp_path: Path) -> None:
        # Even called directly, Cursor renders nothing and sets no env var.
        ctx = _context(tmp_path, all_mcp=("github", "linear"), selected_mcp=("github",))
        result = CursorAdapter().scene_launch_args(ctx)
        assert result.args == ()
        assert dict(result.env) == {}
        assert not (tmp_path / ".crossby").exists()


class TestOpenCodeSceneLaunch:
    def test_opencode_config_env(self, tmp_path: Path) -> None:
        ctx = _context(tmp_path, all_mcp=("github", "linear"), selected_mcp=("github",))
        result = OpenCodeAdapter().scene_launch_args(ctx)
        env = dict(result.env)
        assert "OPENCODE_CONFIG" in env
        cfg = Path(env["OPENCODE_CONFIG"])
        assert cfg == tmp_path / ".crossby" / "scene" / "pr-review" / "launch" / "opencode.json"
        body = json.loads(cfg.read_text())
        # Every discovered server is stated explicitly: selected enabled, the
        # rest enabled=false (rather than omitted) so deselection is authoritative.
        assert set(body["mcp"]) == {"github", "linear"}
        assert body["mcp"]["github"]["type"] == "local"
        assert body["mcp"]["github"]["enabled"] is True
        assert body["mcp"]["linear"]["enabled"] is False


# ---------------------------------------------------------------------------
# Env plumbing through launch()
# ---------------------------------------------------------------------------


class TestLaunchEnvPlumbing:
    def test_scene_env_merged_over_os_environ(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run(cmd: list[str], transcript_path: object, cwd: object, env: object) -> int:
            captured["env"] = env
            return 0

        monkeypatch.setattr("crossby.utils.process.run_with_transcript", fake_run)
        monkeypatch.setenv("PATH_MARKER", "present")

        ctx = _context(tmp_path, all_mcp=("github", "linear"), selected_mcp=("github",))
        OpenCodeAdapter().launch(working_dir=tmp_path, scene=ctx)

        env = captured["env"]
        assert env is not None
        # Scene addition present…
        assert "OPENCODE_CONFIG" in env
        # …and the inherited environment is preserved (merged over os.environ).
        assert env["PATH_MARKER"] == "present"

    def test_no_scene_inherits_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run(cmd: list[str], transcript_path: object, cwd: object, env: object) -> int:
            captured["env"] = env
            return 0

        monkeypatch.setattr("crossby.utils.process.run_with_transcript", fake_run)
        OpenCodeAdapter().launch(working_dir=tmp_path)
        # No scene → env stays None so the child inherits this process's env.
        assert captured["env"] is None


class TestGitExclude:
    """A session launch must never mutate .gitignore (a tracked file)."""

    def test_launch_uses_git_info_exclude_not_gitignore(self, tmp_path: Path) -> None:
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("existing\n")

        ctx = _context(tmp_path, all_mcp=("github", "linear"), selected_mcp=("github",))
        ClaudeAdapter().scene_launch_args(ctx)

        # .gitignore is untouched…
        assert gitignore.read_text() == "existing\n"
        # …and the ignore rule went to the untracked .git/info/exclude instead.
        exclude = (tmp_path / ".git" / "info" / "exclude").read_text()
        assert ".crossby/scene/" in exclude.splitlines()


class TestLocalTreePruning:
    """Pruning removes only crossby-marked launch trees under .crossby/scene/."""

    def test_unmarked_launch_tree_survives(self, tmp_path: Path) -> None:
        # A crossby-owned launch tree for a scene no longer defined.
        owned = _context(tmp_path, name="gone", all_mcp=("a", "b"), selected_mcp=("a",))
        ClaudeAdapter().scene_launch_args(owned)
        owned_tree = tmp_path / ".crossby" / "scene" / "gone" / "launch"
        assert owned_tree.is_dir()

        # A hand-made directory under the same root, without the managed marker.
        handmade = tmp_path / ".crossby" / "scene" / "mine" / "launch"
        handmade.mkdir(parents=True)
        (handmade / "notes.txt").write_text("keep me\n")

        pruned = scene_launch.prune_stale_artifacts(tmp_path, defined_scenes=set())

        assert not owned_tree.exists(), "the marked, undefined scene tree is pruned"
        assert any("gone" in p for p in pruned)
        assert handmade.exists(), "an unmarked hand-made tree is never deleted"
        assert (handmade / "notes.txt").exists()


class TestSceneNameValidation:
    def test_rejects_unsafe_names(self) -> None:
        for bad in ("", ".", "..", "../etc", "a/b", "a\\b", "foo..bar", "active"):
            with pytest.raises(ValueError, match="unsafe scene name"):
                scene_launch.validate_scene_name(bad)

    def test_accepts_safe_names(self) -> None:
        for good in ("pr-review", "base", "review_2", "v1.2", "Scene-A"):
            scene_launch.validate_scene_name(good)  # no raise


class TestAtomicWriteUniqueTemp:
    def test_writes_and_overwrites_without_leftover_tmp(self, tmp_path: Path) -> None:
        from crossby.config.json_utils import atomic_write_text

        target = tmp_path / "sub" / "f.json"
        atomic_write_text(target, "one")
        assert target.read_text() == "one"
        atomic_write_text(target, "two")
        assert target.read_text() == "two"
        # Unique mkstemp names are cleaned up — no leftover temp files.
        assert not list(tmp_path.glob("**/*.tmp"))
