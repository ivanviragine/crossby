"""Tests for the pure scene resolver (``services/scene_resolution``)."""

from __future__ import annotations

import json
from pathlib import Path

from crossby.models.ai import AIToolID
from crossby.models.config import SceneConfig, SceneSelector
from crossby.services.scene_resolution import ResolvedScene, resolve_scene
from crossby.sync.readers import ProjectScan, scan_project


def _make_skill(root: Path, rel_dir: str, name: str) -> None:
    skill_dir = root / rel_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\n", encoding="utf-8")


def _make_agent(root: Path, rel_dir: str, filename: str) -> None:
    agent_dir = root / rel_dir
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / filename).write_text("agent body", encoding="utf-8")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _populate(root: Path) -> None:
    """Build a project inventory across every scene concern."""
    # Skills in Claude's dir and in the shared codex/antigravity dir.
    for name in ("review-skill", "knowledge", "deploy-prod"):
        _make_skill(root, ".claude/skills", name)
    _make_skill(root, ".agents/skills", "shared-skill")

    # Agents (Claude .md files).
    _make_agent(root, ".claude/agents", "code-reviewer.md")
    _make_agent(root, ".claude/agents", "planner.md")

    # MCP servers (project-scope .mcp.json).
    _write_json(
        root / ".mcp.json",
        {"mcpServers": {"github": {"command": "gh-mcp"}, "linear": {"command": "lin-mcp"}}},
    )

    # Permissions + hooks in Claude settings.
    _write_json(
        root / ".claude" / "settings.json",
        {
            "permissions": {"allow": ["Bash(git diff:*)", "Bash(gh pr create)", "Bash(npm test)"]},
            "hooks": {
                "PreToolUse": [{"matcher": "", "hooks": [{"command": "guard.sh"}]}],
                "Stop": [{"matcher": "", "hooks": [{"command": "notify.sh"}]}],
            },
        },
    )


def _scan(root: Path, installed: list[AIToolID] | None = None) -> ProjectScan:
    tools = installed or [AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI]
    return scan_project(root, tools)


class TestSelectorSemantics:
    def test_omitted_concern_selects_everything(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        resolved = resolve_scene(SceneConfig(), _scan(tmp_path), tmp_path)
        # No skills selector → every detected skill is present.
        assert resolved.names("skills") == (
            "deploy-prod",
            "knowledge",
            "review-skill",
            "shared-skill",
        )

    def test_empty_include_selects_nothing(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        scene = SceneConfig(skills=SceneSelector(include=[]))
        resolved = resolve_scene(scene, _scan(tmp_path), tmp_path)
        assert resolved.names("skills") == ()
        assert resolved.groups_for("skills") == ()

    def test_include_globs_union(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        scene = SceneConfig(skills=SceneSelector(include=["review-*", "knowledge"]))
        resolved = resolve_scene(scene, _scan(tmp_path), tmp_path)
        assert resolved.names("skills") == ("knowledge", "review-skill")

    def test_exclude_wins_over_include(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        scene = SceneConfig(skills=SceneSelector(include=["*"], exclude=["deploy-*"]))
        resolved = resolve_scene(scene, _scan(tmp_path), tmp_path)
        assert "deploy-prod" not in resolved.names("skills")
        assert "review-skill" in resolved.names("skills")

    def test_absent_include_with_exclude_means_all_but_excluded(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        scene = SceneConfig(skills=SceneSelector(exclude=["deploy-*"]))
        resolved = resolve_scene(scene, _scan(tmp_path), tmp_path)
        assert "deploy-prod" not in resolved.names("skills")
        assert "knowledge" in resolved.names("skills")


class TestUnmatchedSelectorWarnings:
    def test_unmatched_include_warns_and_does_not_raise(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        scene = SceneConfig(skills=SceneSelector(include=["nope-*"]))
        resolved = resolve_scene(scene, _scan(tmp_path), tmp_path)
        assert resolved.names("skills") == ()
        assert any("nope-*" in w and "skills" in w for w in resolved.warnings)

    def test_unmatched_exclude_warns(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        scene = SceneConfig(skills=SceneSelector(exclude=["ghost-*"]))
        resolved = resolve_scene(scene, _scan(tmp_path), tmp_path)
        assert any("ghost-*" in w for w in resolved.warnings)

    def test_matched_selector_produces_no_warning(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        scene = SceneConfig(skills=SceneSelector(include=["review-*"]))
        resolved = resolve_scene(scene, _scan(tmp_path), tmp_path)
        assert resolved.warnings == ()


class TestSharedTargetPath:
    def test_shared_skills_dir_collapses_with_both_tools(self, tmp_path: Path) -> None:
        """``.agents/skills`` is reported once, attributed to codex + antigravity-cli."""
        _populate(tmp_path)
        scene = SceneConfig(skills=SceneSelector(include=["shared-*"]))
        resolved = resolve_scene(scene, _scan(tmp_path), tmp_path)
        skill_groups = resolved.groups_for("skills")
        assert len(skill_groups) == 1
        group = skill_groups[0]
        assert group.target_path == ".agents/skills"
        assert group.tools == (AIToolID.ANTIGRAVITY_CLI, AIToolID.CODEX)
        assert group.names == ("shared-skill",)

    def test_distinct_dirs_stay_separate(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        # include everything → both .claude/skills and .agents/skills appear.
        resolved = resolve_scene(SceneConfig(), _scan(tmp_path), tmp_path)
        paths = {g.target_path for g in resolved.groups_for("skills")}
        assert paths == {".claude/skills", ".agents/skills"}


class TestToolNarrowing:
    def test_tool_none_spans_all_tools(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        resolved = resolve_scene(SceneConfig(), _scan(tmp_path), tmp_path, tool_id=None)
        tools = {t for g in resolved.groups_for("skills") for t in g.tools}
        assert AIToolID.CLAUDE in tools
        assert AIToolID.CODEX in tools

    def test_tool_id_narrows_to_single_tool(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        resolved = resolve_scene(SceneConfig(), _scan(tmp_path), tmp_path, tool_id=AIToolID.CODEX)
        skill_groups = resolved.groups_for("skills")
        # Only the shared .agents/skills dir involves codex.
        assert all(g.tools == (AIToolID.CODEX,) for g in skill_groups)
        assert {g.target_path for g in skill_groups} == {".agents/skills"}
        assert resolved.tool_id == AIToolID.CODEX

    def test_tool_id_excludes_unrelated_tool(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        # cursor has no skills dir here → no skill groups.
        resolved = resolve_scene(SceneConfig(), _scan(tmp_path), tmp_path, tool_id=AIToolID.CURSOR)
        assert resolved.groups_for("skills") == ()


class TestGlobalConcerns:
    def test_mcp_resolves_from_global_view(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        scene = SceneConfig(mcp=SceneSelector(include=["github"]))
        resolved = resolve_scene(scene, _scan(tmp_path), tmp_path)
        assert resolved.names("mcp") == ("github",)
        groups = resolved.groups_for("mcp")
        assert len(groups) == 1
        assert groups[0].target_path is None

    def test_hooks_use_event_command_names(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        scene = SceneConfig(hooks=SceneSelector(include=["pre_tool_use:*"]))
        resolved = resolve_scene(scene, _scan(tmp_path), tmp_path)
        assert resolved.names("hooks") == ("pre_tool_use:guard.sh",)

    def test_permissions_are_opaque_globs(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        scene = SceneConfig(permissions=SceneSelector(include=["git diff:*", "gh pr *"]))
        resolved = resolve_scene(scene, _scan(tmp_path), tmp_path)
        # 'gh pr *' matches the allowlist text 'gh pr create'; 'npm test' stays out.
        assert set(resolved.names("permissions")) == {"git diff:*", "gh pr create"}

    def test_global_group_names_are_sorted(self, tmp_path: Path) -> None:
        """Every concern's group.names is sorted — permissions included."""
        _populate(tmp_path)
        scene = SceneConfig(permissions=SceneSelector(include=["*"]))
        resolved = resolve_scene(scene, _scan(tmp_path), tmp_path)
        group_names = resolved.groups_for("permissions")[0].names
        assert list(group_names) == sorted(group_names)


class TestAgents:
    def test_agent_names_enumerated_by_stem(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        scene = SceneConfig(agents=SceneSelector(include=["code-*"]))
        resolved = resolve_scene(scene, _scan(tmp_path), tmp_path)
        assert resolved.names("agents") == ("code-reviewer",)


class TestNoFilesystemWrites:
    def test_resolve_scene_writes_nothing(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        scan = scan_project(tmp_path, [AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI])

        def snapshot() -> dict[str, bytes]:
            return {
                str(p.relative_to(tmp_path)): p.read_bytes()
                for p in sorted(tmp_path.rglob("*"))
                if p.is_file()
            }

        before = snapshot()
        scene = SceneConfig(
            skills=SceneSelector(include=["*"]),
            agents=SceneSelector(include=["*"]),
            mcp=SceneSelector(include=["*"]),
            hooks=SceneSelector(include=["*"]),
            permissions=SceneSelector(include=["*"]),
        )
        resolve_scene(scene, scan, tmp_path)
        resolve_scene(scene, scan, tmp_path, tool_id=AIToolID.CODEX)
        after = snapshot()

        assert before == after


class TestResolvedSceneShape:
    def test_returns_resolved_scene_instance(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        resolved = resolve_scene(SceneConfig(), _scan(tmp_path), tmp_path)
        assert isinstance(resolved, ResolvedScene)
        assert resolved.tool_id is None
