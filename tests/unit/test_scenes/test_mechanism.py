"""Mechanism matrix, shared-path precedence, and plan_units."""

from __future__ import annotations

from crossby.models.ai import AIToolID
from crossby.scenes.mechanism import (
    ActivationUnit,
    SceneMechanism,
    base_mechanism,
    plan_units,
)
from crossby.services.scene_resolution import ResolvedGroup, ResolvedScene


class TestBaseMatrix:
    def test_claude_skills_is_declare(self) -> None:
        assert base_mechanism(AIToolID.CLAUDE, "skills") == SceneMechanism.DECLARE

    def test_claude_agents_is_declare(self) -> None:
        assert base_mechanism(AIToolID.CLAUDE, "agents") == SceneMechanism.DECLARE

    def test_codex_skills_is_project(self) -> None:
        # Array-of-tables lever is unsupported/non-authoritative → PROJECT.
        assert base_mechanism(AIToolID.CODEX, "skills") == SceneMechanism.PROJECT

    def test_mcp_declare_tools(self) -> None:
        for tool in (AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI):
            assert base_mechanism(tool, "mcp") == SceneMechanism.DECLARE

    def test_cursor_copilot_mcp_unsupported(self) -> None:
        assert base_mechanism(AIToolID.CURSOR, "mcp") == SceneMechanism.UNSUPPORTED
        assert base_mechanism(AIToolID.COPILOT, "mcp") == SceneMechanism.UNSUPPORTED

    def test_permissions_only_claude_cursor(self) -> None:
        assert base_mechanism(AIToolID.CLAUDE, "permissions") == SceneMechanism.PROJECT
        assert base_mechanism(AIToolID.CURSOR, "permissions") == SceneMechanism.PROJECT
        assert base_mechanism(AIToolID.CODEX, "permissions") == SceneMechanism.UNSUPPORTED

    def test_unknown_cell_defaults_unsupported(self) -> None:
        assert base_mechanism(AIToolID.VSCODE, "skills") == SceneMechanism.UNSUPPORTED


class TestSharedPathPrecedence:
    def _resolved(self, *groups: ResolvedGroup) -> ResolvedScene:
        return ResolvedScene(tool_id=None, groups=tuple(groups), warnings=())

    def test_shared_dir_with_a_project_tool_wins_project(self) -> None:
        # codex(PROJECT) + antigravity(PROJECT) share .agents/skills → one PROJECT unit.
        group = ResolvedGroup(
            concern="skills",
            target_path=".agents/skills",
            tools=(AIToolID.ANTIGRAVITY_CLI, AIToolID.CODEX),
            names=("shared-skill",),
        )
        units = plan_units(self._resolved(group))
        assert units == [
            ActivationUnit(
                concern="skills",
                target_path=".agents/skills",
                tools=(AIToolID.ANTIGRAVITY_CLI, AIToolID.CODEX),
                names=("shared-skill",),
                mechanism=SceneMechanism.PROJECT,
            )
        ]

    def test_claude_only_skills_group_stays_declare(self) -> None:
        group = ResolvedGroup(
            concern="skills",
            target_path=".claude/skills",
            tools=(AIToolID.CLAUDE,),
            names=("review-skill",),
        )
        units = plan_units(self._resolved(group))
        assert units[0].mechanism == SceneMechanism.DECLARE

    def test_global_concern_splits_per_tool(self) -> None:
        group = ResolvedGroup(
            concern="mcp",
            target_path=None,
            tools=(AIToolID.CLAUDE, AIToolID.CURSOR),
            names=("github",),
        )
        units = plan_units(self._resolved(group))
        by_tool = {u.tools[0]: u.mechanism for u in units}
        assert by_tool[AIToolID.CLAUDE] == SceneMechanism.DECLARE
        assert by_tool[AIToolID.CURSOR] == SceneMechanism.UNSUPPORTED
        assert all(len(u.tools) == 1 for u in units)
