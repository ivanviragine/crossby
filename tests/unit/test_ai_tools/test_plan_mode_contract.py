"""Truthful native plan-mode capability and launch contract."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.plan_mode import (
    PlanArtifactLocationError,
    PlanModeConflictError,
    PlanModeUnsupportedError,
)
from crossby.models.ai import (
    AIToolID,
    EffortLevel,
    PlanArtifactLocation,
    PlanModeActivation,
)

SUPPORTED = {
    AIToolID.CLAUDE: ("--permission-mode", "plan"),
    AIToolID.CURSOR: ("--mode", "plan"),
    AIToolID.COPILOT: ("--plan",),
    AIToolID.OPENCODE: ("--agent", "plan"),
    AIToolID.ANTIGRAVITY_CLI: ("--mode", "plan"),
}
UNSUPPORTED = {AIToolID.CODEX, AIToolID.VSCODE, AIToolID.ANTIGRAVITY}


class TestPlanModeCapabilityMatrix:
    def test_every_tool_has_an_explicit_disposition(self) -> None:
        assert set(AbstractAITool.available_tools()) == set(AIToolID)
        for tool_id in AIToolID:
            capability = AbstractAITool.get(tool_id).capabilities().plan_mode
            if tool_id in SUPPORTED:
                assert capability.activation is PlanModeActivation.CLI_ARGUMENT
                assert capability.supported is True
                assert capability.initial_prompt_after_activation is True
                assert capability.verified_version
            else:
                assert tool_id in UNSUPPORTED
                assert capability.activation is PlanModeActivation.UNSUPPORTED
                assert capability.supported is False
                assert capability.remediation

    def test_compatibility_boolean_is_derived_from_typed_capability(self) -> None:
        for tool_id in AIToolID:
            caps = AbstractAITool.get(tool_id).capabilities()
            assert caps.supports_plan_mode is caps.plan_mode.supported

    def test_supported_cli_selectors_match_capability_contract(self) -> None:
        for tool_id, expected in SUPPORTED.items():
            assert tuple(AbstractAITool.get(tool_id).plan_mode_args()) == expected

    def test_antigravity_private_artifact_contract_is_exportable_metadata(self) -> None:
        capability = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).capabilities().plan_mode
        assert capability.artifact_location is PlanArtifactLocation.PRIVATE
        assert capability.artifact_path_template == (
            "~/.gemini/antigravity-cli/brain/<conversation-id>/"
        )
        assert capability.export_command is None
        assert capability.import_command is None
        assert "filesystem output directory" in capability.remediation.lower()

    def test_every_supported_tool_describes_its_artifact_scope(self) -> None:
        expected = {
            AIToolID.CLAUDE: PlanArtifactLocation.REQUESTED_PATH,
            AIToolID.CURSOR: PlanArtifactLocation.SESSION,
            AIToolID.COPILOT: PlanArtifactLocation.PRIVATE,
            AIToolID.OPENCODE: PlanArtifactLocation.WORKSPACE_MANAGED,
            AIToolID.ANTIGRAVITY_CLI: PlanArtifactLocation.PRIVATE,
        }
        for tool_id, location in expected.items():
            capability = AbstractAITool.get(tool_id).capabilities().plan_mode
            assert capability.artifact_location is location
            assert capability.artifact_location_detail


class TestCompletePlanCommands:
    def test_claude(self) -> None:
        assert AbstractAITool.get("claude").build_launch_command(
            model="claude-sonnet-4.6",
            initial_message="Plan this",
            plan_mode=True,
            trusted_dirs=["/tmp/reference"],
            effort=EffortLevel.HIGH,
        ) == [
            "claude",
            "Plan this",
            "--model",
            "claude-sonnet-4-6",
            "--permission-mode",
            "plan",
            "--add-dir",
            "/tmp/reference",
            "--effort",
            "high",
        ]

    def test_cursor(self) -> None:
        assert AbstractAITool.get("cursor").build_launch_command(
            model="sonnet-4.6",
            initial_message="Plan this",
            plan_mode=True,
            effort=EffortLevel.MEDIUM,
        ) == [
            "agent",
            "Plan this",
            "--model",
            "sonnet-4.6",
            "--mode",
            "plan",
            "--sandbox",
            "enabled",
        ]

    def test_copilot(self) -> None:
        assert AbstractAITool.get("copilot").build_launch_command(
            model="gpt-5.4",
            initial_message="Plan this",
            plan_mode=True,
            trusted_dirs=["/tmp/reference"],
            allowed_commands=["git:status"],
        ) == [
            "copilot",
            "-i",
            "Plan this",
            "--model",
            "gpt-5.4",
            "--plan",
            "--add-dir",
            "/tmp/reference",
            "--allow-tool",
            "shell(git:status)",
        ]

    def test_opencode(self) -> None:
        assert AbstractAITool.get("opencode").build_launch_command(
            model="openai/gpt-5.4",
            initial_message="Plan this",
            plan_mode=True,
            effort=EffortLevel.HIGH,
            working_dir=Path("/workspace"),
        ) == [
            "opencode",
            "--prompt",
            "Plan this",
            "--model",
            "openai/gpt-5.4",
            "--agent",
            "plan",
            "--variant",
            "high",
        ]

    def test_antigravity_cli(self) -> None:
        assert AbstractAITool.get("antigravity-cli").build_launch_command(
            model="gemini-3.8-flash",
            initial_message="Plan this",
            plan_mode=True,
            trusted_dirs=["/tmp/reference"],
            effort=EffortLevel.HIGH,
        ) == [
            "agy",
            "--prompt-interactive",
            "Plan this",
            "--model",
            "gemini-3.8-flash-high",
            "--mode",
            "plan",
            "--add-dir",
            "/tmp/reference",
        ]


class TestPlanModeFailures:
    @pytest.mark.parametrize("tool_id", sorted(UNSUPPORTED, key=str))
    def test_direct_builder_rejects_unsupported_tool(self, tool_id: AIToolID) -> None:
        with pytest.raises(PlanModeUnsupportedError) as raised:
            AbstractAITool.get(tool_id).build_launch_command(
                initial_message="/plan is only text",
                plan_mode=True,
            )
        assert raised.value.tool_id is tool_id
        assert AbstractAITool.get(tool_id).capabilities().display_name in str(raised.value)
        assert "Remediation:" in str(raised.value)

    @pytest.mark.parametrize("tool_id", sorted(UNSUPPORTED, key=str))
    def test_direct_launch_rejects_before_process_creation(
        self, tool_id: AIToolID, tmp_path: Path
    ) -> None:
        adapter = AbstractAITool.get(tool_id)
        with (
            patch("crossby.utils.process.run_with_transcript") as base_run,
            patch("crossby.ai_tools.vscode.run_with_transcript") as vscode_run,
            patch("crossby.ai_tools.antigravity.run_with_transcript") as antigravity_run,
            pytest.raises(PlanModeUnsupportedError),
        ):
            adapter.launch(tmp_path, prompt="Plan this", plan_mode=True)
        base_run.assert_not_called()
        vscode_run.assert_not_called()
        antigravity_run.assert_not_called()

    @pytest.mark.parametrize("flag", ["yolo", "auto", "accept_edits"])
    def test_every_superseding_autonomy_flag_is_a_typed_conflict(self, flag: str) -> None:
        with pytest.raises(PlanModeConflictError, match=flag.replace("_", "-")):
            AbstractAITool.get("claude").build_launch_command(plan_mode=True, **{flag: True})

    def test_prompt_text_does_not_activate_codex_plan_mode(self) -> None:
        adapter = AbstractAITool.get("codex")
        assert adapter.build_launch_command(initial_message="/plan inspect this") == [
            "codex",
            "/plan inspect this",
        ]
        with pytest.raises(PlanModeUnsupportedError):
            adapter.build_launch_command(initial_message="/plan inspect this", plan_mode=True)


class TestPlanArtifactRequirements:
    def test_antigravity_rejects_workspace_backed_output(self, tmp_path: Path) -> None:
        with pytest.raises(PlanArtifactLocationError, match="brain directory"):
            AbstractAITool.get("antigravity-cli").build_launch_command(
                plan_mode=True,
                plan_output_dir=tmp_path / "plans",
                working_dir=tmp_path,
            )

    def test_claude_routes_plan_files_to_requested_workspace_directory(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / ".wade" / "plans"
        cmd = AbstractAITool.get("claude").build_launch_command(
            plan_mode=True,
            plan_output_dir=output_dir,
            working_dir=tmp_path,
        )
        assert cmd == [
            "claude",
            "--permission-mode",
            "plan",
            "--settings",
            '{"plansDirectory":"./.wade/plans"}',
        ]

    @pytest.mark.parametrize("tool_id", ["cursor", "copilot", "opencode", "antigravity-cli"])
    def test_non_routable_artifact_scopes_reject_requested_output(
        self, tool_id: str, tmp_path: Path
    ) -> None:
        capability = AbstractAITool.get(tool_id).capabilities().plan_mode
        with pytest.raises(PlanArtifactLocationError, match=capability.artifact_location_detail):
            AbstractAITool.get(tool_id).build_launch_command(
                plan_mode=True,
                plan_output_dir=tmp_path / "plans",
                working_dir=tmp_path,
            )

    def test_claude_rejects_output_outside_project(self, tmp_path: Path) -> None:
        with pytest.raises(PlanArtifactLocationError):
            AbstractAITool.get("claude").build_launch_command(
                plan_mode=True,
                plan_output_dir=tmp_path.parent / "external-plans",
                working_dir=tmp_path,
            )
