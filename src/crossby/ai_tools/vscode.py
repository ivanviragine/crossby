"""VS Code adapter — opens the working directory in VS Code."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import structlog

from crossby.ai_tools.base import AbstractAITool
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    EffortLevel,
    PlanArtifactLocation,
    PlanModeActivation,
    PlanModeCapability,
    TokenUsage,
)
from crossby.utils.process import run_with_transcript

if TYPE_CHECKING:
    from crossby.scenes.launch import SceneLaunchContext

logger = structlog.get_logger()


class VSCodeAdapter(AbstractAITool):
    """Adapter for Visual Studio Code."""

    TOOL_ID: ClassVar[AIToolID] = AIToolID.VSCODE

    def capabilities(self) -> AIToolCapabilities:
        return AIToolCapabilities(
            tool_id=AIToolID.VSCODE,
            display_name="VS Code",
            binary="code",
            tool_type=AIToolType.GUI,
            supports_model_flag=False,
            supports_headless=False,
            supports_initial_message=False,
            blocks_until_exit=False,
            plan_mode=PlanModeCapability(
                activation=PlanModeActivation.UNSUPPORTED,
                activation_detail=(
                    "The VS Code launcher exposes neither a native plan-mode selector nor "
                    "initial-message delivery to an agent session."
                ),
                version_requirement="A VS Code CLI/API with a programmatic native plan selector.",
                verified_version="1.136.1",
                initial_prompt_after_activation=False,
                artifact_location=PlanArtifactLocation.UNAVAILABLE,
                artifact_location_detail="No programmatic native plan session is available.",
                remediation=(
                    "Open VS Code normally and select plan mode manually, or use a supported "
                    "terminal harness for programmatic planning."
                ),
            ),
        )

    def launch(
        self,
        working_dir: Path,
        model: str | None = None,
        prompt: str | None = None,
        detach: bool = False,
        transcript_path: Path | None = None,
        trusted_dirs: list[str] | None = None,
        effort: EffortLevel | None = None,
        allowed_commands: list[str] | None = None,
        yolo: bool = False,
        plan_mode: bool = False,
        accept_edits: bool = False,
        auto: bool = False,
        scene: SceneLaunchContext | None = None,
        network_access: bool = False,
        plan_output_dir: Path | None = None,
        *,
        sandbox: bool = True,
    ) -> int:
        # VS Code is a GUI launcher: `network_access` and `sandbox` are inert
        # (the CLI already warns + ignores unsupported options for GUI tools),
        # and no session-scoped scene lever exists, so `scene` is inert too.
        self.validate_plan_mode_request(
            plan_mode=plan_mode,
            yolo=yolo,
            auto=auto,
            accept_edits=accept_edits,
            initial_message=prompt,
            plan_output_dir=plan_output_dir,
            working_dir=working_dir,
        )
        cmd = ["code", str(working_dir)]
        logger.info("ai_tool.launch", tool="vscode", cwd=str(working_dir))
        return run_with_transcript(cmd, transcript_path, cwd=working_dir)

    def parse_transcript(self, transcript_path: Path) -> TokenUsage:
        return TokenUsage()
