"""VS Code adapter — opens the working directory in VS Code."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import structlog

from crossby.ai_tools.base import AbstractAITool
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    EffortLevel,
    TokenUsage,
)
from crossby.utils.process import run_with_transcript

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
    ) -> int:
        cmd = ["code", str(working_dir)]
        logger.info("ai_tool.launch", tool="vscode", cwd=str(working_dir))
        return run_with_transcript(cmd, transcript_path, cwd=working_dir)

    def parse_transcript(self, transcript_path: Path) -> TokenUsage:
        return TokenUsage()
