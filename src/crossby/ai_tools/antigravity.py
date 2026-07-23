"""Antigravity CLI adapter."""

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


class AntigravityAdapter(AbstractAITool):
    """Adapter for Antigravity CLI."""

    TOOL_ID: ClassVar[AIToolID] = AIToolID.ANTIGRAVITY

    def capabilities(self) -> AIToolCapabilities:
        return AIToolCapabilities(
            tool_id=AIToolID.ANTIGRAVITY,
            display_name="Antigravity",
            binary="antigravity",
            tool_type=AIToolType.TERMINAL,
            supports_model_flag=False,
            headless_flag=None,
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
        plan_mode: bool = False,
        accept_edits: bool = False,
        auto: bool = False,
    ) -> int:
        cmd = [self.capabilities().binary, "."]
        logger.info("ai_tool.launch", tool="antigravity", cwd=str(working_dir))
        return run_with_transcript(cmd, transcript_path, cwd=working_dir)

    def parse_transcript(self, transcript_path: Path) -> TokenUsage:
        return TokenUsage()
