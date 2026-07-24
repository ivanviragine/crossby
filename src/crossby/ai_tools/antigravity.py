"""Antigravity IDE adapter — opens a workspace in the Antigravity 2.0 desktop app.

Distinct from :class:`~crossby.ai_tools.antigravity_cli.AntigravityCLIAdapter`
(``agy``), which is the terminal surface of the same product. This adapter is a
GUI launcher (same shape as :class:`~crossby.ai_tools.vscode.VSCodeAdapter`): it
opens the working directory in the IDE and returns immediately.

Config-sync is intentionally launch-only. The IDE reads the same project-level
``.agents/`` layout as the CLI (``AGENTS.md``, ``.agents/skills``,
``.agents/agents``, ``.agents/mcp_config.json``), which crossby already
provisions through the ``ANTIGRAVITY_CLI`` sync writers. Syncing to
``antigravity-cli`` therefore configures the IDE transitively; registering a
parallel set of IDE writers on those identical paths would only duplicate the
CLI's targets. See ``UNSUPPORTED_TOOLS`` in ``config/instructions.py``.
"""

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
    """Adapter for the Antigravity IDE (Google Antigravity 2.0 desktop app)."""

    TOOL_ID: ClassVar[AIToolID] = AIToolID.ANTIGRAVITY

    def capabilities(self) -> AIToolCapabilities:
        return AIToolCapabilities(
            tool_id=AIToolID.ANTIGRAVITY,
            display_name="Antigravity IDE",
            binary="antigravity",
            tool_type=AIToolType.GUI,
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
        # `antigravity <path>` opens the workspace, mirroring the VS Code-family
        # launcher convention (`code <path>` / `cursor <path>`). Pass the working
        # dir explicitly rather than "." so the target is unambiguous.
        cmd = [self.capabilities().binary, str(working_dir)]
        logger.info("ai_tool.launch", tool="antigravity", cwd=str(working_dir))
        return run_with_transcript(cmd, transcript_path, cwd=working_dir)

    def parse_transcript(self, transcript_path: Path) -> TokenUsage:
        return TokenUsage()
