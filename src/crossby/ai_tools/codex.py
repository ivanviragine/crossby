"""OpenAI Codex CLI adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from crossby.ai_tools.base import AbstractAITool
from crossby.handoff.models import ConversationTranscript, SessionRef
from crossby.handoff.readers import codex as codex_reader
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    EffortLevel,
)

# Codex uses "xhigh" for both our XHIGH and MAX levels
_CODEX_EFFORT_MAP: dict[EffortLevel, str] = {
    EffortLevel.LOW: "low",
    EffortLevel.MEDIUM: "medium",
    EffortLevel.HIGH: "high",
    EffortLevel.XHIGH: "xhigh",
    EffortLevel.MAX: "xhigh",
}


class CodexAdapter(AbstractAITool):
    """Adapter for OpenAI Codex CLI."""

    TOOL_ID: ClassVar[AIToolID] = AIToolID.CODEX

    def capabilities(self) -> AIToolCapabilities:
        return AIToolCapabilities(
            tool_id=AIToolID.CODEX,
            display_name="Codex CLI",
            binary="codex",
            tool_type=AIToolType.TERMINAL,
            supports_model_flag=True,
            headless_flag="exec",
            supports_headless=True,
            supports_effort=True,
            supports_yolo=True,
            supports_resume=True,
            supports_trusted_dirs=True,
            supports_accept_edits=True,
            supports_stop_hook=True,
            supports_session_start_hook=True,
            supports_user_prompt_submit_hook=True,
            sandboxes_writes=True,
            supports_usage_reporting=True,
        )

    def build_resume_command(self, session_id: str) -> list[str] | None:
        """Resume a Codex session: ``codex resume <session_id>``."""
        return ["codex", "resume", session_id]

    def locate_sessions(self, project_path: Path) -> list[SessionRef]:
        return codex_reader.locate_sessions(project_path)

    def read_session(self, ref: SessionRef) -> ConversationTranscript:
        return codex_reader.read_session(ref)

    def initial_message_args(self, prompt: str) -> list[str]:
        """Codex accepts the initial message as a positional argument."""
        return [prompt]

    def plan_dir_args(self, plan_dir: str) -> list[str]:
        """Codex uses --add-dir for plan directory access."""
        return ["--add-dir", plan_dir]

    def trusted_dirs_args(
        self, dirs: list[str], *, autonomy_args: list[str] | None = None
    ) -> list[str]:
        """Codex requires workspace-write sandbox mode for --add-dir to take effect.

        Skip re-emitting ``--sandbox workspace-write`` when the resolved
        autonomy tier already supplied it (accept-edits sets ``-s
        workspace-write``, and auto downgrades to accept-edits on Codex),
        so an accept-edits launch with trusted dirs doesn't pass Codex the
        sandbox option twice.
        """
        sandbox_already_set = "workspace-write" in (autonomy_args or [])
        result: list[str] = [] if sandbox_already_set else ["--sandbox", "workspace-write"]
        for d in dirs:
            result.extend(self.plan_dir_args(d))
        return result

    def is_model_compatible(self, model: str) -> bool:
        """Codex accepts codex-*, gpt-*, and o<digit>* model IDs."""
        lower = model.lower()
        if lower.startswith("codex-") or lower.startswith("gpt-"):
            return True
        # o1, o3, o4-mini etc.
        return bool(re.match(r"^o\d", lower))

    def effort_args(self, effort: EffortLevel) -> list[str]:
        """Codex uses ``-c model_reasoning_effort="<mapped>"``."""
        mapped = _CODEX_EFFORT_MAP.get(effort, effort.value)
        return ["-c", f'model_reasoning_effort="{mapped}"']

    def accept_edits_args(self) -> list[str]:
        """Codex accept-edits: workspace-write sandbox auto-applies edits while
        untrusted shell commands still escalate for approval.

        ``-s workspace-write -a untrusted``. The old ``--approval-mode
        auto-edit`` flag was removed in the Rust CLI (v0.14x) and must not be
        used.
        """
        return ["-s", "workspace-write", "-a", "untrusted"]

    def yolo_args(self) -> list[str]:
        """Codex skips approval prompts with ``-a never`` while keeping its
        sandbox intact.

        ``--yolo`` (an alias for ``--dangerously-bypass-approvals-and-sandbox``)
        is deliberately avoided: it would also disable the OS sandbox
        (Seatbelt/Landlock), making Codex's yolo mode far more permissive than
        the approval-only yolo of every other adapter. Yolo here means "skip
        approval prompts", not "remove the sandbox".
        """
        return ["-a", "never"]
