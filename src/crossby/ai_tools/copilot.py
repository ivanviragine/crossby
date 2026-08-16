"""GitHub Copilot CLI adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from crossby.ai_tools.base import AbstractAITool
from crossby.handoff.models import ConversationTranscript, SessionRef
from crossby.handoff.readers import copilot as copilot_reader
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    HookOutputDialect,
    HookStopDialect,
    TokenUsage,
)

if TYPE_CHECKING:
    from crossby.scenes.launch import SceneLaunchArgs, SceneLaunchContext


class CopilotAdapter(AbstractAITool):
    """Adapter for GitHub Copilot CLI."""

    TOOL_ID: ClassVar[AIToolID] = AIToolID.COPILOT

    def capabilities(self) -> AIToolCapabilities:
        return AIToolCapabilities(
            tool_id=AIToolID.COPILOT,
            display_name="GitHub Copilot",
            binary="copilot",
            tool_type=AIToolType.TERMINAL,
            # `copilot update [channel]` — downloads the latest version.
            update_command=("copilot", "update"),
            supports_model_flag=True,
            headless_flag="--prompt",
            supports_headless=True,
            supports_yolo=True,
            supports_resume=True,
            supports_trusted_dirs=True,
            supports_plan_mode=True,
            supports_accept_edits=True,
            supports_session_start_hook=True,
            supports_stop_hook=True,
            # Copilot's preToolUse has a documented structured stdout schema —
            # flat and top-level, never nested under `hookSpecificOutput` (which
            # appears nowhere in GitHub's hooks docs; it is a Claude/VS Code
            # construct). Modelling this as EXIT_CODE, as crossby did through
            # 0.12.x, threw away the reason string and the `ask` decision.
            hook_output_dialect=HookOutputDialect.PERMISSION_DECISION,
            # ...but its stop channel is a different vocabulary again
            # (`agentStop` reads {"decision": "block", "reason": …}), which is
            # exactly why the two dialects are declared independently.
            hook_stop_dialect=HookStopDialect.BLOCK_DECISION,
            # Non-zero exits other than 2 are fail-closed on preToolUse, but a
            # hook *timeout* is fail-open on every Copilot event, so there is no
            # per-hook fail-closed switch to opt into. Default timeout is 30s
            # (`timeoutSec`).
            hook_fail_open_default=False,
            # Session-scoped scenes: deselected MCP servers become repeated
            # --disable-mcp-server flags; the visibility (--excluded-tools) and
            # approval (--allow-tool) layers are independent, so a scene-excluded
            # tool is also filtered out of any profile-supplied allow entries.
            supports_scene_launch=True,
            scene_tool_denylist_flag="--excluded-tools",
        )

    def build_resume_command(
        self,
        session_id: str,
        *,
        working_dir: Path | None = None,
        network_access: bool = False,
    ) -> list[str] | None:
        """Resume a Copilot session: ``copilot --resume=<session_id>``.

        Accepts and ignores the sandbox context (Copilot does not hard-confine
        writes); the keyword-only params keep polymorphic dispatch TypeError-free.
        """
        return ["copilot", f"--resume={session_id}"]

    def locate_sessions(self, project_path: Path) -> list[SessionRef]:
        return copilot_reader.locate_sessions(project_path)

    def read_session(self, ref: SessionRef) -> ConversationTranscript:
        return copilot_reader.read_session(ref)

    def initial_message_args(self, prompt: str) -> list[str]:
        """Copilot uses -i for the initial message."""
        return ["-i", prompt]

    def parse_transcript(self, transcript_path: Path) -> TokenUsage:
        from crossby.ai_tools.transcript import parse_copilot_transcript

        return parse_copilot_transcript(transcript_path)

    def is_model_compatible(self, model: str) -> bool:
        """Copilot accepts all model IDs."""
        return True

    def plan_mode_args(self) -> list[str]:
        """Copilot supports ``--plan`` (GA'd Jan 2026)."""
        return ["--plan"]

    def plan_dir_args(self, plan_dir: str) -> list[str]:
        """Copilot uses --add-dir for plan directory access."""
        return ["--add-dir", plan_dir]

    def allowed_commands_args(self, commands: list[str]) -> list[str]:
        """Translate canonical patterns to Copilot --allow-tool flags.

        Canonical ``"cmd:args"`` becomes ``--allow-tool "shell(cmd:args)"``.
        """
        result: list[str] = []
        for cmd in commands:
            parts = cmd.split(":", 1)
            binary = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            pattern = f"shell({binary}:{args})" if args else f"shell({binary})"
            result.extend(["--allow-tool", pattern])
        return result

    def accept_edits_args(self) -> list[str]:
        """Copilot auto-approves file writes with ``--allow-tool write`` while
        ``shell`` stays gated."""
        return ["--allow-tool", "write"]

    def yolo_args(self) -> list[str]:
        """Copilot uses ``--yolo`` (alias for ``--allow-all``)."""
        return ["--yolo"]

    def normalize_model_format(self, model_id: str) -> str:
        """Copilot uses dotted format for Claude models."""
        if model_id.startswith("claude-"):
            import re

            # Convert claude-haiku-4-5 -> claude-haiku-4.5
            # Only convert version number separators (digit-digit)
            return re.sub(r"(\d)-(\d)", r"\1.\2", model_id)
        return model_id

    def scene_launch_concerns(self) -> set[str]:
        """Copilot scopes only MCP at launch (per-server disable + allow filter)."""
        return {"mcp"}

    def scene_launch_args(self, scene: SceneLaunchContext) -> SceneLaunchArgs:
        """Scope a scene for one session via Copilot's two independent layers.

        - **Visibility** — a repeated ``--disable-mcp-server <name>`` for each
          deselected MCP server. A server hidden here cannot be re-exposed by the
          approval layer, so this is authoritative.
        - **Approval** — any profile-supplied ``--allow-tool`` entry that names a
          scene-excluded server is dropped before the surviving entries are
          re-emitted, so a profile can never re-allow a tool the scene removed.
          crossby resolves this itself rather than relying on Copilot's internal
          precedence between the two layers.

        Writes no artefact files — Copilot's scene is entirely flag-driven.
        """
        from crossby.scenes.launch import SceneLaunchArgs

        excluded = scene.deselected_mcp()
        args: list[str] = []
        for name in sorted(excluded):
            args += ["--disable-mcp-server", name]
        for entry in scene.allow_tools:
            if not _allow_entry_excluded(entry, excluded):
                args += ["--allow-tool", entry]
        return SceneLaunchArgs(args=tuple(args))


def _allow_entry_excluded(entry: str, excluded_servers: set[str]) -> bool:
    """True when an ``--allow-tool`` *entry* names a scene-excluded MCP server.

    Matches the bare server name and both per-tool spellings Copilot has used —
    the documented ``<server>(<tool>)`` form and the ``<server>__<tool>``
    namespacing — so ``github``, ``github(create_issue)`` and
    ``github__create_issue`` are all dropped when ``github`` is excluded, while
    an unrelated ``shell(git:*)`` survives.
    """
    return any(
        entry == server or entry.startswith(f"{server}__") or entry.startswith(f"{server}(")
        for server in excluded_servers
    )
