"""Claude Code adapter."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import structlog

from crossby.ai_tools.base import AbstractAITool
from crossby.handoff.models import ConversationTranscript, SessionRef
from crossby.handoff.readers import claude as claude_reader
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    EffortLevel,
    HookOutputDialect,
    HookStopDialect,
    TokenUsage,
)

if TYPE_CHECKING:
    from crossby.scenes.launch import SceneLaunchArgs, SceneLaunchContext

logger = structlog.get_logger()


def _encode_claude_path(path: Path) -> str:
    """Encode a filesystem path the way Claude Code does for project directories.

    Claude Code replaces ``/`` with ``-`` **and** ``.`` with ``-``.
    """
    return str(path).replace("/", "-").replace(".", "-")


class ClaudeAdapter(AbstractAITool):
    """Adapter for Claude Code CLI."""

    TOOL_ID: ClassVar[AIToolID] = AIToolID.CLAUDE

    def capabilities(self) -> AIToolCapabilities:
        return AIToolCapabilities(
            tool_id=AIToolID.CLAUDE,
            display_name="Claude Code",
            binary="claude",
            tool_type=AIToolType.TERMINAL,
            supports_model_flag=True,
            headless_flag="--print",
            supports_headless=True,
            supports_effort=True,
            supports_yolo=True,
            supports_resume=True,
            supports_trusted_dirs=True,
            supports_plan_mode=True,
            supports_accept_edits=True,
            supports_auto=True,
            supports_stop_hook=True,
            supports_session_start_hook=True,
            supports_user_prompt_submit_hook=True,
            # Both dialects stated explicitly rather than left to the model
            # default, so the capability matrix reads the same in every adapter.
            hook_output_dialect=HookOutputDialect.HOOK_SPECIFIC_OUTPUT,
            hook_stop_dialect=HookStopDialect.BLOCK_DECISION,
            supports_usage_reporting=True,
            # Session-scoped scenes: Claude takes a whole scene on the command
            # line — an MCP config file (strict), a settings file for skill
            # overrides, and per-agent --disallowedTools entries.
            supports_scene_launch=True,
            scene_settings_flag="--settings",
            scene_mcp_config_flag="--mcp-config",
            scene_mcp_strict_flag="--strict-mcp-config",
            scene_tool_denylist_flag="--disallowedTools",
        )

    def build_resume_command(self, session_id: str) -> list[str] | None:
        """Resume a Claude session: ``claude --resume <session_id>``."""
        return ["claude", "--resume", session_id]

    def locate_sessions(self, project_path: Path) -> list[SessionRef]:
        return claude_reader.locate_sessions(project_path)

    def read_session(self, ref: SessionRef) -> ConversationTranscript:
        return claude_reader.read_session(ref)

    def initial_message_args(self, prompt: str) -> list[str]:
        """Claude accepts the initial message as a positional argument."""
        return [prompt]

    def parse_transcript(self, transcript_path: Path) -> TokenUsage:
        usage = super().parse_transcript(transcript_path)
        # Standardize model IDs from Claude's dashed format to canonical dotted format
        for breakdown in usage.model_breakdown:
            breakdown.model = self.standardize_model_id(breakdown.model)
        return usage

    def is_model_compatible(self, model: str) -> bool:
        """Claude CLI accepts only claude-* model IDs."""
        return model.lower().startswith("claude-")

    def plan_mode_args(self) -> list[str]:
        """Claude supports --permission-mode plan."""
        return ["--permission-mode", "plan"]

    def plan_dir_args(self, plan_dir: str) -> list[str]:
        """Claude uses --add-dir for plan directory access."""
        return ["--add-dir", plan_dir]

    def normalize_model_format(self, model_id: str) -> str:
        """Claude uses dashed format for model IDs."""
        if model_id.startswith("claude-"):
            import re

            # Convert claude-haiku-4.5 -> claude-haiku-4-5
            return re.sub(r"(\d)\.(\d)", r"\1-\2", model_id)
        return model_id

    def standardize_model_id(self, raw_model_id: str) -> str:
        """Convert Claude's dashed format back to the internal dotted format."""
        if raw_model_id.startswith("claude-"):
            import re

            # Convert claude-haiku-4-5 -> claude-haiku-4.5
            return re.sub(r"(\d)-(\d)", r"\1.\2", raw_model_id)
        return raw_model_id

    def preserve_session_data(self, working_dir: Path, main_checkout_path: Path) -> bool:
        """Copy Claude Code session data from source directory to target's project dir.

        Claude Code stores sessions in ``~/.claude/projects/<encoded-path>/``.
        The path encoding replaces every ``/`` with ``-`` **and** every ``.``
        with ``-``, so ``/Users/foo/.worktrees/bar`` becomes
        ``-Users-foo--worktrees-bar``.

        Files are copied without overwriting any that already exist in the
        target's session directory, so existing memory and settings are
        preserved.
        """
        claude_projects_dir = Path.home() / ".claude" / "projects"

        wt_encoded = _encode_claude_path(working_dir)
        main_encoded = _encode_claude_path(main_checkout_path)

        wt_session_dir = claude_projects_dir / wt_encoded
        main_session_dir = claude_projects_dir / main_encoded

        if not wt_session_dir.exists():
            logger.debug(
                "claude.preserve_session_data.no_source",
                working_dir=str(working_dir),
            )
            return True

        main_session_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for item in wt_session_dir.iterdir():
            dest = main_session_dir / item.name
            if dest.exists():
                continue
            if item.is_file():
                shutil.copy2(item, dest)
                copied += 1
            elif item.is_dir():
                shutil.copytree(item, dest)
                copied += 1

        logger.info(
            "claude.preserve_session_data.copied",
            working_dir=str(working_dir),
            main=str(main_checkout_path),
            items=copied,
        )
        return True

    def session_data_dirs(self) -> list[str]:
        return [".claude"]

    def allowed_commands_args(self, commands: list[str]) -> list[str]:
        """Translate canonical patterns to Claude --allowedTools flags.

        Canonical ``"cmd:args"`` becomes ``"Bash(cmd:args)"``.
        """
        from crossby.config.claude_allowlist import canonical_to_claude

        patterns = [canonical_to_claude(cmd) for cmd in commands]
        if not patterns:
            return []
        return ["--allowedTools", *patterns]

    def structured_output_args(self, json_schema: dict[str, Any]) -> list[str]:
        return ["--output-format", "json", "--json-schema", json.dumps(json_schema)]

    def unwrap_structured_output(self, raw: str) -> str:
        """Unwrap Claude's ``--print --output-format json`` envelope.

        Claude wraps the model response in:
        ``{"type": "result", "is_error": bool, "result": "<str>",
           "structured_output": {...}, ...}``

        Prefers ``structured_output`` (schema-conformant object, re-serialized
        to JSON). Falls back to ``result`` (a JSON string for structured
        prompts, plain text otherwise). Returns ``raw`` unchanged when the
        output is not an envelope so plain-text and markdown prompts still work.
        """
        from crossby.handoff.summarizer import SummarizerParseError

        stripped = raw.strip()
        try:
            envelope = json.loads(stripped)
        except json.JSONDecodeError:
            return raw
        if not isinstance(envelope, dict) or envelope.get("type") != "result":
            return raw
        if envelope.get("is_error"):
            msg = envelope.get("result") or "unknown error"
            raise SummarizerParseError(f"Claude reported an error: {msg}")
        structured = envelope.get("structured_output")
        if structured is not None:
            return json.dumps(structured)
        result = envelope.get("result")
        if result is not None:
            return str(result)
        return raw

    def effort_args(self, effort: EffortLevel) -> list[str]:
        """Claude uses the native ``--effort <level>`` flag."""
        return ["--effort", effort.value]

    def yolo_args(self) -> list[str]:
        """Claude uses ``--dangerously-skip-permissions``."""
        return ["--dangerously-skip-permissions"]

    def accept_edits_args(self) -> list[str]:
        """Claude auto-applies edits with ``--permission-mode acceptEdits``."""
        return ["--permission-mode", "acceptEdits"]

    def auto_args(self) -> list[str]:
        """Claude's classifier-mediated guarded autonomy: ``--permission-mode auto``.

        This is a real launch flag. Claude itself reports "unavailable" and
        degrades if the account/model is ineligible — that is a Claude-side
        decision, not a crossby error.
        """
        return ["--permission-mode", "auto"]

    def scene_launch_concerns(self) -> set[str]:
        """Claude scopes MCP, skills, and agents at launch."""
        return {"mcp", "skills", "agents"}

    def scene_launch_args(self, scene: SceneLaunchContext) -> SceneLaunchArgs:
        """Render a scene into session-scoped Claude flags — nothing tracked.

        Three levers, emitted only when the scene actually narrows the concern:

        - **MCP** → an ``.crossby/scene/<name>/launch/mcp.json`` of the selected
          servers, passed as ``--mcp-config <file> --strict-mcp-config`` so
          Claude loads *only* those for the session (mirrors the persistent
          engine, which no-ops when nothing is deselected).
        - **skills** → a ``settings.json`` of ``skillOverrides: {"<name>":
          "off"}`` for each deselected skill, passed as ``--settings <file>``.
          Gated on ``claude >= 2.1.129`` (older builds silently ignore the key);
          on an older/unknown build the overrides are skipped with a warning.
        - **agents** → ``--disallowedTools "Agent(<name>)"`` per deselected
          subagent.

        Writes nothing into ``.claude/`` or ``.mcp.json``.
        """
        import warnings

        from crossby.config.skills import SKILLS_DIR, list_skills
        from crossby.scenes import projection, versioning
        from crossby.scenes.launch import SceneLaunchArgs, mcp_json_config
        from crossby.sync.agents import _AGENT_TARGET_PATHS

        caps = self.capabilities()
        args: list[str] = []

        # MCP — strict config of exactly the selected servers.
        if scene.narrows_mcp():
            path = scene.write_artifact("mcp.json", mcp_json_config(scene.selected_mcp()))
            args += [caps.scene_mcp_config_flag or "--mcp-config", str(path)]
            if caps.scene_mcp_strict_flag:
                args.append(caps.scene_mcp_strict_flag)

        # skills — skillOverrides off for each deselected skill (version-gated).
        # When the scene omits ``skills``, ``resolve_scene`` selects the whole
        # universe (an absent selector filters nothing), so ``selected("skills")``
        # is a superset of the on-disk skills and the disable set is empty — the
        # ``if skills_disable`` guard is the concern-narrowing gate, mirroring MCP.
        skills_dir = scene.project_root / SKILLS_DIR[AIToolID.CLAUDE]
        skills_disable = set(list_skills(skills_dir)) - scene.selected("skills")
        if skills_disable:
            version = versioning.detect_tool_version(AIToolID.CLAUDE)
            if versioning.at_least(version, versioning.CLAUDE_SKILL_OVERRIDES_MIN):
                overrides = {name: "off" for name in sorted(skills_disable)}
                body = {"skillOverrides": overrides}
                settings = json.dumps(body, indent=2, sort_keys=True) + "\n"
                path = scene.write_artifact("settings.json", settings)
                args += [caps.scene_settings_flag or "--settings", str(path)]
            else:
                floor = ".".join(map(str, versioning.CLAUDE_SKILL_OVERRIDES_MIN))
                detected = ".".join(map(str, version)) if version else "unknown"
                warnings.warn(
                    f"skillOverrides needs claude >= {floor} (detected {detected}); "
                    f"{len(skills_disable)} deselected skill(s) not filtered for this launch.",
                    UserWarning,
                    stacklevel=2,
                )

        # agents — block each deselected subagent via --disallowedTools.
        agents_target = _AGENT_TARGET_PATHS.get(str(AIToolID.CLAUDE))
        if agents_target is not None:
            universe = projection.scene_names(scene.project_root, agents_target, "agents")
            agents_disable = universe - scene.selected("agents")
            if agents_disable:
                args.append(caps.scene_tool_denylist_flag or "--disallowedTools")
                args += [f"Agent({name})" for name in sorted(agents_disable)]

        return SceneLaunchArgs(args=tuple(args))
