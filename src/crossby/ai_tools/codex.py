"""OpenAI Codex CLI adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from crossby.ai_tools.base import AbstractAITool
from crossby.handoff.models import ConversationTranscript, SessionRef
from crossby.handoff.readers import codex as codex_reader
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    EffortLevel,
    HookOutputDialect,
    HookStopDialect,
)
from crossby.utils.git_worktree import outside_root_git_metadata_dirs

if TYPE_CHECKING:
    from crossby.scenes.launch import SceneLaunchArgs, SceneLaunchContext

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
            # `codex update` — updates Codex to the latest version. Like Claude,
            # can update a different install than the one on PATH under a
            # package-manager setup; the version-unchanged warning signals it.
            update_command=("codex", "update"),
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
            # Both dialects stated explicitly rather than left to the model
            # default, so the capability matrix reads the same in every adapter.
            hook_output_dialect=HookOutputDialect.HOOK_SPECIFIC_OUTPUT,
            hook_stop_dialect=HookStopDialect.BLOCK_DECISION,
            sandboxes_writes=True,
            supports_network_access=True,
            supports_usage_reporting=True,
            # Session-scoped scenes: Codex takes a named profile that layers a
            # generated ``$CODEX_HOME/<name>.config.toml`` over the base config
            # (requires codex >= 0.134.0 — gated in scene_launch_ready()).
            supports_scene_launch=True,
            scene_profile_flag="--profile",
        )

    def build_resume_command(
        self,
        session_id: str,
        *,
        working_dir: Path | None = None,
        network_access: bool = False,
    ) -> list[str] | None:
        """Resume a Codex session: ``codex resume <session_id>``.

        In a linked worktree, append the sandbox config so git writes work —
        ``--sandbox workspace-write`` + the git-metadata ``writable_roots`` +
        the network pin — but **approval-neutral**: resume deliberately skips
        autonomy resolution, so no ``-a`` flag is injected and the session's
        existing approval policy is preserved (forcing ``-a never`` would disable
        approval prompts for a user who never requested YOLO). A non-worktree
        resume with no ``--network`` composes nothing extra, so it stays
        byte-identical to ``["codex", "resume", <id>]``.
        """
        return [
            "codex",
            "resume",
            session_id,
            *self.sandbox_config_args(
                autonomy_args=[],
                trusted_dirs=None,
                working_dir=working_dir,
                network_access=network_access,
            ),
        ]

    def locate_sessions(self, project_path: Path) -> list[SessionRef]:
        return codex_reader.locate_sessions(project_path)

    def read_session(self, ref: SessionRef) -> ConversationTranscript:
        return codex_reader.read_session(ref)

    def initial_message_args(self, prompt: str) -> list[str]:
        """Codex accepts the initial message as a positional argument."""
        return [prompt]

    def headless_prompt_stdin_args(self) -> list[str] | None:
        """``codex exec`` reads instructions from stdin when no positional prompt
        is passed (a piped stdin is otherwise appended as a ``<stdin>`` block)."""
        return ["exec"]

    def plan_dir_args(self, plan_dir: str) -> list[str]:
        """Codex uses --add-dir for plan directory access."""
        return ["--add-dir", plan_dir]

    def sandbox_config_args(
        self,
        *,
        autonomy_args: list[str],
        trusted_dirs: list[str] | None,
        working_dir: Path | None,
        network_access: bool,
    ) -> list[str]:
        """Single owner of Codex's sandbox / writable-root / network argv.

        Reached from :meth:`build_launch_command` (the launch hook) and, with
        ``autonomy_args=[]``, from :meth:`build_resume_command`.

        Emits, **in order**, a single ``--sandbox workspace-write`` before any
        ``--add-dir``, then one ``--add-dir`` per trusted dir and per
        linked-worktree git-metadata dir (so sandboxed git writes reach the
        external gitdir), and an explicit ``network_access`` pin — but only when
        crossby actually **forces** workspace-write. When nothing forces it,
        returns ``[]`` so the launch/resume stays byte-identical to an unmanaged
        Codex run.

        The metadata dirs go through ``--add-dir`` (which *adds* to the sandbox's
        writable roots) rather than ``-c sandbox_workspace_write.writable_roots``
        (which would *replace* any roots the user configured). The network pin
        is deliberately the replacing form — see below.

        workspace-write is forced by any of: accept-edits/auto (detected as
        ``-a untrusted`` in ``autonomy_args``), one or more ``trusted_dirs``,
        out-of-root worktree metadata, or ``network_access``. YOLO alone
        (``-a never``) does **not** force it. This never emits an ``-a`` flag
        itself: on launch the approval flag already sits in ``autonomy_args``;
        on resume it is intentionally absent (approval-neutral). Treating
        ``--sandbox``/``-s`` as one setting and owning the ordering here is what
        guarantees the mode is emitted exactly once.

        The network pin defends against an ambient ``network_access=true`` in the
        user's config: whenever crossby forces workspace-write it explicitly sets
        the flag (``true`` only with ``--network``, else ``false``), so ambient
        config can never silently enable networking in a crossby-managed sandbox.
        """
        trusted = list(trusted_dirs or [])
        metadata = (
            outside_root_git_metadata_dirs(working_dir) if working_dir is not None else []
        )
        accept_edits = "untrusted" in autonomy_args

        if not (accept_edits or trusted or metadata or network_access):
            return []

        args: list[str] = ["--sandbox", "workspace-write"]
        for d in trusted:
            args.extend(self.plan_dir_args(d))
        for meta_dir in metadata:
            args.extend(self.plan_dir_args(str(meta_dir)))
        args.extend(
            [
                "-c",
                f"sandbox_workspace_write.network_access={'true' if network_access else 'false'}",
            ]
        )
        return args

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
        """Codex accept-edits: auto-apply file edits, still escalate untrusted
        shell commands for approval — the approval half only (``-a untrusted``).

        The workspace-write sandbox that accept-edits needs is emitted by
        :meth:`sandbox_config_args`, which owns sandbox-mode selection so the
        ``--sandbox workspace-write`` flag is passed exactly once (before any
        ``--add-dir``) even when trusted dirs or worktree metadata are also
        present. The old ``--approval-mode auto-edit`` flag was removed in the
        Rust CLI (v0.14x) and must not be used.
        """
        return ["-a", "untrusted"]

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

    def scene_launch_ready(self) -> bool:
        """Codex ``--profile`` scenes need ``codex >= 0.134.0``.

        The legacy in-config ``[profiles.<name>]`` tables were removed in that
        release; only from it on does ``--profile <name>`` layer
        ``$CODEX_HOME/<name>.config.toml`` over the base config. On an older or
        unknown build this returns False so the launch path falls back to
        persistent activation rather than emitting a ``--profile`` the CLI would
        ignore.
        """
        if not self.capabilities().supports_scene_launch:
            return False
        from crossby.scenes import versioning
        from crossby.scenes.launch import CODEX_PROFILE_MIN

        version = versioning.detect_tool_version(AIToolID.CODEX)
        return versioning.at_least(version, CODEX_PROFILE_MIN)

    def scene_launch_concerns(self) -> set[str]:
        """Codex scopes only MCP at launch (via the layered profile)."""
        return {"mcp"}

    def scene_launch_args(self, scene: SceneLaunchContext) -> SceneLaunchArgs:
        """Compile the scene into a namespaced ``$CODEX_HOME`` profile.

        The scene's deselected MCP servers become ``[mcp_servers.<id>] enabled =
        false`` in ``$CODEX_HOME/crossby-<project-slug>-<scene>.config.toml``,
        and ``--profile crossby-<project-slug>-<scene>`` layers it over the base
        config for the session. Emitted only when the scene narrows MCP (the one
        Codex session lever); skills/agents/hooks/permissions have no Codex
        launch flag and are left to persistent ``scene use``.

        The profile lives under ``$CODEX_HOME`` — the documented exception to the
        "everything under ``.crossby/scene/``" rule — because ``--profile`` reads
        nowhere else. It is namespaced by a project-root hash and carries a
        generated-by header so pruning never deletes a hand-written profile.

        If that namespaced path is already occupied by a hand-written profile,
        ``write_codex_profile`` raises :class:`FileExistsError` rather than
        clobber it. Instead of letting that abort ``crossby launch --scene``, the
        launch degrades to persistent ``scene use`` activation for Codex — the
        same fallback taken when the CLI is too old for ``--profile``.
        """
        import warnings

        from crossby.scenes.launch import (
            SceneLaunchArgs,
            codex_profile_name,
            write_codex_profile,
        )

        if not scene.narrows_mcp():
            return SceneLaunchArgs()

        try:
            write_codex_profile(scene.project_root, scene.name, scene.deselected_mcp())
        except FileExistsError as exc:
            from crossby.scenes.engine import apply_scene

            warnings.warn(
                f"{exc} Falling back to persistent 'scene use' activation for scene "
                f"{scene.name!r}; this writes Codex config files.",
                UserWarning,
                stacklevel=2,
            )
            apply_scene(scene.resolved, scene.project_root, tools=(self.TOOL_ID,))
            return SceneLaunchArgs()

        flag = self.capabilities().scene_profile_flag or "--profile"
        return SceneLaunchArgs(args=(flag, codex_profile_name(scene.project_root, scene.name)))
