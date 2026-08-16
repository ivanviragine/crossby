"""crossby launch — launch an AI tool with resolved config."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from crossby.ui.console import console

if TYPE_CHECKING:
    from crossby.ai_tools.base import AbstractAITool
    from crossby.models.ai import AIToolCapabilities
    from crossby.models.config import CrossbyConfig, SceneConfig
    from crossby.scenes.launch import SceneLaunchContext


def launch(
    path: Path = typer.Argument(Path("."), help="Working directory or profile name."),
    tool: str | None = typer.Option(None, "--tool", "-t", help="AI tool to use."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model to use."),
    effort: str | None = typer.Option(None, "--effort", "-e", help="Effort level."),
    yolo: bool | None = typer.Option(None, "--yolo", help="Skip permission prompts."),
    plan: bool = typer.Option(
        False, "--plan", help="Start in the tool's native plan/approval mode."
    ),
    accept_edits: bool | None = typer.Option(
        None,
        "--accept-edits",
        help=(
            "Permission mode (not model selection): auto-approve file edits, "
            "still prompt for shell. Falls back to default prompting where "
            "unsupported (OpenCode, GUIs)."
        ),
    ),
    auto: bool | None = typer.Option(
        None,
        "--auto",
        help=(
            "Permission mode (not model selection): request the classifier-"
            "mediated auto mode where available (Claude); downgrades to accept-"
            "edits, then default prompting, elsewhere. Never escalates to yolo."
        ),
    ),
    command: str | None = typer.Option(
        None, "--command", "-c", help="Command name for config lookup."
    ),
    prompt: str | None = typer.Option(None, "--prompt", "-p", help="Initial prompt to send."),
    complexity: str | None = typer.Option(
        None, "--complexity", help="Task complexity for model selection."
    ),
    transcript: Path | None = typer.Option(
        None, "--transcript", help="Path to save session transcript."
    ),
    profile: str | None = typer.Option(
        None, "--profile", "-P", help="Named launch profile from .crossby.yml."
    ),
    scene: str | None = typer.Option(
        None,
        "--scene",
        "-S",
        help=(
            "Apply a scene from .crossby.yml for this session only — session-scoped "
            "and writes nothing tracked (falls back to persistent activation on "
            "tools without a launch lever)."
        ),
    ),
    resume: str | None = typer.Option(None, "--resume", help="Resume a previous session by ID."),
    trusted_dirs: list[str] | None = typer.Option(
        None, "--trusted-dir", help="Pre-authorize a directory (repeatable)."
    ),
    network: bool = typer.Option(
        False,
        "--network",
        help=(
            "Allow network access inside the tool's sandbox (Codex only). "
            "Security-sensitive: lets sandboxed commands reach the network. "
            "Warned and ignored on tools without a sandbox network opt-in."
        ),
    ),
) -> None:
    """Launch an AI tool with resolved configuration.

    Resolves tool, model, effort, and yolo from CLI flags, profiles,
    config file, and auto-detection. Works without any config file.

    Examples::

        crossby launch                     # auto-detect everything
        crossby launch --tool claude       # specific tool
        crossby launch --profile ccyolo    # use saved profile
    """
    from crossby.ai_tools.base import AbstractAITool
    from crossby.config.json_utils import PathContainmentError
    from crossby.config.loader import ConfigError, load_config
    from crossby.services.ai_resolution import (
        confirm_ai_selection,
        resolve_accept_edits,
        resolve_ai_tool,
        resolve_auto,
        resolve_effort,
        resolve_model,
        resolve_yolo,
    )
    from crossby.services.prompt_delivery import deliver_prompt_if_needed
    from crossby.utils.process import run_with_transcript

    # Apply profile overrides (--profile or positional profile name)
    profile_name = profile
    path_str = str(path)
    if (
        not profile_name
        and not path.exists()
        and path_str != "."
        and not path.is_absolute()
        and len(path.parts) == 1
    ):
        # Simple name without path structure — treat as profile name
        profile_name = path_str
        work_dir = Path(".").resolve()
    else:
        work_dir = path.resolve()

    try:
        config = load_config(work_dir)
    except ConfigError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from exc

    # Validate --scene early. Precedence is explicit CLI flags > scene > profile >
    # ai: defaults, and a scene may name a default `profile:`. An explicit
    # --profile (or a positional profile name) therefore wins over the scene's
    # profile: only fall back to it when no profile was named.
    scene_cfg = None
    if scene is not None:
        if resume is not None:
            console.error("--scene cannot be combined with --resume.")
            raise typer.Exit(1)
        try:
            scene_cfg = config.get_scene(scene)
        except ConfigError as exc:
            console.error(str(exc))
            raise typer.Exit(1) from exc
        if scene_cfg is None:
            console.error(f"Unknown scene: {scene!r}")
            available = sorted(config.scenes)
            console.hint(
                f"Available scenes: {', '.join(available)}"
                if available
                else "No scenes are defined in .crossby.yml."
            )
            raise typer.Exit(1)
        if profile_name is None and scene_cfg.profile:
            profile_name = scene_cfg.profile

    profile_allow_tools: list[str] = []
    if profile_name:
        prof = config.get_profile(profile_name)
        if prof is None:
            console.error(f"Unknown profile: {profile_name!r}")
            console.hint("Check .crossby.yml profiles section")
            raise typer.Exit(1)
        # Profile values serve as defaults — explicit CLI flags take precedence
        if tool is None and prof.tool:
            tool = prof.tool
        if model is None and prof.model:
            model = prof.model
        if effort is None and prof.effort:
            effort = prof.effort
        if yolo is None and prof.yolo is not None:
            yolo = prof.yolo
        if accept_edits is None and prof.accept_edits is not None:
            accept_edits = prof.accept_edits
        if auto is None and prof.auto is not None:
            auto = prof.auto
        profile_allow_tools = list(prof.allow_tools)

    # Resolve relative transcript path against work_dir so that mkdir and the
    # subprocess cwd=work_dir agree on where the file lands.
    if transcript is not None and not transcript.is_absolute():
        transcript = work_dir / transcript

    # Resolve AI selection
    resolved_tool = resolve_ai_tool(tool, config, command or "default")
    if not resolved_tool:
        console.error("No AI tool specified or detected.")
        console.hint("Install an AI tool or specify --tool")
        raise typer.Exit(1)

    # --resume path: short-circuit before model/effort/yolo resolution and
    # interactive confirmation — those flags are irrelevant when resuming.
    if resume is not None:
        resume = resume.strip()
        if not resume:
            console.error("--resume requires a non-empty session ID.")
            raise typer.Exit(1)
        try:
            adapter = AbstractAITool.get(resolved_tool)
        except (ValueError, KeyError) as e:
            console.error(str(e))
            raise typer.Exit(1) from e
        caps = adapter.capabilities()
        if not caps.supports_resume:
            console.error(f"{caps.display_name} does not support session resume.")
            raise typer.Exit(1)
        # Gate --network before the resume short-circuit so a non-Codex resume
        # warns and ignores it rather than silently dropping the request.
        resume_network = _resolve_network_access(network, caps)
        resume_cmd = adapter.build_resume_command(
            resume, working_dir=work_dir, network_access=resume_network
        )
        if resume_cmd is None:
            console.error(
                f"{caps.display_name}.build_resume_command returned None "
                "despite supports_resume=True."
            )
            raise typer.Exit(1)
        console.kv("AI tool", caps.display_name)
        console.kv("Session", resume)
        console.empty()
        if transcript:
            try:
                transcript.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                console.error(f"Cannot create transcript directory: {e}")
                raise typer.Exit(1) from e
        exit_code = run_with_transcript(resume_cmd, transcript, cwd=work_dir)
        if exit_code != 0:
            console.warn(f"AI tool exited with code {exit_code}")
        if transcript and transcript.exists():
            usage = adapter.parse_transcript(transcript)
            if usage.total_tokens is not None:
                console.kv("Tokens", f"{usage.total_tokens:,}")
            if usage.session_id:
                console.kv("Session ID", usage.session_id)
        raise typer.Exit(exit_code)

    try:
        resolved_model = resolve_model(
            model,
            config,
            command or "default",
            tool=resolved_tool,
            complexity=complexity,
            strict=model is not None,
        )
        resolved_effort = resolve_effort(
            effort,
            config,
            command or "default",
            tool=resolved_tool,
            complexity=complexity,
            strict=effort is not None,
        )
        resolved_yolo = resolve_yolo(
            yolo,
            config,
            command or "default",
            tool=resolved_tool,
            strict=yolo is not None,
        )
        resolved_accept_edits = resolve_accept_edits(accept_edits, config, command or "default")
        resolved_auto = resolve_auto(auto, config, command or "default")
    except ValueError as e:
        console.error(str(e))
        raise typer.Exit(1) from e

    # Interactive confirmation
    (
        resolved_tool,
        resolved_model,
        resolved_effort,
        resolved_accept_edits,
        resolved_auto,
        resolved_yolo,
    ) = confirm_ai_selection(
        resolved_tool,
        resolved_model,
        tool_explicit=tool is not None,
        model_explicit=model is not None,
        resolved_effort=resolved_effort,
        effort_explicit=effort is not None,
        resolved_accept_edits=resolved_accept_edits,
        accept_edits_explicit=accept_edits is not None,
        resolved_auto=resolved_auto,
        auto_explicit=auto is not None,
        resolved_yolo=resolved_yolo,
        yolo_explicit=yolo is not None,
    )

    if not resolved_tool:
        console.error("No AI tool selected.")
        raise typer.Exit(1)

    try:
        adapter = AbstractAITool.get(resolved_tool)
    except (ValueError, KeyError) as e:
        console.error(str(e))
        raise typer.Exit(1) from e
    caps = adapter.capabilities()

    normalized_trusted_dirs = list(trusted_dirs) if trusted_dirs else None
    if normalized_trusted_dirs and not caps.supports_trusted_dirs:
        console.error(f"{caps.display_name} does not support --trusted-dir.")
        raise typer.Exit(1)

    # Normalize --network against the tool's capability (warn + ignore on tools
    # with no sandbox network opt-in). Applied here so GUI adapters — which
    # override launch() and never reach the builder — still surface the warning.
    network_effective = _resolve_network_access(network, caps)

    # Normalize accept-edits/auto against the tool's capabilities and surface any
    # downgrade through crossby's own UI. resolve_accept_edits()/resolve_auto()
    # intentionally don't validate tool support — the downgrade lives in
    # build_launch_command(). But GUI adapters (VS Code, Antigravity IDE) override
    # launch() and never reach the builder, so an unsupported flag would be
    # dropped silently; normalizing here keeps the summary honest for every tool.
    if resolved_auto and not caps.supports_auto:
        if caps.supports_accept_edits:
            console.warn(f"{caps.display_name} does not support --auto; using accept-edits.")
            resolved_accept_edits = True
        else:
            console.warn(f"{caps.display_name} does not support --auto; using default prompting.")
        resolved_auto = False
    if resolved_accept_edits and not caps.supports_accept_edits:
        console.warn(
            f"{caps.display_name} does not support --accept-edits; using default prompting."
        )
        resolved_accept_edits = False

    # A higher autonomy tier (yolo/auto/accept-edits) supersedes plan_mode in
    # build_launch_command(), so don't error on tools that lack plan mode when
    # any of those flags are also set.
    if (
        plan
        and not caps.supports_plan_mode
        and not (resolved_yolo or resolved_auto or resolved_accept_edits)
    ):
        console.error(f"{caps.display_name} does not support --plan.")
        raise typer.Exit(1)

    # Display the effective selection. Autonomy tiers are shown in ladder order;
    # the builder resolves precedence (yolo > auto > accept-edits > plan) at launch.
    # The highest requested tier is the effective one ("on"); any lower tier the
    # user also requested is shown as "superseded" so the summary never implies a
    # moot tier is active.
    console.kv("AI tool", caps.display_name)
    if resolved_model:
        console.kv("Model", resolved_model)
    if resolved_effort:
        console.kv("Effort", resolved_effort.value)
    autonomy_tiers = (
        ("YOLO mode", resolved_yolo),
        ("Auto mode", resolved_auto),
        ("Accept-edits mode", resolved_accept_edits),
        ("Plan mode", plan),
    )
    effective_shown = False
    for label, requested in autonomy_tiers:
        if not requested:
            continue
        console.kv(label, "on" if not effective_shown else "superseded")
        effective_shown = True
    console.empty()

    # Resolve a session-scoped scene into a per-launch context (or apply the
    # persistent fallback / warn for GUI tools) before dispatch.
    scene_ctx = None
    if scene is not None and scene_cfg is not None:
        from crossby.services.scene_resolution import scene_root as compute_scene_root

        scene_ctx = _prepare_scene_launch(
            scene_name=scene,
            scene_cfg=scene_cfg,
            resolved_tool=resolved_tool,
            adapter=adapter,
            caps=caps,
            scene_root=compute_scene_root(work_dir),
            config=config,
            allow_tools=profile_allow_tools,
        )

    # Deliver prompt if tool doesn't support initial messages
    if prompt:
        deliver_prompt_if_needed(adapter, prompt)

    # Ensure transcript parent directory exists
    if transcript:
        try:
            transcript.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            console.error(f"Cannot create transcript directory: {e}")
            raise typer.Exit(1) from e

    # Launch. A scene artefact write that would escape the project root raises
    # PathContainmentError from inside adapter.launch → scene_launch_args; abort
    # cleanly here rather than let it become a persistent-write fallback (the
    # adapters catch only their own errors, e.g. Codex's FileExistsError, so a
    # containment error already propagates untouched).
    try:
        exit_code = adapter.launch(
            working_dir=work_dir,
            model=resolved_model,
            prompt=prompt if caps.supports_initial_message else None,
            transcript_path=transcript,
            trusted_dirs=normalized_trusted_dirs,
            effort=resolved_effort,
            yolo=resolved_yolo,
            plan_mode=plan,
            accept_edits=resolved_accept_edits,
            auto=resolved_auto,
            scene=scene_ctx,
            network_access=network_effective,
        )
    except PathContainmentError as exc:
        console.error(f"Refusing to launch scene {scene!r}: {exc}")
        raise typer.Exit(1) from exc

    if exit_code != 0:
        console.warn(f"AI tool exited with code {exit_code}")

    # Parse transcript if captured
    if transcript and transcript.exists():
        usage = adapter.parse_transcript(transcript)
        if usage.total_tokens is not None:
            console.kv("Tokens", f"{usage.total_tokens:,}")
        if usage.session_id:
            console.kv("Session ID", usage.session_id)

    raise typer.Exit(exit_code)


def _resolve_network_access(requested: bool, caps: AIToolCapabilities) -> bool:
    """Return the effective ``--network`` value for *caps*.

    ``--network`` is CLI-only (no ``.crossby.yml`` field). On a tool without a
    sandbox network opt-in (``supports_network_access`` False — everything but
    Codex) the request is warned about and ignored on every path (launch,
    resume, GUI), so a non-Codex tool never receives a network flag it cannot
    honor.
    """
    if requested and not caps.supports_network_access:
        console.warn(f"{caps.display_name} does not support --network; ignoring.")
        return False
    return requested


def _prepare_scene_launch(
    *,
    scene_name: str,
    scene_cfg: SceneConfig,
    resolved_tool: str,
    adapter: AbstractAITool,
    caps: AIToolCapabilities,
    scene_root: Path,
    config: CrossbyConfig,
    allow_tools: list[str],
) -> SceneLaunchContext | None:
    """Resolve *scene_cfg* for one launch and pick how it applies to the tool.

    Everything scene-related — scan, resolve, the persistent-fallback apply,
    stale-artifact pruning, and the :class:`SceneLaunchContext` an adapter
    builds session-scoped launch args from — is rooted at *scene_root*
    (``scene_root(work_dir)``, the found config's parent), never the
    invocation directory. The context's ``project_root`` in particular is not
    just an artefact-write location: adapters also read it back (e.g.
    Claude's ``skills_dir = scene.project_root / SKILLS_DIR[...]``) to build
    the disable set, and that read must land on the same inventory
    ``resolve_scene`` used, or a launch from a subdirectory would silently
    stop narrowing anything. The caller keeps ``work_dir`` (the invocation
    directory) entirely separate — it drives only the subprocess cwd and the
    transcript path, never anything scene-related.

    Returns a :class:`SceneLaunchContext` when the tool can take the scene on the
    command line for the session. Returns ``None`` (after warning) when:

    - the tool is a GUI launcher (VS Code / Antigravity IDE), which overrides
      ``launch()`` and never reaches the launch flags — the scene can't apply, so
      the launch proceeds without it; or
    - the tool has no session-scoped lever (Antigravity CLI, OpenCode), or a
      runtime gate failed (Codex CLI too old) — crossby *attempts* **persistent**
      ``scene use`` activation for that tool and surfaces the results. What that
      writes is tool-dependent: it applies what the tool has a persistent
      mechanism for, reports genuinely-unsupported narrowings (e.g. OpenCode's
      MCP servers that stay enabled), and can be a complete no-op for a tool with
      no persistent mechanism — so the message states the attempt, not a
      guaranteed write.

    The resolver runs across every tool (``tool_id=None``) so its disable sets
    stay anchored on the real project inventory; the adapters and the persistent
    engine narrow to the launch's single tool themselves.
    """
    from crossby.ai_tools.base import AbstractAITool
    from crossby.models.ai import AIToolID, AIToolType
    from crossby.scenes.engine import apply_scene
    from crossby.scenes.launch import (
        SceneLaunchContext,
        prune_stale_artifacts,
        validate_scene_name,
    )
    from crossby.services.scene_resolution import resolve_scene
    from crossby.sync.readers import build_sync_data, scan_project

    # A scene name is interpolated into artefact paths — reject an unsafe one
    # (separators, ``..``, reserved ``active``) before any path is built.
    try:
        validate_scene_name(scene_name)
    except ValueError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from exc

    tool_id = AIToolID(resolved_tool)
    installed = AbstractAITool.detect_installed()
    scan = scan_project(scene_root, installed)
    resolved = resolve_scene(scene_cfg, scan, scene_root, tool_id=None)
    for warning in resolved.warnings:
        console.warn(warning)

    # Clean up artefacts left by scenes since renamed or deleted (best-effort).
    for pruned in prune_stale_artifacts(scene_root, set(config.scenes)):
        console.detail(f"pruned stale scene artefact: {pruned}")

    if caps.tool_type == AIToolType.GUI:
        console.warn(
            f"{caps.display_name} is a GUI tool; scene {scene_name!r} cannot be applied "
            "to it — launching without the scene."
        )
        return None

    if adapter.scene_launch_ready():
        _warn_unsupported_scene_concerns(scene_cfg, adapter, caps, scene_name)
        return SceneLaunchContext(
            name=scene_name,
            resolved=resolved,
            project_root=scene_root,
            sync_data=build_sync_data(scene_root),
            allow_tools=tuple(allow_tools),
        )

    reason = (
        "CLI is too old for a session-scoped scene"
        if caps.supports_scene_launch
        else "has no session-scoped scene lever"
    )
    # Result-dependent: state the *attempt*, then let the surfaced results say
    # what actually happened. For a tool whose narrowed concern has no lever
    # (OpenCode's MCP), persistent activation writes nothing, so a premature
    # "this writes tool config files" claim would be false.
    console.warn(
        f"{caps.display_name} {reason}; attempting persistent activation for scene "
        f"{scene_name!r} instead — results follow."
    )
    results = apply_scene(resolved, scene_root, tools=[tool_id])
    # Surface only genuinely-unsupported outcomes (a narrowing the tool has no
    # lever for, e.g. deselected MCP servers that stay enabled). Benign skips
    # (already linked / already applied) stay quiet.
    for result in results:
        if result.unsupported and result.message:
            console.warn(result.message)
    # A concern the scene declares that this tool can scope neither at launch nor
    # persistently (its cell is UNSUPPORTED) produces no result at all, so it
    # would be silently ignored. Name it. ``mcp`` is excluded — it has its own
    # surfaced result above (and its universe drives whether there is anything to
    # scope). This restores the honesty the pre-de-scope ``scene_launch_ready``
    # path gave, e.g. for OpenCode's skills/agents/hooks/permissions.
    from crossby.models.config import SCENE_CONCERNS
    from crossby.scenes.mechanism import SceneMechanism, base_mechanism

    no_mechanism = [
        concern
        for concern in SCENE_CONCERNS
        if concern != "mcp"
        and getattr(scene_cfg, concern) is not None
        and base_mechanism(tool_id, concern) == SceneMechanism.UNSUPPORTED
    ]
    if no_mechanism:
        console.warn(
            f"{caps.display_name} cannot scope {', '.join(no_mechanism)} for scene "
            f"{scene_name!r} at launch or persistently; "
            f"{'those remain' if len(no_mechanism) > 1 else 'that remains'} unchanged."
        )
    if any(r.action == "error" for r in results):
        console.warn("Scene activation reported errors; launching anyway.")
    return None


def _warn_unsupported_scene_concerns(
    scene_cfg: SceneConfig,
    adapter: AbstractAITool,
    caps: AIToolCapabilities,
    scene_name: str,
) -> None:
    """Warn when the scene declares a concern the tool can't scope at launch.

    A tool with *some* launch lever can still lack one for a specific concern
    (e.g. Cursor scopes MCP but not agents). Rather than silently apply nothing
    for such a concern, name it and point the user at persistent ``scene use``.
    """
    from crossby.models.config import SCENE_CONCERNS

    supported = adapter.scene_launch_concerns()
    unsupported = [
        concern
        for concern in SCENE_CONCERNS
        if getattr(scene_cfg, concern) is not None and concern not in supported
    ]
    if unsupported:
        console.warn(
            f"{caps.display_name} has no session-scoped lever for "
            f"{', '.join(unsupported)}; scene {scene_name!r} scopes "
            f"{'those' if len(unsupported) > 1 else 'that'} only via persistent "
            "'crossby scene use'."
        )
