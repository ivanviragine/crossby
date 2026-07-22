"""crossby handoff — cross-tool AI session handoff via a structured markdown file."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from crossby.ui.console import console

_UNSUPPORTED_SOURCES = {"antigravity-cli", "opencode", "antigravity", "vscode"}
_UNSUPPORTED_REASONS = {
    "antigravity-cli": (
        "Antigravity CLI stores conversations as opaque per-conversation SQLite "
        "databases that are not yet supported (see follow-up issue)."
    ),
    "opencode": (
        "OpenCode stores sessions in a SQLite database that is not yet supported "
        "(see follow-up issue)."
    ),
    "antigravity": "Antigravity sessions are not on disk in a readable form.",
    "vscode": "VS Code chat sessions are not exposed on disk.",
}

# Safety cap: the target's CLI argument must stay well under OS argv limits.
# macOS ARG_MAX is ~256KB; 4KB is a comfortable ceiling for a path-based prompt.
_MAX_INITIAL_MESSAGE_BYTES = 4 * 1024


def handoff(
    from_tool: str | None = typer.Option(
        None, "--from", help="Source tool to read the session from."
    ),
    to_tool: str | None = typer.Option(
        None, "--to", help="Target tool to launch with the handoff as initial prompt."
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help=(
            "Session ID to use. When omitted, an interactive picker is shown on TTY "
            "when multiple matching sessions exist; otherwise the latest session is used."
        ),
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Write the handoff file to this path instead of the default."
    ),
    no_launch: bool = typer.Option(
        False, "--no-launch", help="Write the handoff file but do not launch the target."
    ),
    summarizer_tool: str | None = typer.Option(
        None,
        "--summarizer-tool",
        help="Tool to run the summarization pass (defaults to the source tool).",
    ),
    token_budget: int | None = typer.Option(
        None,
        "--token-budget",
        help="Approximate token budget for the transcript (default: 32000).",
    ),
    prompt: Path | None = typer.Option(
        None, "--prompt", help="Path to a custom summarization prompt (.md)."
    ),
    prompt_preset: str | None = typer.Option(
        None,
        "--prompt-preset",
        help="Bundled prompt preset: default | cc-compact (default: default).",
    ),
    path: Path = typer.Option(Path("."), "--path", help="Project root directory."),
) -> None:
    """Carry session context from one AI CLI to another via a handoff file.

    Examples::

        crossby handoff --from claude --to codex
        crossby handoff --from cursor --to copilot --no-launch
        crossby handoff --from codex --to claude --session-id 019cb497-ec14-...
    """
    from crossby.ai_tools.base import AbstractAITool
    from crossby.config.loader import ConfigError, load_config
    from crossby.handoff.picker import pick_latest_session, prompt_for_session
    from crossby.handoff.prompts import (
        PRESETS,
        PromptNotFoundError,
        load_launch_template,
        load_preset,
        load_user_prompt,
    )
    from crossby.handoff.summarizer import (
        HandoffSummarizer,
        SummarizerParseError,
        SummarizerToolNotInstalledError,
    )
    from crossby.handoff.writer import HandoffWriter
    from crossby.services.handoff_resolution import (
        resolve_handoff_preset,
        resolve_handoff_token_budget,
    )

    project_root = path.resolve()

    # Capture before resolution so a config-injected preset doesn't trip mutual-exclusivity.
    try:
        config = load_config(project_root)
    except ConfigError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from exc
    user_set_preset = prompt_preset is not None
    prompt_preset = resolve_handoff_preset(prompt_preset, config)
    token_budget = resolve_handoff_token_budget(token_budget, config)
    if token_budget <= 0:
        console.error(f"--token-budget must be positive (got {token_budget}).")
        raise typer.Exit(1)

    # --- Resolve the summarization prompt ---------------------------------
    try:
        prompt_template, prompt_source, structured = _resolve_prompt(
            prompt=prompt,
            prompt_preset=prompt_preset,
            user_set_preset=user_set_preset,
            load_preset=load_preset,
            load_user_prompt=load_user_prompt,
            presets=PRESETS,
        )
    except PromptNotFoundError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from None
    except _InvalidPromptFlagsError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from None

    # --- Interactive wizard fills in missing flags -------------------------
    if from_tool is None or to_tool is None:
        wizard = _run_wizard(
            from_tool=from_tool,
            to_tool=to_tool,
            project_root=project_root,
        )
        if wizard is None:
            raise typer.Exit(1)
        from_tool, to_tool = wizard

    source_id = _parse_tool_id(from_tool, "from")
    target_id = _parse_tool_id(to_tool, "to")

    if source_id.value in _UNSUPPORTED_SOURCES:
        reason = _UNSUPPORTED_REASONS.get(
            source_id.value,
            f"{source_id} is not yet supported as a handoff source.",
        )
        console.error(reason)
        raise typer.Exit(1)

    # --- Locate source session --------------------------------------------
    source_adapter = AbstractAITool.get(source_id)
    try:
        sessions = source_adapter.locate_sessions(project_root)
    except NotImplementedError:
        console.error(f"{source_id} does not support session handoff as a source.")
        raise typer.Exit(1) from None

    if session_id is not None:
        chosen = next((s for s in sessions if s.session_id == session_id), None)
        if chosen is None:
            console.error(f"No {source_id} session with id '{session_id}' in {project_root}.")
            raise typer.Exit(1)
    else:
        from crossby.handoff._utils import safe_resolve
        from crossby.ui import prompts

        target = project_root.resolve()
        matching = [s for s in sessions if s.cwd is not None and safe_resolve(s.cwd) == target]
        if prompts.is_tty() and len(matching) > 1:
            chosen = prompt_for_session(sessions, project_root)
        else:
            chosen = pick_latest_session(sessions, project_root)
        if chosen is None:
            console.error(
                f"No {source_id} sessions found for {project_root}. "
                "Start a session in this project first, or pass --session-id."
            )
            raise typer.Exit(1)

    console.step(f"Using {source_id} session {chosen.session_id} from {chosen.started_at}")

    # --- Read & summarize -------------------------------------------------
    transcript = source_adapter.read_session(chosen)
    if not transcript.turns:
        console.error(
            f"{source_id} session {chosen.session_id} has no readable turns — nothing to hand off."
        )
        raise typer.Exit(1)

    if summarizer_tool:
        summarizer_id = _parse_tool_id(summarizer_tool, "summarizer-tool")
    else:
        summarizer_id = source_id
    summarizer_adapter = AbstractAITool.get(summarizer_id)
    summarizer = HandoffSummarizer(
        summarizer_adapter,
        prompt_template=prompt_template,
        token_budget=token_budget,
    )

    def _warn_truncation(total: int, kept: int) -> None:
        console.warn(f"Transcript truncated: kept {kept} of {total} turns to fit token budget.")

    from crossby.handoff.models import HandoffDocument, RawHandoff

    doc: HandoffDocument | RawHandoff
    try:
        console.step(
            f"Summarizing with {summarizer_id} "
            f"(budget={token_budget} tokens, prompt={prompt_source})..."
        )
        if structured:
            doc = summarizer.summarize_structured(
                transcript,
                source_tool=source_id,
                target_tool=target_id,
                on_truncate=_warn_truncation,
            )
        else:
            doc = summarizer.summarize_raw(
                transcript,
                source_tool=source_id,
                target_tool=target_id,
                prompt_source=prompt_source,
                on_truncate=_warn_truncation,
            )
    except SummarizerToolNotInstalledError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from None
    except SummarizerParseError as exc:
        console.error(f"Summarizer failed: {exc}")
        raise typer.Exit(1) from None

    # --- Write markdown --------------------------------------------------
    writer = HandoffWriter(project_root)
    handoff_path = writer.write(doc, output_path=output)
    console.success(f"Wrote handoff file: {handoff_path}")

    if no_launch:
        console.info("--no-launch set; target tool not launched.")
        return

    # --- Launch target ---------------------------------------------------
    target_adapter = AbstractAITool.get(target_id)
    try:
        launch_template = load_launch_template()
    except PromptNotFoundError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from None
    initial_message = launch_template.format(path=handoff_path)
    encoded_len = len(initial_message.encode("utf-8"))
    if encoded_len > _MAX_INITIAL_MESSAGE_BYTES:
        console.error(
            f"Initial-message prompt is {encoded_len} bytes, exceeding safety cap "
            f"of {_MAX_INITIAL_MESSAGE_BYTES}. This should never happen — please file a bug."
        )
        raise typer.Exit(1)
    args = target_adapter.initial_message_args(initial_message)
    if not args:
        console.warn(
            f"{target_id} has no initial-message support; handoff file written but "
            "target cannot be launched with the context pre-loaded. Open it manually: "
            f"{handoff_path}"
        )
        return

    cmd = target_adapter.build_launch_command(initial_message=initial_message)
    console.step(f"Launching {target_id}: {shlex.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
    except FileNotFoundError:
        console.error(f"{target_id} binary not found in PATH.")
        raise typer.Exit(1) from None
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


class _InvalidPromptFlagsError(ValueError):
    """Raised when ``--prompt`` / ``--prompt-preset`` flag combinations are invalid."""


def _resolve_prompt(
    prompt: Path | None,
    prompt_preset: str,
    *,
    user_set_preset: bool,
    load_preset: Callable[[str], str],
    load_user_prompt: Callable[[Path], str],
    presets: dict[str, str],
) -> tuple[str, str, bool]:
    """Return ``(prompt_template, prompt_source, structured)``.

    ``structured`` is True only for the ``default`` preset; any custom prompt
    or non-default preset takes the raw-passthrough path.

    ``user_set_preset`` indicates whether the user actively passed
    ``--prompt-preset``. The flag has to be computed at the call site (before
    resolution) because ``prompt_preset`` here may have been injected by
    config, which must not trip the mutual-exclusivity check.
    """
    custom_prompt_given = prompt is not None
    if custom_prompt_given and user_set_preset:
        raise _InvalidPromptFlagsError("--prompt and --prompt-preset are mutually exclusive.")
    if custom_prompt_given:
        assert prompt is not None
        resolved = prompt.resolve()
        return load_user_prompt(resolved), str(resolved), False
    if prompt_preset not in presets:
        valid = ", ".join(sorted(presets))
        raise _InvalidPromptFlagsError(
            f"Unknown --prompt-preset {prompt_preset!r}. Valid presets: {valid}."
        )
    structured = prompt_preset == "default"
    return load_preset(prompt_preset), prompt_preset, structured


def _parse_tool_id(value: str, flag: str):  # type: ignore[no-untyped-def]
    from crossby.models.ai import AIToolID

    try:
        return AIToolID(value)
    except ValueError:
        valid = ", ".join(t.value for t in AIToolID)
        console.error(f"Unknown tool for --{flag}: {value!r}. Valid: {valid}")
        raise typer.Exit(1) from None


def _run_wizard(
    from_tool: str | None,
    to_tool: str | None,
    project_root: Path,
) -> tuple[str, str] | None:
    """Prompt for missing ``--from``/``--to`` values via ``confirm_defaults``.

    Reads ``handoff_defaults`` from ``.crossby.yml`` via
    :mod:`crossby.services.handoff_resolution`, shows resolved values with
    Proceed / Change entries, then runs a final cancel-able confirm prompt
    (``confirm_defaults`` itself has no cancel path).
    """
    from crossby.ai_tools.base import AbstractAITool
    from crossby.config.loader import ConfigError, load_config
    from crossby.services.confirm import ConfirmField, confirm_defaults
    from crossby.services.handoff_resolution import (
        resolve_handoff_from,
        resolve_handoff_to,
    )
    from crossby.ui import prompts

    installed = [str(t) for t in AbstractAITool.detect_installed()]
    if not installed:
        console.error("No AI tools detected in PATH. Install a supported tool first.")
        return None

    sources = [t for t in installed if t not in _UNSUPPORTED_SOURCES]
    if not sources:
        console.error("None of the installed tools are supported as handoff sources yet.")
        return None

    console.step(f"Handoff wizard — project root: {project_root}")
    console.detail(f"Detected tools: {', '.join(installed)}")

    try:
        config = load_config(project_root)
    except ConfigError as exc:
        console.error(str(exc))
        return None
    resolved_from = resolve_handoff_from(from_tool, config)
    resolved_to = resolve_handoff_to(to_tool, config)

    # Drop resolved values that aren't installed / aren't supported as sources.
    from_value = str(resolved_from) if resolved_from is not None else None
    if from_value not in sources:
        from_value = sources[0]
    to_value = str(resolved_to) if resolved_to is not None else None
    targets_for_default = [t for t in installed if t != from_value]
    if to_value is None or to_value == from_value or to_value not in installed:
        to_value = targets_for_default[0] if targets_for_default else None
    if to_value is None:
        console.error("No alternate installed tool to hand off to.")
        return None

    def _change_from(current: str, _state: dict[str, Any]) -> dict[str, Any]:
        default_idx = sources.index(current) if current in sources else 0
        idx = prompts.select("Source tool (read session from)?", items=sources, default=default_idx)
        new_from = sources[idx]
        updates: dict[str, Any] = {"from": new_from}
        targets = [t for t in installed if t != new_from]
        if _state.get("to") == new_from or _state.get("to") not in targets:
            updates["to"] = targets[0] if targets else None
        return updates

    def _change_to(current: str, state: dict[str, Any]) -> dict[str, Any]:
        targets = [t for t in installed if t != state.get("from")]
        if not targets:
            return {"to": current}
        default_idx = targets.index(current) if current in targets else 0
        idx = prompts.select(
            "Target tool (launch with handoff prompt)?",
            items=targets,
            default=default_idx,
        )
        return {"to": targets[idx]}

    fields = [
        ConfirmField(
            name="from",
            label="Source tool",
            current_value=from_value,
            explicit=from_tool is not None,
            change_fn=_change_from,
        ),
        ConfirmField(
            name="to",
            label="Target tool",
            current_value=to_value,
            explicit=to_tool is not None,
            change_fn=_change_to,
        ),
    ]

    result = confirm_defaults(fields, title="Confirm handoff")
    final_from = result["from"]
    final_to = result["to"]

    # Preserve the explicit cancel step — confirm_defaults has no cancel path.
    if not prompts.confirm(f"Hand off {final_from} → {final_to}?", default=True):
        console.info("Cancelled.")
        return None

    return final_from, final_to
