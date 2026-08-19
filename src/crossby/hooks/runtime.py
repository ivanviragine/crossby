"""Runtime hook I/O contract — the counterpart to the ``sync/hooks.py`` writers.

Where ``sync/hooks.py`` *writes* each tool's hook config (which command runs on
which event), this module handles the *runtime*: it parses the JSON a tool sends
a hook on **stdin** into a normalized :class:`HookEvent`, and serializes a
:class:`HookDecision` back into the stdout shape + exit code that tool expects.

This centralizes the per-tool dialect knowledge that consumers (e.g. wade's
``wade hook`` guard entry point) would otherwise each re-implement. It is kept
deliberately import-light — only ``crossby.models.ai`` plus stdlib/pydantic — so
a pre-tool-use hook that fires on every edit starts fast.

Tool-call dialects (grouped by output *shape*, not tool — see
:class:`HookOutputDialect`):

- Claude / Codex → ``HOOK_SPECIFIC_OUTPUT``
- Cursor → ``PERMISSION``
- Copilot → ``PERMISSION_DECISION``
- Antigravity CLI (``agy``) → ``DECISION``

Stop dialects are tracked separately (see :class:`HookStopDialect`), because a
tool's Stop channel does not follow from its tool-call channel — Copilot reads
the flat permission shape for PreToolUse but ``{"decision": "block"}`` for
``agentStop``:

- Claude / Codex / Copilot → ``BLOCK_DECISION``
- Cursor → ``FOLLOWUP_MESSAGE``
- Antigravity CLI (``agy``) → ``CONTINUE_DECISION``

A deny exits 2 on every dialect **except** ``DECISION`` (agy), so the block is
honored even by a tool that ignores stdout (and a security guard stays
fail-*closed*); the dialect only governs the stdout payload. ``DECISION`` is the
exception: agy reads a non-zero exit as a hook *crash* (raw stderr surfaced,
stdout discarded), so its deny is exit 0 and fail-closed is carried by the
structured ``{"decision": "deny"}`` on stdout, per agy's contract. A *Stop*
decision never exits 2 — that channel stays fail-open so a guard cannot trap the
agent mid-turn.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel

from crossby.models.ai import AIToolID, HookOutputDialect, HookStopDialect

__all__ = [
    "READ_TOOL_NAMES",
    "SHELL_TOOL_NAMES",
    "HookDecision",
    "HookEmission",
    "HookEvent",
    "detect_tool_id",
    "emit_decision",
    "emit_stop_decision",
    "parse_event",
]

# Canonical → PascalCase event name for HOOK_SPECIFIC_OUTPUT tools (Claude/Codex).
_PASCAL_EVENT_NAMES: dict[str, str] = {
    "pre_tool_use": "PreToolUse",
    "post_tool_use": "PostToolUse",
    "session_start": "SessionStart",
    "user_prompt_submit": "UserPromptSubmit",
    "stop": "Stop",
}

# Reverse map: incoming tool event name (any casing) → canonical.
# Gemini's BeforeTool/AfterTool names were dropped with the Gemini CLI (#69);
# agy — its Gemini-family successor — emits Claude-style PreToolUse/PostToolUse,
# so no Gemini-specific entry is needed here.
_CANONICAL_EVENT_NAMES: dict[str, str] = {
    "pretooluse": "pre_tool_use",
    "posttooluse": "post_tool_use",
    "sessionstart": "session_start",
    "userpromptsubmit": "user_prompt_submit",
    "stop": "stop",
}

# Tool-call names that only *read* (lowercased), grouped by the tool that emits
# them so additions stay auditable. Anything not listed here — and not in
# SHELL_TOOL_NAMES — is treated as a write, so a tool name crossby has never
# seen fails *closed* rather than slipping past a guard.
#
# Verified against each tool's current docs / a live probe of the binary. When a
# tool gains a read-only tool call, add it here or its reads start getting
# routed through the write guard.
READ_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # Claude — `task` spawns a subagent whose own tool calls each fire their
        # own PreToolUse hook, so guarding the spawn itself would double-guard.
        # `todowrite` writes the internal todo list, not the filesystem.
        "read",
        "grep",
        "glob",
        "webfetch",
        "websearch",
        "todowrite",
        "task",
        # Cursor (adds `tabread`; shares read/grep with Claude).
        "tabread",
        # Codex — intentionally contributes nothing. Codex ships no read-only
        # tool calls at all; its own instructions route every read through `rg`
        # in the shell, which is why the SHELL_TOOL_NAMES carve-out below is
        # what keeps a Codex session usable under a write guard.
        # Copilot.
        "view",
        "rg",
        "web_fetch",
        "ask_user",
        # Antigravity CLI (agy) — live-captured toolCall.name values.
        "view_file",
        "list_dir",
        "grep_search",
        "read_url_content",
        "search_web",
        "list_permissions",
    }
)

# Shell/command-execution tool names (lowercased).
#
# These are deliberately NOT writes. `is_write` means specifically "a
# path-addressed file write a containment guard can check against
# ``file_path``" — a shell call has no such path, so classifying it as a write
# would hand the guard an unverifiable write and get every shell command
# denied. Shell calls are still guarded, just through the separate
# :attr:`HookEvent.command` channel (command-content policies), not
# ``file_path``.
#
# This is a carve-out, not a hole: leaving these out of the set below would
# brick Codex outright (every Codex read is a shell call) and break any tool
# whose guard is registered unscoped.
SHELL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "bash",  # Claude, Codex, Copilot
        "shell",  # Cursor
        "exec_command",  # Codex
        "powershell",  # Copilot
        "run_command",  # Antigravity CLI
    }
)

# Canonical events on which Cursor actually reads a top-level
# ``additional_context``. Its `beforeSubmitPrompt` output schema is `continue` +
# `user_message` only, so context emitted there is silently dropped.
_CURSOR_CONTEXT_EVENTS: frozenset[str] = frozenset({"session_start", "post_tool_use"})

# Back-compat: map a legacy tool-call dialect to the stop dialect that tool
# uses, so callers still passing a HookOutputDialect to emit_stop_decision keep
# working unchanged (wade does this today at wade/hooks/cli.py:140,153).
_LEGACY_STOP_DIALECTS: dict[HookOutputDialect, HookStopDialect] = {
    HookOutputDialect.HOOK_SPECIFIC_OUTPUT: HookStopDialect.BLOCK_DECISION,
    HookOutputDialect.PERMISSION: HookStopDialect.FOLLOWUP_MESSAGE,
    HookOutputDialect.PERMISSION_DECISION: HookStopDialect.BLOCK_DECISION,
    HookOutputDialect.DECISION: HookStopDialect.CONTINUE_DECISION,
    HookOutputDialect.EXIT_CODE: HookStopDialect.NONE,
}


def _resolve_stop_dialect(dialect: HookStopDialect | HookOutputDialect) -> HookStopDialect:
    """Map a legacy tool-call dialect to its stop dialect; pass stop dialects through."""
    if isinstance(dialect, HookOutputDialect):
        return _LEGACY_STOP_DIALECTS[dialect]
    return dialect


class HookEvent(BaseModel):
    """A normalized hook invocation, parsed from any tool's stdin dialect."""

    event: str | None = None
    """Canonical event name (``pre_tool_use`` / ``stop`` / ``session_start`` …)."""
    tool_name: str | None = None
    """The intercepted tool call, lowercased (``write`` / ``edit`` / ``bash`` …)."""
    file_path: str | None = None
    """Target path for write-family tool calls, if present."""
    command: str | None = None
    """Shell command for bash-family tool calls, if present."""
    cwd: str | None = None
    stop_hook_active: bool = False
    """True when a Stop hook already fired and blocked this turn — a guard must
    not block again (single-shot) or the session loops forever."""
    raw: dict[str, Any] = {}
    """The original decoded payload, for policies needing fields not normalized."""

    @property
    def is_write(self) -> bool:
        """True when this is a path-addressed filesystem write (or intent unknown).

        Fail-*closed* by construction: this is a denylist, not an allowlist. A
        tool name crossby has never seen (a new tool call, a tool crossby does
        not model, Codex's ``apply_patch``, agy's ``write_to_file``) is treated
        as a write, so a guard inspects it rather than waving it through. Only
        the names in :data:`READ_TOOL_NAMES` are known-safe reads.

        Shell calls (:data:`SHELL_TOOL_NAMES`) are deliberately **not** writes —
        see that constant for why. They are guarded through
        :attr:`command` instead of :attr:`file_path`.
        """
        if self.tool_name is None:
            return True  # unknown intent → treat as a write
        if self.tool_name in SHELL_TOOL_NAMES:
            return False  # guarded via the `command` channel, not `file_path`
        return self.tool_name not in READ_TOOL_NAMES


class HookDecision(BaseModel):
    """A tool-neutral hook decision, serialized per dialect by ``emit_decision``.

    Deliberately binary on the gate: there is no ``ask`` action. Copilot's
    ``permissionDecision`` and agy's PreToolUse decision both accept an
    ``ask``/``force_ask`` value that would hand the call back to the user, but a
    crossby guard has no interactive channel — it runs headless, and in batch
    runs a prompt nobody answers is a hang, not a safety net. Guards therefore
    resolve to allow or deny themselves. Deny carries a ``reason``, which is what
    the user sees; that reason is REQUIRED on Copilot deny.

    ``context`` is not a gate decision at all — it injects text on the
    non-blocking events (SessionStart / PostToolUse / prompt-submit).
    """

    action: Literal["allow", "deny", "context"]
    reason: str = ""
    additional_context: str | None = None

    @classmethod
    def allow(cls) -> HookDecision:
        return cls(action="allow")

    @classmethod
    def deny(cls, reason: str) -> HookDecision:
        return cls(action="deny", reason=reason)

    @classmethod
    def context(cls, text: str) -> HookDecision:
        return cls(action="context", additional_context=text)


class HookEmission(BaseModel):
    """What a hook process should write out: stdout JSON, stderr text, exit code.

    The caller does: print ``stdout`` to stdout (if any), ``stderr`` to stderr
    (if any), then ``sys.exit(exit_code)``.
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


_FILE_PATH_KEYS = ("file_path", "filePath", "path", "notebook_path", "notebookPath")

# agy's ``toolCall.args`` uses PascalCase keys the other tools never emit:
# ``TargetFile`` is the write target for ``write_to_file`` /
# ``replace_file_content`` / ``multi_replace_file_content``. Scoped to the agy
# args channel only (see ``_extract_file_path``) rather than widening the global
# ``_FILE_PATH_KEYS`` — so Cursor's top-level fallback and Copilot's ``toolArgs``
# channel can't be spoofed by a PascalCase key. snake_case is kept after it as
# defense-in-depth in case a future agy build switches casing.
_AGY_FILE_PATH_KEYS = ("TargetFile", *_FILE_PATH_KEYS)


def _first_str(source: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty string value among ``keys`` in ``source``."""
    for key in keys:
        val = source.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _tool_call_args(data: dict[str, Any]) -> dict[str, Any]:
    """Return Antigravity CLI's ``toolCall.args`` dict, or ``{}``.

    agy nests the tool-call arguments one level deeper than the other tools:
    ``{"toolCall": {"name": …, "args": {…}}}`` instead of a flat
    ``tool_input``/``toolInput`` wrapper.
    """
    tool_call = data.get("toolCall")
    if isinstance(tool_call, dict):
        args = tool_call.get("args")
        if isinstance(args, dict):
            return args
    return {}


def _extract_file_path(data: dict[str, Any]) -> str | None:
    """Pull the target file path from any supported tool-input dialect.

    Handles Claude/Cursor (``tool_input``/``toolInput`` dict with
    ``file_path``/``filePath``/``path``/``notebook_path``), Copilot (``toolArgs``
    JSON string), Antigravity CLI (``toolCall.args`` dict, whose write target is
    the PascalCase ``TargetFile``), and Cursor's event hooks that place the path
    at the payload *top level* (e.g. ``beforeReadFile``, which has no
    ``tool_input`` wrapper). ``notebook_path`` covers NotebookEdit, whose target
    lives in a differently-named field.
    """
    tool_input = data.get("tool_input") or data.get("toolInput") or {}
    if isinstance(tool_input, dict):
        found = _first_str(tool_input, _FILE_PATH_KEYS)
        if found:
            return found

    # agy's args channel: PascalCase `TargetFile` first, snake_case as fallback.
    found = _first_str(_tool_call_args(data), _AGY_FILE_PATH_KEYS)
    if found:
        return found

    tool_args = data.get("toolArgs")
    if isinstance(tool_args, str):
        try:
            parsed: object = json.loads(tool_args)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            found = _first_str(parsed, ("file", *_FILE_PATH_KEYS))
            if found:
                return found

    # Top-level fallback (Cursor's file-scoped event hooks put the path here,
    # with no tool_input wrapper). Checked last so a wrapped value still wins.
    return _first_str(data, _FILE_PATH_KEYS)


def _extract_command(data: dict[str, Any]) -> str | None:
    """Pull a shell command from any supported tool-input dialect.

    Includes Cursor's ``beforeShellExecution``, which places ``command`` at the
    payload *top level* (no ``tool_input`` wrapper, and often no ``tool_name``).
    """
    tool_input = data.get("tool_input") or data.get("toolInput") or {}
    if isinstance(tool_input, dict):
        val = tool_input.get("command")
        if isinstance(val, str) and val:
            return val
    # Antigravity CLI nests args under toolCall.args; its run_command key is the
    # PascalCase `CommandLine` (snake_case `command` kept as defense-in-depth).
    args_val = _first_str(_tool_call_args(data), ("CommandLine", "command"))
    if args_val:
        return args_val
    tool_args = data.get("toolArgs")
    if isinstance(tool_args, str):
        try:
            parsed: object = json.loads(tool_args)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            val = parsed.get("command")
            if isinstance(val, str) and val:
                return val
    # Top-level fallback (Cursor beforeShellExecution). Checked last.
    top = data.get("command")
    return top if isinstance(top, str) and top else None


def _extract_tool_name(data: dict[str, Any]) -> str | None:
    """Pull the tool-call name (lowercased) from the payload."""
    for key in ("tool_name", "toolName"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return val.lower()
    # Antigravity CLI nests the name under toolCall.name.
    tool_call = data.get("toolCall")
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        if isinstance(name, str) and name:
            return name.lower()
    return None


def _extract_cwd(data: dict[str, Any]) -> str | None:
    """Resolve the command's working directory from any supported dialect.

    A valid top-level lowercase ``cwd`` wins; agy nests its working directory as
    the PascalCase ``toolCall.args.Cwd``, used only as a fallback. Empty /
    non-string values are ignored (mirroring ``_first_str``) so a consumer's
    command guard can resolve a relative shell redirect against the real working
    directory instead of a ``None``.
    """
    top = data.get("cwd")
    if isinstance(top, str) and top:
        return top
    return _first_str(_tool_call_args(data), ("Cwd",))


def _canonical_event(event: str | None) -> str | None:
    """Normalize any casing/spelling of an event name to its canonical form.

    Accepts canonical (``session_start``), tool-native (``sessionStart``,
    ``SessionStart``) or unknown names; unknown names pass through unchanged.
    """
    if not event:
        return None
    return _CANONICAL_EVENT_NAMES.get(event.replace("_", "").lower(), event)


def _extract_event(data: dict[str, Any], override: str | None) -> str | None:
    """Resolve the canonical event name from the payload or an override."""
    if override:
        return _CANONICAL_EVENT_NAMES.get(override.replace("_", "").lower(), override)
    for key in ("hook_event_name", "hookEventName"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return _CANONICAL_EVENT_NAMES.get(val.replace("_", "").lower(), val)
    return None


def detect_tool_id(data: dict[str, Any]) -> AIToolID | None:
    """Best-effort guess of which AI tool sent a hook payload, from its shape.

    A *fallback* for consumers that don't already know the tool — most bake a
    tool id into the hook command at install time, which is more reliable.
    Returns ``None`` when no distinguishing field is present, so the caller can
    apply its own default rather than get a wrong guess.

    Signals, checked in order:

    - **Cursor** — a ``conversation_id`` string, or a non-empty
      ``workspace_roots`` array (Cursor names both differently from the others).
    - **Antigravity CLI** — camelCase ``workspacePaths``/``conversationId`` or an
      ``artifactDirectoryPath``/``toolCall`` wrapper (agy's stdin uses camelCase
      and nests the tool call, so its keys never collide with Cursor's
      snake_case ``workspace_roots``/``conversation_id``). Checked before Codex
      because agy hook stdin can also carry a ``model`` field.
    - **Codex** — a top-level ``model`` string (Codex puts it in hook stdin;
      Claude and Cursor do not).
    - **Claude** — a ``session_id`` string with none of the above (Codex also
      sends ``session_id``, so it is only conclusive once Codex is ruled out).
    """
    if isinstance(data.get("conversation_id"), str):
        return AIToolID.CURSOR
    workspace_roots = data.get("workspace_roots")
    if isinstance(workspace_roots, list) and workspace_roots:
        return AIToolID.CURSOR
    workspace_paths = data.get("workspacePaths")
    if (
        (isinstance(workspace_paths, list) and workspace_paths)
        or isinstance(data.get("conversationId"), str)
        or isinstance(data.get("artifactDirectoryPath"), str)
        or isinstance(data.get("toolCall"), dict)
    ):
        return AIToolID.ANTIGRAVITY_CLI
    if isinstance(data.get("model"), str):
        return AIToolID.CODEX
    if isinstance(data.get("session_id"), str):
        return AIToolID.CLAUDE
    return None


def parse_event(raw_stdin: str, *, event: str | None = None) -> HookEvent:
    """Parse a tool's hook stdin JSON into a normalized :class:`HookEvent`.

    Never raises on malformed input — an empty / non-JSON / non-object payload
    yields a :class:`HookEvent` with all fields ``None`` and ``raw={}`` so the
    caller's policy decides how to treat the absence (fail-open vs fail-closed).

    Args:
        raw_stdin: The raw JSON string the tool wrote to the hook's stdin.
        event: Optional canonical/tool event name to use when the payload omits
            one (e.g. the hook was registered for a known event).
    """
    data: Any = None
    stripped = raw_stdin.strip()
    if stripped:
        try:
            data = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            data = None
    if not isinstance(data, dict):
        return HookEvent(event=_extract_event({}, event))

    return HookEvent(
        event=_extract_event(data, event),
        tool_name=_extract_tool_name(data),
        file_path=_extract_file_path(data),
        command=_extract_command(data),
        cwd=_extract_cwd(data),
        stop_hook_active=bool(data.get("stop_hook_active") or data.get("stopHookActive")),
        raw=data,
    )


def emit_decision(
    decision: HookDecision,
    dialect: HookOutputDialect,
    *,
    event: str | None = None,
) -> HookEmission:
    """Serialize a :class:`HookDecision` into a tool's stdout/stderr/exit contract.

    - ``allow`` → exit 0, no output (every tool treats exit 0 as allow), except
      ``DECISION`` (agy), which gets an explicit ``{"decision": "allow"}``: agy
      marks ``decision`` **required** on a tool-call hook and reads a payload
      with none (a bare ``{}``) as a *deny*, so the allow must name itself.
    - ``deny`` → exit 2 plus the dialect's stdout JSON and the reason on stderr
      (human-readable, honored by ``EXIT_CODE`` tools) — except ``DECISION``
      (agy), which denies at **exit 0** carrying ``{"decision": "deny",
      "reason": …}``: agy reads a non-zero exit as a hook *crash* and surfaces
      raw stderr instead of parsing the structured deny, so the block is carried
      by stdout there, not the exit code.
    - ``context`` → exit 0, with the injection key each tool actually reads:
      ``hookSpecificOutput.additionalContext`` for ``HOOK_SPECIFIC_OUTPUT``
      (Claude/Codex), a **flat** top-level ``additionalContext`` for
      ``PERMISSION_DECISION`` (Copilot), and a top-level ``additional_context``
      for ``PERMISSION`` (Cursor) — the last only on the events Cursor reads it
      on. ``DECISION`` (agy) has no verified context-injection channel, so it
      degrades to an explicit ``{"decision": "allow"}`` proceed (a bare ``{}``
      would read as a deny).

      The shapes are deliberately *not* interchangeable. Claude validates
      PreToolUse output strictly and silently fails open on an unexpected
      root-level key, so its context must be nested and nothing else added;
      Copilot never nests (``hookSpecificOutput`` appears nowhere in GitHub's
      hooks docs) and caps the joined value at 10 KB.

    Args:
        decision: The tool-neutral decision.
        dialect: The tool's output dialect (from ``AIToolCapabilities``).
        event: Canonical event name, used for the ``hookEventName`` field and to
            gate Cursor's context injection to the events that accept it.
    """
    if decision.action == "allow":
        if dialect is HookOutputDialect.DECISION:
            # agy requires an explicit decision; a bare {} reads as a deny.
            return HookEmission(stdout=json.dumps({"decision": "allow"}), exit_code=0)
        return HookEmission(exit_code=0)

    if decision.action == "context":
        if dialect is HookOutputDialect.HOOK_SPECIFIC_OUTPUT and decision.additional_context:
            ctx_payload: dict[str, Any] = {
                "hookSpecificOutput": {
                    "hookEventName": _PASCAL_EVENT_NAMES.get(event or "", event or ""),
                    "additionalContext": decision.additional_context,
                }
            }
            return HookEmission(stdout=json.dumps(ctx_payload), exit_code=0)
        if dialect is HookOutputDialect.PERMISSION_DECISION and decision.additional_context:
            # Copilot reads a flat top-level key — never nested.
            return HookEmission(
                stdout=json.dumps({"additionalContext": decision.additional_context}),
                exit_code=0,
            )
        if (
            dialect is HookOutputDialect.PERMISSION
            and decision.additional_context
            and _canonical_event(event) in _CURSOR_CONTEXT_EVENTS
        ):
            # Cursor accepts `additional_context` on sessionStart/postToolUse
            # only. Its beforeSubmitPrompt output schema is `continue` +
            # `user_message`, so emitting the key there is silently ignored —
            # gate it rather than pretend the injection landed.
            return HookEmission(
                stdout=json.dumps({"additional_context": decision.additional_context}),
                exit_code=0,
            )
        if dialect is HookOutputDialect.DECISION:
            # agy has no verified context channel; degrade to an explicit allow
            # (a bare {} would read as a deny).
            return HookEmission(stdout=json.dumps({"decision": "allow"}), exit_code=0)
        return HookEmission(exit_code=0)

    # deny — exit 2 so the block is honored regardless of dialect, except agy
    # (DECISION), which denies at exit 0 via structured stdout (see below).
    reason = decision.reason
    if dialect is HookOutputDialect.DECISION:
        # agy blocks a tool call via a top-level {"decision": "deny"} at exit 0.
        # A non-zero exit makes agy report a hook *crash* (raw stderr, discarded
        # reason), so the block is carried by stdout here, not the exit code —
        # per agy's contract. `reason` still goes to stderr for humans/logs.
        agy_deny: dict[str, Any] = {"decision": "deny", "reason": reason}
        return HookEmission(stdout=json.dumps(agy_deny), stderr=reason, exit_code=0)
    if dialect is HookOutputDialect.HOOK_SPECIFIC_OUTPUT:
        deny_payload: dict[str, Any] = {
            "hookSpecificOutput": {
                "hookEventName": _PASCAL_EVENT_NAMES.get(event or "", event or ""),
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
        return HookEmission(stdout=json.dumps(deny_payload), stderr=reason, exit_code=2)
    if dialect is HookOutputDialect.PERMISSION_DECISION:
        # Copilot: flat keys, reason REQUIRED on deny. Exactly one JSON object —
        # Copilot strips {"type":"progress"} lines then runs a single
        # JSON.parse, so a second object would concatenate into invalid JSON and
        # be silently ignored. Exit 2 is an additional deny channel that
        # overrides an "allow" in stdout, so it keeps the guard fail-closed even
        # if stdout is dropped.
        copilot_deny: dict[str, Any] = {
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
        return HookEmission(stdout=json.dumps(copilot_deny), stderr=reason, exit_code=2)
    if dialect is HookOutputDialect.PERMISSION:
        perm_payload: dict[str, Any] = {"permission": "deny", "agent_message": reason}
        return HookEmission(stdout=json.dumps(perm_payload), stderr=reason, exit_code=2)
    # EXIT_CODE — the exit code is the only block signal; reason goes to stderr.
    return HookEmission(stderr=reason, exit_code=2)


def emit_stop_decision(
    should_block: bool,
    reason: str,
    dialect: HookStopDialect | HookOutputDialect,
) -> HookEmission:
    """Serialize a session-*Stop* decision into a tool's continue/block contract.

    Unlike PreToolUse (allow/deny a tool call), a Stop hook keeps the agent
    working by *blocking* completion and feeding a message back:

    - ``BLOCK_DECISION`` (Claude, Codex, Copilot):
      ``{"decision": "block", "reason": …}``
    - ``FOLLOWUP_MESSAGE`` (Cursor): ``{"followup_message": …}`` (auto-submitted;
      the tool bounds re-fires via its own ``loop_limit``).
    - ``CONTINUE_DECISION`` (agy): ``{"decision": "continue", "reason": …}`` — agy
      blocks a Stop by telling the agent to *continue*, the inverse polarity of
      the top-level ``continue`` boolean the other stdout dialects use for a
      no-op.
    - ``NONE``: no Stop-block channel — no-op allow. A tool mapped here should
      also report ``supports_stop_hook = False``, so a Stop hook is not installed
      for it in the first place.

    ``should_block=False`` → the turn ends normally, but a *no-op still emits
    JSON on the stdout-reading dialects*: Codex rejects an empty-stdout Stop hook
    with "invalid stop hook JSON output" (confirmed against a live Codex
    session), so a silent no-op would surface an error to the user on every clean
    stop. ``{"continue": true}`` is the no-op for Claude/Codex/Cursor/Copilot;
    agy uses a bare ``{}`` instead, because to agy a top-level ``continue`` is not
    a recognized key and ``{"decision": "continue"}`` would *block* the stop.
    ``NONE`` tools ignore stdout, so they stay truly silent.

    Never exits non-zero, on any dialect: the Stop channel is fail-*open* by
    design, so a guard that misfires can annoy but cannot trap the agent in a
    turn it is unable to end.

    Args:
        should_block: True to keep the agent working instead of ending the turn.
        reason: Message fed back to the agent when blocking.
        dialect: The tool's :class:`HookStopDialect`. A legacy
            :class:`HookOutputDialect` is still accepted and mapped to its stop
            equivalent, so callers written against the old signature keep their
            existing behaviour.
    """
    stop_dialect = _resolve_stop_dialect(dialect)
    if not should_block:
        if stop_dialect is HookStopDialect.NONE:
            return HookEmission(exit_code=0)
        if stop_dialect is HookStopDialect.CONTINUE_DECISION:
            return HookEmission(stdout=json.dumps({}), exit_code=0)
        return HookEmission(stdout=json.dumps({"continue": True}), exit_code=0)
    if stop_dialect is HookStopDialect.BLOCK_DECISION:
        stop_payload: dict[str, Any] = {"decision": "block", "reason": reason}
        return HookEmission(stdout=json.dumps(stop_payload), exit_code=0)
    if stop_dialect is HookStopDialect.FOLLOWUP_MESSAGE:
        followup_payload: dict[str, Any] = {"followup_message": reason}
        return HookEmission(stdout=json.dumps(followup_payload), exit_code=0)
    if stop_dialect is HookStopDialect.CONTINUE_DECISION:
        agy_continue: dict[str, Any] = {"decision": "continue", "reason": reason}
        return HookEmission(stdout=json.dumps(agy_continue), exit_code=0)
    return HookEmission(exit_code=0)
