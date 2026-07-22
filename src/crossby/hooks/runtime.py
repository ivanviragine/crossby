"""Runtime hook I/O contract — the counterpart to the ``sync/hooks.py`` writers.

Where ``sync/hooks.py`` *writes* each tool's hook config (which command runs on
which event), this module handles the *runtime*: it parses the JSON a tool sends
a hook on **stdin** into a normalized :class:`HookEvent`, and serializes a
:class:`HookDecision` back into the stdout shape + exit code that tool expects.

This centralizes the per-tool dialect knowledge that consumers (e.g. wade's
``wade hook`` guard entry point) would otherwise each re-implement. It is kept
deliberately import-light — only ``crossby.models.ai`` plus stdlib/pydantic — so
a pre-tool-use hook that fires on every edit starts fast.

Dialects (grouped by output *shape*, not tool — see :class:`HookOutputDialect`):

- Claude / Codex → ``HOOK_SPECIFIC_OUTPUT``
- Cursor → ``PERMISSION``
- Copilot → ``EXIT_CODE``

A deny always exits 2 regardless of dialect, so the block is honored even by a
tool that ignores stdout; the dialect only governs the stdout payload.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel

from crossby.models.ai import AIToolID, HookOutputDialect

__all__ = [
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
_CANONICAL_EVENT_NAMES: dict[str, str] = {
    "pretooluse": "pre_tool_use",
    "posttooluse": "post_tool_use",
    "beforetool": "pre_tool_use",
    "sessionstart": "session_start",
    "userpromptsubmit": "user_prompt_submit",
    "stop": "stop",
}

# Tool-call names that write to the filesystem (lowercased). Mirrors the set the
# hook writers scope their matchers to; used to tell write intents from reads.
WRITE_TOOL_NAMES: frozenset[str] = frozenset(
    {"write", "edit", "multiedit", "create", "delete", "save", "append", "notebookedit"}
)


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
        """True when this is a filesystem-write tool call (or intent unknown).

        Unknown tool name → treated as a possible write so guards still inspect
        the path (matches the guard scripts' fail-safe behavior).
        """
        return self.tool_name is None or self.tool_name in WRITE_TOOL_NAMES


class HookDecision(BaseModel):
    """A tool-neutral hook decision, serialized per dialect by ``emit_decision``."""

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


def _first_str(source: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty string value among ``keys`` in ``source``."""
    for key in keys:
        val = source.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _extract_file_path(data: dict[str, Any]) -> str | None:
    """Pull the target file path from any supported tool-input dialect.

    Handles Claude/Cursor (``tool_input``/``toolInput`` dict with
    ``file_path``/``filePath``/``path``/``notebook_path``), Copilot (``toolArgs``
    JSON string), and Cursor's event hooks that place the path at the payload
    *top level* (e.g. ``beforeReadFile``, which has no ``tool_input`` wrapper).
    ``notebook_path`` covers NotebookEdit, whose target lives in a
    differently-named field.
    """
    tool_input = data.get("tool_input") or data.get("toolInput") or {}
    if isinstance(tool_input, dict):
        found = _first_str(tool_input, _FILE_PATH_KEYS)
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
    return None


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
        cwd=data.get("cwd") if isinstance(data.get("cwd"), str) else None,
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

    - ``allow`` → exit 0, no output (every tool treats exit 0 as allow).
    - ``deny`` → exit 2 always (universal block), plus the dialect's stdout JSON
      and the reason on stderr (human-readable, honored by ``EXIT_CODE`` tools).
    - ``context`` → exit 0; injects ``additionalContext`` for
      ``HOOK_SPECIFIC_OUTPUT`` (Claude/Codex) and a top-level
      ``additional_context`` for ``PERMISSION`` (Cursor, via its
      ``beforeSubmitPrompt`` event); a no-op allow for ``EXIT_CODE`` (no
      context channel).

    Args:
        decision: The tool-neutral decision.
        dialect: The tool's output dialect (from ``AIToolCapabilities``).
        event: Canonical event name, used for the ``hookEventName`` field.
    """
    if decision.action == "allow":
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
        if dialect is HookOutputDialect.PERMISSION and decision.additional_context:
            return HookEmission(
                stdout=json.dumps({"additional_context": decision.additional_context}),
                exit_code=0,
            )
        return HookEmission(exit_code=0)

    # deny — always exit 2 so the block is honored regardless of dialect.
    reason = decision.reason
    if dialect is HookOutputDialect.HOOK_SPECIFIC_OUTPUT:
        deny_payload: dict[str, Any] = {
            "hookSpecificOutput": {
                "hookEventName": _PASCAL_EVENT_NAMES.get(event or "", event or ""),
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
        return HookEmission(stdout=json.dumps(deny_payload), stderr=reason, exit_code=2)
    if dialect is HookOutputDialect.PERMISSION:
        perm_payload: dict[str, Any] = {"permission": "deny", "agent_message": reason}
        return HookEmission(stdout=json.dumps(perm_payload), stderr=reason, exit_code=2)
    # EXIT_CODE — the exit code is the only block signal; reason goes to stderr.
    return HookEmission(stderr=reason, exit_code=2)


def emit_stop_decision(
    should_block: bool,
    reason: str,
    dialect: HookOutputDialect,
) -> HookEmission:
    """Serialize a session-*Stop* decision into a tool's continue/block contract.

    Unlike PreToolUse (allow/deny a tool call), a Stop hook keeps the agent
    working by *blocking* completion and feeding a message back:

    - ``HOOK_SPECIFIC_OUTPUT`` (Claude, Codex): ``{"decision": "block", "reason": …}``
    - ``PERMISSION`` (Cursor): ``{"followup_message": …}`` (auto-submitted; the
      tool bounds re-fires via its own ``loop_limit``).
    - ``EXIT_CODE`` (Copilot): no Stop-block channel — no-op allow. These
      tools also report ``supports_stop_hook = False``, so a Stop hook should not
      be installed for them in the first place.

    ``should_block=False`` → the turn ends normally, but a *no-op still emits
    ``{"continue": true}`` on the stdout-reading dialects*: Codex rejects an
    empty-stdout Stop hook with "invalid stop hook JSON output" (confirmed
    against a live Codex session), so a silent no-op would surface an error to
    the user on every clean stop. ``{"continue": true}`` is the universal no-op
    — a harmless allow for Claude and Cursor too. ``EXIT_CODE`` tools ignore
    stdout, so they stay truly silent.
    """
    if not should_block:
        if dialect is HookOutputDialect.EXIT_CODE:
            return HookEmission(exit_code=0)
        return HookEmission(stdout=json.dumps({"continue": True}), exit_code=0)
    if dialect is HookOutputDialect.HOOK_SPECIFIC_OUTPUT:
        stop_payload: dict[str, Any] = {"decision": "block", "reason": reason}
        return HookEmission(stdout=json.dumps(stop_payload), exit_code=0)
    if dialect is HookOutputDialect.PERMISSION:
        followup_payload: dict[str, Any] = {"followup_message": reason}
        return HookEmission(stdout=json.dumps(followup_payload), exit_code=0)
    return HookEmission(exit_code=0)
