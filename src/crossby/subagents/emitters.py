"""Per-tool emitters — render a SubagentIR into a tool-specific file body.

Each emitter returns ``(payload, warnings)``.  For four of the five tools
``payload`` is a single string (the markdown file).  For Codex it's a
:class:`CodexEmission` carrying both the agent ``.toml`` body and the
fragment that needs to be merged into ``~/.codex/config.toml`` — Codex is
the only tool with split storage, so it gets its own return type instead of
forcing all emitters to return tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tomli_w
import yaml

from crossby.subagents.ir import ConversionWarning, SubagentIR, WarningSeverity
from crossby.subagents.tool_map import (
    all_read_only,
    from_canonical,
)


@dataclass(frozen=True)
class CodexEmission:
    """Output of the Codex emitter — agent file plus a config.toml fragment.

    ``config_fragment`` is a TOML string suitable for merging into
    ``~/.codex/config.toml`` under an ``[agents.<name>]`` table.  It will be
    empty when the IR carries no fields that need global registration; in
    that case writing only ``agent_toml`` is sufficient.
    """

    agent_toml: str
    config_fragment: str
    suggested_filename: str  # e.g. "researcher.toml"


def emit(tool: str, ir: SubagentIR) -> tuple[Any, list[ConversionWarning]]:
    """Dispatch to the per-tool emitter."""
    emitters = {
        "claude": emit_claude,
        "cursor": emit_cursor,
        "gemini": emit_gemini,
        "copilot": emit_copilot,
        "codex": emit_codex,
    }
    fn = emitters.get(tool)
    if fn is None:
        raise ValueError(f"unknown target tool: {tool!r}")
    return fn(ir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_markdown(fm: dict[str, Any], body: str) -> str:
    """YAML frontmatter + markdown body, without re-sorting keys."""
    yaml_block = yaml.safe_dump(fm, sort_keys=False, default_flow_style=False)
    body = body if body.endswith("\n") or body == "" else body + "\n"
    return f"---\n{yaml_block}---\n{body}"


def _warn_dropped(field: str, target: str, reason: str = "") -> ConversionWarning:
    msg = f"{target} has no equivalent for `{field}`"
    if reason:
        msg += f" ({reason})"
    return ConversionWarning(field=field, severity=WarningSeverity.DROPPED, message=msg)


def _warn_lossy(field: str, message: str) -> ConversionWarning:
    return ConversionWarning(field=field, severity=WarningSeverity.LOSSY, message=message)


def _extras_for(ir: SubagentIR, target: str) -> dict[str, Any]:
    """Re-emit IR.extras only when the target tool matches the original source.

    Tool-specific extras don't generalize, so we only round-trip them back
    into the same tool — otherwise they'd silently leak alien field names
    into a foreign format.
    """
    if ir.source_tool == target and ir.extras:
        return dict(ir.extras)
    return {}


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------


def emit_claude(ir: SubagentIR) -> tuple[str, list[ConversionWarning]]:
    warnings: list[ConversionWarning] = []
    fm: dict[str, Any] = {"name": ir.name}
    if ir.description is not None:
        fm["description"] = ir.description
    if ir.model:
        fm["model"] = ir.model
    if ir.tools is not None:
        fm["tools"] = [from_canonical(t, "claude") for t in ir.tools]
    if ir.disallowed_tools is not None:
        fm["disallowedTools"] = [from_canonical(t, "claude") for t in ir.disallowed_tools]
    if ir.mcp_servers:
        fm["mcpServers"] = ir.mcp_servers
    if ir.max_turns is not None:
        fm["maxTurns"] = ir.max_turns
    if ir.permission_mode:
        fm["permissionMode"] = ir.permission_mode
    if ir.effort:
        fm["effort"] = ir.effort
    if ir.background is not None:
        fm["background"] = ir.background
    if ir.color:
        fm["color"] = ir.color
    if ir.isolation:
        fm["isolation"] = ir.isolation
    if ir.memory:
        fm["memory"] = ir.memory
    if ir.skills is not None:
        fm["skills"] = ir.skills
    if ir.hooks is not None:
        fm["hooks"] = ir.hooks
    if ir.initial_prompt:
        fm["initialPrompt"] = ir.initial_prompt

    # Cross-tool: Codex sandbox_mode → no Claude equivalent (Claude uses permissionMode)
    if ir.sandbox_mode and not ir.permission_mode:
        warnings.append(
            _warn_lossy(
                "sandbox_mode",
                f"Codex sandbox_mode={ir.sandbox_mode!r} has no direct Claude equivalent; "
                "consider setting permissionMode manually.",
            )
        )
    if ir.readonly is True and ir.permission_mode is None:
        # Best-effort: surface the intent as plan mode.
        warnings.append(
            _warn_lossy(
                "readonly",
                "Cursor readonly=true → no exact Claude equivalent; consider permissionMode: plan.",
            )
        )
    if ir.target and ir.source_tool != "claude":
        warnings.append(_warn_dropped("target", "Claude", "Copilot-only field"))

    fm.update(_extras_for(ir, "claude"))
    return _render_markdown(fm, ir.body), warnings


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------


def emit_cursor(ir: SubagentIR) -> tuple[str, list[ConversionWarning]]:
    warnings: list[ConversionWarning] = []
    fm: dict[str, Any] = {"name": ir.name}
    if ir.description is not None:
        fm["description"] = ir.description
    if ir.model:
        fm["model"] = ir.model

    # Cursor has no tool allowlist — collapse to readonly when possible.
    if ir.tools is not None:
        if all_read_only(ir.tools):
            fm["readonly"] = True
            warnings.append(
                _warn_lossy(
                    "tools",
                    "Cursor lacks a tool allowlist; emitted readonly=true since all tools are "
                    "read-only. The fine-grained list was dropped.",
                )
            )
        else:
            warnings.append(
                _warn_dropped(
                    "tools",
                    "Cursor",
                    "no allowlist field — only readonly bool exists; tools list dropped",
                )
            )
    elif ir.readonly is not None:
        fm["readonly"] = ir.readonly

    if ir.is_background is not None:
        fm["is_background"] = ir.is_background
    if ir.background is True and ir.is_background is None:
        fm["is_background"] = True

    # Lossy-warn on fields Cursor doesn't model
    for field, value, reason in [
        ("mcp_servers", ir.mcp_servers, "Cursor agents inherit project MCP config"),
        ("hooks", ir.hooks, "Cursor agent files have no hooks field"),
        ("skills", ir.skills, "Cursor lacks per-agent skill binding"),
        ("permission_mode", ir.permission_mode, "Cursor uses readonly bool only"),
        ("sandbox_mode", ir.sandbox_mode, "Cursor uses readonly bool only"),
        ("disallowed_tools", ir.disallowed_tools, "Cursor has no per-tool denylist"),
        ("target", ir.target, "Copilot-only"),
        ("nickname_candidates", ir.nickname_candidates, "Codex-only"),
    ]:
        if value:
            warnings.append(_warn_dropped(field, "Cursor", reason))

    fm.update(_extras_for(ir, "cursor"))
    return _render_markdown(fm, ir.body), warnings


# ---------------------------------------------------------------------------
# Gemini CLI
# ---------------------------------------------------------------------------


def emit_gemini(ir: SubagentIR) -> tuple[str, list[ConversionWarning]]:
    warnings: list[ConversionWarning] = []
    fm: dict[str, Any] = {"name": ir.name}
    if ir.description is not None:
        fm["description"] = ir.description
    if ir.kind:
        fm["kind"] = ir.kind
    if ir.model:
        fm["model"] = ir.model
    if ir.tools is not None:
        fm["tools"] = [from_canonical(t, "gemini") for t in ir.tools]
    if ir.mcp_servers:
        fm["mcpServers"] = ir.mcp_servers
    if ir.temperature is not None:
        fm["temperature"] = ir.temperature
    if ir.max_turns is not None:
        fm["max_turns"] = ir.max_turns
    if ir.timeout_mins is not None:
        fm["timeout_mins"] = ir.timeout_mins

    # Lossy-warn for Gemini-unsupported fields
    for field, value, reason in [
        ("disallowed_tools", ir.disallowed_tools, "Gemini has no denylist; only an allowlist"),
        ("permission_mode", ir.permission_mode, "Claude-only"),
        ("sandbox_mode", ir.sandbox_mode, "Codex-only"),
        ("readonly", ir.readonly, "Cursor-only"),
        ("hooks", ir.hooks, "not part of Gemini agent frontmatter"),
        ("target", ir.target, "Copilot-only"),
    ]:
        if value:
            warnings.append(_warn_dropped(field, "Gemini", reason))

    fm.update(_extras_for(ir, "gemini"))
    return _render_markdown(fm, ir.body), warnings


# ---------------------------------------------------------------------------
# GitHub Copilot CLI
# ---------------------------------------------------------------------------


def emit_copilot(ir: SubagentIR) -> tuple[str, list[ConversionWarning]]:
    warnings: list[ConversionWarning] = []
    # Copilot requires description; fall back to name when the IR carries none
    # so the file still validates against Copilot's parser.
    fm: dict[str, Any] = {"name": ir.name, "description": ir.description or ir.name}
    if not ir.description:
        warnings.append(
            _warn_lossy(
                "description",
                "Copilot requires a description; falling back to the agent name. "
                "Consider adding one.",
            )
        )
    if ir.target:
        fm["target"] = ir.target
    if ir.model:
        fm["model"] = ir.model
    if ir.tools is not None:
        fm["tools"] = [from_canonical(t, "copilot") for t in ir.tools]
    if ir.disable_model_invocation is not None:
        fm["disable-model-invocation"] = ir.disable_model_invocation
    if ir.user_invocable is not None:
        fm["user-invocable"] = ir.user_invocable
    if ir.mcp_servers:
        fm["mcp-servers"] = ir.mcp_servers
    if ir.metadata:
        fm["metadata"] = ir.metadata

    # 30k char body limit per docs
    if len(ir.body) > 30_000:
        warnings.append(
            _warn_lossy(
                "body",
                f"Copilot caps the agent body at 30,000 characters; "
                f"current body is {len(ir.body)}.",
            )
        )

    for field, value, reason in [
        ("disallowed_tools", ir.disallowed_tools, "Copilot has no denylist field"),
        ("permission_mode", ir.permission_mode, "Claude-only"),
        ("sandbox_mode", ir.sandbox_mode, "Codex-only"),
        ("readonly", ir.readonly, "Cursor-only"),
        ("hooks", ir.hooks, "not part of Copilot agent frontmatter"),
    ]:
        if value:
            warnings.append(_warn_dropped(field, "Copilot", reason))

    fm.update(_extras_for(ir, "copilot"))
    return _render_markdown(fm, ir.body), warnings


# ---------------------------------------------------------------------------
# OpenAI Codex CLI
# ---------------------------------------------------------------------------


def emit_codex(ir: SubagentIR) -> tuple[CodexEmission, list[ConversionWarning]]:
    """Emit a Codex agent file plus a config.toml registration fragment.

    Codex's system prompt lives in ``developer_instructions`` rather than a
    markdown body, and Codex has no per-agent tool allowlist — those fields
    surface as warnings.  The fragment is informational: it shows the
    ``[agents.<name>]`` table the user can append to ``~/.codex/config.toml``
    if they want to register the agent globally (Codex also auto-discovers
    agent files in ``~/.codex/agents/``).
    """
    warnings: list[ConversionWarning] = []

    agent: dict[str, Any] = {
        "name": ir.name,
        "developer_instructions": ir.body or "",
    }
    if not ir.body:
        warnings.append(
            _warn_lossy(
                "body",
                "Codex requires `developer_instructions`; emitting an empty string. "
                "Add a system prompt before using this agent.",
            )
        )
    if ir.description is not None:
        agent["description"] = ir.description
    else:
        warnings.append(
            _warn_lossy(
                "description",
                "Codex agents typically include a description; the IR had none.",
            )
        )
    if ir.model:
        agent["model"] = ir.model
    if ir.effort:
        agent["model_reasoning_effort"] = ir.effort
    # Three signals can drive sandbox_mode, in priority order: an explicit
    # ir.sandbox_mode (Codex source), ir.permission_mode (Claude source —
    # the three values that map cleanly), or the tools/readonly allowlist
    # intent (Cursor / Gemini / Copilot source).
    _claude_to_sandbox = {
        "acceptEdits": "workspace-write",
        "readOnly": "read-only",
        "bypassPermissions": "danger-full-access",
    }
    if ir.sandbox_mode:
        agent["sandbox_mode"] = ir.sandbox_mode
    elif ir.permission_mode in _claude_to_sandbox:
        agent["sandbox_mode"] = _claude_to_sandbox[ir.permission_mode]
    elif ir.tools is not None or ir.readonly is True:
        # Map allowlist intent → coarse sandbox mode.
        if ir.readonly is True or (ir.tools is not None and all_read_only(ir.tools)):
            agent["sandbox_mode"] = "read-only"
        else:
            agent["sandbox_mode"] = "workspace-write"
        if ir.tools is not None:
            warnings.append(
                _warn_lossy(
                    "tools",
                    "Codex has no per-agent tool allowlist; collapsed to "
                    f"sandbox_mode={agent['sandbox_mode']!r}. Fine-grained list dropped.",
                )
            )
    if ir.mcp_servers:
        agent["mcp_servers"] = ir.mcp_servers
    if ir.skills is not None:
        agent["skills"] = ir.skills
    if ir.nickname_candidates:
        agent["nickname_candidates"] = ir.nickname_candidates

    # permission_mode only warrants a dropped-warning when it didn't map
    # cleanly to a sandbox_mode value above (i.e. anything outside the
    # _claude_to_sandbox table — `default`, `dontAsk`, `plan`).
    unmapped_permission_mode = (
        ir.permission_mode
        if ir.permission_mode and ir.permission_mode not in _claude_to_sandbox
        else None
    )
    for field, value, reason in [
        ("disallowed_tools", ir.disallowed_tools, "no denylist field"),
        ("permission_mode", unmapped_permission_mode, "Claude-only"),
        ("hooks", ir.hooks, "not part of Codex agent format"),
        ("target", ir.target, "Copilot-only"),
        ("max_turns", ir.max_turns, "no per-agent turn cap; managed via config.toml"),
        ("temperature", ir.temperature, "Codex has no per-agent temperature"),
        ("timeout_mins", ir.timeout_mins, "Codex uses job_max_runtime_seconds globally"),
    ]:
        if value:
            warnings.append(_warn_dropped(field, "Codex", reason))

    agent.update(_extras_for(ir, "codex"))
    agent_toml = tomli_w.dumps(agent)

    # Build the optional config fragment.  Today we only emit one when Codex
    # source extras reference a registered role path — the agent file itself
    # is the canonical content for everything else.  Always include the bare
    # registration so users see the right shape; emitters that need richer
    # fragments can extend this without touching the agent_toml path.
    suggested_filename = f"{ir.name}.toml"
    config_fragment = tomli_w.dumps(
        {
            "agents": {ir.name: {"path": f"~/.codex/agents/{suggested_filename}"}},
        }
    )

    return (
        CodexEmission(
            agent_toml=agent_toml,
            config_fragment=config_fragment,
            suggested_filename=suggested_filename,
        ),
        warnings,
    )
