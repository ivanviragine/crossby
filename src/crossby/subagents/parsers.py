"""Per-tool parsers — read a tool-specific subagent file and emit a SubagentIR.

Each parser accepts the raw file text and returns ``(SubagentIR, warnings)``.
Parsers are deliberately permissive: missing optional fields are fine, and
unknown fields are preserved into ``IR.extras`` so a same-tool round trip
loses nothing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import yaml

from crossby.subagents.ir import ConversionWarning, SubagentIR, WarningSeverity
from crossby.subagents.tool_map import to_canonical


def parse(
    tool: str, content: str, source_path: Path | None = None
) -> tuple[SubagentIR, list[ConversionWarning]]:
    """Dispatch to the per-tool parser.

    Raises ``ValueError`` for unknown tools; that's a programming error, not
    a user-fixable input issue.
    """
    parsers = {
        "claude": parse_claude,
        "cursor": parse_cursor,
        "gemini": parse_gemini,
        "copilot": parse_copilot,
        "codex": parse_codex,
    }
    fn = parsers.get(tool)
    if fn is None:
        raise ValueError(f"unknown source tool: {tool!r}")
    return fn(content, source_path)


# ---------------------------------------------------------------------------
# Markdown frontmatter helpers
# ---------------------------------------------------------------------------


def _split_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from markdown body. Empty dict if no frontmatter."""
    if not content.startswith("---\n"):
        return {}, content
    end = content.find("\n---\n", 4)
    if end == -1:
        return {}, content
    try:
        raw = yaml.safe_load(content[4:end])
    except yaml.YAMLError:
        return {}, content
    if not isinstance(raw, dict):
        return {}, content
    return raw, content[end + 5 :]


def _normalize_tools(value: Any) -> list[str] | None:
    """Frontmatter ``tools`` can be a YAML list or a comma-separated string."""
    if value is None:
        return None
    if isinstance(value, str):
        items = [t.strip() for t in value.split(",") if t.strip()]
        return items or None
    if isinstance(value, list):
        return [str(t) for t in value]
    return None


def _name_from_path(source_path: Path | None, fallback: str = "agent") -> str:
    if source_path is None:
        return fallback
    name = source_path.name
    # Strip both .md and .agent.md
    if name.endswith(".agent.md"):
        return name[: -len(".agent.md")]
    return source_path.stem


# ---------------------------------------------------------------------------
# Claude Code (.claude/agents/<name>.md)
# ---------------------------------------------------------------------------

_CLAUDE_KNOWN_FIELDS = {
    "name",
    "description",
    "tools",
    "disallowedTools",
    "model",
    "permissionMode",
    "maxTurns",
    "skills",
    "mcpServers",
    "hooks",
    "memory",
    "background",
    "effort",
    "isolation",
    "color",
    "initialPrompt",
}


def parse_claude(
    content: str, source_path: Path | None = None
) -> tuple[SubagentIR, list[ConversionWarning]]:
    fm, body = _split_frontmatter(content)
    warnings: list[ConversionWarning] = []

    raw_tools = _normalize_tools(fm.get("tools"))
    raw_disallowed = _normalize_tools(fm.get("disallowedTools"))

    extras = {k: v for k, v in fm.items() if k not in _CLAUDE_KNOWN_FIELDS}

    ir = SubagentIR(
        name=str(fm.get("name") or _name_from_path(source_path)),
        description=fm.get("description"),
        body=body,
        model=fm.get("model"),
        tools=[to_canonical(t, "claude") for t in raw_tools] if raw_tools else None,
        disallowed_tools=(
            [to_canonical(t, "claude") for t in raw_disallowed] if raw_disallowed else None
        ),
        mcp_servers=fm.get("mcpServers"),
        max_turns=fm.get("maxTurns"),
        effort=fm.get("effort"),
        permission_mode=fm.get("permissionMode"),
        background=fm.get("background"),
        color=fm.get("color"),
        isolation=fm.get("isolation"),
        memory=fm.get("memory"),
        skills=fm.get("skills"),
        hooks=fm.get("hooks"),
        initial_prompt=fm.get("initialPrompt"),
        source_tool="claude",
        source_path=str(source_path) if source_path else None,
        extras=extras,
    )
    return ir, warnings


# ---------------------------------------------------------------------------
# Cursor (.cursor/agents/<name>.md)
# ---------------------------------------------------------------------------

_CURSOR_KNOWN_FIELDS = {"name", "description", "model", "readonly", "is_background"}


def parse_cursor(
    content: str, source_path: Path | None = None
) -> tuple[SubagentIR, list[ConversionWarning]]:
    fm, body = _split_frontmatter(content)
    extras = {k: v for k, v in fm.items() if k not in _CURSOR_KNOWN_FIELDS}
    ir = SubagentIR(
        name=str(fm.get("name") or _name_from_path(source_path)),
        description=fm.get("description"),
        body=body,
        model=fm.get("model"),
        readonly=fm.get("readonly"),
        is_background=fm.get("is_background"),
        source_tool="cursor",
        source_path=str(source_path) if source_path else None,
        extras=extras,
    )
    return ir, []


# ---------------------------------------------------------------------------
# Gemini CLI (.gemini/agents/<name>.md)
# ---------------------------------------------------------------------------

_GEMINI_KNOWN_FIELDS = {
    "name",
    "description",
    "kind",
    "tools",
    "mcpServers",
    "model",
    "temperature",
    "max_turns",
    "timeout_mins",
}


def parse_gemini(
    content: str, source_path: Path | None = None
) -> tuple[SubagentIR, list[ConversionWarning]]:
    fm, body = _split_frontmatter(content)
    warnings: list[ConversionWarning] = []
    raw_tools = _normalize_tools(fm.get("tools"))
    extras = {k: v for k, v in fm.items() if k not in _GEMINI_KNOWN_FIELDS}

    if fm.get("kind") == "remote":
        warnings.append(
            ConversionWarning(
                field="kind",
                severity=WarningSeverity.LOSSY,
                message=(
                    "kind=remote uses Gemini's A2A protocol — "
                    "other tools cannot reproduce remote-agent dispatch."
                ),
            )
        )

    ir = SubagentIR(
        name=str(fm.get("name") or _name_from_path(source_path)),
        description=fm.get("description"),
        body=body,
        model=fm.get("model"),
        tools=[to_canonical(t, "gemini") for t in raw_tools] if raw_tools else None,
        mcp_servers=fm.get("mcpServers"),
        temperature=fm.get("temperature"),
        max_turns=fm.get("max_turns"),
        timeout_mins=fm.get("timeout_mins"),
        kind=fm.get("kind"),
        source_tool="gemini",
        source_path=str(source_path) if source_path else None,
        extras=extras,
    )
    return ir, warnings


# ---------------------------------------------------------------------------
# GitHub Copilot CLI (.github/agents/<name>.agent.md)
# ---------------------------------------------------------------------------

_COPILOT_KNOWN_FIELDS = {
    "name",
    "description",
    "target",
    "tools",
    "model",
    "disable-model-invocation",
    "user-invocable",
    "mcp-servers",
    "metadata",
}


def parse_copilot(
    content: str, source_path: Path | None = None
) -> tuple[SubagentIR, list[ConversionWarning]]:
    fm, body = _split_frontmatter(content)
    raw_tools = _normalize_tools(fm.get("tools"))
    extras = {k: v for k, v in fm.items() if k not in _COPILOT_KNOWN_FIELDS}

    ir = SubagentIR(
        name=str(fm.get("name") or _name_from_path(source_path)),
        description=fm.get("description"),
        body=body,
        model=fm.get("model"),
        tools=[to_canonical(t, "copilot") for t in raw_tools] if raw_tools else None,
        mcp_servers=fm.get("mcp-servers"),
        target=fm.get("target"),
        user_invocable=fm.get("user-invocable"),
        disable_model_invocation=fm.get("disable-model-invocation"),
        metadata=fm.get("metadata"),
        source_tool="copilot",
        source_path=str(source_path) if source_path else None,
        extras=extras,
    )
    return ir, []


# ---------------------------------------------------------------------------
# OpenAI Codex CLI (~/.codex/agents/<name>.toml)
# ---------------------------------------------------------------------------

_CODEX_KNOWN_FIELDS = {
    "name",
    "description",
    "developer_instructions",
    "nickname_candidates",
    "model",
    "model_reasoning_effort",
    "sandbox_mode",
    "mcp_servers",
    "skills",
}


def parse_codex(
    content: str, source_path: Path | None = None
) -> tuple[SubagentIR, list[ConversionWarning]]:
    """Parse a Codex agent .toml file.

    Codex stores the system prompt in ``developer_instructions`` (a TOML
    string) instead of a markdown body — we put it into IR.body so emitters
    don't need to special-case the source format.
    """
    warnings: list[ConversionWarning] = []
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid Codex agent TOML: {exc}") from exc

    extras = {k: v for k, v in data.items() if k not in _CODEX_KNOWN_FIELDS}

    ir = SubagentIR(
        name=str(data.get("name") or _name_from_path(source_path)),
        description=data.get("description"),
        body=str(data.get("developer_instructions", "")),
        model=data.get("model"),
        effort=data.get("model_reasoning_effort"),
        sandbox_mode=data.get("sandbox_mode"),
        mcp_servers=data.get("mcp_servers"),
        skills=data.get("skills"),
        nickname_candidates=data.get("nickname_candidates"),
        source_tool="codex",
        source_path=str(source_path) if source_path else None,
        extras=extras,
    )
    return ir, warnings
