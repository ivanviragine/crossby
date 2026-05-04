"""Canonical agent and skill domain models + parse/render per schema.

Crossby has historically treated agent and skill directories as opaque
trees of files and synced them via directory-level symlinks. That works
when source and target accept the same on-disk schema (Claude, Cursor,
Gemini, Copilot all use ``markdown + YAML frontmatter``); it breaks when
the target uses a different schema (Codex agents are TOML).

This module owns the cross-tool model:

- :class:`AgentDefinition` and :class:`SkillDefinition` are tool-neutral
  representations a writer can render to any supported on-disk shape.
- :func:`parse_*` reads a source file into the canonical form, normalising
  field names and inferring a :class:`AgentSchema` / :class:`SkillSchema`
  marker.
- :func:`render_*` renders the canonical form into a tool-specific shape.
- :func:`agent_schema_for` / :func:`skill_schema_for` answer the
  schema-compatibility question writers use to decide whether a symlink
  preserves fidelity or whether they need to translate.

Lossy fields (e.g. Claude's ``allowed-tools`` for skills, or
``permissionMode: plan``) are returned as ``manual_fix_notes`` so the
caller can append a :mod:`crossby.sync.manual_fix` block to the rendered
artifact.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import tomli_w
import yaml

from crossby.models.ai import AIToolID, EffortLevel
from crossby.sync.manual_fix import ManualFixNote, append_manual_fix_block
from crossby.sync.translation import (
    CLAUDE_PERMISSION_MODES_UNMAPPED,
    map_effort_claude_to_codex,
    map_effort_codex_to_claude,
    map_model_claude_to_codex,
    map_model_codex_to_claude,
    map_permission_mode_claude_to_codex,
    map_permission_mode_codex_to_claude,
)


class AgentSchema(StrEnum):
    """On-disk shape an agent file uses."""

    MARKDOWN = "markdown"  # Claude, Cursor, Gemini, Copilot — YAML frontmatter + body
    TOML = "toml"          # Codex — TOML with developer_instructions multi-line


class SkillSchema(StrEnum):
    """On-disk shape a skill file uses. All supported tools agree today."""

    MARKDOWN = "markdown"  # SKILL.md with YAML frontmatter + body


# Per-tool schema markers.
AGENT_SCHEMA_BY_TOOL: dict[AIToolID, AgentSchema] = {
    AIToolID.CLAUDE: AgentSchema.MARKDOWN,
    AIToolID.CURSOR: AgentSchema.MARKDOWN,
    AIToolID.GEMINI: AgentSchema.MARKDOWN,
    AIToolID.COPILOT: AgentSchema.MARKDOWN,
    AIToolID.CODEX: AgentSchema.TOML,
}

SKILL_SCHEMA_BY_TOOL: dict[AIToolID, SkillSchema] = {
    AIToolID.CLAUDE: SkillSchema.MARKDOWN,
    AIToolID.CURSOR: SkillSchema.MARKDOWN,
    AIToolID.GEMINI: SkillSchema.MARKDOWN,
    AIToolID.COPILOT: SkillSchema.MARKDOWN,
    AIToolID.CODEX: SkillSchema.MARKDOWN,
}


def agent_schema_for(tool: AIToolID) -> AgentSchema:
    return AGENT_SCHEMA_BY_TOOL[tool]


def skill_schema_for(tool: AIToolID) -> SkillSchema:
    return SKILL_SCHEMA_BY_TOOL[tool]


def agents_schema_compatible(source: AIToolID, target: AIToolID) -> bool:
    """True when source and target accept the same agent on-disk shape."""
    return AGENT_SCHEMA_BY_TOOL[source] == AGENT_SCHEMA_BY_TOOL[target]


def skills_schema_compatible(source: AIToolID, target: AIToolID) -> bool:
    """True when source and target accept the same skill on-disk shape."""
    return SKILL_SCHEMA_BY_TOOL[source] == SKILL_SCHEMA_BY_TOOL[target]


# ---------------------------------------------------------------------------
# Canonical models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentDefinition:
    """Tool-neutral representation of one agent.

    ``permission_mode`` carries the *Claude* vocabulary (acceptEdits,
    readOnly, plan, etc.); the renderer translates to Codex's sandbox_mode
    when emitting TOML. ``model`` carries whatever family/id the source
    used; cross-provider mapping happens at render time.
    """

    name: str
    description: str
    body: str = ""
    model: str | None = None
    effort: EffortLevel | None = None
    permission_mode: str | None = None
    skills: tuple[str, ...] = ()
    tools_allow: tuple[str, ...] = ()
    tools_deny: tuple[str, ...] = ()
    extra_frontmatter: dict[str, Any] = field(default_factory=dict)
    manual_fix_notes: tuple[ManualFixNote, ...] = ()

    def with_notes(self, notes: Sequence[ManualFixNote]) -> AgentDefinition:
        if not notes:
            return self
        return AgentDefinition(
            name=self.name,
            description=self.description,
            body=self.body,
            model=self.model,
            effort=self.effort,
            permission_mode=self.permission_mode,
            skills=self.skills,
            tools_allow=self.tools_allow,
            tools_deny=self.tools_deny,
            extra_frontmatter=self.extra_frontmatter,
            manual_fix_notes=tuple([*self.manual_fix_notes, *notes]),
        )


@dataclass(frozen=True)
class SkillDefinition:
    """Tool-neutral representation of one skill (one SKILL.md per skill)."""

    name: str
    description: str
    body: str = ""
    allowed_tools: tuple[str, ...] = ()
    extra_frontmatter: dict[str, Any] = field(default_factory=dict)
    manual_fix_notes: tuple[ManualFixNote, ...] = ()

    def with_notes(self, notes: Sequence[ManualFixNote]) -> SkillDefinition:
        if not notes:
            return self
        return SkillDefinition(
            name=self.name,
            description=self.description,
            body=self.body,
            allowed_tools=self.allowed_tools,
            extra_frontmatter=self.extra_frontmatter,
            manual_fix_notes=tuple([*self.manual_fix_notes, *notes]),
        )


# ---------------------------------------------------------------------------
# Parse: markdown + YAML frontmatter
# ---------------------------------------------------------------------------


_FM_DELIM = "---\n"


def _split_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Pull a YAML mapping out of a markdown file's leading frontmatter.

    Returns ``({}, content)`` when there's no parseable frontmatter, so
    callers can treat the file body as the whole document.
    """
    if not content.startswith(_FM_DELIM):
        return {}, content
    end = content.find("\n---\n", len(_FM_DELIM))
    if end == -1:
        return {}, content
    raw = content[len(_FM_DELIM) : end]
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError:
        return {}, content
    if not isinstance(loaded, dict):
        return {}, content
    body = content[end + len("\n---\n") :]
    return dict(loaded), body


_AGENT_KNOWN_FIELDS = frozenset(
    {
        "name",
        "description",
        "model",
        "effort",
        "permissionMode",
        "permission_mode",
        "skills",
        "tools",
        "disallowedTools",
        "disallowed_tools",
    }
)


def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        # Allow comma-separated singular strings.
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _coerce_effort(value: Any) -> EffortLevel | None:
    if value is None:
        return None
    if isinstance(value, EffortLevel):
        return value
    try:
        return EffortLevel(str(value))
    except ValueError:
        return None


def parse_markdown_agent(content: str, *, fallback_name: str = "agent") -> AgentDefinition:
    """Parse a Claude-style markdown+frontmatter agent file."""
    fm, body = _split_frontmatter(content)
    name = str(fm.get("name") or fallback_name).strip() or fallback_name
    description = str(fm.get("description") or "").strip()

    permission_mode_value = fm.get("permissionMode")
    if permission_mode_value is None:
        permission_mode_value = fm.get("permission_mode")
    permission_mode = (
        str(permission_mode_value).strip() if permission_mode_value is not None else None
    )

    tools_deny_raw = fm.get("disallowedTools")
    if tools_deny_raw is None:
        tools_deny_raw = fm.get("disallowed_tools")

    extra = {k: v for k, v in fm.items() if k not in _AGENT_KNOWN_FIELDS}

    return AgentDefinition(
        name=name,
        description=description,
        body=body,
        model=str(fm.get("model")).strip() if fm.get("model") is not None else None,
        effort=_coerce_effort(fm.get("effort")),
        permission_mode=permission_mode,
        skills=_coerce_str_tuple(fm.get("skills")),
        tools_allow=_coerce_str_tuple(fm.get("tools")),
        tools_deny=_coerce_str_tuple(tools_deny_raw),
        extra_frontmatter=extra,
    )


_SKILL_KNOWN_FIELDS = frozenset(
    {"name", "description", "allowed-tools", "allowed_tools"}
)


def parse_markdown_skill(content: str, *, fallback_name: str = "skill") -> SkillDefinition:
    """Parse a SKILL.md file into a canonical SkillDefinition."""
    fm, body = _split_frontmatter(content)
    name = str(fm.get("name") or fallback_name).strip() or fallback_name
    description = str(fm.get("description") or "").strip()
    allowed_tools_raw = fm.get("allowed-tools")
    if allowed_tools_raw is None:
        allowed_tools_raw = fm.get("allowed_tools")
    extra = {k: v for k, v in fm.items() if k not in _SKILL_KNOWN_FIELDS}

    return SkillDefinition(
        name=name,
        description=description,
        body=body,
        allowed_tools=_coerce_str_tuple(allowed_tools_raw),
        extra_frontmatter=extra,
    )


# ---------------------------------------------------------------------------
# Parse: TOML (Codex agents)
# ---------------------------------------------------------------------------


_CODEX_AGENT_KNOWN_FIELDS = frozenset(
    {
        "name",
        "description",
        "model",
        "model_reasoning_effort",
        "sandbox_mode",
        "developer_instructions",
    }
)


def parse_toml_agent(content: str, *, fallback_name: str = "agent") -> AgentDefinition:
    """Parse a Codex-style TOML agent file."""
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return AgentDefinition(name=fallback_name, description="", body=content)

    name = str(data.get("name") or fallback_name).strip() or fallback_name
    description = str(data.get("description") or "").strip()
    body = str(data.get("developer_instructions") or "")

    sandbox_mode = data.get("sandbox_mode")
    permission_mode = (
        map_permission_mode_codex_to_claude(str(sandbox_mode)) if sandbox_mode else None
    )

    extra = {k: v for k, v in data.items() if k not in _CODEX_AGENT_KNOWN_FIELDS}

    return AgentDefinition(
        name=name,
        description=description,
        body=body,
        model=str(data.get("model")) if data.get("model") is not None else None,
        effort=_coerce_effort(data.get("model_reasoning_effort")),
        permission_mode=permission_mode,
        extra_frontmatter=extra,
    )


# ---------------------------------------------------------------------------
# Render: markdown
# ---------------------------------------------------------------------------


def _yaml_dump(values: dict[str, Any]) -> str:
    # ``sort_keys=False`` preserves insertion order so renders stay stable
    # under round-trips and humans see the most-important fields first.
    return yaml.dump(values, sort_keys=False, default_flow_style=False).rstrip() + "\n"


def render_markdown_agent(
    definition: AgentDefinition,
    *,
    target_tool: AIToolID,
) -> str:
    """Render an AgentDefinition as Claude-style markdown+frontmatter.

    The same shape works for Cursor/Gemini/Copilot. ``target_tool`` is
    accepted so future per-tool tweaks (e.g. Copilot's lower-case tool
    names) can land here without churning the call site.
    """
    fm: dict[str, Any] = {"name": definition.name, "description": definition.description}
    if definition.model is not None:
        # When source was a Codex agent, the canonical model name is GPT-shaped;
        # translate to Claude default for markdown targets.
        if target_tool == AIToolID.CLAUDE:
            fm["model"] = map_model_codex_to_claude(definition.model)
        else:
            fm["model"] = definition.model
    if definition.effort is not None:
        fm["effort"] = definition.effort.value
    if definition.permission_mode is not None:
        fm["permissionMode"] = definition.permission_mode
    if definition.skills:
        fm["skills"] = list(definition.skills)
    if definition.tools_allow:
        fm["tools"] = list(definition.tools_allow)
    if definition.tools_deny:
        fm["disallowedTools"] = list(definition.tools_deny)
    fm.update(definition.extra_frontmatter)

    rendered = f"---\n{_yaml_dump(fm)}---\n{definition.body.lstrip()}"
    if definition.manual_fix_notes:
        rendered = append_manual_fix_block(rendered, list(definition.manual_fix_notes))
    return rendered


def render_markdown_skill(definition: SkillDefinition) -> str:
    """Render a SkillDefinition as SKILL.md.

    Skills are markdown across all supported tools today. Lossy fields
    (e.g. ``allowed-tools`` on a non-Claude target) come in via
    ``manual_fix_notes`` and end up as a block at the bottom; the writer
    decides which fields are lossy for which target.
    """
    fm: dict[str, Any] = {"name": definition.name, "description": definition.description}
    if definition.allowed_tools:
        fm["allowed-tools"] = list(definition.allowed_tools)
    fm.update(definition.extra_frontmatter)

    rendered = f"---\n{_yaml_dump(fm)}---\n{definition.body.lstrip()}"
    if definition.manual_fix_notes:
        rendered = append_manual_fix_block(rendered, list(definition.manual_fix_notes))
    return rendered


# ---------------------------------------------------------------------------
# Render: TOML (Codex agents)
# ---------------------------------------------------------------------------


def render_toml_agent(definition: AgentDefinition) -> str:
    """Render an AgentDefinition as a Codex agent TOML file.

    Maps Claude vocabulary (permissionMode, claude-* model id, max effort)
    to Codex equivalents (sandbox_mode, gpt-5.4 family, xhigh) using
    :mod:`crossby.sync.translation`. Body content goes into the
    ``developer_instructions`` multi-line string. Lossy fields surface as
    :class:`ManualFixNote` items appended to the body.

    Always emits ``name``, ``description``, and ``developer_instructions``
    (Codex requires those three) — using ``""`` when the source was empty.
    Validators downstream still flag empty values as ``error``.
    """
    payload: dict[str, Any] = {
        "name": definition.name,
        "description": definition.description,
    }

    notes: list[ManualFixNote] = list(definition.manual_fix_notes)

    if definition.model is not None:
        mapped_model = map_model_claude_to_codex(definition.model)
        payload["model"] = mapped_model
        if mapped_model == definition.model and not mapped_model.startswith("gpt-"):
            notes.append(
                ManualFixNote(
                    category="model",
                    message=(
                        f"No known Codex equivalent for model `{definition.model}`. "
                        "The id was passed through unchanged; pick a Codex model manually."
                    ),
                )
            )

    if definition.effort is not None:
        codex_effort = map_effort_claude_to_codex(definition.model, definition.effort)
        if codex_effort is not None:
            payload["model_reasoning_effort"] = codex_effort.value

    if definition.permission_mode is not None:
        sandbox = map_permission_mode_claude_to_codex(definition.permission_mode)
        if sandbox is not None:
            payload["sandbox_mode"] = sandbox
        elif definition.permission_mode in CLAUDE_PERMISSION_MODES_UNMAPPED:
            notes.append(
                ManualFixNote(
                    category="permissionMode",
                    message=(
                        f"Claude `permissionMode: {definition.permission_mode}` has no Codex "
                        "equivalent. Pick `sandbox_mode`, `[permissions]`, or app-level filters "
                        "manually."
                    ),
                )
            )

    extra_lossy: list[str] = []
    if definition.skills:
        notes.append(
            ManualFixNote(
                category="skills",
                message=(
                    "Source `skills` declared preload semantics that Codex does not honour at "
                    "spawn time. Listed skills are: "
                    + ", ".join(f"`{s}`" for s in definition.skills)
                    + ". Verify Codex discovers them at runtime."
                ),
            )
        )
    if definition.tools_allow:
        extra_lossy.append(
            "Source `tools` allow-list ("
            + ", ".join(f"`{t}`" for t in definition.tools_allow)
            + ") was preserved as guidance, not as a Codex permission boundary."
        )
    if definition.tools_deny:
        extra_lossy.append(
            "Source `disallowedTools` deny-list ("
            + ", ".join(f"`{t}`" for t in definition.tools_deny)
            + ") was preserved as guidance, not as a Codex permission boundary."
        )
    for message in extra_lossy:
        notes.append(ManualFixNote(category="tools", message=message))

    body = definition.body.rstrip()
    if notes:
        body = append_manual_fix_block(body, notes)
    if not body:
        body = ""

    payload["developer_instructions"] = body

    return tomli_w.dumps(payload)


# ---------------------------------------------------------------------------
# High-level translate helpers
# ---------------------------------------------------------------------------


def translate_skill_for_target(
    definition: SkillDefinition, target: AIToolID
) -> SkillDefinition:
    """Annotate a SkillDefinition with manual-fix notes for ``target``.

    ``allowed-tools`` is a Claude concept; for any non-Claude target, surface
    it as guidance instead of a permission boundary. Other tools today
    accept the same SKILL.md shape so no field rewriting is needed.
    """
    notes: list[ManualFixNote] = []
    if definition.allowed_tools and target != AIToolID.CLAUDE:
        notes.append(
            ManualFixNote(
                category="allowed-tools",
                message=(
                    f"Source `allowed-tools` ("
                    + ", ".join(f"`{t}`" for t in definition.allowed_tools)
                    + f") was kept in frontmatter for reference, but {target} does not enforce "
                    "it. Translate to a tool-specific permission mechanism if hard enforcement "
                    "is needed."
                ),
            )
        )
    return definition.with_notes(notes)


def parse_agent_file(path: Path, *, source_tool: AIToolID) -> AgentDefinition:
    """Convenience parser that picks the right schema by ``source_tool``."""
    content = path.read_text(encoding="utf-8")
    fallback = path.stem
    if AGENT_SCHEMA_BY_TOOL[source_tool] == AgentSchema.TOML:
        return parse_toml_agent(content, fallback_name=fallback)
    return parse_markdown_agent(content, fallback_name=fallback)


def parse_skill_file(path: Path, *, source_tool: AIToolID) -> SkillDefinition:
    """Convenience parser for SKILL.md (markdown for every tool today)."""
    _ = source_tool  # all tools use markdown skills
    content = path.read_text(encoding="utf-8")
    fallback = path.parent.name if path.name == "SKILL.md" else path.stem
    return parse_markdown_skill(content, fallback_name=fallback)


__all__ = [
    "AGENT_SCHEMA_BY_TOOL",
    "SKILL_SCHEMA_BY_TOOL",
    "AgentDefinition",
    "AgentSchema",
    "SkillDefinition",
    "SkillSchema",
    "agent_schema_for",
    "agents_schema_compatible",
    "parse_agent_file",
    "parse_markdown_agent",
    "parse_markdown_skill",
    "parse_skill_file",
    "parse_toml_agent",
    "render_markdown_agent",
    "render_markdown_skill",
    "render_toml_agent",
    "skill_schema_for",
    "skills_schema_compatible",
    "translate_skill_for_target",
]
