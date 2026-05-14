"""Canonical skill domain model + parse/render.

Skills are markdown across every supported tool, so the cross-tool concern
is just lossy-field annotation — Claude's ``allowed-tools`` survives
verbatim in frontmatter but only Claude actually enforces it. This module
provides:

- :class:`SkillDefinition` — tool-neutral SKILL.md representation
- :func:`parse_markdown_skill` / :func:`render_markdown_skill`
- :func:`translate_skill_for_target` — attaches manual-fix notes for
  Claude-only fields when the target tool isn't Claude

Agent cross-tool translation lives in :mod:`crossby.subagents` (rich
``SubagentIR`` with per-tool parsers and emitters, structured
``ConversionWarning`` severity, and a Codex emitter that returns the
agent TOML plus a ``[agents.<name>]`` config fragment). The legacy agent
half of this module has been retired; see :mod:`crossby.sync.agents` for
the sync writers that delegate to ``subagents.api.convert``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from crossby.models.ai import AIToolID
from crossby.sync.manual_fix import ManualFixNote, append_manual_fix_block


class SkillSchema(StrEnum):
    """On-disk shape a skill file uses. All supported tools agree today."""

    MARKDOWN = "markdown"  # SKILL.md with YAML frontmatter + body


# Per-tool schema markers. Kept for API symmetry — all tools use markdown,
# so :func:`skills_schema_compatible` is always ``True``.
SKILL_SCHEMA_BY_TOOL: dict[AIToolID, SkillSchema] = {
    AIToolID.CLAUDE: SkillSchema.MARKDOWN,
    AIToolID.CURSOR: SkillSchema.MARKDOWN,
    AIToolID.GEMINI: SkillSchema.MARKDOWN,
    AIToolID.COPILOT: SkillSchema.MARKDOWN,
    AIToolID.CODEX: SkillSchema.MARKDOWN,
}


def skill_schema_for(tool: AIToolID) -> SkillSchema:
    return SKILL_SCHEMA_BY_TOOL[tool]


def skills_schema_compatible(source: AIToolID, target: AIToolID) -> bool:
    """True when source and target accept the same skill on-disk shape."""
    return SKILL_SCHEMA_BY_TOOL[source] == SKILL_SCHEMA_BY_TOOL[target]


# ---------------------------------------------------------------------------
# Canonical model
# ---------------------------------------------------------------------------


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
# Parse + render
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


_SKILL_KNOWN_FIELDS = frozenset(
    {"name", "description", "allowed-tools", "allowed_tools"}
)


def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


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


def _yaml_dump(values: dict[str, Any]) -> str:
    # ``sort_keys=False`` preserves insertion order so renders stay stable
    # under round-trips and humans see the most-important fields first.
    return yaml.dump(values, sort_keys=False, default_flow_style=False).rstrip() + "\n"


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


def parse_skill_file(path: Path, *, source_tool: AIToolID) -> SkillDefinition:
    """Convenience parser for SKILL.md (markdown for every tool today)."""
    _ = source_tool  # all tools use markdown skills
    content = path.read_text(encoding="utf-8")
    fallback = path.parent.name if path.name == "SKILL.md" else path.stem
    return parse_markdown_skill(content, fallback_name=fallback)


__all__ = [
    "SKILL_SCHEMA_BY_TOOL",
    "SkillDefinition",
    "SkillSchema",
    "parse_markdown_skill",
    "parse_skill_file",
    "render_markdown_skill",
    "skill_schema_for",
    "skills_schema_compatible",
    "translate_skill_for_target",
]
