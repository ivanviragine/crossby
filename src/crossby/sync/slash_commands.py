r"""Convert tool-specific slash commands into one-file skills for other tools.

Several source tools have a ``/<name>`` slash-command primitive that doesn't
exist on the target side of a Crossby sync. Today that includes Claude's
``.claude/commands/<name>.md``, Cursor's ``.cursor/commands/<name>.md``, and
Gemini's ``.gemini/commands/<name>.md``. When syncing one of these to a tool
without a command surface, each command file is wrapped as a single-file
skill at ``<target-skills-dir>/<source>-command-<slug>/SKILL.md`` so the
prompt body survives.

Tool-specific runtime expansion that the target won't honour — Claude's
``$ARGUMENTS`` / ``$1``, ``!`shell```, ``@file-reference``, ``{{template}}``;
Gemini's ``{{args}}`` — is preserved as text and a ``crossby:manual-fix``
block lists the surfaces that need human attention.

This module owns parsing and conversion. Idempotency, write-out, and stale
cleanup happen in the skills writer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from crossby.models.ai import AIToolID
from crossby.sync.agent_models import SkillDefinition, parse_markdown_skill
from crossby.sync.manual_fix import ManualFixNote

# Source-tool → relative directory where its commands live.
_COMMAND_SOURCES: dict[AIToolID, Path] = {
    AIToolID.CLAUDE: Path(".claude") / "commands",
    AIToolID.CURSOR: Path(".cursor") / "commands",
    AIToolID.GEMINI: Path(".gemini") / "commands",
}

# Source-tool → skill-name namespace prefix. Keeps `pr-review.md` from
# colliding across source tools when wrapped as skills under the same target
# (e.g. `claude-command-pr-review` vs `cursor-command-pr-review`).
_SKILL_NAME_PREFIXES: dict[AIToolID, str] = {
    AIToolID.CLAUDE: "claude-command-",
    AIToolID.CURSOR: "cursor-command-",
    AIToolID.GEMINI: "gemini-command-",
}

# Runtime constructs that need a manual-review note when copied verbatim into
# a target tool's skill body. Empty tuple means "no tool-specific runtime
# expansion known; only the generic slash-command caveat is emitted."
_RUNTIME_PATTERNS_BY_TOOL: dict[AIToolID, tuple[tuple[re.Pattern[str], str, str], ...]] = {
    AIToolID.CLAUDE: (
        (
            re.compile(r"\$(?:ARGUMENTS|\d+)\b"),
            "argument-placeholders",
            "Source uses Claude argument placeholders (`$ARGUMENTS` / `$1`). The target "
            "tool does not interpolate them — rewrite as a natural-language instruction "
            "or remove.",
        ),
        (
            re.compile(r"!\s*`"),
            "shell-interpolation",
            "Source uses Claude `!`shell`` interpolation to run a command and embed its "
            "output. The target tool will not execute it; replace with explicit "
            "instructions to run the command when needed.",
        ),
        (
            re.compile(r"(?:^|\s)@[\w./~:-]+"),
            "file-references",
            "Source uses Claude `@file` automatic file expansion. The target tool will "
            "not fetch those files; verify whether they should be read explicitly.",
        ),
        (
            re.compile(r"\{\{[^}]+\}\}"),
            "template-variables",
            "Source uses `{{template}}` placeholders. The target tool does not "
            "interpolate them; rewrite into natural-language instructions.",
        ),
    ),
    AIToolID.CURSOR: (),
    AIToolID.GEMINI: (
        (
            re.compile(r"\{\{\s*args?\s*\}\}", re.IGNORECASE),
            "gemini-args-template",
            "Source uses Gemini `{{args}}` argument template. The target tool does not "
            "interpolate it; rewrite into natural-language instructions.",
        ),
    ),
}


def discover_commands(project_root: Path, source_tool: AIToolID) -> list[Path]:
    """Return every command-file under the given source tool's commands dir.

    Sorted for deterministic output. Empty when the source tool has no command
    primitive or its directory is missing.
    """
    rel = _COMMAND_SOURCES.get(source_tool)
    if rel is None:
        return []
    root = project_root / rel
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.md") if p.is_file())


def command_skill_name(command_path: Path, *, root: Path, source_tool: AIToolID) -> str:
    """Stable skill name for a command file scoped to its source tool.

    Multi-segment commands (``.claude/commands/release/cut.md``) become
    ``claude-command-release-cut`` so the ``/release/cut`` semantics survive
    in the skill name. The prefix is per source tool (see
    :data:`_SKILL_NAME_PREFIXES`).
    """
    relative = command_path.relative_to(root)
    stem_parts = list(relative.with_suffix("").parts)
    prefix = _SKILL_NAME_PREFIXES.get(source_tool, f"{source_tool}-command-")
    return prefix + "-".join(stem_parts)


def detect_runtime_caveats(
    template_body: str,
    *,
    source_tool: AIToolID,
) -> list[ManualFixNote]:
    """Identify the source-tool's runtime constructs in ``template_body``."""
    patterns = _RUNTIME_PATTERNS_BY_TOOL.get(source_tool, ())
    notes: list[ManualFixNote] = []
    for pattern, category, message in patterns:
        if pattern.search(template_body):
            notes.append(ManualFixNote(category=category, message=message))
    return notes


def command_to_skill(
    command_path: Path,
    *,
    root: Path,
    source_tool: AIToolID,
) -> SkillDefinition:
    """Wrap a tool-specific command file as a single-file skill definition.

    The returned :class:`SkillDefinition` carries:

    - ``name``: ``<source>-command-<slug>`` derived from the path under the
      tool's commands directory.
    - ``description``: the frontmatter ``description`` if present, else a
      generic "Run the migrated <tool> ``<name>`` slash command".
    - ``body``: a short skill preamble plus the original command template
      verbatim under a ``## Command Template`` heading.
    - ``manual_fix_notes``: a generic "this was a slash command" note plus
      tool-specific runtime caveats from :data:`_RUNTIME_PATTERNS_BY_TOOL`.
    """
    raw = command_path.read_text(encoding="utf-8")
    parsed = parse_markdown_skill(raw, fallback_name=command_path.stem)
    name = command_skill_name(command_path, root=root, source_tool=source_tool)
    relative = command_path.relative_to(root).with_suffix("").as_posix()
    display = str(source_tool).title()
    description = parsed.description or f"Run the migrated {display} `{relative}` slash command."

    body = (
        f"# {name}\n\n"
        f"Use this skill when the user asks to run the migrated {display} "
        f"`{relative}` slash command.\n\n"
        "## Command Template\n\n"
        f"{parsed.body.strip() or 'No command body was found in the source file.'}\n"
    )

    notes: list[ManualFixNote] = [
        ManualFixNote(
            category="slash-command",
            message=(
                f"Source was the {display} slash command `/{relative}`. The "
                "target tool does not invoke it via slash; the user should ask the "
                "agent to perform this skill explicitly."
            ),
        )
    ]
    notes.extend(detect_runtime_caveats(parsed.body, source_tool=source_tool))

    return SkillDefinition(
        name=name,
        description=description,
        body=body,
        manual_fix_notes=tuple(notes),
    )


def iter_command_skills(
    project_root: Path,
    *,
    source_tools: Iterable[AIToolID] | None = None,
) -> Iterable[tuple[Path, AIToolID, SkillDefinition]]:
    """Yield ``(source_path, source_tool, skill_definition)`` for every command.

    When ``source_tools`` is None, every tool in :data:`_COMMAND_SOURCES` is
    scanned and only those whose commands directory exists yield results.
    """
    tools = tuple(source_tools) if source_tools is not None else tuple(_COMMAND_SOURCES)
    for tool in tools:
        rel = _COMMAND_SOURCES.get(tool)
        if rel is None:
            continue
        root = project_root / rel
        for command_path in discover_commands(project_root, tool):
            yield command_path, tool, command_to_skill(command_path, root=root, source_tool=tool)


__all__ = [
    "command_skill_name",
    "command_to_skill",
    "detect_runtime_caveats",
    "discover_commands",
    "iter_command_skills",
]
