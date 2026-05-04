r"""Convert Claude slash commands into Codex/Cursor/etc one-file skills.

Claude has ``.claude/commands/<name>.md`` slash commands — a markdown file
that the user invokes by typing ``/<name>``. No other supported tool has
the same primitive. When syncing Claude → another tool with the
``translate`` skills strategy, each command file is wrapped as a
single-file skill at ``<target-skills-dir>/claude-command-<name>/SKILL.md``
so the prompt body survives. Tool-specific runtime expansion that the
target won't honour — ``$ARGUMENTS`` / ``$1``, ``!`shell```,
``@file-reference``, ``{{template}}`` — is preserved as text and a
``crossby:manual-fix`` block lists the surfaces that need human
attention.

This module owns parsing and conversion. Idempotency, write-out, and
stale cleanup happen in the skills writer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from crossby.sync.agent_models import SkillDefinition, parse_markdown_skill
from crossby.sync.manual_fix import ManualFixNote


COMMAND_SOURCE_REL = Path(".claude") / "commands"
SKILL_NAME_PREFIX = "claude-command-"


_RUNTIME_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"\$(?:ARGUMENTS|\d+)\b"),
        "argument-placeholders",
        "Source uses Claude argument placeholders (`$ARGUMENTS` / `$1`). The target tool "
        "does not interpolate them — rewrite as a natural-language instruction or remove.",
    ),
    (
        re.compile(r"!\s*`"),
        "shell-interpolation",
        "Source uses Claude `!`shell`` interpolation to run a command and embed its output. "
        "The target tool will not execute it; replace with explicit instructions to run the "
        "command when needed.",
    ),
    (
        re.compile(r"(?:^|\s)@[\w./~:-]+"),
        "file-references",
        "Source uses Claude `@file` automatic file expansion. The target tool will not "
        "fetch those files; verify whether they should be read explicitly.",
    ),
    (
        re.compile(r"\{\{[^}]+\}\}"),
        "template-variables",
        "Source uses `{{template}}` placeholders. The target tool does not interpolate "
        "them; rewrite into natural-language instructions.",
    ),
)


def discover_claude_commands(project_root: Path) -> list[Path]:
    """Return every ``.claude/commands/**/*.md`` file under ``project_root``.

    Sorted for deterministic output. Empty when the directory is missing.
    """
    root = project_root / COMMAND_SOURCE_REL
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.md") if p.is_file())


def command_skill_name(command_path: Path, *, root: Path) -> str:
    """Stable skill name for a command file.

    Multi-segment commands (``.claude/commands/release/cut.md``) become
    ``claude-command-release-cut`` so the ``/release/cut`` semantics survive
    in the skill name.
    """
    relative = command_path.relative_to(root)
    stem_parts = list(relative.with_suffix("").parts)
    return SKILL_NAME_PREFIX + "-".join(stem_parts)


def detect_runtime_caveats(template_body: str) -> list[ManualFixNote]:
    """Identify Claude-runtime constructs in ``template_body`` and return notes."""
    notes: list[ManualFixNote] = []
    for pattern, category, message in _RUNTIME_PATTERNS:
        if pattern.search(template_body):
            notes.append(ManualFixNote(category=category, message=message))
    return notes


def command_to_skill(command_path: Path, *, root: Path) -> SkillDefinition:
    """Wrap a Claude command file as a single-file Codex/etc skill definition.

    The returned :class:`SkillDefinition` carries:

    - ``name``: ``claude-command-<slug>`` derived from the path under
      ``.claude/commands/``.
    - ``description``: the frontmatter ``description`` if present, else a
      generic "Run the migrated Claude `<name>` slash command".
    - ``body``: a short skill preamble plus the original command template
      verbatim under a ``## Command Template`` heading.
    - ``manual_fix_notes``: notes for any Claude runtime construct found in
      the body, plus a generic "this was a slash command" note explaining
      that the target tool does not invoke it via ``/<name>``.
    """
    raw = command_path.read_text(encoding="utf-8")
    parsed = parse_markdown_skill(raw, fallback_name=command_path.stem)
    name = command_skill_name(command_path, root=root)
    relative = command_path.relative_to(root).with_suffix("").as_posix()
    description = parsed.description or f"Run the migrated Claude `{relative}` slash command."

    body = (
        f"# {name}\n\n"
        f"Use this skill when the user asks to run the migrated Claude `{relative}` "
        "slash command.\n\n"
        "## Command Template\n\n"
        f"{parsed.body.strip() or 'No command body was found in the source file.'}\n"
    )

    notes: list[ManualFixNote] = [
        ManualFixNote(
            category="slash-command",
            message=(
                f"Source was the Claude slash command `/{relative}`. The target tool "
                "does not invoke it via slash; the user should ask the agent to perform "
                "this skill explicitly."
            ),
        )
    ]
    notes.extend(detect_runtime_caveats(parsed.body))

    return SkillDefinition(
        name=name,
        description=description,
        body=body,
        manual_fix_notes=tuple(notes),
    )


def iter_command_skills(project_root: Path) -> Iterable[tuple[Path, SkillDefinition]]:
    """Yield ``(source_path, skill_definition)`` for every Claude command."""
    root = project_root / COMMAND_SOURCE_REL
    for command_path in discover_claude_commands(project_root):
        yield command_path, command_to_skill(command_path, root=root)


__all__ = [
    "COMMAND_SOURCE_REL",
    "SKILL_NAME_PREFIX",
    "command_skill_name",
    "command_to_skill",
    "detect_runtime_caveats",
    "discover_claude_commands",
    "iter_command_skills",
]
