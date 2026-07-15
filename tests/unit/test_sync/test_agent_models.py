"""Tests for the SkillDefinition canonical model.

The agent half of this module was retired when sync writers were
refactored to delegate to :mod:`crossby.subagents` (PR #46). The skill
half remains because skills use the same on-disk SKILL.md shape across
every supported tool — only :func:`translate_skill_for_target`'s
manual-fix-on-lossy-field annotation needs a canonical layer.
"""

from __future__ import annotations

import textwrap

from crossby.models.ai import AIToolID
from crossby.sync.agent_models import (
    CLAUDE_ONLY_SKILL_FIELDS,
    SKILL_SCHEMA_BY_TOOL,
    SkillDefinition,
    SkillSchema,
    parse_markdown_skill,
    render_markdown_skill,
    skill_schema_for,
    skills_schema_compatible,
    translate_skill_for_target,
)


class TestSchemaTables:
    def test_skills_all_markdown(self) -> None:
        for tool in SKILL_SCHEMA_BY_TOOL:
            assert skill_schema_for(tool) == SkillSchema.MARKDOWN

    def test_skills_always_compatible(self) -> None:
        for source in SKILL_SCHEMA_BY_TOOL:
            for target in SKILL_SCHEMA_BY_TOOL:
                assert skills_schema_compatible(source, target)


class TestParseMarkdownSkill:
    def test_minimal(self) -> None:
        content = textwrap.dedent(
            """\
            ---
            name: release-notes
            description: Generates release notes.
            ---
            Body.
            """
        )
        d = parse_markdown_skill(content)
        assert d.name == "release-notes"
        assert d.allowed_tools == ()

    def test_allowed_tools(self) -> None:
        content = textwrap.dedent(
            """\
            ---
            name: x
            description: y
            allowed-tools:
              - Read
              - Bash
            ---
            """
        )
        d = parse_markdown_skill(content)
        assert d.allowed_tools == ("Read", "Bash")

    def test_allowed_tools_underscore_alias(self) -> None:
        # Skills sometimes use `allowed_tools` instead of `allowed-tools`.
        content = textwrap.dedent(
            """\
            ---
            name: x
            description: y
            allowed_tools:
              - Read
            ---
            """
        )
        d = parse_markdown_skill(content)
        assert d.allowed_tools == ("Read",)

    def test_unknown_fields_go_to_extra(self) -> None:
        content = textwrap.dedent(
            """\
            ---
            name: x
            description: y
            customField: 42
            ---
            """
        )
        d = parse_markdown_skill(content)
        assert d.extra_frontmatter == {"customField": 42}

    def test_no_frontmatter_uses_fallback_name(self) -> None:
        d = parse_markdown_skill("plain body", fallback_name="my-skill")
        assert d.name == "my-skill"
        assert d.body == "plain body"


class TestRenderMarkdownSkill:
    def test_basic(self) -> None:
        d = SkillDefinition(name="x", description="y", body="Body.\n")
        rendered = render_markdown_skill(d)
        assert "name: x" in rendered
        assert "Body." in rendered

    def test_allowed_tools_emitted(self) -> None:
        d = SkillDefinition(name="x", description="y", allowed_tools=("Read",))
        rendered = render_markdown_skill(d)
        assert "allowed-tools:" in rendered
        assert "- Read" in rendered

    def test_appends_manual_fix_block_when_notes_present(self) -> None:
        from crossby.sync.manual_fix import ManualFixNote

        d = SkillDefinition(
            name="x",
            description="y",
            manual_fix_notes=(ManualFixNote(message="Fix me."),),
        )
        rendered = render_markdown_skill(d)
        assert "<!-- crossby:manual-fix:start -->" in rendered
        assert "Fix me." in rendered


class TestTranslateSkillForTarget:
    def test_no_notes_for_claude_target(self) -> None:
        d = SkillDefinition(name="x", description="y", allowed_tools=("Read",))
        translated = translate_skill_for_target(d, AIToolID.CLAUDE)
        assert translated.manual_fix_notes == ()

    def test_emits_note_for_non_claude_target(self) -> None:
        d = SkillDefinition(name="x", description="y", allowed_tools=("Read",))
        translated = translate_skill_for_target(d, AIToolID.CODEX)
        assert len(translated.manual_fix_notes) == 1
        assert "allowed-tools" in translated.manual_fix_notes[0].message

    def test_no_notes_when_no_allowed_tools(self) -> None:
        d = SkillDefinition(name="x", description="y")
        translated = translate_skill_for_target(d, AIToolID.CODEX)
        assert translated.manual_fix_notes == ()

    def test_flags_claude_only_frontmatter_field_for_non_claude_target(self) -> None:
        # Regression: fields like `model` / `disable-model-invocation` used
        # to pass through extra_frontmatter into every target's SKILL.md
        # completely silently, unlike every other lossy edge Crossby
        # surfaces with a manual-fix note.
        d = SkillDefinition(
            name="x", description="y", extra_frontmatter={"model": "claude-opus-4.7"}
        )
        translated = translate_skill_for_target(d, AIToolID.CODEX)
        assert len(translated.manual_fix_notes) == 1
        assert "`model`" in translated.manual_fix_notes[0].message

    def test_claude_only_frontmatter_field_not_flagged_for_claude_target(self) -> None:
        d = SkillDefinition(
            name="x", description="y", extra_frontmatter={"model": "claude-opus-4.7"}
        )
        translated = translate_skill_for_target(d, AIToolID.CLAUDE)
        assert translated.manual_fix_notes == ()

    def test_unknown_non_claude_only_extra_field_not_flagged(self) -> None:
        # Fields Crossby doesn't recognize as Claude-only pass through
        # without a note — only the known list in CLAUDE_ONLY_SKILL_FIELDS
        # triggers a manual-fix.
        d = SkillDefinition(name="x", description="y", extra_frontmatter={"customField": 42})
        translated = translate_skill_for_target(d, AIToolID.CODEX)
        assert translated.manual_fix_notes == ()

    def test_multiple_claude_only_fields_combine_into_one_note(self) -> None:
        d = SkillDefinition(
            name="x",
            description="y",
            extra_frontmatter={"model": "claude-opus-4.7", "argument-hint": "<file>"},
        )
        translated = translate_skill_for_target(d, AIToolID.CODEX)
        assert len(translated.manual_fix_notes) == 1
        message = translated.manual_fix_notes[0].message
        assert "`argument-hint`" in message
        assert "`model`" in message

    def test_claude_only_skill_fields_constant_matches_known_claude_metadata(self) -> None:
        assert {
            "model",
            "effort",
            "disable-model-invocation",
            "user-invocable",
            "argument-hint",
            "context",
            "agent",
            "hooks",
            "paths",
            "shell",
        } == CLAUDE_ONLY_SKILL_FIELDS
