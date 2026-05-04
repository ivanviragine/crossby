"""Tests for canonical Agent/Skill domain models and translators."""

from __future__ import annotations

import textwrap
import tomllib

from crossby.models.ai import AIToolID, EffortLevel
from crossby.sync.agent_models import (
    AGENT_SCHEMA_BY_TOOL,
    AgentSchema,
    SkillDefinition,
    SkillSchema,
    agent_schema_for,
    agents_schema_compatible,
    parse_markdown_agent,
    parse_markdown_skill,
    parse_toml_agent,
    render_markdown_agent,
    render_markdown_skill,
    render_toml_agent,
    skill_schema_for,
    skills_schema_compatible,
    translate_skill_for_target,
)


class TestSchemaTables:
    def test_codex_is_toml(self) -> None:
        assert agent_schema_for(AIToolID.CODEX) == AgentSchema.TOML

    def test_claude_is_markdown(self) -> None:
        assert agent_schema_for(AIToolID.CLAUDE) == AgentSchema.MARKDOWN

    def test_skills_all_markdown(self) -> None:
        for tool in AGENT_SCHEMA_BY_TOOL:
            assert skill_schema_for(tool) == SkillSchema.MARKDOWN

    def test_compat_within_markdown(self) -> None:
        assert agents_schema_compatible(AIToolID.CLAUDE, AIToolID.CURSOR)
        assert agents_schema_compatible(AIToolID.CURSOR, AIToolID.GEMINI)
        assert agents_schema_compatible(AIToolID.CLAUDE, AIToolID.COPILOT)

    def test_codex_incompatible_with_markdown_tools(self) -> None:
        assert not agents_schema_compatible(AIToolID.CLAUDE, AIToolID.CODEX)
        assert not agents_schema_compatible(AIToolID.CODEX, AIToolID.CURSOR)

    def test_skills_always_compatible(self) -> None:
        for source in AGENT_SCHEMA_BY_TOOL:
            for target in AGENT_SCHEMA_BY_TOOL:
                assert skills_schema_compatible(source, target)


class TestParseMarkdownAgent:
    def test_minimal(self) -> None:
        content = textwrap.dedent(
            """\
            ---
            name: release-lead
            description: Plans and ships releases.
            ---
            Body text.
            """
        )
        d = parse_markdown_agent(content)
        assert d.name == "release-lead"
        assert d.description == "Plans and ships releases."
        assert d.body.strip() == "Body text."

    def test_full_frontmatter(self) -> None:
        content = textwrap.dedent(
            """\
            ---
            name: release-lead
            description: Plans and ships releases.
            model: claude-sonnet-4.6
            effort: high
            permissionMode: acceptEdits
            skills:
              - release-notes
            tools:
              - Read
              - Bash
            disallowedTools:
              - Write
            ---
            Body.
            """
        )
        d = parse_markdown_agent(content)
        assert d.model == "claude-sonnet-4.6"
        assert d.effort == EffortLevel.HIGH
        assert d.permission_mode == "acceptEdits"
        assert d.skills == ("release-notes",)
        assert d.tools_allow == ("Read", "Bash")
        assert d.tools_deny == ("Write",)

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
        d = parse_markdown_agent(content)
        assert d.extra_frontmatter == {"customField": 42}

    def test_no_frontmatter_uses_fallback_name(self) -> None:
        d = parse_markdown_agent("just body", fallback_name="filename-stem")
        assert d.name == "filename-stem"
        assert d.body == "just body"

    def test_invalid_yaml_falls_back(self) -> None:
        d = parse_markdown_agent(
            "---\nname: [unterminated\n---\nbody",
            fallback_name="default",
        )
        assert d.name == "default"


class TestParseTomlAgent:
    def test_minimal(self) -> None:
        content = textwrap.dedent(
            """\
            name = "release-lead"
            description = "Plans and ships releases."
            developer_instructions = "Body."
            """
        )
        d = parse_toml_agent(content)
        assert d.name == "release-lead"
        assert d.description == "Plans and ships releases."
        assert d.body == "Body."

    def test_sandbox_mode_maps_back_to_permission_mode(self) -> None:
        content = textwrap.dedent(
            """\
            name = "x"
            description = "y"
            sandbox_mode = "workspace-write"
            developer_instructions = ""
            """
        )
        d = parse_toml_agent(content)
        assert d.permission_mode == "acceptEdits"

    def test_invalid_toml_returns_minimal(self) -> None:
        d = parse_toml_agent("this is not toml :", fallback_name="oops")
        assert d.name == "oops"


class TestRenderMarkdownAgent:
    def test_round_trip_minimal(self) -> None:
        from crossby.sync.agent_models import AgentDefinition

        d = AgentDefinition(name="x", description="y", body="hello\n")
        rendered = render_markdown_agent(d, target_tool=AIToolID.CLAUDE)
        assert rendered.startswith("---\n")
        assert "name: x" in rendered
        assert "description: y" in rendered
        assert "hello" in rendered

        # Re-parse and check equality of important fields.
        d2 = parse_markdown_agent(rendered)
        assert d2.name == "x"
        assert d2.description == "y"
        assert d2.body.strip() == "hello"

    def test_emits_lists(self) -> None:
        from crossby.sync.agent_models import AgentDefinition

        d = AgentDefinition(
            name="x",
            description="y",
            tools_allow=("Read", "Bash"),
            tools_deny=("Write",),
            skills=("release-notes",),
        )
        rendered = render_markdown_agent(d, target_tool=AIToolID.CURSOR)
        assert "- Read" in rendered
        assert "- Bash" in rendered
        assert "- Write" in rendered
        assert "- release-notes" in rendered

    def test_translates_codex_model_back_to_claude_default(self) -> None:
        from crossby.sync.agent_models import AgentDefinition

        d = AgentDefinition(name="x", description="y", model="gpt-5.4-mini")
        rendered = render_markdown_agent(d, target_tool=AIToolID.CLAUDE)
        # gpt-5.4-mini → claude-sonnet-4.6 (per CODEX_TO_CLAUDE_DEFAULTS)
        assert "model: claude-sonnet-4.6" in rendered

    def test_appends_manual_fix_block(self) -> None:
        from crossby.sync.agent_models import AgentDefinition
        from crossby.sync.manual_fix import ManualFixNote

        d = AgentDefinition(
            name="x",
            description="y",
            manual_fix_notes=(ManualFixNote(message="Fix me."),),
        )
        rendered = render_markdown_agent(d, target_tool=AIToolID.CURSOR)
        assert "<!-- crossby:manual-fix:start -->" in rendered
        assert "Fix me." in rendered


class TestRenderTomlAgent:
    def test_required_fields_present(self) -> None:
        from crossby.sync.agent_models import AgentDefinition

        d = AgentDefinition(name="release-lead", description="Plans releases.", body="Do work.")
        rendered = render_toml_agent(d)
        parsed = tomllib.loads(rendered)
        assert parsed["name"] == "release-lead"
        assert parsed["description"] == "Plans releases."
        assert "Do work." in parsed["developer_instructions"]

    def test_model_and_effort_translated(self) -> None:
        from crossby.sync.agent_models import AgentDefinition

        d = AgentDefinition(
            name="x",
            description="y",
            body="",
            model="claude-sonnet-4.6",
            effort=EffortLevel.HIGH,
        )
        parsed = tomllib.loads(render_toml_agent(d))
        # Sonnet → gpt-5.4-mini, HIGH → XHIGH (one-tier bump for Sonnet).
        assert parsed["model"] == "gpt-5.4-mini"
        assert parsed["model_reasoning_effort"] == "xhigh"

    def test_permission_mode_to_sandbox(self) -> None:
        from crossby.sync.agent_models import AgentDefinition

        d = AgentDefinition(
            name="x",
            description="y",
            body="",
            permission_mode="acceptEdits",
        )
        parsed = tomllib.loads(render_toml_agent(d))
        assert parsed["sandbox_mode"] == "workspace-write"

    def test_unmapped_permission_mode_emits_manual_fix(self) -> None:
        from crossby.sync.agent_models import AgentDefinition

        d = AgentDefinition(
            name="x",
            description="y",
            body="",
            permission_mode="plan",
        )
        rendered = render_toml_agent(d)
        # No sandbox_mode (since plan doesn't map), but a manual-fix block in body.
        parsed = tomllib.loads(rendered)
        assert "sandbox_mode" not in parsed
        assert "Manual migration required" in parsed["developer_instructions"]
        assert "permissionMode: plan" in parsed["developer_instructions"]

    def test_tools_become_manual_fix_notes(self) -> None:
        from crossby.sync.agent_models import AgentDefinition

        d = AgentDefinition(
            name="x",
            description="y",
            body="",
            tools_allow=("Read",),
            tools_deny=("Bash",),
        )
        rendered = render_toml_agent(d)
        parsed = tomllib.loads(rendered)
        instr = parsed["developer_instructions"]
        assert "Source `tools` allow-list" in instr
        assert "Source `disallowedTools` deny-list" in instr

    def test_unknown_model_passes_through_with_note(self) -> None:
        from crossby.sync.agent_models import AgentDefinition

        d = AgentDefinition(name="x", description="y", body="", model="o3-mini")
        rendered = render_toml_agent(d)
        parsed = tomllib.loads(rendered)
        assert parsed["model"] == "o3-mini"
        assert "No known Codex equivalent" in parsed["developer_instructions"]


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
