"""Tests for tool-marker detection in instruction content."""

from __future__ import annotations

from crossby.models.ai import AIToolID
from crossby.sync.instruction_markers import (
    detect_tool_markers,
    foreign_markers,
    is_neutral_for_target,
    manual_fix_notes_for_target,
)


class TestDetectToolMarkers:
    def test_neutral_content(self) -> None:
        assert detect_tool_markers("Just a plain instruction file.") == {}

    def test_claude_subagent(self) -> None:
        found = detect_tool_markers("See `.claude/agents/release.md` for context.")
        assert AIToolID.CLAUDE in found
        assert "Claude subagent paths" in found[AIToolID.CLAUDE]

    def test_claude_exit_plan_mode(self) -> None:
        found = detect_tool_markers("Use ExitPlanMode when ready.")
        assert AIToolID.CLAUDE in found

    def test_claude_permission_mode(self) -> None:
        found = detect_tool_markers("Set permissionMode to acceptEdits.")
        assert AIToolID.CLAUDE in found

    def test_codex_sandbox(self) -> None:
        found = detect_tool_markers("Configure sandbox_mode in .codex/config.toml.")
        assert AIToolID.CODEX in found

    def test_cursor(self) -> None:
        found = detect_tool_markers("Edit .cursorrules to allowlist commands.")
        assert AIToolID.CURSOR in found

    def test_copilot(self) -> None:
        found = detect_tool_markers("See .github/copilot-instructions.md for the contract.")
        assert AIToolID.COPILOT in found

    def test_multi_tool_content(self) -> None:
        found = detect_tool_markers("Use ExitPlanMode in Claude or sandbox_mode in .codex/.")
        assert AIToolID.CLAUDE in found
        assert AIToolID.CODEX in found

    def test_case_insensitive(self) -> None:
        # Match should not depend on original casing.
        found = detect_tool_markers("EXITPLANMODE")
        assert AIToolID.CLAUDE in found

    def test_claude_hooks_slash_command(self) -> None:
        # A standalone `/hooks` slash command is a Claude marker, including when
        # it ends a sentence or is followed by a comma.
        for text in (
            "Run /hooks to configure.",
            "Configure it with /hooks.",  # sentence-ending period
            "See /hooks, then save.",  # comma
        ):
            found = detect_tool_markers(text)
            assert AIToolID.CLAUDE in found, text
            assert "Claude /hooks slash command" in found[AIToolID.CLAUDE]

    def test_hooks_path_segment_is_not_claude(self) -> None:
        # A `/hooks` path segment, URL, or absolute path (not the standalone
        # slash command) must not attribute to Claude.
        for text in (
            "See .agents/hooks.json for guards.",  # relative path
            "Scripts live in tools/hooks/.",  # path segment
            "Fetch https://hooks.example.com/api for webhooks.",  # URL (// and :)
            "Open C:/hooks/config.json on Windows.",  # drive path (:)
            "Edit /hooks/config.json at the root.",  # absolute path (trailing /)
            "Point to /hooks.example.com there.",  # bare domain (trailing .word)
            "Windows path C:/hooks\\config.json here.",  # backslash continuation
            "Config at /hooks-config.json is stale.",  # hyphen continuation
        ):
            assert AIToolID.CLAUDE not in detect_tool_markers(text), text


class TestIsNeutralForTarget:
    def test_pure_neutral(self) -> None:
        assert is_neutral_for_target("Plain content.", AIToolID.CLAUDE) is True

    def test_target_owns_markers(self) -> None:
        # ExitPlanMode is Claude-specific; for a Claude target, that's fine.
        assert is_neutral_for_target("Use ExitPlanMode.", AIToolID.CLAUDE) is True

    def test_target_does_not_own_markers(self) -> None:
        assert is_neutral_for_target("Use ExitPlanMode.", AIToolID.COPILOT) is False

    def test_mixed_with_target_marker_present(self) -> None:
        # Content names both Claude *and* Codex; for a Claude target only the
        # Codex markers are foreign, so it's not neutral.
        text = "Use ExitPlanMode and configure sandbox_mode."
        assert is_neutral_for_target(text, AIToolID.CLAUDE) is False


class TestForeignMarkers:
    def test_returns_only_other_tools(self) -> None:
        text = "Use ExitPlanMode and configure sandbox_mode."
        foreign = foreign_markers(text, AIToolID.CLAUDE)
        assert AIToolID.CODEX in foreign
        assert AIToolID.CLAUDE not in foreign

    def test_empty_when_neutral(self) -> None:
        assert foreign_markers("Plain.", AIToolID.CLAUDE) == {}


class TestManualFixNotes:
    def test_one_note_per_foreign_tool(self) -> None:
        text = "Use ExitPlanMode and configure sandbox_mode."
        notes = manual_fix_notes_for_target(text, AIToolID.COPILOT)
        # Both Claude and Codex are foreign to a Copilot target.
        categories = {note.category for note in notes}
        assert {str(AIToolID.CLAUDE), str(AIToolID.CODEX)} <= categories

    def test_no_notes_when_neutral(self) -> None:
        assert manual_fix_notes_for_target("Plain text.", AIToolID.CLAUDE) == []

    def test_note_text_mentions_target(self) -> None:
        text = "Use ExitPlanMode."
        notes = manual_fix_notes_for_target(text, AIToolID.CURSOR)
        assert len(notes) == 1
        assert "cursor" in notes[0].message


# ---------------------------------------------------------------------------
# Expanded per-tool markers (one positive + one negative case each)
# ---------------------------------------------------------------------------


class TestExpandedCodexMarkers:
    def test_mcp_servers_table_detected(self) -> None:
        found = detect_tool_markers("Configure [mcp_servers.foo] in config.toml.")
        assert AIToolID.CODEX in found

    def test_features_table_detected(self) -> None:
        found = detect_tool_markers("Set [features].codex_hooks = true.")
        assert AIToolID.CODEX in found

    def test_model_reasoning_effort_detected(self) -> None:
        found = detect_tool_markers("Bump model_reasoning_effort to high.")
        assert AIToolID.CODEX in found

    def test_codex_hooks_flag_detected(self) -> None:
        found = detect_tool_markers("Make sure codex_hooks is enabled.")
        assert AIToolID.CODEX in found

    def test_agents_skills_path_detected(self) -> None:
        found = detect_tool_markers("Drop the skill under .agents/skills/foo/.")
        assert AIToolID.CODEX in found

    def test_no_codex_marker(self) -> None:
        found = detect_tool_markers("This file mentions nothing tool-specific.")
        assert AIToolID.CODEX not in found


class TestExpandedCursorMarkers:
    def test_cursor_agents_path(self) -> None:
        found = detect_tool_markers("See .cursor/agents/release-lead.md")
        assert AIToolID.CURSOR in found

    def test_cursor_commands_path(self) -> None:
        found = detect_tool_markers("See .cursor/commands/fmt.md")
        assert AIToolID.CURSOR in found

    def test_cursor_skills_path(self) -> None:
        found = detect_tool_markers("Skills live in .cursor/skills/foo/SKILL.md")
        assert AIToolID.CURSOR in found

    def test_cursor_cli_json(self) -> None:
        found = detect_tool_markers("Allow rules live in .cursor/cli.json")
        assert AIToolID.CURSOR in found

    def test_no_cursor_marker(self) -> None:
        found = detect_tool_markers("Plain prose with no marker.")
        assert AIToolID.CURSOR not in found


class TestExpandedCopilotMarkers:
    def test_workspace_participant(self) -> None:
        found = detect_tool_markers("Ask @workspace for help.")
        assert AIToolID.COPILOT in found

    def test_github_participant(self) -> None:
        found = detect_tool_markers("Run @github to look up the issue.")
        assert AIToolID.COPILOT in found

    def test_workspace_in_email_does_not_match(self) -> None:
        # Negative lookbehind: email addresses must not trip the marker.
        found = detect_tool_markers("Email me@workspace.com for access.")
        assert AIToolID.COPILOT not in found

    def test_github_in_email_does_not_match(self) -> None:
        found = detect_tool_markers("Reach team@github.com offline.")
        assert AIToolID.COPILOT not in found

    def test_github_agents_path(self) -> None:
        found = detect_tool_markers("Agents go under .github/agents/")
        assert AIToolID.COPILOT in found

    def test_vscode_mcp_json(self) -> None:
        found = detect_tool_markers("MCP servers live in .vscode/mcp.json")
        assert AIToolID.COPILOT in found

    def test_no_copilot_marker(self) -> None:
        found = detect_tool_markers("Plain prose.")
        assert AIToolID.COPILOT not in found


class TestExpandedAntigravityCLIMarkers:
    """agy markers cover only agy-*unique* surfaces; shared ones stay off it."""

    def test_hooks_json_path_detected(self) -> None:
        found = detect_tool_markers("Guards live in .agents/hooks.json")
        assert AIToolID.ANTIGRAVITY_CLI in found
        # Regression: the `.agents/hooks.json` path segment must NOT also trip
        # Claude's `/hooks` slash-command marker. When it did, agy-authored
        # hooks content read as foreign-to-agy and forced a needless copy.
        assert AIToolID.CLAUDE not in found

    def test_hooks_json_is_neutral_for_agy_target(self) -> None:
        # Corollary of the collision fix: `.agents/hooks.json` content is neutral
        # for an agy target (no foreign Claude marker leaks in).
        assert is_neutral_for_target("Guards live in .agents/hooks.json", AIToolID.ANTIGRAVITY_CLI)

    def test_mcp_config_path_detected(self) -> None:
        found = detect_tool_markers("MCP servers go in .agents/mcp_config.json")
        assert AIToolID.ANTIGRAVITY_CLI in found

    def test_agents_dir_path_detected(self) -> None:
        found = detect_tool_markers("Subagents live under .agents/agents/")
        assert AIToolID.ANTIGRAVITY_CLI in found

    def test_structured_decision_hook_detected(self) -> None:
        # A structured agy hook payload — the JSON DECISION shape, not prose.
        content = 'The hook returns {"decision": "ask"} to prompt the user.'
        found = detect_tool_markers(content)
        assert AIToolID.ANTIGRAVITY_CLI in found

    def test_agy_decision_variants_detected(self) -> None:
        # agy's PreToolUse allow/deny/ask and Stop continue are all agy-specific.
        for value in ("allow", "deny", "ask", "continue"):
            content = f'Return {{"decision": "{value}"}} from the guard.'
            assert AIToolID.ANTIGRAVITY_CLI in detect_tool_markers(content), value

    def test_decision_block_is_not_agy(self) -> None:
        # `{"decision": "block"}` is the Claude/Codex Stop shape (BLOCK_DECISION),
        # not agy — it must not be attributed to Antigravity CLI.
        content = 'The Stop hook emits {"decision": "block"} to halt.'
        assert AIToolID.ANTIGRAVITY_CLI not in detect_tool_markers(content)

    def test_shared_skills_path_not_attributed_to_agy(self) -> None:
        # `.agents/skills/` is shared with Codex and is a Codex marker; it must
        # NOT be attributed to agy or the shared skills surface would misfire.
        found = detect_tool_markers("Skill lives in .agents/skills/foo/SKILL.md")
        assert AIToolID.CODEX in found
        assert AIToolID.ANTIGRAVITY_CLI not in found

    def test_non_hook_decision_word_not_attributed_to_agy(self) -> None:
        # A bare mention of "decision" in prose is not the structured hook shape.
        content = "Make a decision about the release before Friday."
        assert AIToolID.ANTIGRAVITY_CLI not in detect_tool_markers(content)

    def test_bare_agents_dir_not_attributed_to_agy(self) -> None:
        # A bare `.agents/` prefix (no unique agy surface) must not attribute.
        content = "Config lives somewhere under .agents/ generally."
        assert AIToolID.ANTIGRAVITY_CLI not in detect_tool_markers(content)

    def test_agy_content_is_foreign_for_non_agy_target(self) -> None:
        # agy-authored rules content copied to a Claude target must be flagged.
        # (Uses `.agents/mcp_config.json`, an agy-unique surface that doesn't
        # also trip Claude's pre-existing `/hooks` marker.)
        content = "Register servers in .agents/mcp_config.json before syncing."
        assert is_neutral_for_target(content, AIToolID.CLAUDE) is False
        foreign = foreign_markers(content, AIToolID.CLAUDE)
        assert AIToolID.ANTIGRAVITY_CLI in foreign

    def test_agy_content_neutral_for_agy_target(self) -> None:
        content = "Register servers in .agents/mcp_config.json before syncing."
        assert is_neutral_for_target(content, AIToolID.ANTIGRAVITY_CLI) is True

    def test_no_agy_marker_in_plain_prose(self) -> None:
        assert AIToolID.ANTIGRAVITY_CLI not in detect_tool_markers("Plain prose.")


class TestNeutralStillNeutral:
    """Regression: expanded marker lists must not flag plain documentation."""

    def test_clean_doc_still_clean(self) -> None:
        content = (
            "# Project guidelines\n\n"
            "This codebase is a Python CLI. Follow PEP 8 and write tests.\n"
            "Use idiomatic patterns; prefer composition over inheritance.\n"
        )
        assert detect_tool_markers(content) == {}
