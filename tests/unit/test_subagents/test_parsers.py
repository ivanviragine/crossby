"""Unit tests for subagent parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossby.subagents.parsers import (
    parse_claude,
    parse_codex,
    parse_copilot,
    parse_cursor,
    parse_gemini,
)


class TestParseClaude:
    def test_basic_frontmatter(self) -> None:
        text = (
            "---\n"
            "name: researcher\n"
            "description: Research stuff\n"
            "tools: [Read, Grep, Bash]\n"
            "model: sonnet\n"
            "---\n"
            "Body here.\n"
        )
        ir, warnings = parse_claude(text)
        assert ir.name == "researcher"
        assert ir.description == "Research stuff"
        assert ir.model == "sonnet"
        assert ir.tools == ["read_file", "grep", "bash"]
        assert ir.body == "Body here.\n"
        assert warnings == []

    def test_comma_separated_tools(self) -> None:
        text = "---\nname: a\ndescription: x\ntools: Read, Edit, Bash\n---\nbody\n"
        ir, _ = parse_claude(text)
        assert ir.tools == ["read_file", "edit_file", "bash"]

    def test_unknown_fields_become_extras(self) -> None:
        text = "---\nname: a\ndescription: x\nfooBar: 7\n---\nbody\n"
        ir, _ = parse_claude(text)
        assert ir.extras == {"fooBar": 7}

    def test_no_frontmatter_uses_filename(self, tmp_path: Path) -> None:
        path = tmp_path / "myagent.md"
        path.write_text("just a body")
        ir, _ = parse_claude(path.read_text(), path)
        assert ir.name == "myagent"
        assert ir.body == "just a body"

    def test_disallowed_tools(self) -> None:
        text = "---\nname: a\ndescription: x\ndisallowedTools: [Bash]\n---\nbody\n"
        ir, _ = parse_claude(text)
        assert ir.disallowed_tools == ["bash"]

    def test_explicit_empty_tools_preserved(self) -> None:
        """`tools: []` (explicit empty) must NOT collapse to None.

        Claude/Copilot/Gemini docs treat an empty list as "no tools" while
        an omitted field means "inherit all" — they're semantically distinct.
        """
        text = "---\nname: a\ndescription: x\ntools: []\n---\nbody\n"
        ir, _ = parse_claude(text)
        assert ir.tools == []

    def test_omitted_tools_is_none(self) -> None:
        text = "---\nname: a\ndescription: x\n---\nbody\n"
        ir, _ = parse_claude(text)
        assert ir.tools is None

    def test_crlf_line_endings(self) -> None:
        """Frontmatter parsing tolerates Windows-style line endings."""
        text = "---\r\nname: a\r\ndescription: x\r\n---\r\nbody\r\n"
        ir, _ = parse_claude(text)
        assert ir.name == "a"
        assert ir.description == "x"


class TestParseGemini:
    def test_basic(self) -> None:
        text = (
            "---\n"
            "name: searcher\n"
            "description: Search files\n"
            "tools:\n"
            "  - read_file\n"
            "  - grep_search\n"
            "model: gemini-2.5-pro\n"
            "temperature: 0.5\n"
            "---\n"
            "Body.\n"
        )
        ir, _ = parse_gemini(text)
        assert ir.name == "searcher"
        assert ir.tools == ["read_file", "grep"]  # canonicalized
        assert ir.temperature == 0.5

    def test_remote_kind_warns(self) -> None:
        text = "---\nname: a\ndescription: x\nkind: remote\n---\n"
        _, warnings = parse_gemini(text)
        assert len(warnings) == 1
        assert "A2A" in warnings[0].message


class TestParseCopilot:
    def test_basic(self) -> None:
        text = (
            "---\n"
            "name: helper\n"
            "description: General help\n"
            "tools: [read, edit, search]\n"
            "target: vscode\n"
            "user-invocable: true\n"
            "---\n"
            "Body.\n"
        )
        ir, _ = parse_copilot(text)
        assert ir.name == "helper"
        assert ir.tools == ["read_file", "edit_file", "grep"]
        assert ir.target == "vscode"
        assert ir.user_invocable is True

    def test_double_extension_filename(self, tmp_path: Path) -> None:
        path = tmp_path / "worker.agent.md"
        path.write_text("body only")
        ir, _ = parse_copilot(path.read_text(), path)
        assert ir.name == "worker"


class TestParseCursor:
    def test_basic(self) -> None:
        text = "---\nname: quick\ndescription: Quick agent\nreadonly: true\n---\nBody.\n"
        ir, _ = parse_cursor(text)
        assert ir.name == "quick"
        assert ir.readonly is True
        assert ir.tools is None  # cursor has no allowlist


class TestParseCodex:
    def test_basic(self) -> None:
        text = (
            'name = "worker"\n'
            'description = "A worker"\n'
            'developer_instructions = "Do work."\n'
            'model = "gpt-5"\n'
            'sandbox_mode = "read-only"\n'
            'model_reasoning_effort = "high"\n'
        )
        ir, _ = parse_codex(text)
        assert ir.name == "worker"
        assert ir.body == "Do work."
        assert ir.sandbox_mode == "read-only"
        assert ir.effort == "high"
        assert ir.model == "gpt-5"
        assert ir.tools is None  # codex has no allowlist

    def test_invalid_toml_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid Codex"):
            parse_codex("name = unclosed string")

    def test_unknown_top_level_keys_become_extras(self) -> None:
        text = 'name = "x"\ndescription = "d"\ndeveloper_instructions = "b"\ncustom_thing = 42\n'
        ir, _ = parse_codex(text)
        assert ir.extras == {"custom_thing": 42}
