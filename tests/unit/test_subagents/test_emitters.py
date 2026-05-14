"""Unit tests for subagent emitters."""

from __future__ import annotations

import tomllib

import yaml

from crossby.subagents.emitters import (
    CodexEmission,
    emit_claude,
    emit_codex,
    emit_copilot,
    emit_cursor,
    emit_gemini,
)
from crossby.subagents.ir import SubagentIR, WarningSeverity


def _ir(**kw) -> SubagentIR:  # type: ignore[no-untyped-def]
    base = {"name": "test", "body": "Body.\n", "description": "D"}
    base.update(kw)
    return SubagentIR(**base)


def _frontmatter(text: str) -> dict:  # type: ignore[type-arg]
    assert text.startswith("---\n")
    end = text.find("\n---\n", 4)
    return yaml.safe_load(text[4:end])


class TestEmitClaude:
    def test_translates_canonical_tool_names(self) -> None:
        ir = _ir(tools=["read_file", "bash", "grep"])
        out, _ = emit_claude(ir)
        fm = _frontmatter(out)
        assert fm["tools"] == ["Read", "Bash", "Grep"]

    def test_unknown_tool_passes_through(self) -> None:
        ir = _ir(tools=["read_file", "some-mcp/tool"])
        out, _ = emit_claude(ir)
        fm = _frontmatter(out)
        assert fm["tools"] == ["Read", "some-mcp/tool"]

    def test_extras_round_trip_when_source_matches(self) -> None:
        ir = _ir(source_tool="claude", extras={"customField": 42})
        out, _ = emit_claude(ir)
        assert "customField: 42" in out

    def test_extras_dropped_when_source_differs(self) -> None:
        ir = _ir(source_tool="cursor", extras={"customField": 42})
        out, _ = emit_claude(ir)
        assert "customField" not in out


class TestEmitCursor:
    def test_collapses_readonly_tools(self) -> None:
        ir = _ir(tools=["read_file", "grep"])
        out, warnings = emit_cursor(ir)
        fm = _frontmatter(out)
        assert fm["readonly"] is True
        assert any(w.field == "tools" and w.severity == WarningSeverity.LOSSY for w in warnings)

    def test_drops_write_tools(self) -> None:
        ir = _ir(tools=["read_file", "edit_file"])
        out, warnings = emit_cursor(ir)
        fm = _frontmatter(out)
        assert "readonly" not in fm
        assert any(w.severity == WarningSeverity.DROPPED for w in warnings)


class TestEmitGemini:
    def test_writes_snake_case_tools(self) -> None:
        ir = _ir(tools=["read_file", "bash"])
        out, _ = emit_gemini(ir)
        fm = _frontmatter(out)
        assert fm["tools"] == ["read_file", "run_shell_command"]

    def test_warns_on_disallowed(self) -> None:
        ir = _ir(disallowed_tools=["bash"])
        _, warnings = emit_gemini(ir)
        assert any(w.field == "disallowed_tools" for w in warnings)


class TestEmitCopilot:
    def test_lowercase_tools(self) -> None:
        ir = _ir(tools=["read_file", "bash"])
        out, _ = emit_copilot(ir)
        fm = _frontmatter(out)
        assert fm["tools"] == ["read", "shell"]

    def test_30k_body_warning(self) -> None:
        ir = _ir(body="x" * 30_001)
        _, warnings = emit_copilot(ir)
        assert any("30,000" in w.message for w in warnings)

    def test_missing_description_warns(self) -> None:
        ir = SubagentIR(name="x", body="b")
        out, warnings = emit_copilot(ir)
        fm = _frontmatter(out)
        assert fm["description"] == "x"  # falls back to name
        assert any(w.field == "description" for w in warnings)


class TestEmitCodex:
    def test_emits_developer_instructions_and_fragment(self) -> None:
        ir = _ir(model="gpt-5", effort="high")
        emission, _ = emit_codex(ir)
        assert isinstance(emission, CodexEmission)
        agent = tomllib.loads(emission.agent_toml)
        assert agent["name"] == "test"
        assert agent["developer_instructions"] == "Body.\n"
        assert agent["model"] == "gpt-5"
        assert agent["model_reasoning_effort"] == "high"
        # The fragment registers the agent under [agents.<name>]
        fragment = tomllib.loads(emission.config_fragment)
        assert "test" in fragment["agents"]

    def test_collapses_tools_to_sandbox_mode(self) -> None:
        ir = _ir(tools=["read_file", "edit_file"])
        emission, warnings = emit_codex(ir)
        agent = tomllib.loads(emission.agent_toml)
        assert agent["sandbox_mode"] == "workspace-write"
        assert any(w.field == "tools" and w.severity == WarningSeverity.LOSSY for w in warnings)

    def test_readonly_to_sandbox_mode(self) -> None:
        ir = _ir(tools=["read_file"])
        emission, _ = emit_codex(ir)
        agent = tomllib.loads(emission.agent_toml)
        assert agent["sandbox_mode"] == "read-only"

    def test_filename_suggestion(self) -> None:
        ir = _ir(name="my-worker")
        emission, _ = emit_codex(ir)
        assert emission.suggested_filename == "my-worker.toml"

    def test_empty_body_warns(self) -> None:
        """Codex requires developer_instructions; empty body should surface a warning."""
        ir = SubagentIR(name="x", description="d", body="")
        emission, warnings = emit_codex(ir)
        assert any(w.field == "body" and w.severity == WarningSeverity.LOSSY for w in warnings)
        agent = tomllib.loads(emission.agent_toml)
        assert agent["developer_instructions"] == ""


class TestEmitPreservesEmptyTools:
    """Cross-emitter: explicit empty tools list survives round-trip."""

    def test_claude_emits_empty_list(self) -> None:
        ir = _ir(tools=[])
        out, _ = emit_claude(ir)
        fm = _frontmatter(out)
        assert fm["tools"] == []

    def test_copilot_emits_empty_list(self) -> None:
        ir = _ir(tools=[])
        out, _ = emit_copilot(ir)
        fm = _frontmatter(out)
        assert fm["tools"] == []

    def test_gemini_emits_empty_list(self) -> None:
        ir = _ir(tools=[])
        out, _ = emit_gemini(ir)
        fm = _frontmatter(out)
        assert fm["tools"] == []
