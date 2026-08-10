"""Unit tests for subagent emitters."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

from crossby.subagents.emitters import (
    CodexEmission,
    emit_claude,
    emit_codex,
    emit_copilot,
    emit_cursor,
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
        # The fragment is a global-registration suggestion for the standalone
        # `crossby agents` emitter: it registers the agent under [agents.<name>]
        # at the ~/.codex home path (distinct from the project-local path the
        # sync writer uses, which discards this fragment). #88 §7.
        fragment = tomllib.loads(emission.config_fragment)
        assert "test" in fragment["agents"]
        # Codex registers a role's config layer under `config_file`, not `path`.
        # It must be a real absolute path, not a literal `~/...` string — Codex's
        # config_file is typed AbsolutePathBuf, which `~` does not satisfy.
        expected = str(Path.home() / ".codex" / "agents" / "test.toml")
        assert fragment["agents"]["test"]["config_file"] == expected

    def test_config_file_honors_codex_home_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Codex itself resolves its config dir from $CODEX_HOME when set,
        # falling back to ~/.codex — the suggested config_file must match.
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))
        ir = _ir()
        emission, _ = emit_codex(ir)
        fragment = tomllib.loads(emission.config_fragment)
        expected = str(tmp_path / "agents" / "test.toml")
        assert fragment["agents"]["test"]["config_file"] == expected

    def test_config_file_absolute_even_with_relative_codex_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # $CODEX_HOME could itself be set to a relative value — the
        # suggested config_file must still resolve to an absolute path
        # (Codex's config_file is typed AbsolutePathBuf).
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CODEX_HOME", "relative_codex_dir")
        ir = _ir()
        emission, _ = emit_codex(ir)
        fragment = tomllib.loads(emission.config_fragment)
        config_file = fragment["agents"]["test"]["config_file"]
        assert Path(config_file).is_absolute()
        assert config_file == str(tmp_path / "relative_codex_dir" / "agents" / "test.toml")

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
