"""Tests for the static model registry and adapter get_models() behavior."""

from crossby.ai_tools import AbstractAITool
from crossby.data import MODELS, get_models_for_tool
from crossby.models.ai import AIToolID


class TestModelRegistry:
    def test_get_models_for_tool_returns_list(self) -> None:
        """get_models_for_tool should return lists of strings for known tools."""
        claude_models = get_models_for_tool("claude")
        assert isinstance(claude_models, list)
        assert len(claude_models) > 0
        assert "claude-haiku-4.5" in claude_models

    def test_get_models_for_tool_unknown_returns_empty(self) -> None:
        """get_models_for_tool should return empty list for unknown tool."""
        assert get_models_for_tool("unknown-tool") == []

    def test_registry_contains_no_meta_keys(self) -> None:
        """The loaded MODELS dict should not contain _note or other _ keys."""
        for key in MODELS:
            assert not key.startswith("_")

    def test_claude_registry_includes_current_models(self) -> None:
        """The Claude catalog tracks the current generation (WADE #309 port)."""
        claude_models = get_models_for_tool("claude")
        for model in ("claude-sonnet-5", "claude-opus-4.8", "claude-fable-5"):
            assert model in claude_models


class TestRegistryGetModels:
    """Verify that adapters read correctly from the static registry."""

    def test_claude_adapter_reads_registry(self) -> None:
        adapter = AbstractAITool.get(AIToolID.CLAUDE)
        models = adapter.get_models()
        assert len(models) == len(get_models_for_tool("claude"))
        assert "claude-haiku-4.5" in [m.id for m in models]

    def test_copilot_adapter_reads_registry(self) -> None:
        adapter = AbstractAITool.get(AIToolID.COPILOT)
        models = adapter.get_models()
        assert len(models) == len(get_models_for_tool("copilot"))
        assert "gpt-4.1" in [m.id for m in models]

    def test_gemini_adapter_reads_registry(self) -> None:
        adapter = AbstractAITool.get(AIToolID.GEMINI)
        models = adapter.get_models()
        assert len(models) == len(get_models_for_tool("gemini"))
        assert "gemini-2.5-pro" in [m.id for m in models]

    def test_codex_adapter_reads_registry(self) -> None:
        adapter = AbstractAITool.get(AIToolID.CODEX)
        models = adapter.get_models()
        assert len(models) == len(get_models_for_tool("codex"))
        assert any("codex" in m.id for m in models)

    def test_opencode_adapter_reads_registry(self) -> None:
        adapter = AbstractAITool.get(AIToolID.OPENCODE)
        models = adapter.get_models()
        assert len(models) == len(get_models_for_tool("opencode"))
        # Using suffix for classification, we expect the original string in id
        assert "anthropic/claude-sonnet-4.6" in [m.id for m in models]

    def test_cursor_adapter_reads_registry(self) -> None:
        adapter = AbstractAITool.get(AIToolID.CURSOR)
        models = adapter.get_models()
        assert len(models) == len(get_models_for_tool("cursor"))
        model_ids = [m.id for m in models]
        assert "auto" in model_ids
        assert "claude-opus-4-7-high" in model_ids
