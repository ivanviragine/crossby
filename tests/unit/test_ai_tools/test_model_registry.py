"""Focused tests for static model registry loading."""

from crossby.ai_tools import AbstractAITool
from crossby.data import MODELS, get_models_for_tool
from crossby.models.ai import AIToolID


class TestModelRegistry:
    def test_get_models_for_tool_unknown_returns_empty(self) -> None:
        """get_models_for_tool should return empty list for unknown tool."""
        assert get_models_for_tool("unknown-tool") == []

    def test_registry_contains_no_meta_keys(self) -> None:
        """The loaded MODELS dict should not contain _note or other _ keys."""
        for key in MODELS:
            assert not key.startswith("_")


class TestRegistryGetModels:
    """Verify that adapters read correctly from the static registry."""

    def test_standard_adapter_reads_registry(self) -> None:
        adapter = AbstractAITool.get(AIToolID.CLAUDE)
        models = adapter.get_models()
        assert len(models) == len(get_models_for_tool("claude"))
        assert "claude-haiku-4.5" in [m.id for m in models]

    def test_opencode_adapter_reads_registry(self) -> None:
        adapter = AbstractAITool.get(AIToolID.OPENCODE)
        models = adapter.get_models()
        assert len(models) == len(get_models_for_tool("opencode"))
        # Using suffix for classification, we expect the original string in id
        assert "anthropic/claude-sonnet-4.6" in [m.id for m in models]
