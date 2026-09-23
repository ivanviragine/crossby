"""Tests for the static model registry and adapter get_models() behavior."""

import pytest

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
        for model in (
            "claude-sonnet-5",
            "claude-opus-4.8",
            "claude-opus-5",
            "claude-opus-5.5",
            "claude-fable-5",
            "claude-fable-5.1",
        ):
            assert model in claude_models

    @pytest.mark.parametrize(
        "model",
        [
            "cursor-grok-4.6-high",
            "cursor-grok-4.6-high-fast",
            "cursor-grok-4.6-low",
            "cursor-grok-4.6-low-fast",
            "cursor-grok-4.6-medium",
            "cursor-grok-4.6-medium-fast",
            "cursor-grok-4.6-xhigh",
            "cursor-grok-4.6-xhigh-fast",
            "gemini-3.7-flash-high",
            "gemini-3.7-flash-low",
            "gemini-3.7-flash-medium",
            "gemini-3.8-flash-high",
            "gemini-3.8-flash-low",
            "gemini-3.8-flash-medium",
            "kimi-k3-high",
            "kimi-k3-low",
            "kimi-k3-max",
            "grok-4.7-high",
            "grok-4.7-xhigh-fast",
            "muse-spark-1.3-minimal",
            "muse-spark-1.3-max",
        ],
    )
    def test_cursor_registry_includes_live_cli_models(self, model: str) -> None:
        assert model in get_models_for_tool("cursor")

    def test_cursor_registry_includes_all_opus_5_5_variants(self) -> None:
        # Exactly what a logged-in `agent --list-models` reports: every effort
        # level with a -fast twin, and no -thinking variants.
        efforts = ("low", "medium", "high", "xhigh", "max")
        expected = {e for effort in efforts for e in (effort, f"{effort}-fast")}
        assert {
            model.removeprefix("claude-opus-5-5-")
            for model in get_models_for_tool("cursor")
            if model.startswith("claude-opus-5-5-")
        } == expected

    def test_cursor_registry_includes_all_fable_5_1_variants(self) -> None:
        suffixes = {
            "high",
            "low",
            "max",
            "medium",
            "thinking-high",
            "thinking-low",
            "thinking-max",
            "thinking-medium",
            "thinking-xhigh",
            "xhigh",
        }
        assert {
            model.removeprefix("claude-fable-5-1-")
            for model in get_models_for_tool("cursor")
            if model.startswith("claude-fable-5-1-")
        } == suffixes

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-3.7-flash",
            "gemini-3.8-flash",
            "gpt-5.4",
            "mai-code-1.1-flash",
        ],
    )
    def test_copilot_registry_includes_current_cli_models(self, model: str) -> None:
        assert model in get_models_for_tool("copilot")

    @pytest.mark.parametrize(
        "model",
        [
            "github-copilot/gemini-3.7-flash",
            "github-copilot/grok-4.5",
            "github-copilot/grok-4.6",
            "github-copilot/kimi-k3",
            "github-copilot/mai-code-1.1-flash",
            "google/antigravity-claude-opus-4-6-thinking",
            "google/antigravity-claude-sonnet-4-6",
            "google/gemini-3.7-flash",
            "google/gemini-3.8-flash",
            "opencode/ling-3.0-flash-fin-free",
            "opencode/muse-spark-1.2-contributor-free",
            "opencode/muse-spark-1.3-contributor-free",
            "opencode/mimo-v2.6-flash-free",
            "opencode/nemotron-3.5-lightning-free",
            "github-copilot/claude-opus-5.5",
            "github-copilot/gpt-6-astra",
            "openai/gpt-6-astra",
            "openai/gpt-6-luna",
            "openai/gpt-6-astra-fast",
            "openai/gpt-6-luna-fast",
            "openai/gpt-6-sol-fast",
            "openai/gpt-6-sol",
        ],
    )
    def test_opencode_registry_includes_live_cli_models(self, model: str) -> None:
        assert model in get_models_for_tool("opencode")

    @pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"])
    def test_codex_registry_includes_gpt_5_6(self, model: str) -> None:
        """Codex now ships the gpt-5.6 family (luna/sol/terra) — Issue #112."""
        assert model in get_models_for_tool("codex")

    def test_codex_gpt_5_6_subset_is_exactly_luna_sol_terra(self) -> None:
        """The codex catalog exposes only the three bare gpt-5.6 IDs — no
        gpt-5.6-codex* variant slipped in."""
        codex_5_6 = {m for m in get_models_for_tool("codex") if m.startswith("gpt-5.6")}
        assert codex_5_6 == {"gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"}

    def test_codex_gpt_6_subset_is_exactly_astra_luna_sol(self) -> None:
        """GPT-6 ships as Astra, Sol, and Luna — there is no gpt-6-terra."""
        codex_6 = {m for m in get_models_for_tool("codex") if m.startswith("gpt-6")}
        assert codex_6 == {"gpt-6-astra", "gpt-6-luna", "gpt-6-sol"}

    @pytest.mark.parametrize("model", ["claude-opus-5.5", "gpt-6-astra", "gpt-6-luna", "gpt-6-sol"])
    def test_copilot_registry_includes_new_generation(self, model: str) -> None:
        assert model in get_models_for_tool("copilot")

    @pytest.mark.parametrize(
        ("tool", "model"),
        [
            # Retired from Codex with ChatGPT sign-in (Codex models docs).
            ("codex", "gpt-5.4"),
            ("codex", "gpt-5.4-mini"),
            ("codex", "gpt-5.5"),
            ("codex", "gpt-5.3-codex"),
            # Retired or scheduled for retirement in Copilot's retirement history.
            ("copilot", "claude-sonnet-4.6"),
            ("copilot", "claude-opus-4.7"),
            ("copilot", "gpt-4.1"),
            ("copilot", "mai-code-1-flash"),
            # Retired on the Claude API.
            ("opencode", "anthropic/claude-3-opus-20240229"),
            ("opencode", "anthropic/claude-opus-4.1"),
            # Shut down on the OpenAI API.
            ("cursor", "gpt-5.2-codex"),
            # No longer offered by a logged-in `agent --list-models`.
            ("cursor", "composer-2"),
            ("cursor", "gpt-5.3-codex-spark-preview"),
            ("cursor", "sonnet-4.6"),
            # Still offered by Cursor, but below the 4.6 Claude cutoff.
            ("cursor", "claude-4-sonnet"),
            ("opencode", "openai/codex-mini-latest"),
            # Shut down on the Gemini API.
            ("opencode", "google/gemini-2.0-flash"),
            # Marked deprecated by OpenCode Zen.
            ("opencode", "opencode/hy3-free"),
            # No longer offered by the tool's own CLI (logged-in probe run), and
            # absent from the provider's docs as well.
            ("codex", "gpt-5.1"),
            ("codex", "gpt-5.3-codex-spark"),
            ("antigravity-cli", "gemini-3.5-flash"),
            ("opencode", "google/gemini-1.5-pro"),
            ("opencode", "google/antigravity-claude-sonnet-4.6"),
            ("opencode", "github-copilot/gpt-4o"),
            # Claude models older than 4.6 are out of scope for crossby.
            ("cursor", "claude-4.5-opus-high"),
            ("cursor", "sonnet-4.5"),
            ("opencode", "anthropic/claude-opus-4.5"),
            ("opencode", "anthropic/claude-sonnet-4.5-20250929"),
        ],
    )
    def test_deprecated_models_are_pruned(self, tool: str, model: str) -> None:
        assert model not in get_models_for_tool(tool)


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
        assert "gpt-6-sol" in [m.id for m in models]

    def test_antigravity_cli_adapter_reads_registry(self) -> None:
        adapter = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI)
        models = adapter.get_models()
        model_ids = [m.id for m in models]
        assert len(models) == len(get_models_for_tool("antigravity-cli"))
        # The catalog mixes bare Gemini IDs (effort is baked in at launch, not
        # stored) with fixed provider IDs whose suffix is part of the name.
        assert "gemini-3.8-flash" in model_ids
        assert "gemini-3.8-flash-high" not in model_ids
        assert "gemini-3.8-flash-medium" not in model_ids
        assert "gemini-3.8-flash-low" not in model_ids
        assert "gemini-3.6-flash" in model_ids
        assert "gemini-3.6-flash-high" not in model_ids
        # A fixed provider ID, not an effort variant: `agy models` reports it
        # with the suffix (and no longer offers the bare gpt-oss-120b), so the
        # catalog stores the suffix verbatim.
        assert "gpt-oss-120b-medium" in model_ids
        assert "gpt-oss-120b" not in model_ids

    def test_codex_adapter_reads_registry(self) -> None:
        adapter = AbstractAITool.get(AIToolID.CODEX)
        models = adapter.get_models()
        assert len(models) == len(get_models_for_tool("codex"))
        assert "gpt-6-sol" in [m.id for m in models]

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
