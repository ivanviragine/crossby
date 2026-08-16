"""Tests for scripts/probe_models.py model discovery helpers.

``scripts/`` is not an importable package, so the module is loaded from its
file path.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_PROBE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "probe_models.py"


def _load_probe_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("probe_models", _PROBE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROBE_MODULE = _load_probe_module()
CLAUDE_PATTERN = str(PROBE_MODULE._SCRAPE_PATTERNS["claude"])
COPILOT_PATTERN = str(PROBE_MODULE._SCRAPE_PATTERNS["copilot"])


class TestClaudeScrapePattern:
    """The family-anchored pattern matches current IDs, excludes noisy ones."""

    def test_matches_single_number_versions(self) -> None:
        assert re.findall(CLAUDE_PATTERN, "claude-sonnet-5") == ["claude-sonnet-5"]
        assert re.findall(CLAUDE_PATTERN, "claude-fable-5") == ["claude-fable-5"]
        assert re.findall(CLAUDE_PATTERN, "claude-opus-5") == ["claude-opus-5"]

    def test_matches_dotted_versions(self) -> None:
        assert re.findall(CLAUDE_PATTERN, "claude-opus-4.8") == ["claude-opus-4.8"]
        assert re.findall(CLAUDE_PATTERN, "claude-haiku-4.5") == ["claude-haiku-4.5"]

    def test_excludes_dated_snapshots(self) -> None:
        text = "claude-sonnet-4-5-20250929"
        assert re.findall(CLAUDE_PATTERN, text) == []

    def test_excludes_v1_variants(self) -> None:
        text = "claude-opus-4-1-v1"
        assert re.findall(CLAUDE_PATTERN, text) == []

    def test_excludes_slug_run_ons(self) -> None:
        matches = re.findall(CLAUDE_PATTERN, "claude-sonnet-5-vs-gpt-5")
        assert "claude-sonnet-5" in matches
        assert "claude-sonnet-5-vs-gpt-5" not in matches

    def test_excludes_non_family_prefixes(self) -> None:
        # Legacy dotted IDs like claude-3-5-sonnet-... are not family-anchored.
        assert re.findall(CLAUDE_PATTERN, "claude-3-5-sonnet-20241022") == []


def test_copilot_pattern_captures_mai_id_without_matching_prose() -> None:
    text = "Maintain support for mai-code-1-flash and gemini-3.6-flash in the main catalog."
    assert set(re.findall(COPILOT_PATTERN, text)) == {
        "mai-code-1-flash",
        "gemini-3.6-flash",
    }


def test_copilot_docs_parser_uses_model_table_not_table_of_contents() -> None:
    page = """
    Supported models
    Tool availability values
    unrelated option prose mentioning gpt-noise
    Supported models
    `claude-sonnet-4.6` | General-purpose coding
    `gpt-5.4` | Complex reasoning
    `gemini-3.6-flash` | Fast responses
    `mai-code-1-flash` | Adaptive coding
    Tool availability values
    `gpt-not-a-model-table-entry`
    """

    assert PROBE_MODULE.parse_documented_models("copilot", page) == {
        "claude-sonnet-4.6",
        "gpt-5.4",
        "gemini-3.6-flash",
        "mai-code-1-flash",
    }


def test_catalog_diff_preserves_exact_provider_spelling() -> None:
    registered = {"google/antigravity-claude-sonnet-4.6"}
    discovered = {"google/antigravity-claude-sonnet-4-6"}

    assert PROBE_MODULE.model_catalog_diff(registered, discovered) == (
        registered,
        discovered,
    )


class TestAntigravityModelParsing:
    def test_extracts_gemini_3_7_and_deduplicates_effort_variants(self) -> None:
        output = """
        Available models:
          gemini-3.7-flash-low       Gemini 3.7 Flash (Low)
          gemini-3.7-flash-medium    Gemini 3.7 Flash (Medium)
          gemini-3.7-flash-high      Gemini 3.7 Flash (High)
          gemini-3.7-flash-high      Gemini 3.7 Flash (High)
          claude-opus-4-6-thinking   Claude Opus 4.6 Thinking
          gpt-oss-120b-medium        GPT OSS 120B
        """

        assert PROBE_MODULE.parse_antigravity_models(output) == {
            "gemini-3.7-flash",
            "claude-opus-4-6-thinking",
            "gpt-oss-120b-medium",
        }

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gemini-3.7-flash-low", "gemini-3.7-flash"),
            ("gemini-3.7-flash-medium", "gemini-3.7-flash"),
            ("gemini-3.7-flash-high", "gemini-3.7-flash"),
            ("gemini-3.7-flash-minimal", "gemini-3.7-flash-minimal"),
            ("claude-opus-4-6-thinking", "claude-opus-4-6-thinking"),
            ("gpt-oss-120b-medium", "gpt-oss-120b-medium"),
        ],
    )
    def test_normalizes_only_known_gemini_effort_suffixes(self, model: str, expected: str) -> None:
        assert PROBE_MODULE.normalize_antigravity_model_id(model) == expected

    def test_empty_output_returns_no_models(self) -> None:
        assert PROBE_MODULE.parse_antigravity_models("") == set()
        assert PROBE_MODULE.parse_antigravity_models("Available models:\n") == set()

    def test_failed_command_output_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(PROBE_MODULE.shutil, "which", lambda _binary: "/usr/bin/agy")
        monkeypatch.setattr(
            PROBE_MODULE.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=1,
                stdout="gemini-3.7-flash-high",
                stderr="authentication failed",
            ),
        )

        assert PROBE_MODULE.probe_antigravity_cli() == set()


def test_probe_routing_matches_registry_keys_exactly() -> None:
    registry_path = _PROBE_PATH.parent.parent / "src" / "crossby" / "data" / "models.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_keys = {key for key in registry if not key.startswith("_")}

    assert set(PROBE_MODULE._MODEL_PROBES) == registry_keys
    assert set(PROBE_MODULE._MODEL_PROBE_SOURCES) == registry_keys
