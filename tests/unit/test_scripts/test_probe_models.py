"""Tests for scripts/probe_models.py — the Claude docs scrape pattern.

``scripts/`` is not an importable package, so the module is loaded from its
file path. Only the scrape regex is exercised here; the interactive probe flow
is a developer utility and not unit-tested.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

_PROBE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "probe_models.py"


def _load_claude_pattern() -> str:
    spec = importlib.util.spec_from_file_location("probe_models", _PROBE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module._SCRAPE_PATTERNS["claude"])


CLAUDE_PATTERN = _load_claude_pattern()


class TestClaudeScrapePattern:
    """The family-anchored pattern matches current IDs, excludes noisy ones."""

    def test_matches_single_number_versions(self) -> None:
        assert re.findall(CLAUDE_PATTERN, "claude-sonnet-5") == ["claude-sonnet-5"]
        assert re.findall(CLAUDE_PATTERN, "claude-fable-5") == ["claude-fable-5"]

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
