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
        assert re.findall(CLAUDE_PATTERN, "claude-fable-5-1") == ["claude-fable-5-1"]
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


def test_probe_claude_standardizes_dashed_docs_ids_to_registry_dotted_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The published docs page uses Claude's own dashed ID format
    # (claude-haiku-4-5), but models.json stores the internal dotted
    # convention (claude-haiku-4.5) — probe_claude must bridge the two so
    # the registry diff doesn't report every model as both missing and new.
    monkeypatch.setattr(
        PROBE_MODULE,
        "_scrape_models",
        lambda _tool: {
            "claude-fable-5-1",
            "claude-haiku-4-5",
            "claude-opus-4-6",
            "claude-sonnet-5",
        },
    )
    assert PROBE_MODULE.probe_claude() == {
        "claude-fable-5.1",
        "claude-haiku-4.5",
        "claude-opus-4.6",
        "claude-sonnet-5",
    }


def test_claude_docs_parser_uses_current_lineup_not_embedded_legacy_metadata() -> None:
    page = """
    <section id="latest-models-comparison">
      claude-fable-5-1 claude-opus-5 claude-sonnet-5 claude-haiku-4-5-20251001
    </section>
    <section id="using-the-models-api"></section>
    <script>claude-opus-3 claude-sonnet-3-5 claude-haiku-3</script>
    """

    assert PROBE_MODULE.parse_documented_models("claude", page) == {
        "claude-fable-5-1",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-4-5",
    }


@pytest.mark.parametrize(
    ("dated_id", "expected_alias"),
    [
        # One-part version: the alias regex must not require two components,
        # or the model is dropped from discovery entirely.
        ("claude-sonnet-5-20260101", "claude-sonnet-5"),
        # Dashed two-part version — the shape the docs publish today.
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        # Dotted two-part version, with and without the -v<n> republish suffix.
        ("claude-haiku-4.5-20260101", "claude-haiku-4.5"),
        ("claude-opus-4.8-20260101-v2", "claude-opus-4.8"),
    ],
)
def test_claude_docs_parser_collapses_every_dated_version_shape(
    dated_id: str, expected_alias: str
) -> None:
    page = f'<section id="latest-models-comparison">{dated_id}</section>'
    page += '<section id="using-the-models-api"></section>'

    assert PROBE_MODULE.parse_documented_models("claude", page) == {expected_alias}


def test_claude_docs_parser_does_not_truncate_dotted_dated_snapshots() -> None:
    # Regression: the generic scrape pattern can backtrack past the version's
    # second component when a date follows, yielding a bogus "claude-haiku-4"
    # that the registry diff would report as a brand-new model to add.
    page = """
    <section id="latest-models-comparison">
      claude-haiku-4.5-20260101 claude-sonnet-5-20260101 claude-opus-4.8
    </section>
    <section id="using-the-models-api"></section>
    """

    assert PROBE_MODULE.parse_documented_models("claude", page) == {
        "claude-haiku-4.5",
        "claude-sonnet-5",
        "claude-opus-4.8",
    }


def test_claude_docs_parser_returns_empty_when_current_lineup_anchors_missing() -> None:
    assert PROBE_MODULE.parse_documented_models("claude", "claude-fable-5-1") == set()


def test_copilot_pattern_captures_mai_id_without_matching_prose() -> None:
    text = "Maintain support for mai-code-1-flash and gemini-3.6-flash in the main catalog."
    assert set(re.findall(COPILOT_PATTERN, text)) == {
        "mai-code-1-flash",
        "gemini-3.6-flash",
    }


def test_copilot_docs_parser_uses_model_table_not_table_of_contents() -> None:
    # Mirrors the real GitHub docs page: nav links and prose repeat the
    # "Supported models" / "Tool availability values" text before the real
    # heading, and unrelated prose (an example CLI invocation) mentions a
    # model-like token that isn't actually in the supported-models table.
    page = """
    <a href="#supported-models">Supported models</a>
    <a href="#tool-availability-values">Tool availability values</a>
    Run <code>copilot config --repo model gpt-5.2</code> to override.
    <h2 id="supported-models">Supported models</h2>
    `claude-sonnet-4.6` | General-purpose coding
    `gpt-5.4` | Complex reasoning
    `gemini-3.6-flash` | Fast responses
    `mai-code-1-flash` | Adaptive coding
    <h2 id="tool-availability-values">Tool availability values</h2>
    `gpt-not-a-model-table-entry`
    """

    assert PROBE_MODULE.parse_documented_models("copilot", page) == {
        "claude-sonnet-4.6",
        "gpt-5.4",
        "gemini-3.6-flash",
        "mai-code-1-flash",
    }


def test_copilot_docs_parser_returns_empty_when_headings_missing() -> None:
    assert PROBE_MODULE.parse_documented_models("copilot", "no anchors here") == set()


def test_catalog_diff_preserves_exact_provider_spelling() -> None:
    registered = {"google/antigravity-claude-sonnet-4.6"}
    discovered = {"google/antigravity-claude-sonnet-4-6"}

    assert PROBE_MODULE.model_catalog_diff(registered, discovered) == (
        registered,
        discovered,
    )


class TestAntigravityModelParsing:
    def test_extracts_current_gemini_flash_models_and_deduplicates_effort_variants(
        self,
    ) -> None:
        output = """
        Available models:
          gemini-3.8-flash-low       Gemini 3.8 Flash (Low)
          gemini-3.8-flash-medium    Gemini 3.8 Flash (Medium)
          gemini-3.8-flash-high      Gemini 3.8 Flash (High)
          gemini-3.7-flash-low       Gemini 3.7 Flash (Low)
          gemini-3.7-flash-medium    Gemini 3.7 Flash (Medium)
          gemini-3.7-flash-high      Gemini 3.7 Flash (High)
          gemini-3.7-flash-high      Gemini 3.7 Flash (High)
          claude-opus-4-6-thinking   Claude Opus 4.6 Thinking
          gpt-oss-120b-medium        GPT OSS 120B
        """

        assert PROBE_MODULE.parse_antigravity_models(output) == {
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "claude-opus-4-6-thinking",
            "gpt-oss-120b-medium",
        }

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gemini-3.8-flash-low", "gemini-3.8-flash"),
            ("gemini-3.8-flash-medium", "gemini-3.8-flash"),
            ("gemini-3.8-flash-high", "gemini-3.8-flash"),
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

    def test_extracts_models_without_hard_coded_family_prefix(self) -> None:
        output = """
        Available models:
          mai-code-1-flash           MAI Code 1 (Flash)
          grok-4-fast                Grok 4 Fast
          o3-mini                    O3 Mini
        """

        assert PROBE_MODULE.parse_antigravity_models(output) == {
            "mai-code-1-flash",
            "grok-4-fast",
            "o3-mini",
        }

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


class TestDeprecationParsing:
    """Provider retirement pages feed the probe's deprecation filter."""

    def test_claude_reads_status_table_not_replacements(self) -> None:
        # The history below the status table names each retirement's
        # replacement, which is an active model and must not be picked up.
        page = """
        <a id="model-status" href="#model-status">Model status</a>
        <h2 id="model-status">Model status</h2>
        <table>
        <tr><td>claude-opus-5-5</td><td>Active</td></tr>
        <tr><td>claude-mythos-preview</td><td>Deprecated</td></tr>
        <tr><td>claude-opus-4-1-20250805</td><td>Retired</td></tr>
        <tr><td>claude-haiku-4-5-20251001</td><td>Active</td></tr>
        </table>
        <h2 id="deprecation-history">Deprecation history</h2>
        <tr><td>claude-opus-4-1-20250805</td><td>claude-opus-4-8</td></tr>
        """

        assert PROBE_MODULE.parse_claude_deprecations(page) == {
            "claude-mythos-preview",
            "claude-opus-4.1",
        }

    def test_copilot_retirement_table_slugs_display_names(self) -> None:
        page = """
        <a href="#model-retirement-history">Model retirement history</a>
        <h2 id="model-retirement-history">Model retirement history</h2>
        <table>
        <tr><th>Model name</th><th>Retirement date</th><th>Suggested alternative</th></tr>
        <tr><th scope="row">Claude Sonnet 4.6<sup><a href="#fn">5</a></sup></th>
            <td>2026-09-01</td><td>Claude Sonnet 5</td></tr>
        <tr><th scope="row">GPT-5.2-Codex</th><td>2026-06-05</td><td>GPT-5.3-Codex</td></tr>
        <tr><th scope="row">Claude Opus 4.6 (fast mode) (preview)</th>
            <td>2026-06-29</td><td>Claude Opus 4.8 (fast mode) (preview)</td></tr>
        </table>
        <h2 id="next-steps">Next steps</h2>
        <table><tr><th>Claude Opus 5</th></tr></table>
        """

        # A retired fast-mode variant must not read as its base model retiring,
        # and the suggested alternatives are never collected.
        assert PROBE_MODULE.parse_copilot_retirements(page) == {
            "claude-sonnet-4.6",
            "gpt-5.2-codex",
            "claude-opus-4.6-fast",
        }

    def test_codex_deprecated_section_excludes_recommended_replacements(self) -> None:
        page = """
        <nav><a id="recommended-models"></a><a id="other-models"></a>
        <a id="deprecated-codex-models"></a>
        <a id="configure-your-default-local-model"></a></nav>
        <h2 id="recommended-models">Recommended models</h2>
        <code>codex -m gpt-6-sol</code> <code>codex -m gpt-6-luna</code>
        <h2 id="other-models">Other models</h2>
        <code>codex -m gpt-5.5</code>
        <h2 id="deprecated-codex-models">Deprecated Codex models</h2>
        <p>Replace gpt-5.4 with gpt-6-sol and gpt-5.4-mini with gpt-6-luna.</p>
        <p>The gpt-5.2 and gpt-5.3-codex models are already deprecated.</p>
        <h2 id="configure-your-default-local-model">Configure</h2>
        <p>model = "gpt-6-sol"</p>
        """

        assert PROBE_MODULE.parse_codex_deprecations(page) == {
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.2",
            "gpt-5.3-codex",
        }

    def test_opencode_combines_models_dev_status_and_copilot_retirements(self) -> None:
        catalog = {
            "opencode": {
                "models": {
                    "hy3-free": {"name": "Hy3", "status": "deprecated"},
                    "mimo-v2.6-flash-free": {"name": "MiMo"},
                }
            },
            "github-copilot": {
                "models": {
                    "mai-code-1-flash-picker": {"name": "MAI-Code-1-Flash"},
                    "claude-opus-5": {"name": "Claude Opus 5"},
                }
            },
        }

        deprecated = PROBE_MODULE.parse_opencode_deprecations(
            json.dumps(catalog), {"claude-opus-4.7", "mai-code-1-flash"}
        )

        assert "opencode/hy3-free" in deprecated
        assert "opencode/mimo-v2.6-flash-free" not in deprecated
        assert "github-copilot/claude-opus-4.7" in deprecated
        assert "github-copilot/claude-opus-4.7-fast" in deprecated
        # OpenCode spells MAI-Code-1-Flash differently; matched by display name.
        assert "github-copilot/mai-code-1-flash-picker" in deprecated
        assert "github-copilot/claude-opus-5" not in deprecated

    def test_gemini_counts_only_passed_or_imminent_shutdowns(self) -> None:
        page = """
        <tr><td>gemini-3-pro-preview</td><td>November 18, 2025</td>
            <td>March 9, 2026</td><td>gemini-3.1-pro-preview</td></tr>
        <tr><td>gemini-2.5-flash-image</td><td>October 2, 2025</td>
            <td>October 2, 2026</td><td>gemini-3.1-flash-image-preview</td></tr>
        <tr><td>gemini-3.5-live-translate-preview</td><td>June 2026</td>
            <td>No shutdown date announced</td></tr>
        <tr><td>gemini-3.1-flash-lite</td><td>May 7, 2026</td>
            <td>May 7, 2027</td><td>gemini-3.5-flash-lite</td></tr>
        <tr><td>gemini-embedding-001</td><td>July 14, 2025</td>
            <td>May 14, 2028</td><td>gemini-embedding-2</td></tr>
        """

        # Already shut down, and shutting down within the horizon, count; a
        # launch-time lifecycle date a year or more out does not.
        assert PROBE_MODULE.parse_gemini_deprecations(
            page, PROBE_MODULE.datetime.date(2026, 9, 23)
        ) == {"gemini-3-pro-preview", "gemini-2.5-flash-image"}

    def test_opencode_applies_gemini_shutdowns_to_google_prefix(self) -> None:
        deprecated = PROBE_MODULE.parse_opencode_deprecations(
            json.dumps({}), set(), {"gemini-3-pro-preview"}
        )
        assert deprecated == {"google/gemini-3-pro-preview"}

    def test_antigravity_uses_gemini_deprecations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        urls: list[str] = []

        def fake_fetch(url: str) -> str:
            urls.append(url)
            return "<td>gemini-3-pro-preview</td><td>November 18, 2025</td><td>March 9, 2026</td>"

        monkeypatch.setattr(PROBE_MODULE, "_fetch", fake_fetch)

        assert PROBE_MODULE.probe_deprecations("antigravity-cli") == {"gemini-3-pro-preview"}
        assert urls == [PROBE_MODULE._DEPRECATION_URLS["gemini"]]

    @pytest.mark.parametrize(
        "parser",
        ["parse_claude_deprecations", "parse_copilot_retirements", "parse_codex_deprecations"],
    )
    def test_missing_anchors_yield_nothing(self, parser: str) -> None:
        assert getattr(PROBE_MODULE, parser)("no anchors here") == set()

    def test_malformed_models_dev_payload_yields_nothing(self) -> None:
        assert PROBE_MODULE.parse_opencode_deprecations("<html>", set()) == set()


class TestCliArgExpectations:
    def test_codex_expects_the_flags_the_adapter_emits(self) -> None:
        # CodexAdapter uses ``-a never`` for yolo and ``-c
        # model_reasoning_effort=...`` for effort, never ``--yolo``.
        codex = PROBE_MODULE._EXPECTED_FLAGS["codex"]
        assert codex["yolo"] == "--ask-for-approval"
        assert codex["model_reasoning_effort"] == "--config"

    def test_subcommand_help_is_searched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ``--variant`` only appears in ``opencode run --help``.
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
            calls.append(cmd)
            stdout = "--variant  model variant" if cmd[1:] == ["run", "--help"] else "run  -s"
            stdout += " --model"
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        monkeypatch.setattr(PROBE_MODULE.shutil, "which", lambda _binary: "/usr/bin/opencode")
        monkeypatch.setattr(PROBE_MODULE.subprocess, "run", fake_run)

        assert PROBE_MODULE.probe_cli_args("opencode")["effort"] is True
        assert ["opencode", "run", "--help"] in calls
