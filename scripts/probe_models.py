#!/usr/bin/env python3
"""Developer utility: probe AI CLIs and websites to discover new models.

Compares discovered models against src/crossby/data/models.json and reports
any differences. Exits 1 if updates are needed.

Each provider's own retirement page is read too: newly discovered models that
the provider marks deprecated or retired are reported as ignored rather than
as additions, and catalog entries it marks that way are reported for removal.
"""

from __future__ import annotations

import datetime
import html
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from crossby.ai_tools.base import AbstractAITool

# Must match src/crossby/data/models.json structure
JSON_PATH = Path(__file__).parent.parent / "src" / "crossby" / "data" / "models.json"

_DOCS_URLS: dict[str, str] = {
    "claude": "https://platform.claude.com/docs/en/about-claude/models/overview",
    "copilot": (
        "https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference"
    ),
    "codex": "https://learn.chatgpt.com/docs/models",
}

# Pages where each provider publishes retirements. The model probes report what
# a CLI or docs page *mentions*, and those sources keep listing a model until
# its shutdown date (Codex's "other models" cards, Copilot's CLI reference,
# OpenCode's bundled catalog), so without this list the probe would keep asking
# to re-add models the catalog dropped on purpose.
_DEPRECATION_URLS: dict[str, str] = {
    "claude": "https://platform.claude.com/docs/en/about-claude/model-deprecations",
    "copilot": "https://docs.github.com/en/copilot/reference/ai-models/supported-models",
    "codex": _DOCS_URLS["codex"],
    "opencode": "https://models.dev/api.json",
    # Shared by antigravity-cli (bare Gemini IDs) and OpenCode's google/ prefix.
    "gemini": "https://ai.google.dev/gemini-api/docs/deprecations",
}

# Google publishes a shutdown date for most models at launch, often a year or
# more out. Only a shutdown that has passed or falls within this window counts
# as a retirement; a far-off lifecycle date alone does not.
_GEMINI_SHUTDOWN_HORIZON = datetime.timedelta(days=180)

_SCRAPE_PATTERNS: dict[str, str] = {
    # Family-anchored, word-boundary pattern. Matches single-number families
    # (claude-sonnet-5, claude-fable-5) as well as dotted ones (claude-opus-4.8)
    # while excluding dated snapshots (-20251001), -v1 variants, and docs-page
    # slug run-ons.
    "claude": r"claude-(?:opus|sonnet|haiku|fable)-\d(?:[.-]\d)?(?!\d|-\d|-v\d)\b",
    "copilot": (
        r"(?:claude|gemini|gpt|codex|o[0-9])[a-zA-Z0-9._-]*|"
        r"mai-[a-zA-Z0-9._-]+"
    ),
    "codex": r"gpt-[0-9][.0-9]*[a-zA-Z0-9._-]*",
}

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
# Version part mirrors the family-anchored scrape pattern above, so a dated
# snapshot collapses to its alias whether the docs publish a one-part version
# (claude-sonnet-5-20260101), a dashed two-part one (claude-haiku-4-5-20251001),
# or a dotted one (claude-haiku-4.5-20260101).
_CLAUDE_DATED_ALIAS_RE = re.compile(
    r"(claude-(?:opus|sonnet|haiku|fable)-\d(?:[.-]\d)?)-\d{8}(?:-v\d+)?"
)
# Provider-agnostic: matches the model-ID column of `agy models` output
# regardless of vendor prefix (gemini, claude, gpt, or a future mai/grok/o3
# family), as long as it looks like a multi-segment identifier (e.g.
# "foo-1.2-bar") rather than a plain word from the description column.
_ANTIGRAVITY_MODEL_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[./-][a-z0-9._-]+)+$")
_ANTIGRAVITY_GEMINI_EFFORT_RE = re.compile(
    r"^(gemini-\d+(?:\.\d+)*-(?:flash|pro))-(?:low|medium|high)$"
)

# Per-tool expected CLI flag patterns, keyed by capability name.
# Values are substrings to search for in `--help` / `-h` output.
_EXPECTED_FLAGS: dict[str, dict[str, str]] = {
    "codex": {
        # CodexAdapter.yolo_args() is ``-a never``; ``--yolo`` is a hidden alias
        # that would also drop the sandbox, and crossby never emits it.
        "yolo": "--ask-for-approval",
        "ask_for_approval": "--ask-for-approval",
        "headless": "exec",
        # Effort is passed as ``-c model_reasoning_effort=...``; the key name
        # itself never appears in --help, only the ``--config`` flag does.
        "model_reasoning_effort": "--config",
        "profile": "--profile",
        "image": "--image",
    },
    "antigravity-cli": {
        "headless": "--print",
        "resume": "--conversation",
        "model": "--model",
        "plan_mode": "--mode",
        "sandbox": "--sandbox",
        "yolo": "--dangerously-skip-permissions",
    },
    "claude": {
        "headless": "--print",
        "resume": "--resume",
        "yolo": "--dangerously-skip-permissions",
        "permission_mode": "--permission-mode",
    },
    "copilot": {
        "headless": "--prompt",
        "resume": "--resume",
        "yolo": "--yolo",
        "allow_tool": "--allow-tool",
    },
    "cursor": {
        "list_models": "--list-models",
        "headless": "--print",
        "model": "--model",
        "force": "--force",
    },
    "opencode": {
        "headless": "run",
        "model": "--model",
        "resume": "-s",
        "effort": "--variant",
    },
}

# Subcommands whose ``--help`` is also searched, for flags that live on the
# subcommand crossby actually runs rather than on the top-level command.
_EXTRA_HELP_COMMANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    # ``--variant`` is an option of ``opencode run`` (the headless path).
    "opencode": (("run", "--help"),),
}

# Maps capability names in _EXPECTED_FLAGS to AIToolCapabilities boolean fields.
_CAP_FIELD_MAP: dict[str, str] = {
    "headless": "supports_headless",
    "resume": "supports_resume",
    "yolo": "supports_yolo",
    "effort": "supports_effort",
    "model_reasoning_effort": "supports_effort",
}


def _token_match(pattern: str, text: str) -> bool:
    """Check if a CLI flag or subcommand appears as a standalone token in help text.

    Long flags (``--foo``) are specific enough for direct substring matching.
    Short flags (``-f``) require surrounding whitespace/punctuation so they are
    not confused with options like ``-foo``.  Plain words (subcommands such as
    ``run`` or ``exec``) use word-boundary matching to avoid partial hits like
    ``truncate`` matching ``run``.
    """
    escaped = re.escape(pattern)
    if pattern.startswith("--"):
        return pattern in text
    if pattern.startswith("-"):
        return bool(re.search(r"(?:^|\s)" + escaped + r"(?:\s|,|\[|$)", text, re.MULTILINE))
    return bool(re.search(r"\b" + escaped + r"\b", text))


def _fetch(url: str) -> str | None:
    """Return the body at ``url``, or None when curl is missing or the fetch fails."""
    if not shutil.which("curl"):
        return None
    try:
        result = subprocess.run(
            ["curl", "-fsSL", "--max-time", "10", url],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return None
    return result.stdout if result.returncode == 0 else None


def _scrape_models(tool: str) -> set[str]:
    """Scrape model IDs from docs."""
    if tool not in _DOCS_URLS:
        return set()
    text = _fetch(_DOCS_URLS[tool])
    return parse_documented_models(tool, text) if text is not None else set()


def _html_section(text: str, start_id: str, end_marker: str) -> str | None:
    """Return the HTML between ``id="<start_id>"`` and the next ``end_marker``.

    Docs sites repeat heading IDs in their tables of contents, so the section
    starts at the last occurrence: the heading itself follows its TOC entries.
    """
    start = text.rfind(f'id="{start_id}"')
    if start < 0:
        return None
    end = text.find(end_marker, start)
    return text[start:end] if end >= 0 else None


def _html_to_text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment)))


def parse_claude_deprecations(text: str) -> set[str]:
    """Extract Deprecated/Retired IDs from the Claude model-deprecations page.

    Reads only the "Model status" table: the history below it also names each
    retirement's *replacement*, which is an active model. IDs are returned in
    the registry's dotted alias form (``claude-opus-4.1``).
    """
    section = _html_section(text, "model-status", 'id="deprecation-history"')
    if section is None:
        return set()
    adapter = AbstractAITool.get("claude")
    rows = re.findall(r"(claude-[a-z0-9-]+)\s+(?:Deprecated|Retired)\b", _html_to_text(section))
    return {adapter.standardize_model_id(_CLAUDE_DATED_ALIAS_RE.sub(r"\1", m)) for m in rows}


def _copilot_model_slug(name: str) -> str:
    """Turn a Copilot display name into its CLI model ID.

    ``Claude Sonnet 4.6`` -> ``claude-sonnet-4.6``; ``Claude Opus 4.6 (fast
    mode) (preview)`` -> ``claude-opus-4.6-fast``, so retiring a fast-mode
    variant never reads as retiring its base model.
    """
    lower = name.lower()
    fast = "(fast mode)" in lower
    lower = re.sub(r"\([^)]*\)", " ", lower)
    slug = "-".join(lower.split())
    return f"{slug}-fast" if fast else slug


def parse_copilot_retirements(text: str) -> set[str]:
    """Extract retired and scheduled-for-retirement IDs from Copilot's
    supported-models page ("Model retirement history" table)."""
    section = _html_section(text, "model-retirement-history", "</table>")
    if section is None:
        return set()
    names = re.findall(r"<tr>\s*<t[hd][^>]*>(.*?)</t[hd]>", section, flags=re.S)
    slugs = {
        _copilot_model_slug(_html_to_text(re.sub(r"<sup>.*?</sup>", "", n, flags=re.S)))
        for n in names
    }
    return {s for s in slugs if s and s != "model-name"}


def parse_codex_deprecations(text: str) -> set[str]:
    """Extract model IDs named in the Codex docs' "Deprecated Codex models" section.

    That prose also names the replacements ("Replace gpt-5.4 with gpt-6-sol"),
    so IDs offered under "Recommended models" are never treated as deprecated.
    """
    section = _html_section(
        text, "deprecated-codex-models", 'id="configure-your-default-local-model"'
    )
    recommended = _html_section(text, "recommended-models", 'id="other-models"')
    if section is None or recommended is None:
        return set()
    mentioned = set(re.findall(r"\b(gpt-\d[a-z0-9.-]*[a-z0-9])", _html_to_text(section)))
    return mentioned - set(re.findall(r"codex -m (gpt-[a-z0-9._-]+)", recommended))


def parse_gemini_deprecations(text: str, today: datetime.date) -> set[str]:
    """Extract Google models whose shutdown date has passed or is imminent.

    Each row of the Gemini API deprecations tables reads "<model> <release
    date> <shutdown date> <replacement>"; rows whose shutdown column says "No
    shutdown date announced" never match.
    """
    date = r"[A-Z][a-z]+ \d{1,2}, \d{4}"
    rows = re.findall(
        rf"\b((?:gemini|gemma|veo|lyria|imagen)-[a-z0-9.-]*[a-z0-9]) "
        rf"(?:{date}|[A-Z][a-z]+ \d{{4}}) ({date})",
        _html_to_text(text),
    )
    cutoff = today + _GEMINI_SHUTDOWN_HORIZON
    return {
        model
        for model, shutdown in rows
        if datetime.datetime.strptime(shutdown, "%B %d, %Y").date() <= cutoff
    }


def parse_opencode_deprecations(
    text: str, copilot_retired: set[str], gemini_retired: frozenset[str] | set[str] = frozenset()
) -> set[str]:
    """Deprecated ``provider/model`` IDs for OpenCode.

    models.dev (OpenCode's catalog source) flags deprecated models directly.
    Google's shutdowns apply to the ``google/`` prefix, and its
    ``github-copilot`` provider lags GitHub's own retirement table, so
    Copilot retirements apply to that prefix too: by exact ID, by the fast-mode
    variant of a retired model, and by display name for IDs OpenCode spells
    differently (``mai-code-1-flash-picker`` is "MAI-Code-1-Flash").
    """
    try:
        catalog: dict[str, Any] = json.loads(text)
    except ValueError:
        return set()
    deprecated: set[str] = set()
    for provider, entry in catalog.items():
        if not isinstance(entry, dict):
            continue
        for model_id, model in entry.get("models", {}).items():
            if not isinstance(model, dict):
                continue
            retired_by_name = (
                provider == "github-copilot"
                and _copilot_model_slug(str(model.get("name", ""))) in copilot_retired
            )
            if model.get("status") == "deprecated" or retired_by_name:
                deprecated.add(f"{provider}/{model_id}")
    for model in copilot_retired:
        deprecated |= {f"github-copilot/{model}", f"github-copilot/{model}-fast"}
    return deprecated | {f"google/{model}" for model in gemini_retired}


def probe_deprecations(tool: str) -> set[str]:
    """Return the model IDs ``tool``'s provider marks deprecated or retired."""
    if tool == "antigravity-cli":
        text = _fetch(_DEPRECATION_URLS["gemini"])
        return parse_gemini_deprecations(text, datetime.date.today()) if text else set()
    if tool == "opencode":
        text = _fetch(_DEPRECATION_URLS["opencode"])
        if not text:
            return set()
        return parse_opencode_deprecations(
            text, probe_deprecations("copilot"), probe_deprecations("antigravity-cli")
        )
    parsers: dict[str, Callable[[str], set[str]]] = {
        "claude": parse_claude_deprecations,
        "copilot": parse_copilot_retirements,
        "codex": parse_codex_deprecations,
    }
    if tool not in parsers:
        return set()
    text = _fetch(_DEPRECATION_URLS[tool])
    return parsers[tool](text) if text else set()


def _pattern_matches(pattern: str, text: str) -> set[str]:
    matches = re.findall(pattern, text)
    return {match if isinstance(match, str) else match[0] for match in matches}


def parse_documented_models(tool: str, text: str) -> set[str]:
    """Extract model IDs from a tool's published documentation.

    GitHub's page repeats section titles in its table of contents and other
    prose, so this anchors on the unique heading-id attributes rather than
    the visible heading text to isolate the actual table.
    """
    pattern = _SCRAPE_PATTERNS[tool]

    if tool == "claude":
        # The model hub embeds legacy model metadata after the visible page.
        # Restrict discovery to the current-lineup comparison so retired IDs
        # do not appear as newly available merely because their model cards
        # remain linked from the page payload.
        start = text.find('id="latest-models-comparison"')
        end = text.find('id="using-the-models-api"', start if start >= 0 else 0)
        if start < 0 or end < 0:
            return set()
        current_lineup = text[start:end]
        # The current table may publish only a dated snapshot (currently Haiku
        # 4.5), while Crossby's Claude adapter stores and launches the stable
        # alias. Collapse that date here before adapter punctuation handling.
        models = set(_CLAUDE_DATED_ALIAS_RE.findall(current_lineup))
        # Consume the dated tokens before the generic pass. Left in place, a
        # dotted snapshot lets the generic pattern backtrack past the version's
        # second component and emit a truncated ID (claude-haiku-4.5-20260101 ->
        # "claude-haiku-4"), which would be reported as a new catalog model.
        models.update(_pattern_matches(pattern, _CLAUDE_DATED_ALIAS_RE.sub(" ", current_lineup)))
        return models

    if tool == "codex":
        full_matches = re.findall(r"codex -m (gpt-[a-z0-9._-]+)", text)
        if full_matches:
            return set(full_matches)

    if tool == "copilot":
        start = text.find('id="supported-models"')
        end = text.find('id="tool-availability-values"', start if start >= 0 else 0)
        if start < 0 or end < 0:
            return set()
        return _pattern_matches(pattern, text[start:end])

    return _pattern_matches(pattern, text)


def probe_claude() -> set[str]:
    """Discover Claude models from the published docs.

    Claude Code has no non-interactive model-list command: ``claude models`` is
    interpreted as a *prompt* and returns prose, not a model list, so it cannot
    be scraped. The published models docs page is therefore the sole source of
    truth for the Claude catalog.

    The docs page uses Claude's own dashed ID format (``claude-haiku-4-5``),
    but ``models.json`` stores the internal dotted convention
    (``claude-haiku-4.5``, see ``AbstractAITool.standardize_model_id``), so
    scraped IDs are converted before comparison.
    """
    adapter = AbstractAITool.get("claude")
    return {adapter.standardize_model_id(model) for model in _scrape_models("claude")}


def probe_copilot() -> set[str]:
    if shutil.which("copilot"):
        try:
            res = subprocess.run(
                ["copilot", "--model", "x"], capture_output=True, text=True, timeout=15
            )
            out = res.stdout + res.stderr
            matches = re.findall(_SCRAPE_PATTERNS["copilot"], out)
            models = {re.sub(r"[.,;]+$", "", m) for m in matches if not m.startswith(".")}
            if models:
                return models
        except Exception:
            pass
    return _scrape_models("copilot")


def probe_cursor() -> set[str]:
    """Probe Cursor CLI (agent) via ``agent --list-models``."""
    if not shutil.which("agent"):
        return set()
    try:
        res = subprocess.run(["agent", "--list-models"], capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            models = set()
            for line in res.stdout.splitlines():
                # Lines like: "sonnet-4.6 - Claude 4.6 Sonnet"
                stripped = line.strip()
                if stripped and " - " in stripped:
                    model_id = stripped.split(" - ")[0].strip()
                    if model_id and not model_id.startswith(("Available", "Tip:")):
                        models.add(model_id)
            if models:
                return models
    except Exception:
        pass
    return set()


def normalize_antigravity_model_id(model: str) -> str:
    """Collapse only agy's known Gemini effort-expanded IDs to base IDs.

    Non-Gemini suffixes such as ``claude-opus-4-6-thinking`` and
    ``gpt-oss-120b-medium`` are part of those tools' model names and must stay
    intact. Gemini suffixes other than low/medium/high are also preserved so a
    newly introduced variant is reported rather than silently rewritten.
    """
    match = _ANTIGRAVITY_GEMINI_EFFORT_RE.fullmatch(model)
    return match.group(1) if match else model


def parse_antigravity_models(output: str) -> set[str]:
    """Extract and canonicalize model IDs from ``agy models`` output.

    Takes the first whitespace-delimited token of each line — the model-ID
    column — without assuming which vendor families can appear there.
    """
    models: set[str] = set()
    for line in output.splitlines():
        clean_line = _ANSI_ESCAPE_RE.sub("", line).strip()
        if not clean_line:
            continue
        first_token = clean_line.split(maxsplit=1)[0]
        if _ANTIGRAVITY_MODEL_ID_RE.match(first_token):
            models.add(normalize_antigravity_model_id(first_token))
    return models


def probe_antigravity_cli() -> set[str]:
    """Probe Antigravity CLI via its authoritative ``agy models`` command."""
    if not shutil.which("agy"):
        return set()
    try:
        res = subprocess.run(["agy", "models"], capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            return parse_antigravity_models(res.stdout)
    except Exception:
        pass
    return set()


def probe_codex() -> set[str]:
    """Read available models from codex's local cache file (~/.codex/models_cache.json).

    Falls back to web scraping if the cache doesn't exist or yields no models.
    """
    cache = Path.home() / ".codex" / "models_cache.json"
    if cache.exists():
        try:
            with open(cache, encoding="utf-8") as f:
                data = json.load(f)
            models = {m["slug"] for m in data.get("models", []) if m.get("visibility") == "list"}
            if models:
                return models
        except Exception:
            pass
    return _scrape_models("codex")


def probe_opencode() -> set[str]:
    try:
        res = subprocess.run(["opencode", "models"], capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            models = set()
            for line in res.stdout.splitlines():
                if line.strip() and not line.startswith(("#", "-")):
                    parts = line.split()
                    if parts and "/" in parts[0]:
                        models.add(parts[0])
            if models:
                return models
    except Exception:
        pass
    return set()


# These keys deliberately match every non-meta key in data/models.json. Keeping
# routing in one table makes a renamed/replaced tool visible to unit tests rather
# than silently skipping its registry section.
_MODEL_PROBES: dict[str, Callable[[], set[str]]] = {
    "claude": probe_claude,
    "cursor": probe_cursor,
    "copilot": probe_copilot,
    "antigravity-cli": probe_antigravity_cli,
    "codex": probe_codex,
    "opencode": probe_opencode,
}

_MODEL_PROBE_SOURCES: dict[str, str] = {
    "claude": "published Claude model documentation",
    "cursor": "agent --list-models",
    "copilot": "Copilot CLI picker or published CLI reference",
    "antigravity-cli": "agy models",
    "codex": "Codex local model cache or published Codex guidance",
    "opencode": "opencode models",
}


def model_catalog_diff(registered: set[str], discovered: set[str]) -> tuple[set[str], set[str]]:
    """Return retained-but-unseen and newly discovered exact model IDs.

    Provider prefixes and punctuation are part of a tool's accepted model ID,
    so this comparison deliberately does not run adapter normalization.
    """
    return registered - discovered, discovered - registered


def probe_cli_args(tool: str) -> dict[str, bool]:
    """Run ``<tool> --help`` and check for expected flag patterns.

    Tools listed in ``_EXTRA_HELP_COMMANDS`` also have those subcommands'
    ``--help`` searched.

    Returns a dict mapping capability_name -> found (bool) for each entry in
    ``_EXPECTED_FLAGS[tool]``.  Returns an empty dict when the tool is not in
    ``_EXPECTED_FLAGS``, its binary is not installed, or the help output is empty.
    """
    expected = _EXPECTED_FLAGS.get(tool)
    if not expected:
        return {}

    try:
        adapter = AbstractAITool.get(tool)
    except ValueError:
        return {}

    binary = adapter.capabilities().binary
    if not shutil.which(binary):
        return {}

    combined = ""
    help_commands = [("--help",), ("-h",), *_EXTRA_HELP_COMMANDS.get(tool, ())]
    for help_args in help_commands:
        try:
            result = subprocess.run(
                [binary, *help_args],
                capture_output=True,
                text=True,
                timeout=15,
            )
            combined += result.stdout + result.stderr
        except Exception:
            pass

    if not combined.strip():
        return {}

    return {cap_name: _token_match(pattern, combined) for cap_name, pattern in expected.items()}


def report_cli_args() -> bool:
    """Compare probed CLI flags against adapter capabilities and print a report.

    For each tool in ``_EXPECTED_FLAGS``:
    - MISSING: expected flags not found in ``--help`` (possible deprecation/rename)
    - CAPABILITY MISMATCH: flag found but the adapter declares no support for it
    - Reports a clean status if all flags match expectations

    Returns ``True`` if any issues (missing flags or capability mismatches) were
    detected, ``False`` when everything looks correct.
    """
    from crossby.ui.console import console

    console.header("CLI Arguments")

    has_cli_diff = False

    for tool in _EXPECTED_FLAGS:
        try:
            adapter = AbstractAITool.get(tool)
        except ValueError:
            continue

        caps = adapter.capabilities()
        if not shutil.which(caps.binary):
            console.warn(f"[{tool}] CLI not installed, skipping argument probe.")
            continue

        found_flags = probe_cli_args(tool)
        if not found_flags:
            console.warn(f"[{tool}] Could not probe CLI arguments (--help returned no output).")
            continue

        console.header(f"Tool: {tool}")

        missing = [cap for cap, found in found_flags.items() if not found]
        present = [cap for cap, found in found_flags.items() if found]

        if missing:
            has_cli_diff = True
            console.warn("MISSING (expected flags not found in --help):")
            for cap in sorted(missing):
                flag = _EXPECTED_FLAGS[tool][cap]
                console.detail(f"  - {cap} ({flag})")

        mismatches = []
        for cap in present:
            field = _CAP_FIELD_MAP.get(cap)
            if field and not getattr(caps, field, True):
                mismatches.append((cap, _EXPECTED_FLAGS[tool][cap], field))

        if mismatches:
            has_cli_diff = True
            console.warn("CAPABILITY MISMATCH (flag found but adapter declares no support):")
            for cap, flag, field in sorted(mismatches):
                console.detail(f"  ! {cap}: {flag} found but {field}=False")

        if not missing and not mismatches:
            console.detail("✓ CLI args match expectations.")

        console.empty()

    return has_cli_diff


def main() -> int:
    from crossby.ui.console import console

    with open(JSON_PATH, encoding="utf-8") as f:
        registry_raw: Mapping[str, list[str]] = json.load(f)

    registry: dict[str, set[str]] = {
        k: set(v) for k, v in registry_raw.items() if not k.startswith("_")
    }

    with console.status("Probing external AI providers..."):
        found: dict[str, set[str]] = {
            tool: _MODEL_PROBES[tool]() for tool in registry if tool in _MODEL_PROBES
        }
        deprecations: dict[str, set[str]] = {tool: probe_deprecations(tool) for tool in registry}

    has_diff = False
    console.empty()
    diff_summary: list[str] = []

    for tool, expected in registry.items():
        actual_raw = found.get(tool, set())
        deprecated = deprecations.get(tool, set())

        # Checked even when discovery is unavailable: retirement evidence comes
        # from the provider's own deprecation page, not from the model probe.
        still_listed = expected & deprecated
        if still_listed:
            has_diff = True
            console.warn(
                f"[{tool}] DEPRECATED (provider marks retired/deprecated; "
                f"remove from {JSON_PATH.name}):"
            )
            for m in sorted(still_listed):
                console.detail(f"  - {m}")
            diff_summary.append(
                f"For the '{tool}' tools list, REMOVE these items: {sorted(still_listed)}"
            )

        # An unavailable source is not evidence that every registered model was
        # removed. Make the affected tool/source explicit and skip its diff.
        if not actual_raw:
            source = _MODEL_PROBE_SOURCES.get(tool, "model source")
            console.warn(
                f"[{tool}] SKIPPED: {source} returned no models "
                "(missing CLI, authentication, network, cache, or command failure)."
            )
            continue

        not_returned, new = model_catalog_diff(expected, actual_raw)
        ignored = new & deprecated
        new -= deprecated

        console.header(f"Provider: {tool}")
        if ignored:
            console.warn("IGNORED (in probe, but the provider marks these retired/deprecated):")
            for m in sorted(ignored):
                console.detail(f"  · {m}")
        if not not_returned and not new:
            console.detail("✓ Up to date.")
        else:
            if new:
                has_diff = True
                console.warn(f"NEW (found in probe but not in {JSON_PATH.name}):")
                for m in sorted(new):
                    console.detail(f"  + {m}")
                diff_summary.append(f"For the '{tool}' tools list, ADD these items: {sorted(new)}")
            if not_returned:
                console.warn(
                    "NOT RETURNED (retained; absence alone is not tool-specific "
                    "retirement evidence):"
                )
                for m in sorted(not_returned):
                    console.detail(f"  - {m}")
        console.empty()

    has_diff = report_cli_args() or has_diff

    if not has_diff:
        console.success("All models and CLI args match expectations!")
        return 0

    from crossby.ui import prompts

    if not sys.stdin.isatty():
        console.error(f"Differences found. Please update {JSON_PATH} manually.")
        return 1

    msg = f"\nWould you like to use an AI agent to auto-correct {JSON_PATH.name}?"
    if not prompts.confirm(msg, default=False):
        console.error(f"Differences found. Please update {JSON_PATH} manually.")
        return 1

    installed = []
    for tool_id in AbstractAITool.detect_installed():
        try:
            adapter = AbstractAITool.get(tool_id)
            if adapter.capabilities().supports_headless:
                installed.append((tool_id, adapter))
        except ValueError:
            pass

    if not installed:
        console.error("No compatible headless AI tools installed to perform auto-correction.")
        return 1

    items = [f"{t[1].capabilities().display_name} ({t[0]})" for t in installed]
    idx = prompts.select("Select AI tool to use for correction", items)
    tool_id, adapter = installed[idx]

    prompt = (
        "You are tasked with updating a JSON file based on some diff instructions.\n"
        "Output ONLY valid JSON. Do not include markdown formatting (like ```json), "
        "intro, or outro text. Output raw JSON only.\n\n"
        "Here is the current JSON:\n"
        f"{json.dumps(registry_raw, indent=2)}\n\n"
        "Please apply the following changes to the lists:\n" + "\n".join(diff_summary)
    )

    env = os.environ.copy()
    # Claude Code exports these sentinels inside an active session. Strip them so
    # a nested probe subprocess doesn't mis-detect itself as already running
    # inside Claude Code (which crashes the nested launch).
    for sentinel in ("CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_ENTRYPOINT"):
        env.pop(sentinel, None)

    expected_schema = {
        "type": "object",
        "additionalProperties": {
            "type": "array",
            "items": {"type": "string"},
        },
    }

    cmd = adapter.build_launch_command(prompt=prompt, json_schema=expected_schema)
    with console.status(f"Asking {adapter.capabilities().display_name} to fix {JSON_PATH.name}..."):
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)

    out = res.stdout.strip()

    # Try multiple strategies to find valid JSON
    def extract_json(text: str) -> str | None:
        def extract_payload(parsed: Any) -> str | None:
            # Claude and Copilot `--json-schema` wraps the answer inside `structured_output`
            if isinstance(parsed, dict) and "structured_output" in parsed:
                parsed = parsed["structured_output"]
            # Require the full top-level key set so a partial payload (e.g.
            # {"copilot": [...]}), wrapped or not, is rejected instead of
            # silently deleting the omitted providers.
            if isinstance(parsed, dict) and set(parsed) == set(registry_raw):
                return json.dumps(parsed, indent=2)
            return None

        # Strategy 1: The whole thing might be valid JSON
        try:
            parsed = json.loads(text)
            if payload := extract_payload(parsed):
                return payload
        except ValueError:
            pass

        # Strategy 2: Remove markdown formatting (leading and trailing fence only)
        cleaned = re.sub(r"^```(?:json)?\n", "", text, count=1)
        cleaned = re.sub(r"\n```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
            if payload := extract_payload(parsed):
                return payload
        except ValueError:
            pass

        # Strategy 3: Find first { and last }
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start < end:
                substring = text[start:end]
                try:
                    parsed = json.loads(substring)
                    if payload := extract_payload(parsed):
                        return payload
                except ValueError:
                    pass
        return None

    valid_json = extract_json(out)
    if not valid_json:
        console.error("AI tool did not return valid JSON.")
        if res.returncode != 0:
            console.error(f"Command failed with exit code: {res.returncode}")
        console.detail(f"Raw stdout was:\n{out}")
        if res.stderr:
            console.detail(f"Raw stderr was:\n{res.stderr.strip()}")
        return 1

    out = valid_json

    from rich.console import Console
    from rich.syntax import Syntax

    rc = Console()
    console.empty()
    console.header(f"Proposed {JSON_PATH.name}")
    rc.print(Syntax(out, "json", theme="monokai", word_wrap=True))
    console.empty()

    if prompts.confirm(f"Overwrite {JSON_PATH.name} with this new content?", default=True):
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        console.success(f"Successfully updated {JSON_PATH.name}.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
