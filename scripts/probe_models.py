#!/usr/bin/env python3
"""Developer utility: probe AI CLIs and websites to discover new models.

Compares discovered models against src/crossby/data/models.json and reports
any differences. Exits 1 if updates are needed.
"""

from __future__ import annotations

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
        "yolo": "--yolo",
        "ask_for_approval": "--ask-for-approval",
        "headless": "exec",
        "model_reasoning_effort": "model_reasoning_effort",
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


def _scrape_models(tool: str) -> set[str]:
    """Scrape model IDs from docs."""
    if tool not in _DOCS_URLS or not shutil.which("curl"):
        return set()

    url = _DOCS_URLS[tool]

    try:
        result = subprocess.run(
            ["curl", "-fsSL", "--max-time", "10", url],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return set()

        return parse_documented_models(tool, result.stdout)
    except Exception:
        return set()


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
    for help_flag in ["--help", "-h"]:
        try:
            result = subprocess.run(
                [binary, help_flag],
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

    has_diff = False
    console.empty()
    diff_summary: list[str] = []

    for tool, expected in registry.items():
        actual_raw = found.get(tool, set())

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

        console.header(f"Provider: {tool}")
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
