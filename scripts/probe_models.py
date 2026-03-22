#!/usr/bin/env python3
"""Probe installed AI CLIs against CROSSBY's bundled model registry.

This is a developer utility, not a test. It checks two things:

1. Which models the installed CLIs report, then compares them with
   ``src/crossby/data/models.json``.
2. Whether the CLI help output still exposes the flags that the adapter
   layer expects for the supported terminal tools.

The script is intentionally read-only. It reports drift and exits non-zero
when differences are found, but it does not try to patch files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import crossby.ai_tools  # noqa: F401 - registers adapters
from crossby.ai_tools.base import AbstractAITool
from crossby.models.ai import AIToolID

MODELS_JSON = SRC_ROOT / "crossby" / "data" / "models.json"
SUPPORTED_TOOLS = ("claude", "copilot", "gemini", "codex", "cursor", "opencode")
_MODEL_PROBE_TIMEOUTS = {
    "claude": 30,
    "copilot": 20,
    "gemini": 20,
    "codex": 20,
    "cursor": 20,
    "opencode": 20,
}
_DOCS_URLS: dict[str, str] = {
    "claude": "https://docs.anthropic.com/en/docs/about-claude/models/overview",
    "gemini": "https://geminicli.com/docs/cli/model/",
    "codex": "https://developers.openai.com/codex/models",
}
_SCRAPE_PATTERNS: dict[str, str] = {
    "claude": r"claude-[a-z]+-[0-9]+[-\.][0-9]+[a-zA-Z0-9._-]*",
    "gemini": r"gemini-[0-9][.0-9]*-(?:flash|pro|ultra)[a-z0-9._-]*",
    "codex": r"gpt-[0-9][.0-9]*[a-zA-Z0-9._-]*",
}
_HELP_COMMANDS: dict[str, list[list[str]]] = {
    "claude": [["claude", "--help"]],
    "copilot": [["copilot", "--help"]],
    "gemini": [["gemini", "--help"]],
    "codex": [["codex", "--help"], ["codex", "exec", "--help"]],
    "cursor": [["agent", "--help"]],
    "opencode": [["opencode", "--help"], ["opencode", "run", "--help"]],
}

_ENV_ALLOWLIST = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LOGNAME",
    "PATH",
    "PATHEXT",
    "SHELL",
    "SYSTEMROOT",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "USER",
)

_EXPECTED_FLAGS: dict[str, dict[str, str]] = {
    "claude": {
        "headless": "--print",
        "resume": "--resume",
        "yolo": "--dangerously-skip-permissions",
        "effort": "--effort",
        "structured_output": "--json-schema",
        "allowed_tools": "--allowedTools",
        "model": "--model",
    },
    "copilot": {
        "headless": "--prompt",
        "resume": "--resume",
        "yolo": "--yolo",
        "allow_tool": "--allow-tool",
        "model": "--model",
    },
    "gemini": {
        "headless": "-p",
        "resume": "--resume",
        "yolo": "--yolo",
        "allowed_tools": "--allowed-tools",
        "output_format": "--output-format",
        "model": "--model",
    },
    "codex": {
        "headless": "exec",
        "sandbox": "--sandbox",
        "resume": "resume",
        "yolo": "--dangerously-bypass-approvals-and-sandbox",
        "effort": "model_reasoning_effort",
        "cd": "--cd",
        "add_dir": "--add-dir",
        "model": "--model",
    },
    "cursor": {
        "headless": "--print",
        "model": "--model",
        "force": "--force",
        "list_models": "--list-models",
        "plan_mode": "--mode",
    },
    "opencode": {
        "headless": "run",
        "model": "--model",
        "resume": "-s",
        "effort": "--variant",
        "prompt": "--prompt",
    },
}


@dataclass
class ToolReport:
    tool: str
    installed: bool
    binary: str | None = None
    models_found: set[str] = field(default_factory=set)
    models_missing: set[str] = field(default_factory=set)
    models_new: set[str] = field(default_factory=set)
    help_missing: list[str] = field(default_factory=list)
    help_matched: list[str] = field(default_factory=list)
    probe_failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_diff(self) -> bool:
        return bool(self.models_missing or self.models_new or self.help_missing or self.probe_failures)


class ProbeFailure(RuntimeError):
    """Raised when a probe could not determine the requested information."""


def _read_registry() -> dict[str, set[str]]:
    raw = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    return {tool: set(models) for tool, models in raw.items() if not tool.startswith("_")}


def _clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", env["LANG"])
    env.setdefault("NO_COLOR", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("TERM", "xterm")
    if extra:
        env.update(extra)
    return env


def _run(
    command: list[str],
    timeout: int = 20,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_clean_env(extra_env),
    )


def _scrape_models(tool: str) -> set[str]:
    """Best-effort fallback for tools with official model docs."""
    if tool not in _DOCS_URLS or not shutil.which("curl"):
        return set()

    try:
        result = subprocess.run(
            ["curl", "-fsSL", "--max-time", "10", _DOCS_URLS[tool]],
            capture_output=True,
            text=True,
            timeout=15,
            env=_clean_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()

    if result.returncode != 0:
        return set()

    if tool == "codex":
        codex_matches = re.findall(r"codex -m (gpt-[a-z0-9._-]+)", result.stdout)
        if codex_matches:
            return set(codex_matches)

    matches = re.findall(_SCRAPE_PATTERNS[tool], result.stdout)
    return {m if isinstance(m, str) else m[0] for m in matches}


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _token_match(pattern: str, text: str) -> bool:
    escaped = re.escape(pattern)
    if pattern.startswith("--"):
        return pattern in text
    if pattern.startswith("-"):
        return bool(re.search(r"(?:^|\s)" + escaped + r"(?:\s|,|\[|$)", text, re.MULTILINE))
    return bool(re.search(r"\b" + escaped + r"\b", text))


def _get_adapter(tool: str) -> AbstractAITool:
    return AbstractAITool.get(AIToolID(tool))


def _parse_choice_list(text: str) -> set[str]:
    choices: set[str] = set()
    pattern = re.compile(r"Allowed choices are\s+(.*?)(?:\.\s*$|$)", re.DOTALL)
    match = pattern.search(text)
    if not match:
        return choices
    for raw in match.group(1).split(","):
        item = raw.strip().strip(".")
        if item:
            choices.add(item)
    return choices


def _probe_claude_models() -> set[str]:
    try:
        result = _run(["claude", "models"], timeout=_MODEL_PROBE_TIMEOUTS["claude"])
    except subprocess.TimeoutExpired as exc:
        fallback_reason = f"claude models timed out: {exc}"
    else:
        models: set[str] = set()
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "-")):
                    continue
                parts = stripped.split()
                if parts and parts[0].lower() not in {"model", "name", "id"}:
                    models.add(parts[0])
        if models:
            return models
        fallback_reason = (
            "claude models returned no models"
            if result.returncode == 0
            else f"claude models exited with code {result.returncode}"
        )

    fallback = _scrape_models("claude")
    if fallback:
        return fallback
    raise ProbeFailure(fallback_reason)


def _probe_copilot_models() -> set[str]:
    result = _run(["copilot", "--model", "nope"], timeout=_MODEL_PROBE_TIMEOUTS["copilot"])
    models = _parse_choice_list(result.stdout + result.stderr)
    if models:
        return models
    raise ProbeFailure("copilot --model probe returned no model choices")


def _probe_gemini_models() -> set[str]:
    result = _run(["gemini", "--list-models"], timeout=_MODEL_PROBE_TIMEOUTS["gemini"])
    models: set[str] = set()
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "-")):
                continue
            parts = stripped.split()
            if parts and parts[0].lower() not in {"model", "name", "id"}:
                models.add(parts[0])
        if models:
            return models

    fallback = _scrape_models("gemini")
    if fallback:
        return fallback
    raise ProbeFailure(f"gemini --list-models exited with code {result.returncode}")


def _probe_codex_models() -> set[str]:
    cache = Path.home() / ".codex" / "models_cache.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        models = {
            model["slug"]
            for model in data.get("models", [])
            if isinstance(model, dict) and model.get("visibility") == "list" and model.get("slug")
        }
        normalized = {m for m in models if isinstance(m, str)}
        if normalized:
            return normalized

    fallback = _scrape_models("codex")
    if fallback:
        return fallback
    raise ProbeFailure("codex model cache missing/empty and docs scrape returned no models")


def _probe_cursor_models() -> set[str]:
    result = _run(["agent", "--list-models"], timeout=_MODEL_PROBE_TIMEOUTS["cursor"])
    if result.returncode != 0:
        raise ProbeFailure(f"agent --list-models exited with code {result.returncode}")
    models: set[str] = set()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or " - " not in stripped:
            continue
        model_id = stripped.split(" - ", 1)[0].strip()
        if model_id and not model_id.startswith(("Available", "Tip:")):
            models.add(model_id)
    if models:
        return models
    raise ProbeFailure("agent --list-models returned no models")


def _probe_opencode_models() -> set[str]:
    result = _run(["opencode", "models"], timeout=_MODEL_PROBE_TIMEOUTS["opencode"])
    if result.returncode != 0:
        raise ProbeFailure(f"opencode models exited with code {result.returncode}")
    models: set[str] = set()
    for line in _strip_ansi(result.stdout).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        models.add(stripped.split()[0])
    if models:
        return models
    raise ProbeFailure("opencode models returned no models")


_MODEL_PROBES = {
    "claude": _probe_claude_models,
    "copilot": _probe_copilot_models,
    "gemini": _probe_gemini_models,
    "codex": _probe_codex_models,
    "cursor": _probe_cursor_models,
    "opencode": _probe_opencode_models,
}


def _probe_models(tool: str) -> set[str]:
    return _MODEL_PROBES[tool]()


def _probe_help_output(tool: str, binary: str) -> str:
    combined = []
    commands = _HELP_COMMANDS.get(tool, [[binary, "--help"]])
    for command in commands:
        result = _run(command)
        combined.append(result.stdout)
        combined.append(result.stderr)
    return _strip_ansi("\n".join(combined))


def _probe_cli_flags(tool: str) -> tuple[list[str], list[str]]:
    adapter = _get_adapter(tool)
    caps = adapter.capabilities()
    help_text = _probe_help_output(tool, caps.binary)
    if not help_text.strip():
        raise ProbeFailure("no CLI help output returned")
    expected = _EXPECTED_FLAGS.get(tool, {})
    matched: list[str] = []
    missing: list[str] = []
    for cap_name, pattern in expected.items():
        if _token_match(pattern, help_text):
            matched.append(f"{cap_name} ({pattern})")
        else:
            missing.append(f"{cap_name} ({pattern})")
    return matched, missing


def _normalize_models(tool: str, models: set[str]) -> set[str]:
    adapter = _get_adapter(tool)
    return {adapter.standardize_model_id(model) for model in models}


def _build_report(tool: str, registry: dict[str, set[str]]) -> ToolReport:
    adapter = _get_adapter(tool)
    caps = adapter.capabilities()
    binary = caps.binary
    installed = shutil.which(binary) is not None
    report = ToolReport(tool=tool, installed=installed, binary=binary if installed else None)

    if not installed:
        report.notes.append(f"{binary} not found in PATH")
        return report

    try:
        raw_models = _probe_models(tool)
    except Exception as exc:  # pragma: no cover - best-effort probe
        report.probe_failures.append(f"model probe failed: {exc}")
    else:
        normalized = _normalize_models(tool, raw_models)
        report.models_found = normalized
        expected = registry.get(tool, set())
        report.models_missing = expected - normalized
        report.models_new = normalized - expected

    try:
        report.help_matched, report.help_missing = _probe_cli_flags(tool)
    except Exception as exc:  # pragma: no cover - best-effort probe
        report.probe_failures.append(f"help probe failed: {exc}")

    return report


def _format_models(models: set[str], limit: int = 10) -> str:
    if not models:
        return "-"
    ordered = sorted(models)
    if len(ordered) <= limit:
        return ", ".join(ordered)
    head = ", ".join(ordered[:limit])
    return f"{head}, ... (+{len(ordered) - limit} more)"


def _print_report(report: ToolReport) -> None:
    status = "OK" if not report.has_diff else "DIFF"
    binary = report.binary or "-"
    print(f"[{status}] {report.tool} ({binary})")
    if report.models_found:
        print(f"  models: {_format_models(report.models_found)}")
    if report.models_missing:
        print(f"  missing from registry: {_format_models(report.models_missing)}")
    if report.models_new:
        print(f"  new in probe: {_format_models(report.models_new)}")
    if report.help_missing:
        print(f"  missing flags: {', '.join(report.help_missing)}")
    if report.help_matched:
        print(f"  matched flags: {', '.join(report.help_matched)}")
    for failure in report.probe_failures:
        print(f"  probe failure: {failure}")
    for note in report.notes:
        print(f"  note: {note}")


def _json_ready(report: ToolReport) -> dict[str, Any]:
    return {
        "tool": report.tool,
        "installed": report.installed,
        "binary": report.binary,
        "models_found": sorted(report.models_found),
        "models_missing": sorted(report.models_missing),
        "models_new": sorted(report.models_new),
        "help_missing": report.help_missing,
        "help_matched": report.help_matched,
        "probe_failures": report.probe_failures,
        "notes": report.notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool",
        action="append",
        choices=SUPPORTED_TOOLS,
        help="Limit probing to one or more tools.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable output.",
    )
    args = parser.parse_args(argv)

    registry = _read_registry()
    tools = tuple(args.tool) if args.tool else SUPPORTED_TOOLS
    reports = [_build_report(tool, registry) for tool in tools]
    has_diff = any(report.has_diff for report in reports)

    if args.json:
        payload = {"tools": [_json_ready(report) for report in reports], "has_diff": has_diff}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("CROSSBY model probe")
        print(f"registry: {MODELS_JSON.relative_to(REPO_ROOT)}")
        print()
        for report in reports:
            _print_report(report)
            print()
        if has_diff:
            print("Differences found. Update the registry or the adapter contract as needed.")
        else:
            print("All probed tools match the bundled registry and expected CLI flags.")

    return 1 if has_diff else 0


if __name__ == "__main__":
    raise SystemExit(main())
