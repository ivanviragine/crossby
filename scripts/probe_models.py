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
        "approval": "--ask-for-approval",
        "full_auto": "--full-auto",
        "json": "--json",
        "cd": "--cd",
        "add_dir": "--add-dir",
        "model": "--model",
    },
    "cursor": {
        "headless": "--print",
        "resume": "--resume",
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
    notes: list[str] = field(default_factory=list)

    @property
    def has_diff(self) -> bool:
        return bool(self.models_missing or self.models_new or self.help_missing)


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
    result = _run(["claude", "models"])
    if result.returncode != 0:
        return set()
    models: set[str] = set()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        parts = stripped.split()
        if parts and parts[0].lower() not in {"model", "name", "id"}:
            models.add(parts[0])
    return models


def _probe_copilot_models() -> set[str]:
    result = _run(["copilot", "--model", "nope"])
    return _parse_choice_list(result.stdout + result.stderr)


def _probe_gemini_models() -> set[str]:
    result = _run(["gemini", "--list-models"])
    if result.returncode != 0:
        return set()
    models: set[str] = set()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        parts = stripped.split()
        if parts and parts[0].lower() not in {"model", "name", "id"}:
            models.add(parts[0])
    return models


def _probe_codex_models() -> set[str]:
    cache = Path.home() / ".codex" / "models_cache.json"
    if not cache.exists():
        return set()
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    models = {
        model["slug"]
        for model in data.get("models", [])
        if isinstance(model, dict) and model.get("visibility") == "list" and model.get("slug")
    }
    return {m for m in models if isinstance(m, str)}


def _probe_cursor_models() -> set[str]:
    result = _run(["agent", "--list-models"])
    if result.returncode != 0:
        return set()
    models: set[str] = set()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or " - " not in stripped:
            continue
        model_id = stripped.split(" - ", 1)[0].strip()
        if model_id and not model_id.startswith(("Available", "Tip:")):
            models.add(model_id)
    return models


def _probe_opencode_models() -> set[str]:
    result = _run(["opencode", "models"])
    if result.returncode != 0:
        return set()
    models: set[str] = set()
    for line in _strip_ansi(result.stdout).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        models.add(stripped.split()[0])
    return models


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


def _probe_help_output(binary: str) -> str:
    combined = []
    for flag in ("--help", "-h"):
        result = _run([binary, flag])
        combined.append(result.stdout)
        combined.append(result.stderr)
    return "\n".join(combined)


def _probe_cli_flags(tool: str) -> tuple[list[str], list[str]]:
    adapter = _get_adapter(tool)
    caps = adapter.capabilities()
    help_text = _probe_help_output(caps.binary)
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
        report.notes.append(f"model probe failed: {exc}")
        raw_models = set()

    if raw_models:
        normalized = _normalize_models(tool, raw_models)
        report.models_found = normalized
        expected = registry.get(tool, set())
        report.models_missing = expected - normalized
        report.models_new = normalized - expected
    else:
        report.notes.append("no model list returned")

    try:
        report.help_matched, report.help_missing = _probe_cli_flags(tool)
    except Exception as exc:  # pragma: no cover - best-effort probe
        report.notes.append(f"help probe failed: {exc}")

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
