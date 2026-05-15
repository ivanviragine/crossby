"""Project config detection — scan a source tool for all config types."""

from __future__ import annotations

import contextlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from crossby.config.instructions import INSTRUCTIONS_FILE
from crossby.config.skills import SKILLS_DIR
from crossby.models.ai import AIToolID


@dataclass
class DetectedConfig:
    """A single detected configuration item in a project."""

    # One of: instructions, skills, allowlist, hooks, mcp_servers, custom_commands
    config_type: str
    detail: str  # human-readable summary, e.g. "CLAUDE.md", "4 skills"
    portable: bool  # whether crossby can sync this today
    reason: str = ""  # why it can't be synced (empty if portable)


def detect_source_configs(tool_id: AIToolID, root: Path) -> list[DetectedConfig]:
    """Detect all config types present for *tool_id* in *root*.

    Returns only items that actually exist on disk.
    """
    items: list[DetectedConfig] = []
    _detect_instructions(tool_id, root, items)
    _detect_skills(tool_id, root, items)
    _detect_allowlist(tool_id, root, items)
    _detect_hooks(tool_id, root, items)
    _detect_mcp_servers(tool_id, root, items)
    _detect_custom_commands(tool_id, root, items)
    return items


# ------------------------------------------------------------------
# Internal detectors
# ------------------------------------------------------------------


def _detect_instructions(tool_id: AIToolID, root: Path, items: list[DetectedConfig]) -> None:
    rel = INSTRUCTIONS_FILE.get(tool_id)
    if rel is None:
        return
    path = root / rel
    if path.is_file() or (path.is_symlink() and path.exists()):
        items.append(
            DetectedConfig(
                config_type="instructions",
                detail=rel,
                portable=True,
            )
        )


def _detect_skills(tool_id: AIToolID, root: Path, items: list[DetectedConfig]) -> None:
    rel = SKILLS_DIR.get(tool_id)
    if rel is None:
        return
    path = root / rel
    if not (path.is_dir() or (path.is_symlink() and path.exists())):
        return
    resolved = path.resolve() if path.is_symlink() else path
    count = 0
    with contextlib.suppress(OSError):
        count = sum(1 for d in resolved.iterdir() if d.is_dir() and (d / "SKILL.md").is_file())
    label = "1 skill" if count == 1 else f"{count} skills"
    items.append(
        DetectedConfig(
            config_type="skills",
            detail=f"{rel} ({label})",
            portable=True,
        )
    )


def _detect_allowlist(tool_id: AIToolID, root: Path, items: list[DetectedConfig]) -> None:
    n = 0

    if tool_id == AIToolID.CLAUDE:
        n = len(
            _read_json_list(
                root / ".claude" / "settings.json",
                ["permissions", "allow"],
                prefix="Bash(",
            )
        )
    elif tool_id == AIToolID.CURSOR:
        n = len(
            _read_json_list(
                root / ".cursor" / "cli.json",
                ["permissions", "allow"],
                prefix="Shell(",
            )
        )
    elif tool_id == AIToolID.GEMINI:
        n = _count_gemini_shell_allow_rules(root / ".gemini" / "policies" / "crossby.toml")

    if n == 0:
        return

    label = "1 pattern" if n == 1 else f"{n} patterns"
    items.append(
        DetectedConfig(
            config_type="allowlist",
            detail=label,
            portable=True,
        )
    )


def _count_gemini_shell_allow_rules(path: Path) -> int:
    if not path.is_file():
        return 0
    with contextlib.suppress(OSError, tomllib.TOMLDecodeError):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        rules = data.get("rule", [])
        if not isinstance(rules, list):
            return 0
        return sum(
            1
            for r in rules
            if isinstance(r, dict)
            and r.get("toolName") == "run_shell_command"
            and r.get("decision") == "allow"
            and isinstance(r.get("commandPrefix"), str)
            and r["commandPrefix"]
        )
    return 0


def _detect_hooks(tool_id: AIToolID, root: Path, items: list[DetectedConfig]) -> None:
    count = 0

    if tool_id == AIToolID.CLAUDE:
        hooks = _read_json_key(root / ".claude" / "settings.json", "hooks")
        if isinstance(hooks, dict):
            count = sum(len(v) for v in hooks.values() if isinstance(v, list))

    elif tool_id == AIToolID.CURSOR:
        data = _read_json_file(root / ".cursor" / "hooks.json")
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    count += len(v)

    elif tool_id == AIToolID.COPILOT:
        data = _read_json_file(root / ".github" / "hooks" / "hooks.json")
        if isinstance(data, dict):
            hooks = data.get("hooks", {})
            if isinstance(hooks, dict):
                count = sum(len(v) for v in hooks.values() if isinstance(v, list))

    elif tool_id == AIToolID.GEMINI:
        hooks = _read_json_key(root / ".gemini" / "settings.json", "hooks")
        if isinstance(hooks, dict):
            count = sum(len(v) for v in hooks.values() if isinstance(v, list))
        elif isinstance(hooks, list):
            count = len(hooks)

    if count == 0:
        return

    label = "1 hook" if count == 1 else f"{count} hooks"
    items.append(
        DetectedConfig(
            config_type="hooks",
            detail=label,
            portable=True,
        )
    )


def _detect_mcp_servers(tool_id: AIToolID, root: Path, items: list[DetectedConfig]) -> None:
    n = 0

    if tool_id == AIToolID.CLAUDE:
        n = _count_json_dict(root / ".claude" / "settings.json", "mcpServers")
    elif tool_id == AIToolID.CURSOR:
        n = _count_json_dict(root / ".cursor" / "mcp.json", "mcpServers")
    elif tool_id == AIToolID.COPILOT:
        n = _count_json_dict(root / ".vscode" / "mcp.json", "servers")
    elif tool_id == AIToolID.GEMINI:
        n = _count_json_dict(root / ".gemini" / "settings.json", "mcpServers")
    elif tool_id == AIToolID.CODEX:
        n = _count_codex_mcp_servers(root / ".codex" / "config.toml")

    if n == 0:
        return

    label = "1 MCP server" if n == 1 else f"{n} MCP servers"
    items.append(
        DetectedConfig(
            config_type="mcp_servers",
            detail=label,
            portable=True,
        )
    )


def _count_json_dict(path: Path, key: str) -> int:
    value = _read_json_key(path, key)
    return len(value) if isinstance(value, dict) else 0


def _count_codex_mcp_servers(path: Path) -> int:
    if not path.is_file():
        return 0
    with contextlib.suppress(OSError, tomllib.TOMLDecodeError):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        section = data.get("mcp_servers", {})
        if isinstance(section, dict):
            return len(section)
    return 0


def _detect_custom_commands(tool_id: AIToolID, root: Path, items: list[DetectedConfig]) -> None:
    if tool_id != AIToolID.CLAUDE:
        return
    cmds_dir = root / ".claude" / "commands"
    if not cmds_dir.is_dir():
        return
    count = sum(1 for _ in cmds_dir.rglob("*.md"))
    if count == 0:
        return
    label = "1 custom command" if count == 1 else f"{count} custom commands"
    items.append(
        DetectedConfig(
            config_type="custom_commands",
            detail=label,
            portable=False,
            reason="tool-specific, no cross-tool equivalent yet",
        )
    )


# ------------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------------


def _read_json_file(path: Path) -> object:
    if not path.is_file():
        return None
    with contextlib.suppress(json.JSONDecodeError, OSError):
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _read_json_key(path: Path, key: str) -> object:
    data = _read_json_file(path)
    if isinstance(data, dict):
        return data.get(key)
    return None


def _read_json_list(path: Path, keys: list[str], *, prefix: str = "") -> list[str]:
    data = _read_json_file(path)
    for key in keys:
        if not isinstance(data, dict):
            return []
        data = data.get(key)
    if not isinstance(data, list):
        return []
    if prefix:
        return [s for s in data if isinstance(s, str) and s.startswith(prefix)]
    return [s for s in data if isinstance(s, str)]
