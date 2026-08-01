"""Post-sync target validators.

Re-parse each tool's output files and surface structural issues that
``run_sync`` itself can't catch:

- ``.codex/config.toml`` parses as TOML; every ``[mcp_servers.<name>]``
  whose ``command`` is set has the binary on ``PATH``.
- ``.codex/agents/*.toml`` parses and carries ``name`` /
  ``description`` / ``developer_instructions``.
- ``<tool>/skills/<name>/SKILL.md`` carries ``name`` and ``description``
  frontmatter.
- ``AGENTS.md`` / ``CLAUDE.md`` / ``.cursorrules`` /
  ``.github/copilot-instructions.md`` are under a 32KB review threshold.
- Tool-specific JSON config files (``.claude/settings.json``,
  ``.cursor/cli.json``, ``.cursor/mcp.json``, ``.vscode/mcp.json``,
  ``.agents/mcp_config.json``) parse cleanly.

Findings are tool-neutral; the CLI renders them as a Rich table.
"""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern

ValidationLevel = Literal["ok", "warning", "error"]


@dataclass(frozen=True)
class ValidationFinding:
    """One outcome from a validator: a fact about a path."""

    tool_id: AIToolID | None
    concern: SyncConcern | None
    level: ValidationLevel
    path: Path
    detail: str


# Files that should stay under this size threshold for an instructions file.
INSTRUCTION_SIZE_LIMIT_BYTES = 32 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(
    project_root: Path,
    path: Path,
    detail: str,
    *,
    tool_id: AIToolID | None = None,
    concern: SyncConcern | None = None,
) -> ValidationFinding:
    return ValidationFinding(
        tool_id=tool_id,
        concern=concern,
        level="ok",
        path=_relative(path, project_root),
        detail=detail,
    )


def _warn(
    project_root: Path,
    path: Path,
    detail: str,
    *,
    tool_id: AIToolID | None = None,
    concern: SyncConcern | None = None,
) -> ValidationFinding:
    return ValidationFinding(
        tool_id=tool_id,
        concern=concern,
        level="warning",
        path=_relative(path, project_root),
        detail=detail,
    )


def _error(
    project_root: Path,
    path: Path,
    detail: str,
    *,
    tool_id: AIToolID | None = None,
    concern: SyncConcern | None = None,
) -> ValidationFinding:
    return ValidationFinding(
        tool_id=tool_id,
        concern=concern,
        level="error",
        path=_relative(path, project_root),
        detail=detail,
    )


def _relative(path: Path, project_root: Path) -> Path:
    try:
        return path.relative_to(project_root)
    except ValueError:
        return path


def _parse_skill_frontmatter(skill_md: Path) -> dict[str, object] | None:
    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    try:
        data = yaml.safe_load(text[4:end])
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    return data


# ---------------------------------------------------------------------------
# Shared MCP PATH check
# ---------------------------------------------------------------------------


def _validate_mcp_commands_on_path(
    findings: list[ValidationFinding],
    project_root: Path,
    config_path: Path,
    *,
    tool_id: AIToolID,
    servers: Mapping[str, object],
) -> None:
    """Append one ``ok`` / ``warning`` finding per server-with-a-command.

    Servers without a ``command`` (e.g. pure HTTP/SSE transports) are
    skipped. Env-var-templated commands are expanded via
    :func:`os.path.expandvars` before the :func:`shutil.which` lookup so
    ``${HOME}/bin/foo`` resolves the same way the shell would when Codex /
    Claude / Cursor actually launches the server.
    """
    for name, server in sorted(servers.items()):
        if not isinstance(server, dict):
            continue
        command = server.get("command")
        if not command:
            continue
        command_str = str(command)
        resolved = os.path.expandvars(command_str)
        if shutil.which(resolved):
            findings.append(
                _ok(
                    project_root,
                    config_path,
                    f"MCP `{name}` command `{command_str}` on PATH",
                    tool_id=tool_id,
                    concern=SyncConcern.MCP,
                )
            )
        else:
            findings.append(
                _warn(
                    project_root,
                    config_path,
                    f"MCP `{name}` command `{command_str}` not on PATH",
                    tool_id=tool_id,
                    concern=SyncConcern.MCP,
                )
            )


# ---------------------------------------------------------------------------
# Codex validators
# ---------------------------------------------------------------------------


def validate_codex_config(project_root: Path) -> list[ValidationFinding]:
    config_path = project_root / ".codex" / "config.toml"
    if not config_path.is_file():
        return []
    findings: list[ValidationFinding] = []
    try:
        parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        findings.append(
            _error(
                project_root,
                config_path,
                f"invalid TOML: {exc}",
                tool_id=AIToolID.CODEX,
                concern=SyncConcern.MCP,
            )
        )
        return findings

    findings.append(
        _ok(
            project_root,
            config_path,
            "valid TOML",
            tool_id=AIToolID.CODEX,
            concern=SyncConcern.MCP,
        )
    )
    mcp_servers = parsed.get("mcp_servers")
    if isinstance(mcp_servers, dict):
        _validate_mcp_commands_on_path(
            findings,
            project_root,
            config_path,
            tool_id=AIToolID.CODEX,
            servers=mcp_servers,
        )
    return findings


def validate_codex_agents(project_root: Path) -> list[ValidationFinding]:
    agents_dir = project_root / ".codex" / "agents"
    if not agents_dir.is_dir():
        return []
    required = ("name", "description", "developer_instructions")
    findings: list[ValidationFinding] = []
    for agent_file in sorted(agents_dir.glob("*.toml")):
        try:
            parsed = tomllib.loads(agent_file.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            findings.append(
                _error(
                    project_root,
                    agent_file,
                    f"invalid TOML: {exc}",
                    tool_id=AIToolID.CODEX,
                    concern=SyncConcern.AGENTS,
                )
            )
            continue
        missing = [key for key in required if not parsed.get(key)]
        if missing:
            findings.append(
                _error(
                    project_root,
                    agent_file,
                    "missing required field(s): " + ", ".join(missing),
                    tool_id=AIToolID.CODEX,
                    concern=SyncConcern.AGENTS,
                )
            )
            continue
        findings.append(
            _ok(
                project_root,
                agent_file,
                "agent TOML has required fields",
                tool_id=AIToolID.CODEX,
                concern=SyncConcern.AGENTS,
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Skills validators (markdown frontmatter, applies to every tool)
# ---------------------------------------------------------------------------


_SKILLS_LOCATIONS: dict[AIToolID, Path] = {
    AIToolID.CLAUDE: Path(".claude") / "skills",
    AIToolID.CURSOR: Path(".cursor") / "skills",
    AIToolID.CODEX: Path(".agents") / "skills",
    AIToolID.ANTIGRAVITY_CLI: Path(".agents") / "skills",
    AIToolID.COPILOT: Path(".github") / "skills",
}


def validate_skill_frontmatter(project_root: Path) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    seen: set[Path] = set()
    for tool, rel in _SKILLS_LOCATIONS.items():
        root = project_root / rel
        if not root.is_dir():
            continue
        # Resolve symlinks so we walk the underlying tree once even if multiple
        # tool dirs symlink to the same canonical source (or, as with Codex and
        # Antigravity CLI, share the same literal .agents/skills path).
        try:
            real_root = root.resolve()
        except OSError:
            real_root = root
        if real_root in seen:
            continue
        seen.add(real_root)
        for skill_md in sorted(real_root.glob("*/SKILL.md")):
            data = _parse_skill_frontmatter(skill_md)
            relative = _relative(skill_md, project_root)
            if data is None:
                findings.append(
                    _error(
                        project_root,
                        skill_md,
                        "skill frontmatter missing or unparseable",
                        tool_id=tool,
                        concern=SyncConcern.SKILLS,
                    )
                )
                continue
            missing = [key for key in ("name", "description") if not data.get(key)]
            if missing:
                findings.append(
                    _error(
                        project_root,
                        skill_md,
                        "skill frontmatter missing " + ", ".join(missing),
                        tool_id=tool,
                        concern=SyncConcern.SKILLS,
                    )
                )
                continue
            findings.append(
                ValidationFinding(
                    tool_id=tool,
                    concern=SyncConcern.SKILLS,
                    level="ok",
                    path=relative,
                    detail="skill has name and description",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Instructions validator (size threshold)
# ---------------------------------------------------------------------------


_INSTRUCTION_FILES: dict[AIToolID, Path] = {
    AIToolID.CLAUDE: Path("CLAUDE.md"),
    AIToolID.CODEX: Path("AGENTS.md"),
    AIToolID.ANTIGRAVITY_CLI: Path("AGENTS.md"),
    AIToolID.CURSOR: Path(".cursorrules"),
    AIToolID.COPILOT: Path(".github") / "copilot-instructions.md",
}


def validate_instruction_sizes(project_root: Path) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    seen: set[Path] = set()
    for tool, rel in _INSTRUCTION_FILES.items():
        path = project_root / rel
        if not path.is_file():
            continue
        # Resolve so symlinked targets aren't double-counted.
        try:
            real = path.resolve()
        except OSError:
            real = path
        if real in seen:
            continue
        seen.add(real)
        size = path.stat().st_size
        kb = size / 1024
        if size > INSTRUCTION_SIZE_LIMIT_BYTES:
            findings.append(
                _warn(
                    project_root,
                    path,
                    f"{kb:.1f}KB exceeds the 32KB review threshold",
                    tool_id=tool,
                    concern=SyncConcern.RULES,
                )
            )
        else:
            findings.append(
                _ok(
                    project_root,
                    path,
                    f"{kb:.1f}KB",
                    tool_id=tool,
                    concern=SyncConcern.RULES,
                )
            )
    return findings


# ---------------------------------------------------------------------------
# JSON-config MCP PATH validators (every tool that lists MCP commands in JSON)
# ---------------------------------------------------------------------------


# Where each tool stores MCP server entries in a JSON file, and under which
# top-level key the server table lives. ``.vscode/mcp.json`` (Copilot) uses
# ``servers``; every other JSON-shape store uses ``mcpServers``. Only the
# project-relative, project-scope files live here; Claude's user-scope
# ``~/.claude.json`` is validated separately and only under user scope (see
# :func:`_global_claude_json_path`). The project-local ``.claude.json`` that
# used to sit here is *not* a file Claude Code reads MCP servers from, so it
# was dropped — discovery never read it either.
_JSON_MCP_LOCATIONS: dict[AIToolID, list[tuple[Path, str]]] = {
    AIToolID.CLAUDE: [
        (Path(".mcp.json"), "mcpServers"),
        (Path(".claude") / "settings.json", "mcpServers"),
    ],
    AIToolID.CURSOR: [
        (Path(".cursor") / "mcp.json", "mcpServers"),
    ],
    AIToolID.COPILOT: [
        (Path(".vscode") / "mcp.json", "servers"),
    ],
    AIToolID.ANTIGRAVITY_CLI: [
        (Path(".agents") / "mcp_config.json", "mcpServers"),
    ],
}


def _global_claude_json_path() -> Path:
    """Claude's user-scope ``~/.claude.json``, resolved at call time.

    Reads the discovery module's path attribute rather than capturing it, so a
    test that monkeypatches ``mcp_discovery._GLOBAL_CLAUDE_JSON_PATH`` steers
    both discovery and validation at the same fake home — keeping the two in
    agreement, which is the whole point.
    """
    from crossby.sync.mcp_discovery import _GLOBAL_CLAUDE_JSON_PATH

    return _GLOBAL_CLAUDE_JSON_PATH


def validate_mcp_command_paths(
    project_root: Path, *, include_user_scope: bool = False
) -> list[ValidationFinding]:
    """Validate that every tool's MCP server ``command`` resolves on ``PATH``.

    Codex is handled separately by :func:`validate_codex_config` because its
    config lives in TOML rather than JSON. This walker covers the JSON-shape
    tools — Claude, Cursor, Copilot (via the VS Code mcp.json), and Antigravity CLI.
    JSON parse failures are silently skipped here; :func:`validate_json_configs`
    surfaces them with full detail.

    Claude's user-scope ``~/.claude.json`` is only checked when
    *include_user_scope* is set, matching what discovery reads for that flag.
    """
    findings: list[ValidationFinding] = []
    locations: list[tuple[AIToolID, Path, str]] = [
        (tool, project_root / rel, key)
        for tool, entries in _JSON_MCP_LOCATIONS.items()
        for rel, key in entries
    ]
    if include_user_scope:
        locations.append((AIToolID.CLAUDE, _global_claude_json_path(), "mcpServers"))

    for tool, path, key in locations:
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(raw, dict):
            continue
        servers = raw.get(key)
        if not isinstance(servers, dict):
            continue
        _validate_mcp_commands_on_path(
            findings,
            project_root,
            path,
            tool_id=tool,
            servers=servers,
        )
    return findings


# ---------------------------------------------------------------------------
# JSON config validators
# ---------------------------------------------------------------------------


# Each entry is (relative path, concern). Claude's settings.json holds both
# MCP servers and hooks, so we tag it MCP (validation is just JSON parseability).
# `.mcp.json` ships with the MCP PATH walker, so we list it here too — otherwise
# malformed JSON in it would be silently dropped by the walker without any
# "invalid JSON" finding from elsewhere. User-scope `~/.claude.json` is
# parse-checked separately, only under user scope.
_JSON_CONFIGS: dict[AIToolID, list[tuple[Path, SyncConcern]]] = {
    AIToolID.CLAUDE: [
        (Path(".mcp.json"), SyncConcern.MCP),
        (Path(".claude") / "settings.json", SyncConcern.MCP),
    ],
    AIToolID.CURSOR: [
        (Path(".cursor") / "cli.json", SyncConcern.MCP),
        (Path(".cursor") / "mcp.json", SyncConcern.MCP),
        (Path(".cursor") / "hooks.json", SyncConcern.HOOKS),
    ],
    AIToolID.ANTIGRAVITY_CLI: [(Path(".agents") / "mcp_config.json", SyncConcern.MCP)],
    AIToolID.COPILOT: [
        (Path(".vscode") / "mcp.json", SyncConcern.MCP),
        (Path(".github") / "hooks" / "hooks.json", SyncConcern.HOOKS),
    ],
}


def validate_json_configs(
    project_root: Path, *, include_user_scope: bool = False
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    configs: list[tuple[AIToolID, Path, SyncConcern]] = [
        (tool, project_root / rel, concern)
        for tool, entries in _JSON_CONFIGS.items()
        for rel, concern in entries
    ]
    if include_user_scope:
        configs.append((AIToolID.CLAUDE, _global_claude_json_path(), SyncConcern.MCP))

    for tool, path, concern in configs:
        if not path.is_file():
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            findings.append(
                _error(
                    project_root,
                    path,
                    f"invalid JSON: {exc}",
                    tool_id=tool,
                    concern=concern,
                )
            )
            continue
        except (OSError, UnicodeDecodeError) as exc:
            # UnicodeDecodeError (a ValueError, not an OSError) fires before
            # json.loads on non-UTF-8 bytes — treat it as unreadable too, the
            # same way read_json_file does.
            findings.append(
                _error(
                    project_root,
                    path,
                    f"could not be read: {exc}",
                    tool_id=tool,
                    concern=concern,
                )
            )
            continue
        findings.append(
            _ok(
                project_root,
                path,
                "valid JSON",
                tool_id=tool,
                concern=concern,
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def validate_target(
    project_root: Path, *, include_user_scope: bool = False
) -> list[ValidationFinding]:
    """Run every validator against ``project_root`` and return findings.

    MCP PATH checks run BEFORE :func:`validate_json_configs` so a malformed
    JSON file is reported as a parse error by the JSON validator rather than
    silently swallowed by the PATH walker.

    When *include_user_scope* is set, Claude's user-scope ``~/.claude.json`` is
    validated too — matching the files discovery reads under that flag.
    """
    findings: list[ValidationFinding] = []
    findings.extend(validate_codex_config(project_root))
    findings.extend(validate_codex_agents(project_root))
    findings.extend(validate_skill_frontmatter(project_root))
    findings.extend(validate_instruction_sizes(project_root))
    findings.extend(validate_mcp_command_paths(project_root, include_user_scope=include_user_scope))
    findings.extend(validate_json_configs(project_root, include_user_scope=include_user_scope))
    return findings


def has_errors(findings: list[ValidationFinding]) -> bool:
    return any(f.level == "error" for f in findings)


__all__ = [
    "INSTRUCTION_SIZE_LIMIT_BYTES",
    "ValidationFinding",
    "ValidationLevel",
    "has_errors",
    "validate_codex_agents",
    "validate_codex_config",
    "validate_instruction_sizes",
    "validate_json_configs",
    "validate_mcp_command_paths",
    "validate_skill_frontmatter",
    "validate_target",
]
