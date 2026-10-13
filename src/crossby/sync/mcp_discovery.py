"""MCP server discovery — scan existing tool configs for mcp_servers entries.

Used by `crossby init` to propose a unified mcp_servers section from
servers already configured in any of the supported tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from crossby.sync.base import SyncConcern, SyncResult

Scope = Literal["project", "user"]


@dataclass
class DiscoveredServer:
    """A single MCP server entry found in a tool config."""

    name: str
    source_tool: str
    data: dict[str, Any]
    # "project" for a repo-local file (``.mcp.json``, ``.cursor/mcp.json``, …);
    # "user" only for the home-directory ``~/.claude.json``. Lets callers label
    # each server and keep user-scoped ones out of committed project files.
    scope: Scope = "project"
    # Names only: never include configuration values in a manual-fix report.
    unsupported_fields: tuple[str, ...] = ()
    # Native activation is independent of whether the connection can be ported.
    # None means malformed/unknown, never permission to provision or revoke.
    enabled: bool | None = True


@dataclass
class DiscoveryResult:
    """Result of scanning all tool configs for MCP servers."""

    servers: dict[str, DiscoveredServer] = field(default_factory=dict)
    conflicts: list[tuple[str, str, str]] = field(default_factory=list)  # (name, tool1, tool2)
    # (name, source_tool) for every kept server whose raw entry declared an
    # ``oauth`` block. No ``MCPServerConfig`` field represents OAuth config
    # (callbackPort, clientId, authServerMetadataUrl, ...), so every writer
    # silently drops it today; this list lets callers surface that instead
    # of losing it with no trace. See :func:`report_oauth_configs`.
    oauth_servers: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# Claude's user-scope config file (``claude mcp add --scope user``). Module-level
# so tests can monkeypatch it, matching the pattern used for Cursor's global
# config path in sync/permissions.py.
_GLOBAL_CLAUDE_JSON_PATH = Path.home() / ".claude.json"


def _read_json_section(
    path: Path, key: str, errors: list[str] | None = None
) -> dict[str, Any] | None:
    """Read a JSON file and return the value at ``key``, or None on error."""
    from crossby.sync.mcp_activation import read_activation_json

    problems = errors if errors is not None else []
    raw = read_activation_json(path, problems)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        problems.append(f"{path}: MCP configuration must be an object")
        return None
    section = raw.get(key, {})
    if not isinstance(section, dict):
        problems.append(f"{path}: {key} must be an object")
        return None
    return section


def _entry_enabled(
    entry: dict[str, Any], field_name: str, label: str, errors: list[str], *, invert: bool = False
) -> bool | None:
    if field_name not in entry:
        return True
    value = entry[field_name]
    if not isinstance(value, bool):
        errors.append(f"{label}: {field_name} must be a boolean")
        return None
    return not value if invert else value


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize a tool-specific entry to the canonical crossby format."""
    result: dict[str, Any] = {}

    # Copilot adds a "type" field for transport — translate back. Claude Code
    # accepts "streamable-http" as an alias for "http"; crossby's model only
    # knows "http"/"sse"/"stdio", so fold the alias here or the server would
    # fail validation and be silently dropped.
    transport = entry.get("type") or entry.get("transport")
    if transport == "streamable-http":
        transport = "http"
    if transport and transport != "stdio":
        result["transport"] = transport

    if "command" in entry:
        result["command"] = entry["command"]
    if entry.get("args"):
        result["args"] = entry["args"]
    if entry.get("env"):
        result["env"] = entry["env"]
    if entry.get("headers"):
        result["headers"] = entry["headers"]
    # Antigravity CLI uses "serverUrl" instead of "url" for remote servers.
    if "url" in entry:
        result["url"] = entry["url"]
    elif "serverUrl" in entry:
        result["url"] = entry["serverUrl"]

    return result


def discover_mcp_servers(
    project_root: Path, *, include_user_scope: bool = False, from_tool: str | None = None
) -> DiscoveryResult:
    """Scan tool config files for MCP server definitions.

    Scans:
    - .mcp.json → mcpServers (Claude's project-scope file, e.g. ``claude mcp
      add --scope project``; this is the canonical location for most real
      projects, checked into version control)
    - .claude/settings.json → mcpServers (legacy/manual location)
    - ~/.claude.json → mcpServers (Claude's user-scope file) — **only when
      ``include_user_scope`` is set**; by default definitions are skipped so a
      bare ``crossby sync`` never rakes personal servers into committed project
      files. Project activation metadata is still read from it. Servers from
      it are tagged ``scope="user"``. Native config-dir/legacy paths also apply.
    - .cursor/mcp.json → mcpServers
    - .vscode/mcp.json → servers (Copilot format)
    - .agents/mcp_config.json → mcpServers (Antigravity CLI)
    - .codex/config.toml → mcp_servers

    Claude sources are scanned most-specific-first (project .mcp.json, then
    .claude/settings.json, then the user-scope ~/.claude.json) so the
    first-seen-wins merge below prefers the project-scoped definition when
    the same server name appears in more than one Claude source.

    When *from_tool* is set, only that tool's source(s) are scanned — conflict
    resolution never crosses tool boundaries, so a server name that exists in
    both Claude's and Codex's configs still yields Codex's own definition for
    ``from_tool="codex"`` instead of losing to whichever tool the unscoped scan
    order happened to see first. Without this, a caller that scans everything
    and *then* filters by tool (the naive approach) would silently drop a
    tool's own server whenever another tool defined the same name earlier in
    the scan order.

    Returns:
        DiscoveryResult with merged servers (first-seen wins) and conflicts.
    """
    result = DiscoveryResult()

    sources: list[tuple[str, Path, str, Scope]] = [
        ("claude", project_root / ".mcp.json", "mcpServers", "project"),
        ("claude", project_root / ".claude" / "settings.json", "mcpServers", "project"),
    ]
    if include_user_scope:
        from crossby.sync.mcp_activation import claude_config_paths

        global_config, _settings = claude_config_paths(_GLOBAL_CLAUDE_JSON_PATH)
        sources.append(("claude", global_config, "mcpServers", "user"))
    sources += [
        ("cursor", project_root / ".cursor" / "mcp.json", "mcpServers", "project"),
        ("copilot", project_root / ".vscode" / "mcp.json", "servers", "project"),
        ("antigravity-cli", project_root / ".agents" / "mcp_config.json", "mcpServers", "project"),
    ]
    if from_tool is not None:
        sources = [s for s in sources if s[0] == from_tool]

    for tool, path, key, scope in sources:
        section = _read_json_section(path, key, result.errors)
        if section is None:
            continue
        for name, entry in section.items():
            if not isinstance(entry, dict):
                result.errors.append(f"{path}: MCP server `{name}` must be an object")
                continue
            normalized = _normalize_entry(entry)
            # Antigravity's serverUrl is a native remote declaration without
            # an explicit transport discriminator (unlike Claude's url).
            if tool == "antigravity-cli" and "serverUrl" in entry and "command" not in entry:
                normalized.setdefault("transport", "http")
            enabled = (
                _entry_enabled(
                    entry, "disabled", f"{path}: MCP server `{name}`", result.errors, invert=True
                )
                if tool == "antigravity-cli"
                else True
            )
            if enabled is not None and not enabled:
                normalized["enabled"] = False
            if name in result.servers:
                existing_tool = result.servers[name].source_tool
                # Multiple Claude scopes (.mcp.json, .claude/settings.json,
                # ~/.claude.json) share the "claude" tool label; a name
                # collision between them is scope precedence, not a
                # cross-tool conflict, so it's resolved silently
                # (first-seen — i.e. most specific scope — wins).
                if existing_tool != tool:
                    result.conflicts.append((name, existing_tool, tool))
            else:
                result.servers[name] = DiscoveredServer(
                    name=name, source_tool=tool, data=normalized, scope=scope, enabled=enabled
                )
                if isinstance(entry.get("oauth"), dict):
                    result.oauth_servers.append((name, tool))

    # Codex TOML
    codex_path = project_root / ".codex" / "config.toml"
    if (from_tool is None or from_tool == "codex") and (
        codex_path.exists() or codex_path.is_symlink()
    ):
        toml_section = _read_codex_mcp(codex_path, result.errors)
        if toml_section is not None:
            for name, entry in toml_section.items():
                if not isinstance(entry, dict):
                    result.errors.append(f"{codex_path}: MCP server `{name}` must be a table")
                    continue
                if name in result.servers:
                    result.conflicts.append((name, result.servers[name].source_tool, "codex"))
                else:
                    from crossby.sync.mcp_codex import normalize_codex_mcp

                    normalized, unsupported = normalize_codex_mcp(entry)
                    result.servers[name] = DiscoveredServer(
                        name=name,
                        source_tool="codex",
                        data=normalized,
                        unsupported_fields=unsupported,
                        enabled=_entry_enabled(
                            entry, "enabled", f"{codex_path}: MCP server `{name}`", result.errors
                        ),
                    )

    from crossby.sync.mcp_activation import claude_disabled_names, cursor_disabled_names

    if any(server.source_tool == "claude" for server in result.servers.values()):
        rejected, toggled = claude_disabled_names(
            project_root, _GLOBAL_CLAUDE_JSON_PATH, result.errors
        )
        for name, server in result.servers.items():
            if server.source_tool == "claude" and (
                name in toggled or (server.scope == "project" and name in rejected)
            ):
                server.enabled = False
                server.data["enabled"] = False
    if any(server.source_tool == "cursor" for server in result.servers.values()):
        disabled = cursor_disabled_names(project_root, result.errors)
        for name, server in result.servers.items():
            if server.source_tool == "cursor" and name in disabled:
                server.enabled = False
                server.data["enabled"] = False

    return result


def _read_codex_mcp(path: Path, errors: list[str] | None = None) -> dict[str, Any] | None:
    """Read mcp_servers from a Codex TOML config file."""
    try:
        import tomllib

        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        section = raw.get("mcp_servers", {})
        if isinstance(section, dict):
            return section
        if errors is not None:
            errors.append(f"{path}: mcp_servers must be a table")
    except Exception:
        if errors is not None:
            errors.append(f"{path}: unreadable or invalid MCP TOML")
    return None


def report_oauth_configs(
    project_root: Path, *, include_user_scope: bool = False
) -> list[SyncResult]:
    """Report MCP servers whose source entry has an ``oauth`` block.

    No writer in :mod:`crossby.sync.mcp` ports OAuth config (``callbackPort``,
    ``clientId``, ``authServerMetadataUrl``, ...) across tools — it's dropped
    silently during discovery/normalization today because
    :class:`crossby.models.config.MCPServerConfig` has no field for it. This
    turns that silent drop into a manual-fix report row instead, mirroring
    :func:`crossby.sync.plugins.report_plugins`'s detect-only pattern.

    ``file_path=None`` is deliberate (there is no per-target artifact this
    row is about) — it's what makes :func:`crossby.sync.report.classify_status`
    read the row as ``Not Added`` and :func:`crossby.sync.plan.summarize_plan`
    count it toward the doctor readiness score.

    When *include_user_scope* is set the scan covers ``~/.claude.json`` too, so
    a user-scope server being ported doesn't get its ``oauth`` block dropped
    without a trace — matching the scope the sync itself is running at.
    """
    discovery = discover_mcp_servers(project_root, include_user_scope=include_user_scope)
    return [
        SyncResult(
            tool_id=None,
            concern=SyncConcern.MCP,
            action="skipped",
            file_path=None,
            message=(
                f"MCP server `{name}` (from {source_tool}) has an `oauth` block Crossby "
                "does not port across tools; this is a manual-fix — configure OAuth "
                "directly in each target tool's native config."
            ),
        )
        for name, source_tool in discovery.oauth_servers
    ]


def report_duplicate_claude_servers(project_root: Path) -> list[SyncResult]:
    """One informational row per server defined in *both* Claude MCP files.

    crossby writes servers to ``.mcp.json`` and deliberately leaves any copy in
    ``.claude/settings.json`` (where an older crossby or a manual edit may have
    put it) untouched — it never silently edits a key it no longer owns. A name
    living in both files is a stale duplicate the user should resolve, so this
    surfaces it rather than resolving it silently the way discovery's
    first-seen-wins merge does.
    """
    mcp_json = _read_json_section(project_root / ".mcp.json", "mcpServers") or {}
    settings = _read_json_section(project_root / ".claude" / "settings.json", "mcpServers") or {}
    return [
        SyncResult(
            tool_id=None,
            concern=SyncConcern.MCP,
            action="skipped",
            file_path=None,
            message=(
                f"MCP server `{name}` is defined in both `.mcp.json` and "
                "`.claude/settings.json`; crossby writes `.mcp.json` and leaves the "
                "settings.json copy in place — remove one to clear the duplicate."
            ),
        )
        for name in sorted(set(mcp_json) & set(settings))
    ]
