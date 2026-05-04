"""Discover Claude plugin trees and marketplaces; report as manual-fix.

Claude plugins (``.claude/plugins/``), plugin-marketplace registries
(``.claude/plugin-marketplaces.json``), and plugin marketplace manifests
(``.claude-plugin/marketplace.json``) bundle commands, agents, MCP
servers, skills, and hooks with provider-specific metadata. No other
supported tool has the same primitive, and the bundles can include
sub-resources Crossby doesn't have a faithful translator for. This
module detects them and emits one ``SyncResult(action="skipped",
message="manual setup required …")`` per finding so users see what
their next sync didn't touch.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from crossby.sync.base import SyncConcern, SyncResult


@dataclass(frozen=True)
class PluginFinding:
    """One plugin / marketplace entry that needs manual migration."""

    path: Path
    label: str
    detail: str


# Path → human label, in detection order.
_PLUGIN_PATHS: tuple[tuple[Path, str, str], ...] = (
    (
        Path(".claude") / "plugins",
        "Claude plugins directory",
        "Plugin tree needs manual migration; bundled commands, agents, MCP "
        "servers, skills, and hooks must be ported by hand.",
    ),
    (
        Path(".claude") / "plugin-marketplaces.json",
        "Claude plugin marketplace registry",
        "Plugin marketplace registry needs manual setup; install or copy "
        "any referenced plugins to the target tool by hand.",
    ),
    (
        Path(".claude-plugin") / "marketplace.json",
        "Claude plugin marketplace manifest",
        "Plugin marketplace manifest needs manual setup.",
    ),
)


def _read_plugin_dir_names(plugins_dir: Path) -> list[str]:
    """Return immediate subdirectory names of a plugins/ tree (sorted)."""
    if not plugins_dir.is_dir():
        return []
    return sorted(child.name for child in plugins_dir.iterdir() if child.is_dir())


def _read_marketplace_entries(path: Path) -> list[str]:
    """Best-effort extract of named entries from a marketplace JSON file."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    entries: list[str] = []
    if isinstance(data, dict):
        plugins = data.get("plugins")
        if isinstance(plugins, list):
            for entry in plugins:
                if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                    entries.append(str(entry["name"]))
    return sorted(set(entries))


def discover_plugins(project_root: Path) -> list[PluginFinding]:
    """Walk known plugin paths under ``project_root`` and return findings."""
    findings: list[PluginFinding] = []

    plugins_dir = project_root / Path(".claude") / "plugins"
    plugin_names = _read_plugin_dir_names(plugins_dir)
    if plugin_names:
        for name in plugin_names:
            findings.append(
                PluginFinding(
                    path=plugins_dir / name,
                    label=f"plugin `{name}`",
                    detail=(
                        "Plugin needs manual migration; bundled commands, agents, "
                        "MCP servers, skills, and hooks are not auto-converted."
                    ),
                )
            )
    elif plugins_dir.is_dir():
        # Empty plugins directory still warrants a row so users know we saw it.
        findings.append(
            PluginFinding(
                path=plugins_dir,
                label="Claude plugins directory",
                detail="Plugins directory present but empty.",
            )
        )

    for relative, label_template, detail_template in _PLUGIN_PATHS[1:]:
        path = project_root / relative
        if not path.exists():
            continue
        entries = _read_marketplace_entries(path)
        if entries:
            for entry in entries:
                findings.append(
                    PluginFinding(
                        path=path,
                        label=f"marketplace plugin `{entry}`",
                        detail=detail_template,
                    )
                )
        else:
            findings.append(
                PluginFinding(
                    path=path,
                    label=label_template,
                    detail=detail_template,
                )
            )

    return findings


def findings_to_results(findings: Iterable[PluginFinding]) -> list[SyncResult]:
    """Render each finding as a ``SyncResult(action="skipped")`` row."""
    return [
        SyncResult(
            tool_id=None,
            concern=SyncConcern.PLUGINS,
            action="skipped",
            file_path=finding.path,
            message=f"{finding.label}: {finding.detail}",
        )
        for finding in findings
    ]


def report_plugins(project_root: Path) -> list[SyncResult]:
    """Convenience: discover + render in one call."""
    return findings_to_results(discover_plugins(project_root))


def has_plugin_findings(results: Sequence[SyncResult]) -> bool:
    return any(r.concern == SyncConcern.PLUGINS for r in results)


__all__ = [
    "PluginFinding",
    "discover_plugins",
    "findings_to_results",
    "has_plugin_findings",
    "report_plugins",
]
