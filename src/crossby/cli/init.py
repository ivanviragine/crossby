"""crossby init — initialize CROSSBY config in a project."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from crossby.ai_tools.base import AbstractAITool
from crossby.models.config import ComplexityModelMapping, CrossbyConfig
from crossby.ui.console import console


def init(
    path: Path = typer.Argument(Path("."), help="Project directory to initialize."),
) -> None:
    """Initialize CROSSBY config in a project.

    Detects installed AI tools, reads their known models from the
    bundled registry, and generates a .crossby.yml config file.
    """
    project_root = path.resolve()
    config_path = project_root / ".crossby.yml"

    if config_path.exists():
        console.warn(f".crossby.yml already exists at {config_path}")
        raise typer.Exit(1)

    console.step("Detecting installed AI tools...")
    installed = AbstractAITool.detect_installed()

    if not installed:
        console.error("No AI tools found in PATH.")
        console.hint("Install at least one AI tool (claude, copilot, gemini, codex, cursor, etc.)")
        raise typer.Exit(1)

    console.success(f"Found {len(installed)} AI tool(s): {', '.join(str(t) for t in installed)}")

    # Build model mappings for each installed tool
    models: dict[str, ComplexityModelMapping] = {}
    for tool_id in installed:
        adapter = AbstractAITool.get(tool_id)
        mapping = adapter.get_recommended_mapping()
        if mapping.easy or mapping.medium or mapping.complex or mapping.very_complex:
            models[str(tool_id)] = mapping

    # Pick default tool
    default_tool = str(installed[0])
    if len(installed) > 1:
        from crossby.ui import prompts

        if prompts.is_tty():
            idx = prompts.select(
                "Select default AI tool",
                [str(t) for t in installed],
            )
            default_tool = str(installed[idx])

    # Build config
    config = CrossbyConfig(
        ai=CrossbyConfig.model_fields["ai"].default.__class__(
            default_tool=default_tool,
        ),
        models=models,
    )

    # Write YAML
    config_dict: dict[str, object] = {"version": config.version}

    ai_dict: dict[str, object] = {}
    if config.ai.default_tool:
        ai_dict["default_tool"] = config.ai.default_tool
    config_dict["ai"] = ai_dict

    if models:
        models_dict: dict[str, dict[str, str | None]] = {}
        for tool_name, mapping in models.items():
            models_dict[tool_name] = {
                k: v for k, v in mapping.model_dump().items() if v is not None
            }
        config_dict["models"] = models_dict

    config_dict["permissions"] = {"allowed_commands": []}

    config_dict["sync"] = {"auto": True, "tools": []}

    # Discover existing MCP servers from tool configs
    from crossby.sync.mcp_discovery import discover_mcp_servers

    discovery = discover_mcp_servers(project_root)
    if discovery.servers:
        from crossby.models.config import MCPServerConfig

        mcp_dict: dict[str, object] = {}
        for name, discovered in discovery.servers.items():
            entry = {k: v for k, v in discovered.data.items() if v is not None and v != []}
            try:
                MCPServerConfig(**entry)
                mcp_dict[name] = entry
            except Exception:
                console.warn(f"Skipping discovered MCP server '{name}' — invalid config from {discovered.source_tool}")
        if mcp_dict:
            config_dict["mcp_servers"] = mcp_dict
            console.success(f"Discovered {len(mcp_dict)} MCP server(s) from existing tool configs")
        if discovery.conflicts:
            for server_name, tool1, tool2 in discovery.conflicts:
                console.warn(
                    f"MCP server '{server_name}' found in both {tool1} and {tool2} — kept {tool1} definition"
                )

    # Detect existing instruction files and propose rules config
    rules_dict = _prompt_rules_config(project_root)
    if rules_dict:
        config_dict["rules"] = rules_dict

    config_path.write_text(
        yaml.dump(config_dict, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    console.success(f"Created {config_path}")
    console.hint("Edit .crossby.yml to customize models, commands, and permissions")
    if discovery.servers:
        console.hint("Run 'crossby sync mcp' to sync MCP servers to all tools")
    if rules_dict:
        console.hint("Run 'crossby sync rules' to sync instruction files")


def _prompt_rules_config(project_root: Path) -> dict[str, object] | None:
    """Detect existing instruction files and build a rules config dict."""
    from crossby.sync.rules import detect_existing_rules, suggest_source
    from crossby.ui import prompts

    existing = detect_existing_rules(project_root)
    if not existing:
        return None

    console.step(
        f"Found instruction file(s): {', '.join(str(p.name) for p in existing.values())}"
    )

    suggested = suggest_source(existing)

    if prompts.is_tty():
        seen = {suggested}
        source_choices = [suggested]
        for p in existing.values():
            s = str(p.relative_to(project_root))
            if s not in seen:
                seen.add(s)
                source_choices.append(s)

        if len(source_choices) > 1:
            idx = prompts.select("Select canonical source file", source_choices)
            source = source_choices[idx]
        else:
            source = suggested

        strategy_idx = prompts.select("Sync strategy", ["symlink", "copy"])
        strategy = ["symlink", "copy"][strategy_idx]
    else:
        source = suggested
        strategy = "symlink"

    rules: dict[str, object] = {"source": source, "strategy": strategy}
    console.success(f"Rules: source={source}, strategy={strategy}")
    return rules
