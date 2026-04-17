"""crossby convert — translate allowlist patterns between AI tool formats."""

from __future__ import annotations

import typer

from crossby.ui.console import console

# Tool format translators
_TRANSLATORS: dict[str, dict[str, object]] = {}


def _get_to_canonical(tool: str) -> object:
    """Get the function that converts a tool-specific pattern to canonical."""
    # For now, support stripping the wrapper
    wrappers = {
        "claude": "Bash(",
        "cursor": "Shell(",
        "copilot": "shell(",
    }
    prefix = wrappers.get(tool)
    if not prefix:
        return None

    def _strip(pattern: str) -> str:
        if pattern.startswith(prefix) and pattern.endswith(")"):
            return pattern[len(prefix) : -1]
        return pattern

    return _strip


def _get_from_canonical(tool: str) -> object:
    """Get the function that converts a canonical pattern to tool-specific."""
    translators: dict[str, object] = {}

    from crossby.config.claude_allowlist import canonical_to_claude
    from crossby.config.cursor_allowlist import canonical_to_cursor

    translators["claude"] = canonical_to_claude
    translators["cursor"] = canonical_to_cursor

    def _to_shell(pattern: str) -> str:
        parts = pattern.split(":", 1)
        binary = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        return f"shell({binary}:{args})" if args else f"shell({binary})"

    translators["copilot"] = _to_shell

    return translators.get(tool)


def convert(
    pattern: str = typer.Argument(help="Allowlist pattern to translate."),
    from_tool: str = typer.Option(..., "--from", help="Source tool format."),
    to_tool: str = typer.Option(..., "--to", help="Target tool format."),
) -> None:
    """Translate allowlist patterns between AI tool formats.

    Examples:

        crossby convert "Bash(myapp:*)" --from claude --to cursor
        crossby convert "myapp:*" --from canonical --to claude
    """
    supported = {"claude", "cursor", "copilot", "gemini", "canonical"}

    if from_tool not in supported:
        supported_str = ", ".join(sorted(supported))
        console.error(f"Unknown source tool: {from_tool}. Supported: {supported_str}")
        raise typer.Exit(1)

    if to_tool not in supported:
        supported_str = ", ".join(sorted(supported))
        console.error(f"Unknown target tool: {to_tool}. Supported: {supported_str}")
        raise typer.Exit(1)

    # Step 1: Convert to canonical format
    if from_tool == "canonical":
        canonical = pattern
    elif from_tool == "gemini":
        console.error(
            "Gemini CLI no longer uses --allowed-tools flags. "
            "Permissions are managed via .gemini/policies/crossby.toml "
            "(written automatically by 'crossby sync')."
        )
        raise typer.Exit(1)
    else:
        strip_fn = _get_to_canonical(from_tool)
        canonical = strip_fn(pattern) if strip_fn and callable(strip_fn) else pattern

    # Step 2: Convert from canonical to target format
    if to_tool == "canonical":
        result = canonical
    elif to_tool == "gemini":
        console.error(
            "Gemini CLI no longer uses --allowed-tools flags. "
            "Permissions are managed via .gemini/policies/crossby.toml "
            "(written automatically by 'crossby sync')."
        )
        raise typer.Exit(1)
    else:
        translate_fn = _get_from_canonical(to_tool)
        result = translate_fn(canonical) if translate_fn and callable(translate_fn) else canonical

    console.plain(result)
