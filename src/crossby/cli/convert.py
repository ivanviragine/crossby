"""crossby convert — translate allowlist patterns between AI tool formats."""

from __future__ import annotations

from collections.abc import Callable

import typer

from crossby.config.patterns import canonical_to_shell, shell_to_canonical, wrapped_to_canonical
from crossby.ui.console import console

PatternTranslator = Callable[[str], str]


def _get_to_canonical(tool: str) -> PatternTranslator | None:
    """Get the function that converts a tool-specific pattern to canonical."""
    translators: dict[str, PatternTranslator] = {
        "claude": lambda pattern: wrapped_to_canonical(pattern, "Bash"),
        "cursor": lambda pattern: wrapped_to_canonical(pattern, "Shell"),
        "copilot": shell_to_canonical,
        "gemini": shell_to_canonical,
    }
    return translators.get(tool)


def _get_from_canonical(tool: str) -> PatternTranslator | None:
    """Get the function that converts a canonical pattern to tool-specific."""
    translators: dict[str, PatternTranslator] = {}

    from crossby.config.claude_allowlist import canonical_to_claude
    from crossby.config.cursor_allowlist import canonical_to_cursor

    translators["claude"] = canonical_to_claude
    translators["cursor"] = canonical_to_cursor

    # Copilot and Gemini use the same shell() format
    translators["copilot"] = canonical_to_shell
    translators["gemini"] = canonical_to_shell

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
    else:
        strip_fn = _get_to_canonical(from_tool)
        canonical = strip_fn(pattern) if strip_fn else pattern

    # Step 2: Convert from canonical to target format
    if to_tool == "canonical":
        result = canonical
    else:
        translate_fn = _get_from_canonical(to_tool)
        result = translate_fn(canonical) if translate_fn else canonical

    console.plain(result)
