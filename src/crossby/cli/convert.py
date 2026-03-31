"""crossby convert — translate allowlist patterns between AI tool formats."""

from __future__ import annotations

from collections.abc import Callable

import typer

from crossby.ui.console import console

PatternTranslator = Callable[[str], str]


def _strip_wrapper(pattern: str, prefix: str) -> str:
    """Strip a tool-specific wrapper like ``Bash(…)`` or ``Shell(…)``."""
    if pattern.startswith(prefix) and pattern.endswith(")"):
        return pattern[len(prefix) : -1]
    return pattern


def _to_shell(pattern: str) -> str:
    """Convert a canonical pattern to shell() format (Copilot/Gemini)."""
    parts = pattern.split(":", 1)
    binary = parts[0]
    args = parts[1] if len(parts) > 1 else ""
    return f"shell({binary}:{args})" if args else f"shell({binary})"


_TO_CANONICAL: dict[str, PatternTranslator] = {
    "claude": lambda p: _strip_wrapper(p, "Bash("),
    "cursor": lambda p: _strip_wrapper(p, "Shell("),
    "copilot": lambda p: _strip_wrapper(p, "shell("),
    "gemini": lambda p: _strip_wrapper(p, "shell("),
}

_FROM_CANONICAL: dict[str, PatternTranslator] = {}


def _init_from_canonical() -> dict[str, PatternTranslator]:
    """Build the canonical→tool translator map (lazy, cached)."""
    if _FROM_CANONICAL:
        return _FROM_CANONICAL

    from crossby.sync.permissions import canonical_to_claude, canonical_to_cursor

    _FROM_CANONICAL["claude"] = canonical_to_claude
    _FROM_CANONICAL["cursor"] = canonical_to_cursor
    _FROM_CANONICAL["copilot"] = _to_shell
    _FROM_CANONICAL["gemini"] = _to_shell
    return _FROM_CANONICAL


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
        strip_fn = _TO_CANONICAL.get(from_tool)
        canonical = strip_fn(pattern) if strip_fn else pattern

    # Step 2: Convert from canonical to target format
    if to_tool == "canonical":
        result = canonical
    else:
        translators = _init_from_canonical()
        translate_fn = translators.get(to_tool)
        result = translate_fn(canonical) if translate_fn else canonical

    console.plain(result)
