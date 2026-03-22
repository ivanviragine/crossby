"""Shared helpers for translating canonical command patterns."""

from __future__ import annotations


def canonical_to_wrapped(pattern: str, wrapper: str) -> str:
    """Wrap a canonical command pattern in a tool-specific function call."""
    return f"{wrapper}({pattern})"


def wrapped_to_canonical(pattern: str, wrapper: str) -> str:
    """Strip a tool-specific wrapper back to the canonical command pattern."""
    prefix = f"{wrapper}("
    if pattern.startswith(prefix) and pattern.endswith(")"):
        return pattern[len(prefix) : -1]
    return pattern


def canonical_to_shell(pattern: str) -> str:
    """Translate a canonical command pattern to shell(...) syntax."""
    return canonical_to_wrapped(pattern, "shell")


def shell_to_canonical(pattern: str) -> str:
    """Strip shell(...) syntax back to the canonical command pattern."""
    return wrapped_to_canonical(pattern, "shell")
