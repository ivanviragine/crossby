"""A bounded Cursor/Claude MCP reference bridge, never a value evaluator.

Cursor documents ${env:NAME} in command, args, env, url and headers:
https://cursor.com/docs/mcp#config-interpolation
Only ordinary environment names are portable. Other ${...} forms (including
Cursor literals shaped like Claude references) cannot enter the canonical
Claude-shaped model without losing their identity, so refuse the server.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from crossby.models.config import MCPServerConfig

_CLAUDE_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_CURSOR_REF = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


def _translate(value: str, *, to_cursor: bool) -> str:
    pattern = _CLAUDE_REF if to_cursor else _CURSOR_REF
    # No documented escape equivalence: do not guess about doubled dollars or
    # backslash-prefixed expressions. Check remaining text too, so defaults,
    # nested, malformed and other native interpolation cannot be partly ported.
    for match in pattern.finditer(value):
        if match.start() and value[match.start() - 1] in {"$", "\\"}:
            raise ValueError("unsupported reference escape")
    if "${" in pattern.sub("", value):
        raise ValueError("unsupported interpolation or interpolation-shaped literal")
    prefix = "${env:" if to_cursor else "${"
    return pattern.sub(lambda m: prefix + m.group(1) + "}", value)


def _map_fields(
    entry: dict[str, Any], transform: Callable[[str], str]
) -> tuple[dict[str, Any], tuple[str, ...]]:
    result = dict(entry)
    unsupported: list[str] = []
    for field in ("command", "args", "env", "url", "headers"):
        value = entry.get(field)
        try:
            if isinstance(value, str):
                result[field] = transform(value)
            elif isinstance(value, list):
                result[field] = [transform(v) if isinstance(v, str) else v for v in value]
            elif isinstance(value, dict):
                result[field] = {
                    k: transform(v) if isinstance(v, str) else v for k, v in value.items()
                }
        except ValueError:
            unsupported.append(field)
    return result, tuple(unsupported)


def normalize_cursor_mcp(entry: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Normalize supported references; return field-name-only refusals."""
    return _map_fields(entry, lambda v: _translate(v, to_cursor=False))


def render_cursor_mcp(entry: dict[str, Any]) -> dict[str, Any]:
    """Render validated canonical references in Cursor's native syntax."""
    rendered, unsupported = _map_fields(entry, lambda v: _translate(v, to_cursor=True))
    if unsupported:
        raise ValueError("Unsupported Cursor interpolation in " + ", ".join(unsupported))
    return rendered


def cursor_target_unsupported(server: MCPServerConfig) -> tuple[str, ...]:
    """Preflight the same fields that rendering uses, before any target writes."""
    _, unsupported = _map_fields(server.model_dump(), lambda v: _translate(v, to_cursor=True))
    return unsupported


def cursor_codex_unsupported(server: MCPServerConfig) -> tuple[str, ...]:
    """Codex can express self env refs, whole headers and exact Bearer refs.

    Prefixes/suffixes, aliases and references in executable/URL/arguments lack
    native equivalents. Never turn them into Codex literals on reverse sync.
    """
    unsupported: set[str] = set()
    for field, values in (
        ("command", [server.command]),
        ("args", server.args),
        ("url", [server.url]),
    ):
        if any(isinstance(v, str) and "${" in v for v in values):
            unsupported.add(field)
    for key, value in server.env.items():
        if "${" in value and value != "${" + key + "}":
            unsupported.add("env")
    folded_names: set[str] = set()
    for key, value in server.headers.items():
        folded = key.lower()
        if folded in folded_names:
            unsupported.add("headers")
        folded_names.add(folded)
        if "${" not in value:
            continue
        if _CLAUDE_REF.fullmatch(value):
            continue
        if (
            folded == "authorization"
            and value.startswith("Bearer ")
            and _CLAUDE_REF.fullmatch(value[len("Bearer ") :])
        ):
            continue
        unsupported.add("headers")
    return tuple(sorted(unsupported))
