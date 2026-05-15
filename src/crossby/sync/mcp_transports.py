"""Parse env-var indirection out of MCP server header/env values.

Claude and Cursor MCP configs use literal ``"${VAR}"`` values inside the
``headers`` and ``env`` tables. Codex's ``.codex/config.toml`` has dedicated
fields that name the env var without interpolation:

- ``Authorization: Bearer ${VAR}`` → ``bearer_token_env_var = "VAR"``
- ``X-Header: ${VAR}``             → ``env_http_headers = { "X-Header" = "VAR" }``
- Other static headers              → ``http_headers = { ... }``
- ``env: {NAME: "${NAME}"}``       → ``env_vars = ["NAME"]``  (passthrough)
- ``env: {KEY: "literal"}``        → ``env = { KEY = "literal" }``

Crossby keeps the original Claude/Cursor shape internally (under the
``MCPServerConfig.headers``/``env`` fields), and only the Codex writer
applies these rewrites at render time. Other writers preserve the literal
``${VAR}`` form because the source tools interpret it themselves.

``${VAR:-default}`` fallbacks are intentionally not preserved — neither the
Claude nor Codex schemas have first-class support for them. The literal is
written out and a manual-fix note is emitted by the caller.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

# ``${VAR}`` standalone — captures VAR. Allows ``${VAR:-default}`` syntax
# but does *not* preserve the default; callers should warn separately.
_ENV_VAR_RE = re.compile(r"\A\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}\Z")
# ``Bearer ${VAR}`` — captures VAR.
_BEARER_RE = re.compile(r"\ABearer\s+\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}\Z")


def parse_env_var_ref(value: str) -> str | None:
    """If ``value`` is a pure ``${VAR}`` reference, return ``VAR``.

    Returns ``None`` for literal values, prefixed strings, or anything else.
    Default fallbacks (``${VAR:-default}``) match but the default is
    discarded — callers should emit a manual-fix note.
    """
    match = _ENV_VAR_RE.match(value)
    return match.group(1) if match else None


def parse_bearer_env_var(value: str) -> str | None:
    """If ``value`` is a ``Bearer ${VAR}`` token, return ``VAR``."""
    match = _BEARER_RE.match(value)
    return match.group(1) if match else None


def has_default_fallback(value: str) -> bool:
    """True if ``value`` uses ``${VAR:-default}`` syntax (lossy in rewrite)."""
    return ":-" in value and "${" in value


@dataclass(frozen=True)
class CodexHeaderRewrite:
    """Result of refactoring a Claude/Cursor ``headers`` map for Codex.

    Each output field corresponds to a Codex MCP server config key. Empty
    fields should be omitted from the rendered TOML to avoid noisy output.
    """

    bearer_token_env_var: str | None = None
    http_headers: dict[str, str] = field(default_factory=dict)
    env_http_headers: dict[str, str] = field(default_factory=dict)
    dropped_default_fallbacks: tuple[str, ...] = ()


def rewrite_headers_for_codex(headers: Mapping[str, object]) -> CodexHeaderRewrite:
    """Refactor a flat ``headers`` map into Codex-shaped fields.

    Authorization-Bearer-of-env-var maps to ``bearer_token_env_var``;
    plain ``${VAR}`` headers map to ``env_http_headers``; everything else
    is treated as a literal and goes into ``http_headers``.
    """
    bearer_env_var: str | None = None
    static: dict[str, str] = {}
    env: dict[str, str] = {}
    dropped: list[str] = []

    for raw_key, raw_value in headers.items():
        key = str(raw_key)
        value = str(raw_value)
        if has_default_fallback(value):
            dropped.append(key)

        if key.lower() == "authorization":
            captured = parse_bearer_env_var(value)
            if captured is not None:
                bearer_env_var = captured
                continue

        env_var = parse_env_var_ref(value)
        if env_var is not None:
            env[key] = env_var
            continue

        static[key] = value

    return CodexHeaderRewrite(
        bearer_token_env_var=bearer_env_var,
        http_headers=static,
        env_http_headers=env,
        dropped_default_fallbacks=tuple(dropped),
    )


@dataclass(frozen=True)
class CodexEnvRewrite:
    """Result of refactoring a Claude/Cursor ``env`` map for Codex.

    ``env_vars`` is a list of names whose values should be inherited from
    the parent process; ``env`` keeps literal key=value pairs.
    """

    env: dict[str, str] = field(default_factory=dict)
    env_vars: list[str] = field(default_factory=list)
    dropped_default_fallbacks: tuple[str, ...] = ()


def rewrite_env_for_codex(env: Mapping[str, object]) -> CodexEnvRewrite:
    """Refactor a flat ``env`` map into Codex ``env`` + ``env_vars``.

    Self-references (``KEY = "${KEY}"``) become ``env_vars = ["KEY"]``;
    everything else stays as a literal.
    """
    static: dict[str, str] = {}
    passthrough: list[str] = []
    dropped: list[str] = []

    for raw_key, raw_value in env.items():
        key = str(raw_key)
        value = str(raw_value)
        if has_default_fallback(value):
            dropped.append(key)
        captured = parse_env_var_ref(value)
        if captured is not None and captured == key:
            passthrough.append(key)
            continue
        static[key] = value

    return CodexEnvRewrite(
        env=static,
        env_vars=passthrough,
        dropped_default_fallbacks=tuple(dropped),
    )


__all__ = [
    "CodexEnvRewrite",
    "CodexHeaderRewrite",
    "has_default_fallback",
    "parse_bearer_env_var",
    "parse_env_var_ref",
    "rewrite_env_for_codex",
    "rewrite_headers_for_codex",
]
