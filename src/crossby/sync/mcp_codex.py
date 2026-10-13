"""Reverse native Codex MCP fields without reading the process environment.

Native ``env``/``http_headers`` are literal; ``env_vars`` is an allowlist,
with explicit ``env`` overriding inherited values. See OpenAI's
https://learn.chatgpt.com/docs/config-file/config-reference and
https://github.com/openai/codex/blob/main/codex-rs/rmcp-client/src/utils.rs.
Conditional header fallbacks and remote environment sources cannot be
represented by Crossby's flat, source-shaped maps; reject the whole server
so a partial translation cannot replace a working target's authentication.
"""

from __future__ import annotations

import re
from typing import Any

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_COMMON_FIELDS = {"command", "args", "url", "transport", "enabled"}
_STDIO_FIELDS = {"env", "env_vars"}
_HTTP_FIELDS = {"bearer_token_env_var", "http_headers", "env_http_headers"}


def _env_name(value: object) -> bool:
    return isinstance(value, str) and _ENV_NAME.fullmatch(value) is not None


def _string_map(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    )


def normalize_codex_mcp(entry: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return canonical data and field names whose semantics cannot be ported.

    No values appear in diagnostics. Keep ``transport`` for compatibility with
    Crossby-written TOML. Activation is audited separately by discovery.
    Native URL-only entries imply HTTP.
    """
    result = {k: v for k, v in entry.items() if k in _COMMON_FIELDS}
    is_http = "url" in entry and "command" not in entry
    allowed = _COMMON_FIELDS | (_HTTP_FIELDS if is_http else _STDIO_FIELDS)
    unsupported = set(entry) - allowed
    for field in ("url", "command", "args"):
        value = entry.get(field)
        values = value if isinstance(value, list) else [value]
        if any(isinstance(v, str) and "${" in v for v in values):
            unsupported.add(field)
    if is_http:
        result.setdefault("transport", "http")
        if "args" in entry:
            unsupported.add("args")
        headers: dict[str, str] = {}
        # HTTP header names are case-insensitive. Multiple definitions depend
        # on runtime availability of env vars, which a flat map cannot express.
        header_names: set[str] = set()
        for field in ("http_headers", "env_http_headers"):
            values = entry.get(field, {})
            if not _string_map(values):
                unsupported.add(field)
                continue
            for name, value in values.items():
                folded = name.lower()
                if folded in header_names:
                    unsupported.update(("http_headers", "env_http_headers"))
                header_names.add(folded)
                if field == "env_http_headers":
                    if not _env_name(value):
                        unsupported.add(field)
                    headers[name] = "${" + value + "}"
                else:
                    # A literal ${...} would be expanded by the target or
                    # reclassified by the Codex writer on a return sync.
                    if "${" in value:
                        unsupported.add(field)
                    headers[name] = value
        if "bearer_token_env_var" in entry:
            value = entry["bearer_token_env_var"]
            if not _env_name(value) or "authorization" in header_names:
                unsupported.add("bearer_token_env_var")
            else:
                headers["Authorization"] = "Bearer ${" + value + "}"
        if headers:
            result["headers"] = headers
    else:
        env: dict[str, str] = {}
        env_vars = entry.get("env_vars", [])
        if not isinstance(env_vars, list):
            unsupported.add("env_vars")
        else:
            for item in env_vars:
                name = item
                if isinstance(item, dict):
                    # OpenAI supports structured local entries too. Remote
                    # sources must never be rewritten as local references.
                    if set(item) - {"name", "source"} or item.get("source", "local") != "local":
                        unsupported.add("env_vars")
                        continue
                    name = item.get("name")
                if not _env_name(name):
                    unsupported.add("env_vars")
                    continue
                env[name] = "${" + name + "}"
        explicit = entry.get("env", {})
        if not _string_map(explicit):
            unsupported.add("env")
        else:
            if any("${" in value for value in explicit.values()):
                unsupported.add("env")
            # Native Codex literals override inherited variables.
            env.update(explicit)
        if env:
            result["env"] = env
    return result, tuple(sorted(unsupported))
