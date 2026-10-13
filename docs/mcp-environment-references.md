# MCP environment-reference portability

Crossby translates reference syntax without reading environment values. Sync,
`--plan`, and `--dry-run` do not launch MCP servers or test credentials; each
native client resolves the references when it connects. Missing-variable and
credential-filtering behavior belongs to that client, not Crossby.

## Supported bridge

| Native source | Cursor target | Reverse target |
| --- | --- | --- |
| Claude `${NAME}` in `command`, `args`, `env`, `url`, `headers` | `${env:NAME}`, retaining prefixes, suffixes and multiple references | Claude `${NAME}` |
| Codex `env_vars` (local names), `env_http_headers`, `bearer_token_env_var` | Self env references, whole header references, `Bearer ${env:NAME}` | Codex's dedicated reference fields |
| Plain strings, including empty strings and `$NAME` without braces | Preserved | Preserved |

Names use `[A-Za-z_][A-Za-z0-9_]*`. For example, Claude
`"Authorization": "Bearer ${API_TOKEN}"` becomes Cursor
`"Authorization": "Bearer ${env:API_TOKEN}"`. Native Cursor URL-only entries
imply HTTP. Cursor-to-Claude supports surrounding text; Cursor-to-Codex supports
only self env references, a whole header reference, and an Authorization value
with the exact `Bearer ${env:NAME}` form. Codex has no equivalent for env aliases,
prefixes/suffixes, multiple references, or command/argument/URL interpolation.

## Refusals preserve the target

Crossby reports a manual-fix (`Not Added`) and preserves the affected target
server when it cannot carry the connection faithfully. Supported siblings still
sync. Refusals include:

- Defaults such as Claude `${NAME:-fallback}` when targeting Cursor. Cursor's
  documentation does not establish equivalent fallback syntax.
- Cursor `${workspaceFolder}`, `${userHome}`, other non-environment expressions,
  malformed/nested expressions, undocumented escapes, and environment names
  outside the supported grammar.
- Interpolation-shaped literals: Cursor `${NAME}` is not a Cursor environment
  reference; copying it to Claude would turn a literal into a reference.
  Claude `${env:NAME}` cannot be copied to Cursor as a literal either. Crossby's
  flat canonical model cannot safely round-trip these literals, so it refuses
  the server rather than interpreting them.
- Cursor fields outside the portable connection model, including `envFile` and
  `auth`, which must be configured in the target's native file.
- Cursor references targeting Copilot or Antigravity CLI. Their writers remain
  conservative; this bridge does not establish their interpolation semantics.

Native Codex literal `${...}` values retain the existing refusal behavior.
Claude-to-Codex's existing default-fallback rewriting and reporting are
unchanged: where that older rewrite drops a default, the report still says so.

Portability and activation are separate. Valid explicit source disables can
still revoke Crossby-owned target copies even for unsupported connections;
user-authored same-name entries are never claimed or revoked. Existing target
native disables remain in place. Malformed source or activation files refuse
MCP sync before writes. See [activation discovery](mcp-activation-discovery.md).

## Native-format audit

Audited on 2026-10-13:

- [Cursor config interpolation](https://cursor.com/docs/mcp#config-interpolation)
  documents `${env:NAME}` in the five connection fields and distinguishes its
  workspace/home/path expressions. It gives both a self env example and a
  Bearer header example, but does not document defaults or escaping.
- [Claude environment expansion](https://code.claude.com/docs/en/mcp#environment-variable-expansion-in-mcpjson)
  documents `${NAME}`, defaults, and the same five fields. Claude also documents
  runtime restrictions for some credential variables; a reference translation
  does not override the native client's security policy.
- [Codex configuration reference](https://developers.openai.com/codex/config-reference)
  and [native MCP client source](https://github.com/openai/codex/blob/main/codex-rs/rmcp-client/src/utils.rs)
  distinguish literal maps from named inherited variables and named HTTP
  headers. Explicit env literals override inherited values, and HTTP header
  overlaps can depend on runtime availability; the #166 refusal of ambiguous
  fallbacks remains intact.
