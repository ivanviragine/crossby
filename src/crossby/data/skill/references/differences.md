# crossby cross-tool differences

Per-surface mapping table for the eight tools `crossby` supports today:
Claude, Cursor, Gemini, Codex, Copilot, OpenCode, VS Code, and
Antigravity. Direct 1:1 mappings (e.g. `Bash(myapp:*)` ↔
`Shell(myapp:*)` ↔ `myapp:*`) are listed once; lossy or unsupported
edges are flagged.

Docs last checked: 2026-05-04. If today's date is later, re-open each
tool's docs and confirm the schemas before trusting these rows.

## Rules / instructions

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| `CLAUDE.md` / `AGENTS.md` / `.cursorrules` / `GEMINI.md` / `.github/copilot-instructions.md` | every other tool's instruction file | symlink (default) | All tools accept the same plain-markdown body; symlink keeps every target in lockstep with the source. |
| Source content with Claude-only markers (`/hooks`, `.claude/agents/`, `Subagent`, `permissionMode`, `ExitPlanMode`, `TodoWrite`) | every non-Claude target | force-copy with `<!-- crossby:manual-fix -->` | Crossby refuses to symlink so target-tool semantics aren't silently overridden. Edit the copy and remove the manual-fix block when done. |
| Source content with Codex-only markers (`.codex/`, `sandbox_mode`, `developer_instructions`) | every non-Codex target | force-copy with manual-fix | Same idea, reversed direction. |

## Agents

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| `.claude/agents/<name>.md` (markdown + YAML frontmatter) | `.cursor/agents/`, `.gemini/agents/`, `.github/agents/` | directory symlink | All four markdown-shape tools accept the same on-disk format. |
| `.claude/agents/<name>.md` | `.codex/agents/<name>.toml` | per-file translate | TOML schema differs; `name`, `description`, `developer_instructions`, plus mapped `model`, `model_reasoning_effort`, `sandbox_mode`. |
| Frontmatter `permissionMode: acceptEdits` / `readOnly` | `sandbox_mode: workspace-write` / `read-only` | direct mapping | Other Claude modes (`default`, `dontAsk`, `plan`, `bypassPermissions`) carry through as a `<!-- crossby:manual-fix -->` block — Codex has no equivalent. |
| Frontmatter `model: claude-opus-*` | `model = "gpt-5.4"` | family mapping | `claude-sonnet-*` and `claude-haiku-*` map to `gpt-5.4-mini`. |
| Frontmatter `model: claude-sonnet-*` + `effort` | `model_reasoning_effort` (one tier higher) | family-aware effort bias | Sonnet shifts up: `low → medium`, `medium → high`, `high → xhigh`. Opus and Haiku map 1:1, `max → xhigh`. |
| Frontmatter `tools` / `disallowedTools` / `skills` | None | manual-fix only | Preserved as guidance under the `## Manual migration required` block; Codex doesn't enforce these as a permission boundary. |
| Frontmatter `name` / `description` missing | inferred | filename slug + first H1 fallback | Every translated TOML still carries the three Codex-required keys. |

## Skills

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| `<tool>/skills/<name>/SKILL.md` | every other tool's skills dir | directory symlink (default) | All tools accept SKILL.md verbatim. |
| Source has `allowed-tools` and target ≠ Claude | per-tool copy with manual-fix | translate strategy (`--strategy translate`) | `allowed-tools` is Claude-only; non-Claude targets see a manual-fix note explaining the field isn't enforced. |
| Source `scripts/`, `references/`, `assets/` subdirs | mirrored into the translated target skill dir | translate strategy | Support files are copied verbatim. |
| `.claude/commands/<name>.md` (Claude slash command) | `<target-skills-dir>/claude-command-<slug>/SKILL.md` | translate strategy, only when target ≠ Claude | The body is wrapped under a `## Command Template` heading; runtime expansion (`$ARGUMENTS`, `!\`shell\``, `@file`, `{{tpl}}`) becomes a manual-fix note. |

## MCP and config

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| `.claude/settings.json` `mcpServers` | `.cursor/mcp.json`, `.gemini/settings.json`, `.vscode/mcp.json`, `.codex/config.toml` | non-destructive merge | Existing entries the user added by hand are preserved. |
| `headers: {Authorization: "Bearer ${TOKEN}"}` | Codex `bearer_token_env_var = "TOKEN"` | regex rewrite | Only the `Bearer ${VAR}` shape is rewritten; `${VAR:-default}` fallbacks are dropped and the key is reported. |
| `headers: {X-Foo: "${VAR}"}` | Codex `env_http_headers = {X-Foo = "VAR"}` | regex rewrite | Static headers stay in `http_headers`. |
| `env: {KEY: "${KEY}"}` (self-reference) | Codex `env_vars = ["KEY"]` | regex rewrite | Other `${VAR}` env values stay literal so source tools can interpolate them. |
| `enabled: false` / `disabled: true` | dropped from the target | merge | Disabled servers in the source are removed from every target. |

## Permissions

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| `.claude/settings.json` `permissions.allow` (`Bash(cmd:*)`) | canonical `cmd:*` | reverse parse | Only `Bash(...)` entries are read; non-Bash patterns are skipped. |
| Canonical `cmd:*` | `Bash(cmd:*)` (Claude), `Shell(cmd:*)` (Cursor), `commandPrefix = "cmd"` (Gemini policy TOML) | per-tool translator | Symmetrical: every supported tool can be source or target. |
| Cursor / Gemini specific shapes (`Shell(cmd:*)`, policy TOML rules) | canonical | reverse parse | Allows any tool to seed the canonical pattern set. |

## Hooks

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| `.claude/settings.json` `hooks.PreToolUse` | `.cursor/hooks.json` `preToolUse`, Copilot `preToolUse`, Gemini `BeforeTool` | event-name translation | Tool name strings (`Bash`, `Edit`, `Write`) are also translated per tool. |
| Multiple sources defining the same `(event, command)` | matcher widened (union of tools) | merge | Re-running with a broader matcher upgrades coverage instead of duplicating entries. |
| `Stop`, `Notification`, `SessionStart`, `UserPromptSubmit` | per-tool equivalents where they exist | partial mapping | Unsupported events are dropped with a warning. |

## Plugins

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| `.claude/plugins/<name>/` | none | report-only | Each plugin becomes a `Not Added` row in the persistent report; bundled commands/agents/MCP/skills/hooks must be migrated by hand. |
| `.claude/plugin-marketplaces.json` | none | report-only | Marketplace registries don't have an equivalent; install referenced plugins manually for the target tool. |
| `.claude-plugin/marketplace.json` | none | report-only | Marketplace manifests are detected and listed; the entries inside them are surfaced individually. |

## Models

| Source family | Codex default | Effort bias | Notes |
| --- | --- | --- | --- |
| `claude-opus-*` | `gpt-5.4` | 1:1 (`max → xhigh`) | Reverse: `gpt-5.4 → claude-opus-4.7` (latest alias). |
| `claude-sonnet-*` | `gpt-5.4-mini` | shift up one tier | Coding-agent bias; reverse picks the lowest source tier that maps to the given Codex tier (`xhigh → high`). |
| `claude-haiku-*` | `gpt-5.4-mini` | 1:1 (`max → xhigh`) | Reverse: `gpt-5.4-mini → claude-sonnet-4.6` by default. |

## Validation

| Check | Level | What it catches |
| --- | --- | --- |
| `.codex/config.toml` parses as TOML | error | Manual edits that broke the file. |
| `.codex/agents/*.toml` carries `name` / `description` / `developer_instructions` | error | Translated agent files that lost a required field. |
| `<tool>/skills/<name>/SKILL.md` carries `name` / `description` | error | Stripped or hand-edited skill frontmatter. |
| Each `mcp_servers.<name>.command` is on `PATH` | warning | Missing binary on the host; users see this before the first invocation fails. |
| `AGENTS.md` / `CLAUDE.md` / `GEMINI.md` / `.cursorrules` / `.github/copilot-instructions.md` ≤ 32KB | warning | Instructions creeping past the size threshold beyond which review becomes painful. |
| Tool-specific JSON files parse as JSON | error | `.claude/settings.json`, `.cursor/cli.json`, `.cursor/mcp.json`, `.vscode/mcp.json`, `.gemini/settings.json`. |

## Sources

- https://code.claude.com/docs/en/skills
- https://code.claude.com/docs/en/sub-agents
- https://code.claude.com/docs/en/hooks
- https://code.claude.com/docs/en/mcp
- https://code.claude.com/docs/en/settings
- https://code.claude.com/docs/en/plugins
- https://developers.openai.com/codex/config-reference
- https://developers.openai.com/codex/mcp
- https://developers.openai.com/codex/skills
- https://developers.openai.com/codex/subagents
- https://developers.openai.com/codex/hooks
- https://docs.cursor.com/cli
- https://docs.github.com/en/copilot/using-github-copilot/copilot-cli
