# crossby cross-tool differences

Per-surface mapping table for the eight tools `crossby` supports today:
Claude, Cursor, Gemini, Codex, Copilot, OpenCode, VS Code, and
Antigravity. Direct 1:1 mappings (e.g. `Bash(myapp:*)` ↔
`Shell(myapp:*)` ↔ `myapp:*`) are listed once; lossy or unsupported
edges are flagged.

Docs last checked: 2026-05-15. If today's date is later, re-open each
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
| Source has `model`, `effort`, `disable-model-invocation`, `user-invocable`, `argument-hint`, `context`, `agent`, `hooks`, or `paths`/`shell` and target ≠ Claude | per-tool copy with manual-fix | translate strategy | These Claude-only skill fields are kept in frontmatter for reference (no data loss, round-trips back to Claude cleanly) but flagged with one combined manual-fix note since no other tool interprets them. |
| Source `scripts/`, `references/`, `assets/` subdirs | mirrored into the translated target skill dir | translate strategy | Support files are copied verbatim. |
| `.claude/commands/<name>.md` (Claude slash command) | `<target-skills-dir>/claude-command-<slug>/SKILL.md` | translate strategy, only when target ≠ Claude | The body is wrapped under a `## Command Template` heading; runtime expansion (`$ARGUMENTS`, `!\`shell\``, `@file`, `{{tpl}}`) becomes a manual-fix note. |
| `.cursor/commands/<name>.md` (Cursor slash command) | `<target-skills-dir>/cursor-command-<slug>/SKILL.md` | translate strategy, only when target ≠ Cursor | Plain-markdown body wrapped under `## Command Template`; no Cursor runtime expansion is currently detected, so only the generic "this was a slash command" manual-fix note is emitted. |
| `.gemini/commands/<name>.md` (Gemini slash command) | `<target-skills-dir>/gemini-command-<slug>/SKILL.md` | translate strategy, only when target ≠ Gemini | The Gemini `{{args}}` template placeholder becomes a manual-fix note; everything else passes through verbatim. |

## MCP and config

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| `.mcp.json` (project scope), `.claude/settings.json`, `~/.claude.json` (user scope) `mcpServers` | `.cursor/mcp.json`, `.gemini/settings.json`, `.vscode/mcp.json`, `.codex/config.toml` | non-destructive merge | Existing entries the user added by hand are preserved. All three Claude sources are scanned most-specific-first (project `.mcp.json` wins on a name collision), so name collisions between them are resolved silently rather than reported as a conflict. |
| `headers: {Authorization: "Bearer ${TOKEN}"}` | Codex `bearer_token_env_var = "TOKEN"` | regex rewrite | Only the `Bearer ${VAR}` shape is rewritten; `${VAR:-default}` fallbacks are dropped and the key is reported. |
| `headers: {X-Foo: "${VAR}"}` | Codex `env_http_headers = {X-Foo = "VAR"}` | regex rewrite | Static headers stay in `http_headers`. |
| `env: {KEY: "${KEY}"}` (self-reference) | Codex `env_vars = ["KEY"]` | regex rewrite | Other `${VAR}` env values stay literal so source tools can interpolate them. |
| `enabled: false` / `disabled: true` | dropped from the target | merge | Disabled servers in the source are removed from every target. |
| `oauth: {...}` (`callbackPort`, `clientId`, `authServerMetadataUrl`, ...) | not written to any target | report-only | No writer ports OAuth config across tools; a `manual-fix` row is reported per server instead of silently dropping it (`Not Added` in the sync report). Configure OAuth manually for each target. |

## Permissions

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| `.claude/settings.json` `permissions.allow` (`Bash(cmd:*)`) | canonical `cmd:*` | reverse parse | Only `Bash(...)` entries are read; non-Bash patterns are skipped. |
| Canonical `cmd:*` | `Bash(cmd:*)` (Claude), `Shell(cmd:*)` (Cursor), `commandPrefix = "cmd"` (Gemini policy TOML) | per-tool translator | Symmetrical: every supported tool can be source or target. |
| Cursor / Gemini specific shapes (`Shell(cmd:*)`, policy TOML rules) | canonical | reverse parse | Allows any tool to seed the canonical pattern set. |

## Hooks

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| Canonical events `pre_tool_use`, `post_tool_use`, `session_start`, `user_prompt_submit`, `stop`, `notification` | per-tool event names (PascalCase / camelCase / BeforeTool / etc.) | event-name translation | Each writer also drops events its tool can't represent and records a `manual_fix` note in the report row. |
| Source hook with `tools` filter | `.cursor/hooks.json` only honours `tools` on its tool-execution events; Cursor `stop`, Codex `Stop` / `UserPromptSubmit` ignore matcher | partial mapping | The `tools` / `matcher` field is stripped on write and the dropped scope shows up as a `hooks.<event>.matcher` manual-fix note. |
| Source hook of unsupported event for the target (e.g. Claude `Notification` → Codex, anything but `pre_tool_use` → Copilot) | dropped from the target | manual-fix | Each unique unsupported event produces one `hooks.<event>` manual-fix note so the user knows what didn't make it across. |
| Any hook written to `.codex/hooks.json` | inert until `[features].codex_hooks = true` is set in `.codex/config.toml` | always-on manual-fix | Codex won't load the file otherwise. `CodexHooksWriter` always emits the `features.codex_hooks` reminder, even when every event mapped cleanly. |
| Multiple sources defining the same `(event, command)` | matcher widened (union of tools) | merge | Re-running with a broader matcher upgrades coverage instead of duplicating entries. |

Per-tool supported events:

| Tool | Supported canonical events | Honours `matcher` / `tools` on |
| --- | --- | --- |
| Claude | `pre_tool_use`, `post_tool_use`, `session_start`, `user_prompt_submit`, `stop`, `notification` | every event |
| Codex | `pre_tool_use`, `post_tool_use`, `session_start`, `user_prompt_submit`, `stop` | `pre_tool_use`, `post_tool_use`, `session_start` only |
| Cursor | `pre_tool_use`, `user_prompt_submit`, `stop` | `pre_tool_use` only |
| Copilot | `pre_tool_use` | none — Copilot hooks apply to every tool |
| Gemini | `pre_tool_use` (`BeforeTool`), `post_tool_use` (`AfterTool`) | both |

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
| Every MCP server `command` is on `PATH` across `.codex/config.toml`, `.claude.json`, `.mcp.json`, `.claude/settings.json`, `.cursor/mcp.json`, `.vscode/mcp.json`, `.gemini/settings.json` | warning | Missing binary on the host; users see this before the first invocation fails. Env-var-templated commands like `${HOME}/bin/foo` are expanded via `os.path.expandvars` before the lookup. |
| `AGENTS.md` / `CLAUDE.md` / `GEMINI.md` / `.cursorrules` / `.github/copilot-instructions.md` ≤ 32KB | warning | Instructions creeping past the size threshold beyond which review becomes painful. |
| Tool-specific JSON files parse as JSON | error | `.claude/settings.json`, `.cursor/cli.json`, `.cursor/mcp.json`, `.vscode/mcp.json`, `.gemini/settings.json`. |

## Directional caveats

The lossy edges Crossby actually emits today, grouped by source → target. A
missing row means the source ↔ target pair is non-lossy for that concern;
absent pairs (OpenCode / VS Code / Antigravity as direct source/target for
the concerns below) aren't wired yet.

| Source → Target | Concern | What gets dropped or rewritten |
| --- | --- | --- |
| Claude → Codex | agents | `permissionMode: plan` / `dontAsk` / `bypassPermissions`; `tools` / `disallowedTools` / `skills` (become prompt guidance only) |
| Claude → Codex | hooks | `Notification`; `matcher` on `UserPromptSubmit` / `Stop`; non-`command` hook types (`prompt`, `agent`, `http`, `async`) |
| Claude → Codex | mcp | `headers` with `${VAR:-default}` fallbacks; `oauth` (whole block — reported as a manual-fix row, not written to any target); `type: sse` |
| Claude → Cursor | hooks | every event except `pre_tool_use`, `user_prompt_submit`, and `stop`; `tools` filter on `user_prompt_submit` / `stop` |
| Claude → Copilot | hooks | every event except `pre_tool_use`; the `tools` filter (Copilot has no per-tool scope) |
| Claude → Gemini | hooks | every event except `pre_tool_use` and `post_tool_use` (kept as `BeforeTool` / `AfterTool`) |
| Codex → Claude | agents | `model_reasoning_effort` (Claude has no equivalent); `[permissions]` table |
| Codex → Cursor / Gemini / Copilot | mcp | TOML-specific `bearer_token_env_var` → header rewrite back into `Authorization: Bearer ${VAR}` form |
| Cursor / Gemini → Codex | hooks | same drops as Claude → Codex (matcher on `UserPromptSubmit` / `Stop`, unsupported events) |
| Cursor → any non-Cursor | commands | wrapped as `cursor-command-<slug>` skill; slash invocation is lost |
| Gemini → any non-Gemini | commands | wrapped as `gemini-command-<slug>` skill; `{{args}}` template flagged for manual rewrite |
| Any tool with Claude-only markers in its instructions file → any non-Claude target | rules | symlink downgraded to copy with a `crossby:manual-fix` block; per-tool marker list lives in `instruction_markers.py` |
| Any source → any target | plugins | `.claude/plugins/`, `.claude/plugin-marketplaces.json`, and `.claude-plugin/marketplace.json` are reported as `Not Added`; migrate by hand |

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
