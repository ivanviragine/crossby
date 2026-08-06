# crossby cross-tool differences

Per-surface mapping table for the eight tools `crossby` supports today:
Claude, Cursor, Codex, Copilot, OpenCode, VS Code, Antigravity IDE, and
Antigravity CLI. Direct 1:1 mappings (e.g. `Bash(myapp:*)` ↔
`Shell(myapp:*)` ↔ `myapp:*`) are listed once; lossy or unsupported
edges are flagged.

Docs last checked: 2026-05-15. If today's date is later, re-open each
tool's docs and confirm the schemas before trusting these rows.

## Rules / instructions

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| `CLAUDE.md` / `AGENTS.md` / `.cursorrules` / `.github/copilot-instructions.md` | every other tool's instruction file (`AGENTS.md` is shared by Codex and Antigravity CLI) | symlink (default) | All tools accept the same plain-markdown body; symlink keeps every target in lockstep with the source. |
| Source content with Claude-only markers (`/hooks`, `.claude/agents/`, `Subagent`, `permissionMode`, `ExitPlanMode`, `TodoWrite`) | every non-Claude target | force-copy with `<!-- crossby:manual-fix -->` | Crossby refuses to symlink so target-tool semantics aren't silently overridden. Edit the copy and remove the manual-fix block when done. |
| Source content with Codex-only markers (`.codex/`, `sandbox_mode`, `developer_instructions`) | every non-Codex target | force-copy with manual-fix | Same idea, reversed direction. |

## Agents

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| `.claude/agents/<name>.md` (markdown + YAML frontmatter) | `.cursor/agents/`, `.agents/agents/` (Antigravity CLI), `.github/agents/` | directory symlink | All four markdown-shape tools accept the same on-disk format. |
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

## MCP and config

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| `.mcp.json` (project scope), `.claude/settings.json`, and — only with `--include-user-scope` — `~/.claude.json` (user scope) `mcpServers` | `.mcp.json` (Claude, plus a narrow `enabledMcpjsonServers` approval in `.claude/settings.json`), `.cursor/mcp.json`, `.agents/mcp_config.json` (Antigravity CLI), `.vscode/mcp.json`, `.codex/config.toml` | non-destructive merge | Existing entries the user added by hand are preserved. Claude reads project servers from `.mcp.json` (not `settings.json`), and remote entries are written with `type`, not `transport`. User scope is off by default so personal `~/.claude.json` servers don't leak into committed project files. A name defined in both `.mcp.json` and `.claude/settings.json` is reported as a duplicate to clean up. |
| `headers: {Authorization: "Bearer ${TOKEN}"}` | Codex `bearer_token_env_var = "TOKEN"` | regex rewrite | Only the `Bearer ${VAR}` shape is rewritten; `${VAR:-default}` fallbacks are dropped and the key is reported. |
| `headers: {X-Foo: "${VAR}"}` | Codex `env_http_headers = {X-Foo = "VAR"}` | regex rewrite | Static headers stay in `http_headers`. |
| `env: {KEY: "${KEY}"}` (self-reference) | Codex `env_vars = ["KEY"]` | regex rewrite | Other `${VAR}` env values stay literal so source tools can interpolate them. |
| `enabled: false` / `disabled: true` | dropped from the target | merge | Disabled servers in the source are removed from every target. |
| `oauth: {...}` (`callbackPort`, `clientId`, `authServerMetadataUrl`, ...) | not written to any target | report-only | No writer ports OAuth config across tools; a `manual-fix` row is reported per server instead of silently dropping it (`Not Added` in the sync report). Configure OAuth manually for each target. |
| Remote (http/sse) server `url` | Antigravity CLI `serverUrl` | key rename | Antigravity CLI's `.agents/mcp_config.json` uses `serverUrl` instead of `url`/`httpUrl` for remote servers; stdio servers are unaffected. |

## Permissions

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| `.claude/settings.json` `permissions.allow` (`Bash(cmd:*)`) | canonical `cmd:*` | reverse parse | Only `Bash(...)` entries are read; non-Bash patterns are skipped. |
| Canonical `cmd:*` | `Bash(cmd:*)` (Claude), `Shell(cmd:*)` (Cursor) | per-tool translator | Symmetrical: every supported tool can be source or target. Antigravity CLI has no persistent allowlist file — permissions are launch-time flags (`--dangerously-skip-permissions`/`--sandbox`), so it's outside this table (same as Codex's sandbox mode). |
| Cursor-specific shape (`Shell(cmd:*)`) | canonical | reverse parse | Allows any tool to seed the canonical pattern set. |

## Hooks

| Source | Target | Strategy | Caveat |
| --- | --- | --- | --- |
| Canonical events `pre_tool_use`, `post_tool_use`, `session_start`, `user_prompt_submit`, `stop`, `notification` | per-tool event names (PascalCase `PreToolUse` / camelCase `preToolUse` / etc.) | event-name translation | Each writer also drops events its tool can't represent and records a `manual_fix` note in the report row. |
| Source hook with `tools` filter | Cursor honours a scope only on `preToolUse`/`postToolUse`, and as a single `matcher` **regex string** (its schema has no `tools` array); Cursor `stop`, Codex `Stop` / `UserPromptSubmit` ignore matcher | partial mapping | The scope is stripped on write where unsupported. The manual-fix note is named after the key each tool actually uses: `hooks.<event>.tools` for Cursor, `hooks.<event>.matcher` for Codex and Antigravity CLI, and a single un-suffixed `hooks.tools` for Copilot, whose scope crossby drops on every event alike. |
| Source hook of unsupported event for the target (e.g. Claude `Notification` → Codex) | dropped from the target | manual-fix | Each unique unsupported event produces one `hooks.<event>` manual-fix note so the user knows what didn't make it across. |
| Any hook written to `.codex/hooks.json` | `[features].hooks = true` plus the deprecated `codex_hooks` alias written to `.codex/config.toml` | automatic, defensive | The hooks feature is stable and **on by default** since Codex 0.146.0, so the flag is no longer required; `CodexHooksWriter` still writes both keys so a project pinned to an older Codex isn't left with inert hooks. A manual-fix note appears only if that file can't be written. |
| `pre_tool_use` hook scoped to a shell tool → Cursor | a `preToolUse` entry *and* an unscoped `beforeShellExecution` entry | fan-out | Cursor is the only tool with a dedicated shell event. Registering both means callers register once and get shell coverage everywhere; both fire for one shell call, which is safe because a guard decision is idempotent. `beforeShellExecution` matches the command string, not a tool name, so it is written without a matcher. |
| Any hook written to `.agents/hooks.json` (Antigravity CLI) | container-wrapped `{name: {Event: [...]}}`; `Stop` handlers sit directly under the event key, `PreToolUse`/`PostToolUse` wrap them in `{matcher, hooks}` | shape split | agy nests every hook under an arbitrary container name and drops the matcher on `Stop`; the runtime emits agy's `{"decision": …}` dialect (a `Stop` blocks via `{"decision": "continue"}`, the inverse of the other tools' `continue` boolean). |
| Multiple sources defining the same `(event, command)` | matcher widened (union of tools) | merge | Re-running with a broader matcher upgrades coverage instead of duplicating entries. |

Per-tool supported events:

| Tool | Supported canonical events | Honours `matcher` / `tools` on |
| --- | --- | --- |
| Claude | `pre_tool_use`, `post_tool_use`, `session_start`, `user_prompt_submit`, `stop`, `notification` | every event |
| Codex | `pre_tool_use`, `post_tool_use`, `session_start`, `user_prompt_submit`, `stop` | `pre_tool_use`, `post_tool_use`, `session_start` only |
| Cursor | `pre_tool_use`, `post_tool_use`, `session_start`, `user_prompt_submit`, `stop` | `pre_tool_use`, `post_tool_use` only (as a `matcher` regex) |
| Copilot | `pre_tool_use`, `post_tool_use`, `session_start`, `stop` | none — crossby writes Copilot hooks unscoped (Copilot does support a `matcher` regex; wiring it up is tracked separately) |
| Antigravity CLI | `pre_tool_use`, `post_tool_use`, `stop` | `pre_tool_use`, `post_tool_use` only |

Antigravity CLI (`agy`) has no `session_start` / `user_prompt_submit` equivalent
(its `PreInvocation` / `PostInvocation` fire per model call, not once at session
start), so those events are dropped with a manual-fix note. Its own bundled
plugin registers no `PreToolUse` hook — so `PreToolUse` guards are best-effort
there and `Stop` is the reliable enforcement surface. agy's `Stop` stdin carries
no `stop_hook_active`-style re-entry flag, so a single-shot `Stop` guard must
track its own "already nudged" state rather than rely on the payload.

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
| Every MCP server `command` is on `PATH` across `.codex/config.toml`, `.mcp.json`, `.claude/settings.json`, `.cursor/mcp.json`, `.vscode/mcp.json`, `.agents/mcp_config.json` (and `~/.claude.json` under `--include-user-scope`) | warning | Missing binary on the host; users see this before the first invocation fails. Env-var-templated commands like `${HOME}/bin/foo` are expanded via `os.path.expandvars` before the lookup. |
| `AGENTS.md` / `CLAUDE.md` / `.cursorrules` / `.github/copilot-instructions.md` ≤ 32KB | warning | Instructions creeping past the size threshold beyond which review becomes painful. |
| Tool-specific JSON files parse as JSON | error | `.claude/settings.json`, `.cursor/cli.json`, `.cursor/mcp.json`, `.vscode/mcp.json`, `.agents/mcp_config.json`. |

## Directional caveats

The lossy edges Crossby actually emits today, grouped by source → target. A
missing row means the source ↔ target pair is non-lossy for that concern;
absent pairs (OpenCode / VS Code / Antigravity IDE as direct source/target for
the concerns below) aren't wired yet.

| Source → Target | Concern | What gets dropped or rewritten |
| --- | --- | --- |
| Claude → Codex | agents | `permissionMode: plan` / `dontAsk` / `bypassPermissions`; `tools` / `disallowedTools` / `skills` (become prompt guidance only) |
| Claude → Codex | hooks | `Notification`; `matcher` on `UserPromptSubmit` / `Stop`; non-`command` hook types (`prompt`, `agent`, `http`, `async`) |
| Claude → Codex | mcp | `headers` with `${VAR:-default}` fallbacks; `oauth` (whole block — reported as a manual-fix row, not written to any target); `type: sse` |
| Claude → Cursor | hooks | `notification` (no Cursor equivalent); `tools` filter on `session_start` / `user_prompt_submit` / `stop` |
| Claude → Copilot | hooks | `user_prompt_submit` and `notification`; the `tools` filter (crossby writes Copilot hooks unscoped — see the per-tool table above) |
| Claude → Antigravity CLI | hooks | `session_start` / `user_prompt_submit` / `notification` (no agy equivalent); `matcher` on `Stop`; non-`command` hook types |
| Codex → Claude | agents | `model_reasoning_effort` (Claude has no equivalent); `[permissions]` table |
| Codex → Cursor / Copilot / Antigravity CLI | mcp | TOML-specific `bearer_token_env_var` → header rewrite back into `Authorization: Bearer ${VAR}` form |
| Cursor → Codex | hooks | same drops as Claude → Codex (matcher on `UserPromptSubmit` / `Stop`, unsupported events) |
| Cursor → any non-Cursor | commands | wrapped as `cursor-command-<slug>` skill; slash invocation is lost |
| Any tool with Claude-only markers in its instructions file → any non-Claude target | rules | symlink downgraded to copy with a `crossby:manual-fix` block; per-tool marker list lives in `instruction_markers.py` |
| Any source → any target | plugins | `.claude/plugins/`, `.claude/plugin-marketplaces.json`, and `.claude-plugin/marketplace.json` are reported as `Not Added`; migrate by hand |

## Scene activation mechanisms

A scene narrows what each tool sees per concern. Because the levers are uneven
across tools, crossby picks the least-invasive mechanism per `(tool, concern)`
cell (`scenes/mechanism.py`):

- **DECLARE** — write the tool's own disable key, leaving the user's real files
  intact. Only these surfaces exist today:

  | Concern | Tool | Key written |
  | --- | --- | --- |
  | skills | Claude | `skillOverrides: {"<name>": "off"}` (needs `claude >= 2.1.129`; never affects plugin skills) |
  | agents | Claude | `permissions.deny: ["Agent(<name>)"]` |
  | mcp | Claude | `disabledMcpjsonServers: ["<name>"]` |
  | mcp | Codex | `mcp_servers.<id>.enabled = false` (silently ignored on an untrusted project) |
  | mcp | Antigravity CLI | `mcpServers.<name>.disabled = true` |

- **PROJECT** — for skills/agents, re-point the tool's directory at a filtered
  symlink tree under `.crossby/scene/active/`; for hooks/permissions, filter the
  list and revoke the deselected remainder through the ownership ledger. Used
  wherever no DECLARE key exists, and made **authoritative** whenever tools share
  a resolved path and any of them lacks a lever (Codex + Antigravity CLI both use
  `.agents/skills`, so PROJECT wins there).

- **UNSUPPORTED** — no per-item lever at all: Cursor and Copilot MCP (deleting a
  server would be destructive), and permissions on Codex / Copilot / Antigravity
  CLI (their autonomy is a launch flag, not a per-project policy file). Reported,
  never faked.

Hooks have **no** per-item disable lever on any tool crossby supports (Claude and
Copilot are all-or-nothing via `disableAllHooks`; Cursor and Antigravity have no
toggle), so a scene can only narrow hooks by removing crossby-owned entries.

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
