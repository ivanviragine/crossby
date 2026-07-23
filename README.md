# crossby

**One config. Every AI tool.**

Stop re-writing your rules, permissions, and agents for every CLI. `crossby` syncs each surface across the tools that support it — including Claude, Copilot, Codex, Cursor, OpenCode, VS Code, the Antigravity IDE, and Antigravity CLI — and lets you hand off a live session from one tool to another without losing context.

```
$ crossby sync --from codex

✓  rules         AGENTS.md         →  CLAUDE.md, .cursorrules, +1 more
✓  agents        .agents/          →  .claude/agents/, .cursor/agents/, +2 more
✓  skills        .agents/skills/   →  .claude/skills/, .cursor/skills/, +1 more
✓  permissions                     →  translated for Claude, Cursor
✓  hooks                           →  written for Claude, Cursor, Copilot
✓  mcp servers                     →  merged into Claude, Cursor, Codex, Copilot, Antigravity CLI
```

Already on Claude? `crossby sync --from claude` works the same way — any tool can be the source.

---

## Why crossby?

- **Every new tool inherits your setup.** Install a new AI CLI tomorrow and one `crossby sync` gives it your rules, agents, permissions, hooks, and MCP servers — translated into whatever format that tool expects.
- **Pick any tool as your source.** `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, Copilot instructions — whatever you already write in becomes canonical. No migration, no lock-in.
- **One set of launch flags for every tool.** `--model`, `--effort`, `--plan`, `--accept-edits`, `--auto`, `--yolo`, `--resume` — crossby handles the per-tool translation. The four-tier autonomy ladder (`plan < accept-edits < auto < yolo`) maps to each tool's native flag or degrades gracefully (see [Launch autonomy modes](#launch-autonomy-modes)).
- **Cross-tool session handoff.** Mid-session in one CLI and want to continue in another? `crossby handoff` summarizes the transcript and hands it off — so you never re-explain what you're doing.
- **Stateless by default.** Works without a config file: `crossby sync` reads directly from each tool's standard paths. Drop in a `.crossby.yml` only when you want saved profiles or defaults.

## Install

```bash
pip install crossby
# or
uv tool install crossby
# or
pipx install crossby
```

Requires Python 3.11+.

## Quick start

```bash
# Don't know where to start? Run crossby with no args for an interactive menu.
crossby

# Scaffold a .crossby.yml with your defaults (launch, sync, handoff).
crossby init

# Sync your setup to every installed tool (replace codex with claude, cursor, antigravity-cli, copilot…)
crossby sync --from codex

# Or launch the interactive wizard
crossby sync

# Launch a saved profile (e.g. Claude + Sonnet + high effort + YOLO — see below)
crossby launch ccyolo

# …or spell it out with unified flags
crossby launch --tool claude --model claude-sonnet-4.6 --effort high --yolo

# Inspect before writing — plan, doctor, validate
crossby sync --plan --from claude
crossby sync --doctor --from claude
crossby sync --validate-target

# Hand the current session off to another tool
crossby handoff --from claude --to codex

# Parse a session transcript for token usage
crossby stats /path/to/transcript.txt
```

Every command with missing arguments drops into a "Proceed / Change X" review so you can accept the resolved defaults with one keystroke or tweak any single value before it runs.

## What gets synced

| Config      | Strategy             | Notes                                                                                        |
| ----------- | -------------------- | -------------------------------------------------------------------------------------------- |
| Rules       | Symlink (auto-copy)  | `AGENTS.md` ↔ `CLAUDE.md` ↔ `.cursorrules` ↔ `.github/copilot-instructions.md` (`AGENTS.md` is shared by Codex and Antigravity CLI). Falls back to copy with a `<!-- crossby:manual-fix -->` block when the source mentions surfaces specific to a different tool (`/hooks`, `ExitPlanMode`, `permissionMode`, …). |
| Agents      | Symlink / translate  | Markdown-shape tools (Claude / Cursor / Copilot / Antigravity CLI) symlink directories. Codex translates per file into `.codex/agents/<name>.toml` with `permissionMode → sandbox_mode`, `model + effort` family-mapped to GPT, lossy fields preserved as a manual-fix block. |
| Skills      | Symlink / translate  | All five tools accept the same `SKILL.md` shape, so symlink is the default. `--strategy translate` rewrites per tool with manual-fix notes for Claude `allowed-tools` on non-Claude targets, and converts Claude slash commands (`.claude/commands/*.md`) into `claude-command-<slug>` skills for every other tool. |
| Permissions | Convert              | Canonical `cmd:args` ↔ `Bash()` / `Shell()` / `shell()` format per tool                       |
| Hooks       | Write                | Per-tool native hook schema; matcher widens on re-runs                                       |
| MCP servers | Merge                | Source tool's MCP config → each target's; `Authorization: Bearer ${VAR}`, `${VAR}` headers, and env-var self-references are rewritten into Codex `bearer_token_env_var` / `env_http_headers` / `env_vars` |
| Plugins     | Detect (manual)      | `.claude/plugins/`, `plugin-marketplaces.json`, and `.claude-plugin/marketplace.json` are reported as `Not Added`; bundled commands/agents/MCP servers must be migrated by hand |

Before writing anything, `crossby sync --plan` shows a stage-by-concern dry-run summary; `--doctor` adds a readiness rating (`high` / `medium` / `low`) plus the target-validation checks that would run after; `--validate-target` re-parses already-synced files (TOML / JSON parseability, agent required fields, skill frontmatter, `AGENTS.md` size threshold, MCP `command` on `PATH`). Use `--dry-run` to run a real sync in shadow mode.

After every real sync, the result table is also written to `.crossby/sync-report.md` — a portable `| Status | Item | Notes |` markdown table you can paste into a PR description. Pass `--no-persist-report` to skip, or `--report-format markdown-table` to render the same shape on stdout.

> Need to translate a single allowlist pattern by hand (e.g. while editing a config file)? `crossby convert "Bash(myapp:*)" --from claude --to cursor` prints the equivalent pattern for the target tool. To translate a single subagent file between tools (Claude / Cursor / Copilot / Codex), use `crossby agents convert --from claude --to codex .claude/agents/researcher.md`.

## Translate strategy and manual-fix blocks

Default strategy is `symlink` (with content-aware copy fallback for rules). Pass `--strategy translate` to do per-file rewriting that preserves intent across tools whose semantics diverge:

```bash
crossby sync --from claude --strategy translate
```

When a field doesn't have a faithful equivalent on the target — e.g. Claude `permissionMode: plan` going to Codex, or `allowed-tools` going to a tool that doesn't enforce them — the rendered file gets a clearly-marked block:

```markdown
<!-- crossby:manual-fix:start -->
## Manual migration required

- Claude-specific agent semantics carried over verbatim. The target tool does not enforce them — review and rewrite or remove as needed. Fields preserved: `permissionMode: plan`, `skills` preload list (`release-notes`).
<!-- crossby:manual-fix:end -->
```

Re-running `crossby sync` replaces the block in lockstep with the source — no stacking. Removing the block once you've addressed the note is supported; the next sync only re-emits if the source still triggers it.

## Cross-provider model translation

`crossby launch` translates model ids across families when the target tool wouldn't accept the source family natively:

```bash
# Pass a Claude model id to Codex — translated to gpt-5.4-mini under the hood
crossby launch --tool codex --model claude-sonnet-4.6 --effort high
# → codex --model gpt-5.4-mini -c model_reasoning_effort=xhigh
```

Sonnet shifts effort up one tier (low→medium, medium→high, high→xhigh) for coding-agent behavior. The reverse direction (`gpt-5.4` → Claude) picks the lowest source tier so users don't accidentally over-bill. A `UserWarning` fires whenever a translation happens; pass a native id to silence it.

## Launch autonomy modes

`crossby launch` exposes a four-tier **autonomy ladder** — how much the agent may do without asking. These are *permission* modes, not model selection:

```
--plan  <  --accept-edits  <  --auto  <  --yolo
read-only   auto-edit,        classifier-       skip all
            ask shell         guarded           prompts
```

- `--plan` — read-only planning; the agent proposes but doesn't act.
- `--accept-edits` — auto-approve file edits, still prompt for shell/commands. Broadly portable (5 of 6 CLIs support it at launch).
- `--auto` — Claude Code's classifier-mediated guarded autonomy (a separate model reviews each non-read action). **Claude-only** among the CLIs crossby drives; on other tools it **downgrades to that tool's accept-edits**, then to default prompting — never to `--yolo`.
- `--yolo` — skip all permission prompts.

**Precedence (most permissive wins):** `yolo > auto > accept-edits > plan`. If you pass several, the highest applies. A requested tier that a tool doesn't support downgrades to the next lower *autonomy* tier it does support (emitting a `UserWarning`), stopping at default prompting — it never escalates.

Per-tool mapping (verified against official docs, July 2026 — flags drift between versions, so crossby re-checks against `<tool> --help`):

| Tool            | `--accept-edits`                      | `--auto` (classifier)                     |
| --------------- | ------------------------------------- | ----------------------------------------- |
| Claude          | `--permission-mode acceptEdits`       | `--permission-mode auto`                  |
| Codex           | `-s workspace-write -a untrusted`     | ↓ downgrades to accept-edits              |
| Cursor CLI      | *(none — its default Agent mode already **is** accept-edits)* | ↓ downgrades to accept-edits |
| Copilot         | `--allow-tool write`                  | ↓ downgrades to accept-edits              |
| Antigravity CLI | `--mode accept-edits`                 | ↓ downgrades to accept-edits              |
| OpenCode        | ↓ default prompting (config-only)     | ↓ default prompting                       |
| VS Code, Antigravity IDE | ↓ default prompting (GUI)    | ↓ default prompting                       |

> Codex's old `--approval-mode auto-edit` was **removed** in the Rust CLI — crossby never emits it. Note Cursor CLI's default *is* accept-edits (the inverse of the Cursor IDE default), so `--accept-edits` is honored with no extra flag and no warning.

## Agent-readable runbook

`crossby init --install-skill` copies the bundled `crossby-sync` skill into every installed tool's skills directory. From inside Claude Code / Codex / Cursor / etc., the LLM can drive the full sync loop end-to-end — scan, plan, fix manual-fix blocks, validate — without leaving the session. The bundle is at `src/crossby/data/skill/`; the `references/differences.md` file in the bundle has the per-surface mapping table.

The bundle follows the [Agent Skills](https://agentskills.io) standard layout (`SKILL.md`, `agents/openai.yaml`, `references/`), so Codex users can also install it via the upstream `$skill-installer` skill without touching crossby:

```
$skill-installer install https://github.com/ivanviragine/crossby/tree/main/src/crossby/data/skill
```

That installs it globally under `$CODEX_HOME/skills/` instead of per-project. Use whichever model fits — `crossby init --install-skill` for a project-scoped install that travels with the repo, or `$skill-installer` for a one-time user-scoped install.

## Session handoff

```bash
# Hand off the latest session from the source tool
crossby handoff --from claude --to codex

# Or pick a specific session by id
crossby handoff --from claude --to codex --session-id 019cb497-ec14-7453-9224

# Write the handoff file but don't launch — review before switching tools
crossby handoff --from cursor --to copilot --no-launch

# Use the bundled Claude Code "compact" prompt instead of the default summary
crossby handoff --from claude --to codex --prompt-preset cc-compact

# Or supply your own summarization prompt
crossby handoff --from claude --to codex --prompt ./my-prompt.md
```

`crossby` reads the chosen session from the source tool, asks an LLM to summarize it into a structured handoff document, writes it to `.crossby/handoffs/HANDOFF-<timestamp>.md`, and launches the target with the file **path** (not its contents) as the initial prompt — so it fits under OS argv limits regardless of transcript size.

The default preset produces a structured six-section handoff (current task, key decisions, modified files, blockers, next steps, critical context). Pass `--prompt-preset cc-compact` to use Claude Code's partial-compaction prompt, or `--prompt <path>` to supply your own; both paths skip structured parsing and write the summarizer's output verbatim. The two flags are mutually exclusive.

Supported sources: Claude, Cursor, Codex, Copilot. Supported targets: all of the above plus Antigravity CLI, OpenCode, Antigravity IDE, VS Code.

## Optional: `.crossby.yml`

Run `crossby init` to scaffold this file interactively, or hand-author it to set defaults and save launch profiles:

```yaml
version: 1
ai:
  default_tool: claude
  default_model: claude-sonnet-4.6
  effort: medium

profiles:
  ccyolo:                         # → crossby launch ccyolo
    tool: claude
    model: claude-sonnet-4.6
    effort: high
    yolo: true
  quick:                          # → crossby launch quick
    tool: cursor
    model: haiku
    effort: low

sync_defaults:                    # fed into `crossby sync`
  from: claude
  to: cursor

handoff_defaults:                 # fed into `crossby handoff`
  from: claude
  to: codex
  prompt_preset: default
  token_budget: 32000
```

Profiles are just named bundles of `--tool` / `--model` / `--effort` / `--accept-edits` / `--auto` / `--yolo`. Run them by name (`crossby launch ccyolo`) or with `--profile ccyolo`. Explicit flags on the command line still override the profile. The autonomy fields (`accept_edits`, `auto`, `yolo`) also work under `ai:` as global or per-command defaults.

`sync_defaults` and `handoff_defaults` feed the interactive prompts for those commands — CLI flags still win, and you always get the "Proceed / Change X" review before anything runs. `crossby sync` does **not** require this file — it reads directly from each tool's standard paths.

## Supported tools

| Tool           | Sync | Launch | Handoff (source) | Handoff (target) |
| -------------- | ---- | ------ | ---------------- | ---------------- |
| Claude         | ✓    | ✓      | ✓                | ✓                |
| Copilot        | ✓    | ✓      | ✓                | ✓                |
| Codex          | ✓    | ✓      | ✓                | ✓                |
| Cursor         | ✓    | ✓      | ✓                | ✓                |
| Antigravity CLI| ✓    | ✓      | —                 | ✓                |
| OpenCode       | ✓    | ✓      | —                 | ✓                |
| VS Code        | ✓    | ✓      | —                 | ✓                |
| Antigravity IDE| ✓    | ✓      | —                 | ✓                |

The **Antigravity IDE** is a launch-only GUI tool. It reads the same project-level `.agents/` config as **Antigravity CLI** (`AGENTS.md`, `.agents/skills`, `.agents/agents`, `.agents/mcp_config.json`), so syncing to `antigravity-cli` provisions the IDE too — there is no separate IDE sync target.

Per-tool flag mappings and adapter details live in [CONTRIBUTING.md](CONTRIBUTING.md#tool-reference).

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) — architecture, how to add a new tool, per-tool flag reference, release process.

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and architecture.

## License

MIT
