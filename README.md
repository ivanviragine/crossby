# crossby

**Write your AI-tool setup once. Reuse it, focus it, and carry it between coding agents.**

`crossby` is a local interoperability layer for AI coding agents. You keep writing your rules, subagents, skills, permissions, hooks, and MCP servers in whatever format your main tool already uses — and crossby handles three jobs on top of that:

- **Reuse one configuration across compatible agents.** One `crossby sync` translates your setup into each tool's native format, so a new CLI inherits it instead of starting blank. Direct sync targets are **Claude, Cursor, Copilot, Codex, and Antigravity CLI**.
- **Focus an agent on a task-shaped set of capabilities.** A **scene** narrows the installed tools down to just the skills, agents, MCP servers, hooks, and permissions a task needs — persistently, or for a single launch.
- **Carry a live session to another agent.** `crossby handoff` summarizes your current transcript and continues it in another tool, so you never re-explain what you were doing.

```console
$ crossby sync --from claude

✓  rules         CLAUDE.md          →  AGENTS.md, .cursorrules, +1 more
✓  agents        .claude/agents/    →  .cursor/agents/, .codex/agents/, +2 more
✓  skills        .claude/skills/    →  .cursor/skills/, .agents/skills/, +2 more
✓  permissions                      →  translated for Cursor
✓  hooks                            →  written for Cursor, Codex, Copilot, Antigravity CLI
✓  mcp servers                      →  merged into Cursor, Codex, Copilot, Antigravity CLI
```

Any of the five direct-sync tools can be the source — `crossby sync --from cursor` works the same way. (No tool holds every surface, though: permissions live only in Claude's and Cursor's config, so a sync *from* Codex, Copilot, or Antigravity CLI has no permissions to read, and a sync *to* them writes none.) crossby is **stateless by default** — it reads directly from each tool's standard paths, so no config file is required.

## Which workflow do you want?

| I want to… | Use | Start with |
| --- | --- | --- |
| Give my other tools the same rules/agents/skills/MCP/hooks I already wrote | **Sync** | `crossby sync --plan --from claude` |
| Run a tool with only the capabilities one task needs | **Scenes** | `crossby scene list` / `crossby launch --scene <name>` |
| Continue my current session in a different tool | **Handoff** | `crossby handoff --from claude --to codex` |

Everything else in this README expands one of these three. Jump to [What crossby supports](#what-crossby-supports) for the exact per-tool coverage.

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

Lead with the read-only inspection commands to see what crossby would do before it writes: `sync --plan` / `--doctor`, `scene show`, `scene use --plan`. Commands that write or launch (handoff, `scene add` / `use`) note their side effects in the sections below.

```bash
# Don't know where to start? Run crossby with no args for an interactive menu (TTY only).
crossby
```

### 1. Reuse your setup — sync

```bash
# See exactly what a sync would write, without touching any file.
crossby sync --plan --from claude

# Add a readiness rating and the post-sync validation checks it would run.
crossby sync --doctor --from claude

# Happy with the plan? Run it for real. Any direct-sync tool can be the source
# (claude, cursor, copilot, codex, antigravity-cli).
crossby sync --from claude

# Prefer to be walked through it? Omit --from for the interactive wizard.
crossby sync
```

`--plan` and `--doctor` never write; `--dry-run` runs a real sync in shadow mode. See [Syncing configuration](#syncing-configuration) for what gets written and what stays additive.

### 2. Focus a session — scenes

```bash
# List the scenes defined in .crossby.yml, with per-concern counts.
crossby scene list

# No scenes yet? Drop in opinionated starters and tweak them.
crossby scene install-starters   # pr-review, deploy-watch, write-docs, presentation

# Preview what a scene resolves to per tool before applying it.
crossby scene show pr-review

# Apply it for one launch only — nothing tracked is touched, no cleanup needed.
crossby launch --scene pr-review --tool claude
```

Use `crossby scene use <name>` to persist a scene across tools, or `crossby launch --scene <name>` for a session-scoped one. See [Scenes](#scenes).

### 3. Continue elsewhere — handoff

```bash
# Write the handoff summary to .crossby/handoffs/ and stop — review it first.
crossby handoff --from cursor --to copilot --no-launch

# Or summarize the latest Claude session and continue it straight in Codex
# (this writes the handoff file and launches Codex with it pre-loaded).
crossby handoff --from claude --to codex
```

Handoff sources are the tools with readable transcripts (**Claude, Cursor, Codex, Copilot**). See [Session handoff](#session-handoff).

```bash
# A few more one-offs:
crossby launch ccyolo                       # launch a saved profile (see .crossby.yml)
crossby stats /path/to/transcript.txt       # parse a transcript for token usage
crossby convert "Bash(myapp:*)" --from claude --to cursor   # translate one allowlist pattern
crossby tools update                        # update your installed AI CLIs
```

Every command with missing arguments drops into a "Proceed / Change X" review, so you can accept the resolved defaults with one keystroke or tweak any single value before it runs.

## What crossby supports

crossby drives eight tools, but **not every tool does every job.** The tables below are the source of truth for every support claim in this README.

### Direct sync targets

A sync **writer** exists for five tools. These are the only tools crossby writes configuration *into*:

| Surface | Claude | Cursor | Copilot | Codex | Antigravity CLI |
| --- | :---: | :---: | :---: | :---: | :---: |
| Rules (`AGENTS.md` ↔ `CLAUDE.md` ↔ `.cursorrules` ↔ Copilot) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Agents (subagents) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Skills | ✓ | ✓ | ✓ | ✓ | ✓ |
| MCP servers | ✓ | ✓ | ✓ | ✓ | ✓ |
| Hooks | ✓ | ✓ | ✓ | ✓ | ✓ |
| Permissions | ✓ | ✓ | — | — | — |

A checkmark is **per surface, not per tool** — the exceptions are real:

- **Permissions sync only to Claude and Cursor** — the tools with a persistent per-project permission file that crossby writes. Copilot, Codex, and Antigravity CLI gate command permissions through launch-time flags or sandbox modes (`--allow-tool`, `--sandbox`, `--mode`) rather than a synced policy file, so they have no permission writer.
- **Plugins are detected, never written.** `.claude/plugins/`, `.claude/plugin-marketplaces.json`, and `.claude-plugin/marketplace.json` are reported as `Not Added`; their bundled commands/agents/MCP servers must be migrated by hand.

### Launch and handoff

Launching and handoff reach a wider set of tools than sync:

| Tool | Direct sync | Launch | Handoff source | Handoff target |
| --- | :---: | :---: | :---: | :---: |
| Claude | ✓ | ✓ | ✓ | ✓ auto |
| Cursor | ✓ | ✓ | ✓ | ✓ auto |
| Copilot | ✓ | ✓ | ✓ | ✓ auto |
| Codex | ✓ | ✓ | ✓ | ✓ auto |
| Antigravity CLI | ✓ | ✓ | — | ✓ auto |
| OpenCode | — | ✓ | — | ✓ auto |
| Antigravity IDE | via Antigravity CLI | ✓ (GUI) | — | manual |
| VS Code | — | ✓ (GUI) | — | manual |

- **OpenCode and VS Code are launch adapters, not sync targets.** crossby can launch them (and hand off *to* them), but it writes no rules/agents/skills/MCP/hooks/permissions into either.
- **The Antigravity IDE consumes the CLI's configuration transitively.** It reads the same project-level `.agents/` layout as **Antigravity CLI** (`AGENTS.md`, `.agents/skills`, `.agents/agents`, `.agents/mcp_config.json`), so syncing to `antigravity-cli` provisions the IDE too. There is no separate IDE sync target.
- **"✓ auto" vs "manual" handoff.** For CLI targets, crossby launches the tool with the handoff summary pre-loaded as the initial prompt. GUI tools (VS Code, Antigravity IDE) can't take an initial message, so crossby **writes the handoff file and prints its path for you to open by hand** — it does not launch them with the context loaded.

Per-tool flag mappings and adapter internals live in [CONTRIBUTING.md](CONTRIBUTING.md#tool-reference).

## Syncing configuration

### What gets synced

| Config      | Strategy             | Notes                                                                                        |
| ----------- | -------------------- | -------------------------------------------------------------------------------------------- |
| Rules       | Symlink (auto-copy)  | `AGENTS.md` ↔ `CLAUDE.md` ↔ `.cursorrules` ↔ `.github/copilot-instructions.md` (`AGENTS.md` is shared by Codex and Antigravity CLI). Falls back to copy with a `<!-- crossby:manual-fix -->` block when the source mentions surfaces specific to a different tool (`/hooks`, `ExitPlanMode`, `permissionMode`, …). |
| Agents      | Symlink / translate  | Markdown-shape tools (Claude / Cursor / Copilot / Antigravity CLI) symlink directories. Codex translates per file into `.codex/agents/<name>.toml` with `permissionMode → sandbox_mode`, `model + effort` family-mapped to GPT, lossy fields preserved as a manual-fix block. |
| Skills      | Symlink / translate  | All five tools accept the same `SKILL.md` shape, so symlink is the default. `--strategy translate` rewrites per tool with manual-fix notes for Claude `allowed-tools` on non-Claude targets, and converts Claude slash commands (`.claude/commands/*.md`) into `claude-command-<slug>` skills for every other tool. |
| Permissions | Convert              | Canonical `cmd:args` ↔ `Bash()` / `Shell()` per tool — **Claude and Cursor only** (see [above](#direct-sync-targets)). Revocable: a pattern crossby wrote is removed when the source drops it (see [Revocable sync](#revocable-sync)). |
| Hooks       | Write                | Per-tool native hook schema; a crossby-written hook's matcher narrows as well as widens on re-runs, and the hook is revoked when the source drops it. |
| MCP servers | Merge                | Source tool's MCP config → each target's; `Authorization: Bearer ${VAR}`, `${VAR}` headers, and env-var self-references are rewritten into Codex `bearer_token_env_var` / `env_http_headers` / `env_vars`. |
| Plugins     | Detect (manual)      | `.claude/plugins/`, `.claude/plugin-marketplaces.json`, and `.claude-plugin/marketplace.json` are reported as `Not Added`; bundled commands/agents/MCP servers must be migrated by hand. |

### Preview before you write

crossby is built to be inspected before it touches a file:

- `crossby sync --plan` shows a stage-by-concern dry-run summary and writes nothing.
- `--doctor` adds a readiness rating (`high` / `medium` / `low`) plus the target-validation checks that would run afterward.
- `--validate-target` re-parses already-synced files (TOML / JSON parseability, agent required fields, skill frontmatter, `AGENTS.md` size threshold, MCP `command` on `PATH`).
- `--dry-run` runs a real sync in shadow mode.

After every real sync, the result table is also written to `.crossby/sync-report.md` — a portable `| Status | Item | Notes |` markdown table you can paste into a PR. A row can be `Added`, `Removed` (a revocation), `Check before using` (a lossy translation), or `Not Added`. Pass `--no-persist-report` to skip it, or `--report-format markdown-table` to render the same shape on stdout.

### Translate strategy and manual-fix blocks

Default strategy is `symlink` (with content-aware copy fallback for rules). Pass `--strategy translate` to do per-file rewriting that preserves intent across tools whose semantics diverge:

```bash
crossby sync --from claude --strategy translate
```

When a field has no faithful equivalent on the target — e.g. Claude `permissionMode: plan` going to Codex, or `allowed-tools` going to a tool that doesn't enforce them — the rendered file gets a clearly-marked block:

```markdown
<!-- crossby:manual-fix:start -->
## Manual migration required

- Claude-specific agent semantics carried over verbatim. The target tool does not enforce them — review and rewrite or remove as needed. Fields preserved: `permissionMode: plan`, `skills` preload list (`release-notes`).
<!-- crossby:manual-fix:end -->
```

Re-running `crossby sync` replaces the block in lockstep with the source — no stacking. Removing the block once you've addressed the note is supported; the next sync only re-emits it if the source still triggers it.

> Need to translate a single allowlist pattern by hand? `crossby convert "Bash(myapp:*)" --from claude --to cursor` prints the equivalent pattern for the target tool. To translate a single subagent file (Claude / Cursor / Copilot / Codex), use `crossby agents convert --from claude --to codex .claude/agents/researcher.md`.

### Revocable sync

crossby never *removes* configuration you own — it only ever *revokes an entry it recorded writing*, tracked in a per-machine, gitignored `.crossby/owned.json` ledger. A hand-authored entry that merely shares a name with a source entry is never revoked, though a normal same-named merge can still overwrite its contents — this guarantee bounds *removal*, not additive or update writes.

Within that boundary, hooks and permissions are *additive by default but revocable*: syncing `--from A` then `--from B` leaves each target reflecting B, not the union of both — a hook or permission pattern crossby wrote for A is taken back once it's gone from the source. MCP servers are narrower: crossby removes a server only when it wrote that server *and* the source marks it disabled (`enabled: false`); a server merely dropped from the source is left in place, so for MCP the two syncs leave the union. A same-named MCP server you wrote by hand is never deleted, though a normal MCP merge can still overwrite its config — the ledger bounds *removal*, not additive or update writes. A fresh clone starts with an empty ledger, so it never revokes until it has recorded writes of its own.

## Scenes

A **scene** is a task-shaped slice of the project's skills, agents, MCP servers, hooks, and permissions. Activating one filters each installed tool down to just the selected capabilities, using the least-invasive mechanism per tool — a native disable key where one exists (Claude `skillOverrides`, MCP toggles), otherwise a re-pointed, filtered projection of the source directory.

```bash
# List the scenes defined in .crossby.yml, with per-concern counts
crossby scene list

# Show what a scene resolves to per tool, the mechanism each would use,
# and any selectors that matched nothing
crossby scene show pr-review

# Apply a scene to every installed tool
crossby scene use pr-review

# Preview without writing, or scope to one tool
crossby scene use pr-review --plan
crossby scene use pr-review --tool cursor

# Revert to the pre-scene baseline
crossby scene clear

# What's active, per-tool mechanism, and whether any managed file drifted
crossby scene status
```

Key behaviours:

- **Switching restores the true baseline.** `use B` while `A` is active reverts `A` first, then applies `B` from the original pre-`A` state — so a later `clear` restores your settings, not `A`'s (or `B`'s) output.
- **Reverting is ledger-driven.** `clear` only undoes what crossby wrote (tracked in `.crossby/owned.json`); a `skillOverrides`, `deny`, or MCP-`disabled` entry you authored by hand is left untouched. `clear` works even after the active scene is renamed or deleted from `.crossby.yml`. One exception: a scene that *narrows* hooks or permissions removes those crossby-synced entries through the revocable-sync channel, and `clear` does **not** put them back — `use` warns when this happens; re-run `crossby sync` to restore them.
- **A corrupt ledger fails closed.** If `.crossby/owned.json` can't be parsed as the expected structure — invalid JSON, a non-object root, a symlink, or a malformed scene-tracking section (the authority for a revert) — both `use` and `clear`, including `--plan`, refuse (exit 1) rather than revert from an empty view. (A valid ledger that simply owns nothing is fine, as is a genuinely absent one on a fresh per-machine clone.) Restore it from backup or revert the applied settings by hand; **never delete it** — a *missing* ledger reads as "crossby owns nothing," which re-opens the exact gap.
- **Drift is detected, not clobbered.** `status` compares a per-tool content hash captured at apply time against the current file (normalised, so a semantically-neutral reformat is not flagged). `use` and `clear` refuse to revert a scene whose managed files have drifted — pass `--force` to proceed anyway.
- **`--tool`** scopes `list` / `show` / `use` / `clear` / `status` (output for the read commands, effect for the write ones); `--plan` previews `use` / `clear` without writing.

Activation state is recorded in `.crossby/scene-state.json` (gitignored) — the active scene, when it was applied, the per-tool mechanism, and the drift hashes. It is bookkeeping for `status`; the authority for reverting is the ownership ledger.

### Authoring scenes

You don't have to hand-write `scenes:` YAML. `crossby scene create` walks a wizard over the skills, agents, MCP servers, hooks, and permissions it actually finds in the project, and `add` / `remove` edit a scene's selectors from the command line:

```bash
# Interactive wizard — multi-select each concern, then a review step
crossby scene create pr-review

# Or build the exact same scene non-interactively (required when stdin is not a
# TTY — the wizard refuses rather than silently selecting everything)
crossby scene create pr-review \
  --skill "review-*" --skill knowledge --agent code-reviewer \
  --mcp github --exclude-mcp linear \
  --description "Review a pull request" --extends base --profile ccyolo

# Append to / remove from an existing scene's selectors (idempotent)
crossby scene add pr-review --permission "gh pr *"
crossby scene remove pr-review --exclude-mcp linear

# Print the scene block to stdout instead of writing it
crossby scene create pr-review --skill "review-*" --print

# Delete a scene (refused while it is active — clear it first, or --force)
crossby scene delete pr-review

# Drop in opinionated presets and tweak them
crossby scene install-starters   # pr-review, deploy-watch, write-docs, presentation
```

Every selector flag has an `--exclude-*` counterpart (`--skill` / `--exclude-skill`, and the same for `--agent`, `--mcp`, `--hook`, `--permission`). Adding a pattern to one channel removes it from the other, so include and exclude can never contradict — the move is reported when it happens.

Writes are **surgical**: only the edited `scenes.<name>` entry is rewritten, located by parsing the YAML rather than line-scanning. Everything *outside* that entry's span — every comment and every other section, including sibling scenes — is preserved byte-for-byte. Each write is backed up, re-parsed, and rolled back if it would produce an invalid config. Starter scenes skip any same-named scene you already have and are idempotent on re-run; because they use glob selectors, unmatched selectors warn rather than error.

### Session-scoped scenes — `crossby launch --scene`

`crossby scene use` **persists** a scene into each tool's config files. When you instead want a scene to apply to **one launch only** — touching nothing tracked and needing no `clear` afterward — pass `--scene` to `crossby launch`. This session-scoped guarantee holds only for tools that expose a launch-time lever (Claude, Codex ≥ 0.134.0, Copilot); a tool without one doesn't get session isolation. crossby warns and does what it can: a CLI tool with no launch lever falls back to persistent `scene use` activation — needing a later `clear` for whatever that actually writes, though for a tool like OpenCode whose every concern is unsupported it writes nothing and the deselected capabilities stay enabled — and a GUI tool just launches without the scene. Narrowing can therefore be partial — see the per-tool table below.

```bash
# Launch Claude with the pr-review scene for this session only.
# Emits --mcp-config <file> --strict-mcp-config and a --settings file, and
# writes nothing into .claude/ or .mcp.json.
crossby launch --scene pr-review --tool claude

# A scene may name a default profile; --scene alone picks it up.
crossby launch --scene pr-review

# Precedence is explicit flags > scene > profile > ai: defaults, so an explicit
# --profile or --model still wins over the scene's profile.
crossby launch --scene pr-review --tool codex --model gpt-5.2
```

`--scene` targets exactly **one** tool (resolved from `--tool`, the scene's `profile:`, or `ai.default_tool`); it does not fan out — that's what `crossby scene use` is for. Rendered artefacts live under `.crossby/scene/<name>/launch/`, written atomically and kept out of git via `.git/info/exclude`. **One exception:** Codex's `--profile` reads only from `$CODEX_HOME` (usually `~/.codex`, shared across projects), so its generated profile is written there as `crossby-<project-slug>-<scene>.config.toml` — namespaced by a project-root hash and carrying a generated-by header, so pruning stale profiles never touches a hand-written one.

**Not every tool has a session-scoped lever.** Where a tool can't scope a scene (or a specific concern) at launch, crossby warns rather than applying nothing silently — but the outcome varies: a CLI tool without a launch lever falls back to persistent activation, a concern with no lever at all can be left wide open, and a GUI tool just launches without the scene:

| Tool | Session-scoped lever |
| --- | --- |
| Claude | `--mcp-config <file> --strict-mcp-config`, a `--settings` file of `skillOverrides` (needs `claude ≥ 2.1.129`), and `--disallowedTools "Agent(<name>)"` per deselected agent |
| Codex | `--profile <name>` layering a generated `$CODEX_HOME/<name>.config.toml` (needs `codex ≥ 0.134.0`) |
| Copilot | `--disable-mcp-server <name>` per deselected server; a profile's `--allow-tool` entries naming an excluded tool are dropped |
| Cursor | none — falls back to persistent activation (its only knob relocates the whole config base including auth) |
| OpenCode | none — persistent-activation fallback writes nothing (no sync writer for any concern), so deselected servers stay enabled and there's nothing to `clear` |
| Antigravity CLI | none — falls back to persistent activation, warning that config was written |
| VS Code / Antigravity IDE | none (GUI) — warns that the scene cannot apply, and launches without it |

If a tool has a session lever but not for a concern the scene narrows (e.g. Codex scopes MCP but not agents), crossby warns and applies what it can.

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

# Or supply your own summarization prompt (mutually exclusive with --prompt-preset)
crossby handoff --from claude --to codex --prompt ./my-prompt.md
```

crossby reads the chosen session from the source tool, asks an LLM to summarize it into a structured handoff document, writes it to `.crossby/handoffs/HANDOFF-<timestamp>.md`, and — for a CLI target — launches the tool with the file **path** (not its contents) as the initial prompt, so it fits under OS argv limits regardless of transcript size.

The default preset produces a structured six-section handoff (current task, key decisions, modified files, blockers, next steps, critical context). Pass `--prompt-preset cc-compact` to use Claude Code's partial-compaction prompt, or `--prompt <path>` to supply your own; both paths skip structured parsing and write the summarizer's output verbatim. The two flags are mutually exclusive.

**Sources** are the tools whose transcripts crossby can read: **Claude, Cursor, Codex, Copilot.** **Targets** are every supported tool — but the two GUI tools are a **manual continuation path, not an automatic launch**:

- **Claude, Cursor, Codex, Copilot, OpenCode, Antigravity CLI** are launched with the handoff pre-loaded.
- **VS Code and the Antigravity IDE** can't receive an initial message, so crossby writes the handoff file and prints its path for you to open by hand.

## Launch options

`crossby launch` runs any supported tool with one unified set of flags — crossby translates each into the target's native syntax (or degrades gracefully when the tool lacks it).

### Autonomy modes

`crossby launch` exposes a four-tier **autonomy ladder** — how much the agent may do without asking. These are *permission* modes, not model selection:

```text
--plan  <  --accept-edits  <  --auto  <  --yolo
read-only   auto-edit,        classifier-       skip all
            ask shell         guarded           prompts
```

- `--plan` — read-only planning; the agent proposes but doesn't act.
- `--accept-edits` — auto-approve file edits, still prompt for shell/commands. Broadly portable (5 of the 6 CLIs support it at launch; OpenCode falls back to default prompting).
- `--auto` — Claude Code's classifier-mediated guarded autonomy (a separate model reviews each non-read action). **Claude-only** among the CLIs crossby drives; on other tools it **downgrades to that tool's accept-edits**, then to default prompting — never to `--yolo`.
- `--yolo` — skip all permission prompts.

**Precedence (most permissive wins):** `yolo > auto > accept-edits > plan`. If you pass several, the highest applies. A requested tier a tool doesn't support downgrades to the next lower *autonomy* tier it does support (emitting a `UserWarning`), stopping at default prompting — it never escalates.

Per-tool mapping (verified against official docs, July 2026; CLI flags can drift between versions, so treat the table as a point-in-time snapshot):

| Tool            | `--accept-edits`                      | `--auto` (classifier)                     |
| --------------- | ------------------------------------- | ----------------------------------------- |
| Claude          | `--permission-mode acceptEdits`       | `--permission-mode auto`                  |
| Codex           | `--sandbox workspace-write -a untrusted` | ↓ downgrades to accept-edits           |
| Cursor CLI      | *(none — its default Agent mode already **is** accept-edits)* | ↓ downgrades to accept-edits |
| Copilot         | `--allow-tool write`                  | ↓ downgrades to accept-edits              |
| Antigravity CLI | `--mode accept-edits`                 | ↓ downgrades to accept-edits              |
| OpenCode        | ↓ default prompting (config-only)     | ↓ default prompting                       |
| VS Code, Antigravity IDE | ↓ default prompting (GUI)    | ↓ default prompting                       |

> Codex's old `--approval-mode auto-edit` was **removed** in the Rust CLI — crossby never emits it. Note Cursor CLI's default *is* accept-edits (the inverse of the Cursor IDE default), so `--accept-edits` is honored with no extra flag and no warning.

### Cross-provider model translation

`crossby launch` translates model ids across families when the target tool wouldn't accept the source family natively:

```bash
# Pass a Claude model id to Codex — translated to gpt-5.4-mini under the hood
crossby launch --tool codex --model claude-sonnet-4.6 --effort high
# → codex --model gpt-5.4-mini -c model_reasoning_effort=xhigh
```

Sonnet shifts effort up one tier (low→medium, medium→high, high→xhigh) for coding-agent behavior. The reverse direction (`gpt-5.4` → Claude) picks the lowest source tier so users don't accidentally over-bill. A `UserWarning` fires whenever a translation happens; pass a native id to silence it.

### Codex sandbox: linked worktrees & `--network`

Codex confines writes with an OS sandbox (`--sandbox workspace-write` — Seatbelt on macOS, Landlock on Linux). crossby **keeps that sandbox on every path** — it never emits `--yolo` / `--dangerously-bypass-approvals-and-sandbox`; Codex's yolo is approval-skipping only (`-a never`), and approval `never` appears **only** when you actually request `--yolo`.

- **Linked worktrees & submodules just work.** In a linked worktree the working tree's `.git` is a *file* pointing at metadata that lives **outside** the working directory, which the sandbox would otherwise block. crossby detects this and grants only the real git-metadata dirs outside the root to the sandbox with `--add-dir` — which *adds* to the writable roots, preserving any `sandbox_workspace_write.writable_roots` you configured — so sandboxed git operations succeed while the sandbox stays on. A normal checkout grants nothing. This applies to launch, `--resume` (approval-neutral: no `-a` injected), and the headless handoff summarizer.
- **`--network` (Codex only).** `crossby launch --network` allows network access inside the sandbox (package installs, remote fetch/push). It is **security-sensitive** and off by default. On tools without a sandbox network opt-in it is **warned and ignored** on every path (launch, resume, GUI).
- **Explicit network pin.** Whenever crossby forces `workspace-write` (a worktree, `--network`, `--accept-edits`, or `--trusted-dir`), it also emits an explicit `-c sandbox_workspace_write.network_access=<true|false>` (`true` only with `--network`) so an ambient `network_access = true` in your Codex config can never silently enable networking in a crossby-managed sandbox. A plain, unmanaged launch emits no sandbox flag and stays byte-identical.

## Update installed tools

Keep your AI CLIs current without remembering each tool's own updater (`claude update`, `codex update`, `agent update`, `agy update`, `copilot update`, `opencode upgrade`, …):

```bash
# Pick which installed tools to update (default all), then run each updater
crossby tools update

# Update specific tools only
crossby tools update --tool claude --tool codex

# See the resolved command per tool without running anything
crossby tools update --dry-run

# Skip crossby's confirmation prompt (e.g. in a script)
crossby tools update --yes
```

crossby lists the installed, updatable tools, runs each tool's own updater sequentially — continuing past any failure — and prints a report of `Tool · Version (before → after) · ✓/✗`.

**v1 limitations.** Each tool declares one **static** update command; there is no detection of the install method (npm / brew / standalone). A tool that updates a *different* installation than the one on `PATH` can report success without changing the active version (surfaced as a "version did not change" warning). GUI tools (the Antigravity IDE, VS Code) self-update through their IDE and are never offered. This updates the *managed AI tools*, not the crossby CLI itself.

## Optional: `.crossby.yml`

crossby is stateless by default — `crossby sync` reads directly from each tool's standard paths and needs no config file. Add a `.crossby.yml` only when you want saved profiles, per-tier model defaults, scenes, or command defaults. Run `crossby init` to scaffold it interactively, or hand-author it:

```yaml
version: 1
ai:
  default_tool: claude
  default_model: claude-sonnet-4.6
  effort: medium

models:                           # per-tool, per-complexity-tier overrides
  claude:
    easy: claude-haiku-4.5        # `crossby launch --complexity easy`
    complex: claude-sonnet-4.6    # `crossby launch --complexity complex`
    complex_effort: high          # …and raise effort to `high` for that tier
    very_complex_effort: xhigh
  codex:
    complex: gpt-5.4
    complex_effort: xhigh

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

scenes:                           # task-shaped bundles of capabilities
  base:
    skills:
      exclude: [deploy-*]
  pr-review:
    description: Review a pull request
    extends: base                 # single-parent composition
    profile: ccyolo               # default launch profile for this scene
    skills:
      include: [review-*, knowledge]
    agents:
      include: [code-reviewer]
    mcp:
      include: [github]
    hooks:
      include: ["pre_tool_use:*"]
    permissions:
      include: ["git diff:*", "gh pr *"]

sync_defaults:                    # fed into `crossby sync`
  from: claude
  to: cursor

handoff_defaults:                 # fed into `crossby handoff`
  from: claude
  to: codex
  prompt_preset: default
  token_budget: 32000
```

Profiles are named bundles of `--tool` / `--model` / `--effort` / `--accept-edits` / `--auto` / `--yolo`. Run them by name (`crossby launch ccyolo`) or with `--profile ccyolo`. Explicit flags on the command line still override the profile.

The `models:` section maps a tool + complexity tier to a model id. Each tier (`easy` / `medium` / `complex` / `very_complex`) also takes an optional `<tier>_effort` override. Effort resolution order is `--effort` flag → `CROSSBY_EFFORT` env → per-command `ai.<command>.effort` → per-tier `<tier>_effort` → global `ai.effort`. Values must be one of `low` / `medium` / `high` / `xhigh` / `max`.

`sync_defaults` and `handoff_defaults` feed the interactive prompts for those commands — CLI flags still win, and you always get the "Proceed / Change X" review before anything runs.

## Agent-readable runbook

`crossby init --install-skill` copies the bundled `crossby-sync` skill into every installed tool's skills directory. From inside Claude Code / Codex / Cursor / etc., the LLM can drive the full sync loop end-to-end — scan, plan, fix manual-fix blocks, validate — without leaving the session. The bundle is at `src/crossby/data/skill/`; its `references/differences.md` file has the per-surface mapping table.

The bundle follows the [Agent Skills](https://agentskills.io) standard layout (`SKILL.md`, `agents/openai.yaml`, `references/`), so Codex users can also install it via the upstream `$skill-installer` skill:

```shell
$skill-installer install https://github.com/ivanviragine/crossby/tree/main/src/crossby/data/skill
```

That installs it globally under `$CODEX_HOME/skills/` instead of per-project. Use whichever fits — `crossby init --install-skill` for a project-scoped install that travels with the repo, or `$skill-installer` for a one-time user-scoped install.

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) — architecture, how to add a new tool, per-tool flag reference, release process.

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and architecture.

## License

MIT
