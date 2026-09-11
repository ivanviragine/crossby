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

- **Permissions sync only to Claude and Cursor** — the tools with a persistent per-project permission file that crossby writes. Copilot, Codex, and Antigravity CLI gate command permissions through launch-time flags or sandbox modes (`--allow-tool`, `--dangerously-skip-permissions`) rather than a synced policy file, so they have no permission writer. (Antigravity CLI's `--sandbox` is a terminal-restriction flag, not a permission grant, and crossby does not emit it.)
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

- **OpenCode and VS Code are launch adapters, not sync targets.** crossby can launch them (and hand off *to* them), but neither has its own sync writer, so a sync never targets them directly. (One shared-file caveat: Copilot's MCP config lives at `.vscode/mcp.json`, so a sync *to Copilot* does write into that workspace file — as the Copilot target, not a VS Code one.)
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

- **Switching restores the true baseline.** `use B` while `A` is active reverts `A` first, then applies `B` from the original pre-`A` state — so a later `clear` restores your settings, not `A`'s (or `B`'s) output. For each physical skills/agents target, `.crossby/owned.json` records whether it was absent, its exact literal symlink target, the exact Crossby-allocated backup holding a displaced real directory, or a canonical source that a scene deliberately left untouched. Shared targets such as `.agents/skills` have one baseline record. Clear and switch use that record directly; they never rediscover an original source or choose a backup by scanning `.bak*` names, so unrelated user backups stay untouched.
- **Reverting is ledger-driven.** `clear` only undoes what crossby wrote (tracked in `.crossby/owned.json`); a `skillOverrides`, `deny`, or MCP-`disabled` entry you authored by hand is left untouched. Path-baseline records survive partial apply/clear failures and are removed only after the exact path is restored. `clear` works even after the active scene is renamed or deleted from `.crossby.yml`. One exception: a scene that *narrows* hooks or permissions removes those crossby-synced entries through the revocable-sync channel, and `clear` does **not** put them back — `use` warns when this happens, and `clear` repeats the warning before discarding its recovery state; re-run `crossby sync` to restore them.
- **Missing or corrupt recovery authority fails closed.** If `.crossby/owned.json` can't be parsed as the expected structure — invalid JSON, a non-object root, a symlink, malformed scene tracking, an unsafe target/backup path, or an unknown descriptor — both `use` and `clear`, including `--plan`, refuse (exit 1) rather than revert from an empty view. Ordinary `sync` leaves the ledger bytes untouched and refuses hooks, permissions, and MCP writes before they can create ownership that cannot be persisted. An active PROJECT scene created by an older Crossby without a path-baseline record also refuses clear/switch with manual-recovery guidance; Crossby will not infer a source or claim a neighboring backup. (A valid ledger that simply owns nothing is fine, as is a genuinely absent one on a fresh per-machine clone.) Restore the ledger from backup or restore the applied paths/settings by hand; **never delete it** — a *missing* ledger reads as "crossby owns nothing," which re-opens the exact gap.
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

`crossby scene use` **persists** a scene into each tool's config files. When you instead want a scene to apply to **one launch only**, pass `--scene` to `crossby launch`. Tools with a launch-time lever (Claude, Codex ≥ 0.134.0, Copilot) use untracked launch artefacts and need no later `clear`. A terminal tool without one falls back to persistent activation; crossby warns, records successful or partial activation in `scene status`, and tells you to run `scene clear` afterward. A concern with no persistent mechanism remains a true no-op (and crossby warns that it was not narrowed), while a GUI tool launches without the scene. Narrowing can therefore be partial — see the per-tool table below.

Persistent launch fallback follows the same lifecycle as `scene use`: a shared skills directory expands only the recorded skills scope, active-scene switches cannot strand another tool, and corrupt ownership provenance fails closed before mutation. A scoped reapply checks every recorded path for the launch tool plus the shared skills path for a skills-only co-sharer; drift in an unrelated co-sharer concern is left untouched and does not block the launch. Launch has no `--force` escape hatch, so an applicable drift or another failed precondition aborts before the child process starts. Error rows are recorded as `status: partial` and launch continues with a prominent recovery warning; an apply exception or inability to record recoverable state aborts launch instead. If writing the state record fails after mutation, crossby reverts reversible changes, removes stale state for the rolled-back scope, and retains recovery state for any untouched active tools rather than start a child with untracked restrictions. Hook or permission removals cannot be restored by `scene clear`; clear repeats the `crossby sync` guidance before discarding their recovery record. An apply exception records completed removals and directs you to clear the scene, then run `crossby sync`. A state-write failure after such a removal directs you to run `crossby sync`, then fix the state path.

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

`--scene` selects exactly **one** launch tool (resolved from `--tool`, the scene's `profile:`, or `ai.default_tool`). A persistent fallback may additionally record another installed tool that shares the same physical capability directory (currently Codex and Antigravity CLI share `.agents/skills`), because re-pointing that directory necessarily affects both. Rendered session artefacts live under `.crossby/scene/<name>/launch/`, written atomically and kept out of git via `.git/info/exclude`. **One exception:** Codex's `--profile` reads only from `$CODEX_HOME` (usually `~/.codex`, shared across projects), so its generated profile is written there as `crossby-<project-slug>-<scene>.config.toml` — namespaced by a project-root hash and carrying a generated-by header, so pruning stale profiles never touches a hand-written one. If that exact path contains a hand-written profile, crossby preserves it byte-for-byte and routes the launch through the recoverable persistent fallback instead.

**Not every tool has a session-scoped lever.** Where a tool can't scope a scene (or a specific concern) at launch, crossby warns rather than applying nothing silently — but the outcome varies: a CLI tool without a launch lever falls back to persistent activation, a concern with no lever at all can be left wide open, and a GUI tool just launches without the scene:

| Tool | Session-scoped lever |
| --- | --- |
| Claude | `--mcp-config <file> --strict-mcp-config`, a `--settings` file of `skillOverrides` (needs `claude ≥ 2.1.129`), and `--disallowedTools "Agent(<name>)"` per deselected agent |
| Codex | `--profile <name>` layering a generated `$CODEX_HOME/<name>.config.toml` (needs `codex ≥ 0.134.0`) |
| Copilot | `--disable-mcp-server <name>` per deselected server; a profile's `--allow-tool` entries naming an excluded tool are dropped |
| Cursor | none — falls back to persistent activation (its only knob relocates the whole config base including auth) |
| OpenCode | none — persistent fallback records the lifecycle but has no tool-config mechanism, so deselected servers stay enabled; `scene clear` removes the fallback state |
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

`crossby launch` runs any supported tool with one unified set of flags. Ordinary
autonomy options may degrade when a tool lacks an exact equivalent; native plan
mode is stricter and fails before launch when Crossby cannot guarantee it.

### Programmatic sandbox selection

Library consumers can choose sandbox confinement independently from autonomy by passing the keyword-only `sandbox=` argument to an adapter's `launch()`, `build_launch_command()`, or `build_resume_command()` method. This is an adapter API only; there is no `crossby launch` CLI flag or persisted config field for it.

| Adapter | `sandbox=True` (default) | `sandbox=False` |
| --- | --- | --- |
| Codex | Preserves the existing conditional `workspace-write` composition described below | `--sandbox danger-full-access` |
| Cursor | `--sandbox enabled` | `--sandbox disabled` |
| All others | No sandbox-selection flag | No sandbox-selection flag |

The setting never changes approval behavior: Codex `danger-full-access` does not imply `-a never`, and yolo does not imply an unrestricted sandbox. Cursor now explicitly enables its sandbox on the default adapter path instead of inheriting a potentially disabled user setting. The static `sandboxes_writes` capability still describes an adapter's normal confinement; it is not a guarantee for a particular invocation made with `sandbox=False`.

### Native plan mode

Native planning has two deliberately separate surfaces:

- `launch(..., plan_mode=True)` enters native plan mode for a human and keeps the
  historical exit-code return value. `supports_plan_mode` is this
  **activation-only** compatibility view.
- `run_plan_session(PlanSessionRequest(...))` owns activation, interaction,
  exact-session collection, validation, and cleanup, then returns normalized
  Markdown plus provenance. `supports_plan_session` is true only when that full
  lifecycle is deterministic.

Neither surface treats prompt text such as `/plan` as activation. Plan mode,
sandbox confinement, and approval policy are independent request dimensions; a
collector either preserves a supported choice or rejects it before spawning.
Unknown and below-floor CLI versions also fail before a harness process starts.
`PlanSessionRequest` rejects unknown fields instead of silently applying a
default. Cursor and Antigravity CLI encode effort in model IDs, so an explicit
`effort` also requires an explicit `model`. Antigravity further requires a
compatible Gemini model whose native effort tier matches the request; missing,
non-Gemini, unavailable, or conflicting model tiers are rejected before the
collector launches. OpenCode collection accepts only `low`, `medium`, and
`high`: interactive launches retain the legacy `xhigh`/`max` → `high`
normalization, but a collected session rejects tiers the native `--variant`
argument cannot preserve exactly.

Support matrix (contracts verified against the listed builds on 2026-09-10):

| Tool | Native selector | Collector / exact binding | Interaction | Sandbox / approval | Verified floor | Remediation |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Code | `--permission-mode plan` | Interactive CLI; one `.md` in a fresh UUID `plansDirectory` | Native terminal | Tool-managed / tool-managed | 2.1.263 | Use a project-contained `plan_output_dir`; ambiguous, symlinked, blank, or missing output fails |
| Codex CLI | `collaborationMode.mode = "plan"` | Headerless app-server JSONL; exact thread + turn + completed plan-item IDs, then exact-turn interrupt | Callback | Preserved / preserved | 0.153.4 | The bound completed plan item is terminal; ordinary interactive launch has no pre-prompt selector and remains activation-only unsupported |
| Cursor CLI | ACP `session/set_mode` → `plan` | ACP; exact session + blocking `cursor/create_plan` request ID | Callback, including separate final plan outcome | Preserved / preserved (`on-request`, `never`) | 2026.09.02-c22c1a3 | Supply a handler for questions and the non-executing final outcome |
| GitHub Copilot CLI | `--plan` | Headless CLI; assigned UUID + unique local `--share` export | Resumable callback | Tool-managed / preserved (`on-request`, `never`) | 1.0.83 | Collection disables remote sharing and removes only its temporary export after normalization |
| OpenCode | `run --agent plan --dir <workspace>` | Headless JSONL; emitted session ID + `export <exact-id>` + plan-mode assistant text | Resumable callback | Tool-managed / tool-managed | 1.18.29 | Exported session/message directories must match the request; no latest-session lookup is used |
| Antigravity CLI | `--mode plan` | Headless JSON; case-insensitive terminal status + exact conversation ID + requested schema echo + `structured_output.plan` | Resumable callback | Tool-managed / tool-managed | 1.2.0 | Free text and private brain storage are not artifact fallbacks |
| VS Code | Unsupported | None | None | Unsupported | 1.136.1 | Select plan mode manually or use a complete terminal collector |
| Antigravity IDE | Unsupported | None | None | Unsupported | — | Select plan mode manually or use a complete terminal collector |

`tool-managed` means the harness's native plan posture owns that dimension; only
its safe default is accepted. `preserved` means Crossby enforces the listed
caller choices explicitly; an unlisted approval policy is rejected before
collection. Protocol and resumable collectors never invent an answer or
auto-approve implementation. A missing handler produces
`PlanInteractionRequiredError`; final plan approval is represented separately
and an `APPROVED` response is refused by collectors where it would transition
into execution.

The collected API is the automation surface:

```python
from pathlib import Path

from crossby.ai_tools import (
    AbstractAITool,
    PlanInteractionKind,
    PlanInteractionOutcome,
    PlanInteractionRequiredError,
    PlanInteractionResponse,
    PlanSessionError,
    PlanSessionRequest,
)


def answer(interaction):
    if interaction.kind is PlanInteractionKind.PLAN_APPROVAL:
        return PlanInteractionResponse(outcome=PlanInteractionOutcome.DENIED)
    # A real integration should obtain this from its user or workflow.
    return PlanInteractionResponse(
        outcome=PlanInteractionOutcome.ANSWERED,
        option_id=interaction.options[0].option_id if interaction.options else None,
        answer=None if interaction.options else "Use the existing public API",
    )


adapter = AbstractAITool.get("codex")
try:
    result = adapter.run_plan_session(
        PlanSessionRequest(
            prompt="Plan issue #176",
            working_dir=Path.cwd(),
        ),
        answer,
    )
except PlanInteractionRequiredError as exc:
    print(f"Waiting for {exc.interaction.question_id}: {exc.interaction.prompt}")
except PlanSessionError as exc:
    print(f"Plan collection failed: {exc}")
else:
    Path("PLAN.md").write_text(result.plan, encoding="utf-8")
    print(result.session_id, result.artifact_source, result.artifact_id)
```

Crossby returns Markdown and the available session, turn, item, conversation,
or path evidence. It does not impose WADE validation and does not require the
harness itself to create `PLAN.md`. `timeout_seconds` is one collector deadline
shared by the initial invocation, question continuations, protocol waits, and
subprocess-backed export. Caller-supplied artifact-location failures remain
`PlanArtifactLocationError` and are also caught by the collected API's
`PlanSessionError` integration boundary.

For activation-only CLI use, continue calling `adapter.launch(...,
plan_mode=True)` or `adapter.build_launch_command(..., plan_mode=True)` and
handle `PlanModeLaunchError`. `--plan` remains mutually exclusive with
`--yolo`, `--auto`, and `--accept-edits`. `--plan-output-dir <dir>` remains a
launch option only for Claude. The collected Claude API also accepts a
project-contained `plan_output_dir`, but creates a unique run-owned child
directory within it so a concurrent or newer artifact cannot be selected.

When a Claude scene also narrows skills, Crossby combines `plansDirectory` and
the scene's `skillOverrides` into one `--settings` JSON source. Claude treats
repeated `--settings` occurrences as replacement, so emitting two would discard
the requested plan destination.

### Autonomy modes

The remaining flags form the autonomy ladder (how much the agent may do without
asking). They are permission modes, not model selection:

- `--accept-edits` — auto-approve file edits, still prompt for shell/commands. Broadly portable (5 of the 6 CLIs support it at launch; OpenCode falls back to default prompting). *(Codex is the exception — its accept-edits is sandbox-confined rather than per-command-prompted; see the note below the table.)*
- `--auto` — Claude Code's classifier-mediated guarded autonomy (a separate model reviews each non-read action). **Claude-only** among the CLIs crossby drives; on other tools it **downgrades to that tool's accept-edits**, then to default prompting — never to `--yolo`.
- `--yolo` — skip all permission prompts.

**Precedence (most permissive wins):** `yolo > auto > accept-edits`. If you pass
several of these three, the highest applies. A requested tier a tool doesn't
support downgrades to the next lower autonomy tier it does support (emitting a
`UserWarning`), stopping at default prompting — it never escalates. None can be
combined with `--plan`.

Per-tool mapping (verified against official docs, July 2026; CLI flags can drift between versions, so treat the table as a point-in-time snapshot):

| Tool            | `--accept-edits`                      | `--auto` (classifier)                     |
| --------------- | ------------------------------------- | ----------------------------------------- |
| Claude          | `--permission-mode acceptEdits`       | `--permission-mode auto`                  |
| Codex           | `-a on-request --sandbox workspace-write` | ↓ downgrades to accept-edits              |
| Cursor CLI      | *(none — its default Agent mode already **is** accept-edits)* | ↓ downgrades to accept-edits |
| Copilot         | `--allow-tool write`                  | ↓ downgrades to accept-edits              |
| Antigravity CLI | `--mode accept-edits`                 | ↓ downgrades to accept-edits              |
| OpenCode        | ↓ default prompting (config-only)     | ↓ default prompting                       |
| VS Code, Antigravity IDE | ↓ default prompting (GUI)    | ↓ default prompting                       |

> Codex's old `--approval-mode auto-edit` was **removed** in the Rust CLI — crossby never emits it. Codex CLI 0.152 also removed the per-command `untrusted` approval policy, so Codex accept-edits maps to `-a on-request` (Codex's native "Auto" posture): the agent runs edits **and** commands freely inside the `workspace-write` sandbox and prompts you only before an action that would **escape** it (network, writes outside the workspace). Unlike the other tools, Codex's safety boundary here is the OS sandbox, not a per-command prompt — see the "Codex sandbox: linked worktrees & `--network`" section below. `--approve-for-me` is deliberately avoided (it would route even those escapes through automatic review with no prompt at all). Note Cursor CLI's default *is* accept-edits (the inverse of the Cursor IDE default), so `--accept-edits` is honored with no extra flag and no warning.

### Cross-provider model translation

`crossby launch` translates model ids across families when the target tool wouldn't accept the source family natively:

```bash
# Pass a Claude model id to Codex — translated to gpt-5.4-mini under the hood
crossby launch --tool codex --model claude-sonnet-4.6 --effort high
# → codex --model gpt-5.4-mini -c model_reasoning_effort=xhigh
```

Sonnet shifts effort up one tier (low→medium, medium→high, high→xhigh) for coding-agent behavior. The reverse direction (`gpt-5.4` → Claude) picks the lowest source tier so users don't accidentally over-bill. A `UserWarning` fires whenever a translation happens; pass a native id to silence it.

### Codex sandbox: linked worktrees & `--network`

Codex can confine writes with an OS sandbox (`--sandbox workspace-write` — Seatbelt on macOS, Landlock on Linux). With the default programmatic `sandbox=True`, crossby preserves its existing conditional composition. An explicit `sandbox=False` emits only `--sandbox danger-full-access` from the sandbox composer: no trusted or Git-metadata `--add-dir` roots and no workspace-write network pin. This remains independent from approvals—crossby never emits `--dangerously-bypass-approvals-and-sandbox`; Codex yolo is approval-skipping only (`-a never`), and approval `never` appears only when yolo is requested.

- **Linked worktrees & submodules just work when sandboxed.** In a linked worktree the working tree's `.git` is a *file* pointing at metadata that lives **outside** the working directory, which the sandbox would otherwise block. crossby detects this and grants only the real git-metadata dirs outside the root to the sandbox with `--add-dir` — which *adds* to the writable roots, preserving any `sandbox_workspace_write.writable_roots` you configured — so sandboxed git operations succeed while the sandbox stays on. A normal checkout grants nothing. This applies to launch, `--resume` (approval-neutral: no `-a` injected), and the headless handoff summarizer.
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

crossby lists the installed, updatable tools, runs each tool's own updater sequentially — continuing past any failure — and prints a report of `Tool · Version (before → after) · Status (updated / version unchanged / ✓ / ✗)`.

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
