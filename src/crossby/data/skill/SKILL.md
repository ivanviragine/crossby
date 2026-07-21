---
name: crossby-sync
description: Use when the user asks to sync, mirror, or port AI tool configuration — rules, agents, skills, MCP servers, hooks, and permissions — across Claude Code, Codex, Cursor, GitHub Copilot, OpenCode, VS Code, Antigravity, or Antigravity CLI. Also handles cross-tool translation with manual-fix notes for lossy fields (e.g. Claude `permissionMode: plan` → Codex), Claude slash commands as namespaced skills, and pre-write inspection via `--plan` / `--doctor` / `--validate-target`.
metadata:
  short-description: Sync AI tool config across every installed CLI
---

# crossby sync runbook

`crossby` is a CLI that keeps AI tool config (rules / agents / skills /
MCP servers / hooks / permissions) consistent across every supported AI
tool. This skill drives it without a human at the keyboard: it inspects
the project, runs sync, fixes manual-fix items inside generated
artifacts, validates, and re-runs until the result is clean.

## Autonomy

Keep going until the selected sync is complete: run the inspector, run
the migrator, fix every `<!-- crossby:manual-fix:start -->` block in
generated files, and re-run checks without stopping for confirmation.
If the user has not selected a source tool, infer one from the project
(prefer Codex's `AGENTS.md`, then `CLAUDE.md`) and proceed. Do not edit
the source tool's files (`.claude/settings.json`, `.claude/agents/`,
etc.); manual fixes belong in the **generated** Codex / Cursor / Copilot /
Antigravity CLI artifacts. Preserve unrelated existing config entries in
target files (e.g. `[mcp_servers]` Crossby didn't write, hand-curated
hook entries, custom JSON keys) — do not ask about them unless they
fail validation or directly conflict with the sync.

## Sync order

Run in this order for each project:

1. Use the host tool's TODO/task-list. Don't create `SYNC_TODOS.md` or
   any other todo file unless the user asks. Make TODOs concrete with
   literal source → target labels, e.g.

   - Inspect rules sources via `crossby sync --plan`
   - Inspect Codex readiness via `crossby sync --doctor`
   - Sync rules from `<source>` to all installed tools
   - Sync agents from `<source>` (translate to `.codex/agents/*.toml`)
   - Sync skills from `<source>` with `--strategy translate` if any
     source skill has `allowed-tools`
   - Validate the target via `crossby sync --validate-target`
   - Report the final markdown table

   Before finishing, update the TODO list so every finished step is
   marked `completed` and no step remains `in_progress`.

2. Read `references/differences.md` (and refresh the upstream tool
   docs if its `Docs last checked` date is more than ~3 months old).

3. Inspect before writing:

   ```bash
   crossby sync --plan --from <source>
   crossby sync --doctor --from <source>
   crossby sync --dry-run --from <source>
   ```

4. Convert / sync surfaces in the order Crossby's writers run:

   - **rules**: source instruction file → every other tool's path,
     symlinked when content is neutral, copied with a manual-fix block
     when the content references another tool's surfaces (`/hooks`,
     `ExitPlanMode`, `permissionMode`, `.claude/agents/`, etc.).
   - **agents**: directory symlink between markdown-shape tools
     (Claude / Cursor / Copilot / Antigravity CLI); per-file translation
     to `.codex/agents/<name>.toml` for Codex.
   - **skills**: directory symlink by default; per-file translation
     when `--strategy translate` is set so `allowed-tools` etc. become
     manual-fix notes for non-Claude targets. Claude slash commands
     under `.claude/commands/` become `claude-command-<slug>` skills
     for non-Claude targets.
   - **mcp**: merged into each tool's native shape (JSON for Claude /
     Cursor / Copilot / Antigravity CLI — the latter using `serverUrl`
     instead of `url` for remote servers; TOML for Codex). `Authorization:
     Bearer ${VAR}` becomes Codex `bearer_token_env_var`; `${VAR}` headers
     become `env_http_headers`; env-var self-references become
     `env_vars`.
   - **permissions**: per-tool allowlist translation between canonical
     `cmd:args` and `Bash(...)` / `Shell(...)`. Antigravity CLI has no
     persistent allowlist file — permissions are launch-time flags
     (`--dangerously-skip-permissions`/`--sandbox`), so it's skipped
     here (same as Codex's sandbox mode).
   - **hooks**: dedup by `(event, command)` with matcher widening.
   - **plugins**: detect-only; emits `Not Added` rows for `.claude/
     plugins/`, `plugin-marketplaces.json`, and `.claude-plugin/
     marketplace.json`. Migrate by hand.

5. Run the real sync, then inspect for manual-fix items:

   ```bash
   crossby sync --from <source>
   grep -rn "<!-- crossby:manual-fix:start -->" .
   cat .crossby/sync-report.md
   ```

6. Open every artifact that contains a `<!-- crossby:manual-fix:start
   -->` block and rewrite the indicated content for the target tool's
   semantics. Remove the block when the rewrite is done; if the source
   changes later, the next sync will re-emit a fresh block, so leaving
   stale blocks in place is safe but noisy.

7. Run `--validate-target` after each fix:

   ```bash
   crossby sync --validate-target
   ```

8. Re-run inspection and validation until both report no actionable
   findings.

9. Return the final report as a markdown table whose **Status** column
   uses `Added`, `Check before using`, or `Not Added`. Use `Added`
   when an artifact was created or unchanged, `Check before using`
   when a manual-fix block remains in the artifact, `Not Added` when
   no target was written. Wrap the artifact type in inline code (e.g.
   `` `Skill` ``); item name is plain text after it; notes are short
   and literal. Same shape as the persistent report at
   `.crossby/sync-report.md`:

   ```markdown
   | Status | Item | Notes |
   | --- | --- | --- |
   | `Added` | `Rule` .cursorrules | foreign markers in source |
   | `Added` | `Agent` release-lead | translated to TOML |
   | `Check before using` | `Skill` claude-command-pr-review | Converted from a Claude slash command |
   | `Not Added` | `Plugin` team-macros | Plugin needs manual setup |
   ```

## Self-healing loop

Repeat until clean:

1. `crossby sync --doctor --from <source>`
2. `crossby sync --dry-run --from <source>`
3. `crossby sync --from <source>`
4. Open every `<!-- crossby:manual-fix:start -->` block in generated
   artifacts; rewrite for target semantics; remove the block.
5. `crossby sync --validate-target`
6. Re-run from step 1 until `--doctor` reports `readiness: high` and
   `--validate-target` shows no errors.

Do not edit source-tool files. If a manual-fix block names a behaviour
that requires source-side changes (e.g. dropping `permissionMode:
plan`), leave the generated artifact with literal guidance and surface
the requirement to the user.

## Commands

   ```bash
   CROSSBY=crossby
   ```

Inspect first.

   ```bash
   $CROSSBY sync --plan --from <source>
   $CROSSBY sync --doctor --from <source>
   $CROSSBY sync --dry-run --from <source>
   ```

Run the sync.

   ```bash
   $CROSSBY sync --from <source>
   $CROSSBY sync --from <source> --to <target>          # narrow target
   $CROSSBY sync rules --from <source>                  # one concern
   ```

Validate after edits.

   ```bash
   $CROSSBY sync --validate-target
   ```

Get a portable result table.

   ```bash
   $CROSSBY sync --from <source> --report-format markdown-table
   ```

Run `$CROSSBY sync --help` for the full flag reference. The deeper
schema mapping table lives in `references/differences.md`.
