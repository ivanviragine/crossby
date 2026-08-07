# Contributing to crossby

Thanks for your interest in contributing. This document is the maintainer/developer guide — architecture, conventions, and how to extend crossby safely. If you're looking for usage docs, see [README.md](README.md).

## Development Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ivanviragine/crossby
cd crossby
uv sync --extra dev
```

## Running Checks

| Command                | What it does                                 |
| ---------------------- | -------------------------------------------- |
| `./scripts/test.sh`    | Run the full test suite                      |
| `./scripts/check.sh`   | Lint (ruff) + type check (mypy strict)       |
| `./scripts/fmt.sh`     | Auto-format with ruff                        |
| `./scripts/check-all.sh` | Tests + lint + format check + type check  |

Run `./scripts/check-all.sh` before opening a PR.

`tomli-w` is a base dependency (required for Codex MCP sync), so a plain `uv sync` pulls in everything the test suite needs.

## Architecture

### Directory Layout

```
src/crossby/
├── cli/          # Typer commands (entry point: cli/main.py:cli_main)
├── services/    # High-level orchestrators (sync, launch, handoff)
├── ai_tools/     # Per-tool adapters (Claude, Copilot, Codex, Antigravity CLI, …)
├── sync/         # Sync writers — translate and write config per tool
├── subagents/    # Subagent format translation (canonical IR + parsers/emitters)
├── handoff/      # Session readers, summarizer, prompt loader, handoff writer
├── config/       # .crossby.yml loading and Pydantic models
├── models/       # Shared data models (AIToolID, capabilities, …)
├── data/         # Static model catalog and bundled prompt presets
├── ui/           # Rich/questionary UI components
└── logging/      # structlog configuration
```

### Subagent format translation

`src/crossby/subagents/` translates a single subagent definition between
Claude Code, Cursor, Copilot, and Codex.  The architecture is a
canonical intermediate representation (`SubagentIR`) with one parser and one
emitter per tool — no pairwise converters.  Tool-specific fields that don't
generalize live in `SubagentIR.extras` and are only re-emitted when the
target tool matches the original source.

Lossy translations surface as `ConversionWarning(severity=lossy|dropped)`
rather than silent drops — Cursor (no tool allowlist), Codex (only
`sandbox_mode`), and Copilot's required `description` are the main offenders.

Codex is the asymmetric case: its emitter returns a `CodexEmission`
containing both the agent `.toml` body and a `[agents.<name>]` fragment for
`~/.codex/config.toml`.  Orchestration features (`/fleet`, `/multitask`,
Codex `max_depth`) are out of scope and documented as not translatable.

CLI: `crossby agents convert --from <tool> --to <tool> <input>`.

### Request Flow

```
CLI command
  → service (e.g. run_sync)
    → AI tool adapter (AbstractAITool — auto-registered via __init_subclass__)
      → sync writer (AbstractSyncWriter, keyed by (tool_id, concern) in SyncRegistry)
```

### Key Concepts

- **`AIToolID`** (`models/ai.py`) — a `StrEnum`. Works as both an enum member and a string key.
- **`AbstractAITool`** (`ai_tools/base.py`) — every adapter subclasses this. Setting the `TOOL_ID` class variable auto-registers the adapter via `__init_subclass__` — no other file needs to change.
- **`SyncRegistry`** (`sync/base.py`) — maps `(tool_id, concern)` → writer instance. Populated in `sync/__init__.py`; `run_sync()` orchestrates matching writers and collects `SyncResult`s.
- **`SyncConcern`** — enumeration of what a writer handles: `RULES`, `AGENTS`, `SKILLS`, `PERMISSIONS`, `HOOKS`, `MCP`, `PLUGINS`. `PLUGINS` is detect-only — `run_sync()` injects findings via `sync/plugins.py` after the regular writer pass.
- **Canonical agent IR** lives in `subagents/` (PR #46): `SubagentIR` plus one parser and one emitter per tool. `sync.agents._sync_translate` / `CodexAgentsWriter` delegate to `subagents.api.convert` for cross-tool translation; `ConversionWarning`s with `severity=lossy|dropped` are turned into `<!-- crossby:manual-fix -->` blocks by `_ir_body_with_manual_fix` before emit so the lossy edge surfaces inside the artifact, not just on the terminal.
- **Canonical skill model** (`sync/agent_models.py`) — `SkillDefinition` is a tool-neutral dataclass plus `parse_markdown_skill` / `render_markdown_skill` / `translate_skill_for_target`. Skills use the same on-disk SKILL.md shape across every tool today, so the canonical layer exists only to attach manual-fix notes for fields the target tool doesn't honour (Claude `allowed-tools` on non-Claude targets).
- **Manual-fix block** (`sync/manual_fix.py`) — when a writer can't faithfully translate a source field, it embeds a stable `<!-- crossby:manual-fix:start --> ... <!-- crossby:manual-fix:end -->` block in the rendered file. The block survives markdown rendering, sits inside TOML multi-line strings without escaping, and is replaced 1:1 on re-runs (`strip_manual_fix_blocks` + `append_manual_fix_block` + `find_manual_fix_blocks`).
- **Cross-provider mappings** (`sync/translation.py`) — Claude↔Codex family table for `model`, family-aware `effort` bias, and `permissionMode` ↔ `sandbox_mode`. Used by both the agents writer and `crossby launch`'s `build_launch_command` for cross-provider model translation.
- **Pre-write inspection** (`sync/plan.py`, `sync/validate.py`) — `--plan` summarizes a dry-run by concern + manual-fix count; `--doctor` adds validation findings and a coarse `high`/`medium`/`low` readiness rating; `--validate-target` re-parses every synced file (TOML / JSON parseability, agent required fields, skill frontmatter, MCP `command` on PATH, instruction file size).
- **Persistent reports** (`sync/report.py`) — every real (non-dry-run) sync writes `.crossby/sync-report.md` with a portable `| Status | Item | Notes |` table. Statuses: `Added`, `Check before using`, `Not Added` — driven by `(action, file_path)` rather than message-substring matching.
- **`.crossby.yml`** is loaded by `config/loader.py` into Pydantic v2 models. **Sync does not depend on it** — it reads each tool's native config directly from standard paths. The config is consulted by `crossby launch` for defaults and by `crossby scene` for the `scenes:` definitions it activates.
- **Symlinks are always relative** (`os.path.relpath`, `config/linker.py`) so they survive repo moves.
- **Sync is idempotent** — re-running on already-linked files is a no-op. Translate writers hash-compare rendered output before deciding `created` / `updated` / `skipped`.

### Headless vs. interactive launches

`build_launch_command` takes two distinct prompt-related parameters:
- `prompt` — used for batch/headless invocations (one-shot runs, CI), gated by the tool's `headless_flag`.
- `initial_message` — used for interactive sessions, placed as the first positional arg before any flags.

Keep these separate when adding launch logic.

## Adding a New AI Tool

The adapter pattern is designed so adding a tool is a single-file change.

1. Add the tool ID to `AIToolID` in `src/crossby/models/ai.py`.
2. Create `src/crossby/ai_tools/<tool>.py` subclassing `AbstractAITool`:
   - Set `TOOL_ID = AIToolID.<TOOL>` (this auto-registers the adapter).
   - Implement `capabilities()` returning an `AIToolCapabilities` with at minimum `binary`, `display_name`, `model_flag`, `supports_*` booleans.
   - Override the optional hooks that apply — e.g. `yolo_args()`, `effort_args()`, `trusted_dirs_args()`, `normalize_model_format()`, `resolve_effort_model()`, `initial_message_args()`.
3. If the tool should participate in `crossby sync`, add writers under `src/crossby/sync/<concern>.py` for each concern it supports (see below) and register them in `sync/__init__.py`.
4. If the tool should be a handoff **source**, override `locate_sessions()` and `read_session()` in the adapter.
5. Add static model entries to `src/crossby/data/` if the tool has a known model catalog.
6. Add tests under `tests/` — unit tests for the adapter, and integration tests for any sync writers.

Adapters are imported via `src/crossby/ai_tools/__init__.py`. Make sure to add your import there so `__init_subclass__` runs.

## Adding a New Sync Writer

Sync writers live in `src/crossby/sync/<concern>.py` and subclass `AbstractSyncWriter` (see `sync/base.py`). Each writer:

1. Sets `tool_id: AIToolID` and `concern: SyncConcern`.
2. Implements `sync(data, project_root, *, dry_run, force) -> SyncResult`.
3. Must be idempotent — re-running on unchanged state should return `action="skipped"` (with `file_path` set when the artifact is already in place; `file_path=None` means "nothing was synced for this concern", which the report renderer maps to `Not Added`).
4. Must respect `dry_run` — compute the intended change but make no filesystem writes.
5. On write conflicts, honor `force` (backup + overwrite) vs. raising.

Register the instance in `src/crossby/sync/__init__.py` alongside the other writers. `SyncRegistry` enforces uniqueness by `(tool_id, concern)`.

### Revocation and the ownership ledger

crossby's sync is **additive by default but revocable**: it can take back an entry it wrote earlier (e.g. after `sync --from A` then `sync --from B`, a target reflects B's hooks/permissions, not the union of both). The mechanism is a provenance ledger (`sync/ownership.py` → `.crossby/owned.json`) that records what crossby wrote per `(tool_id, concern)` — hook `(event, command)` pairs, canonical permission patterns, and MCP server names.

The load-bearing rule when adding removal to a writer:

- **Writers must never infer "absent from `data` ⇒ remove".** Revocation is computed **only** in `run_sync()` as `ledger_owned − current_set` and handed to the writer through an explicit, default-empty field on `SyncData` (`hooks_remove`, `permissions_remove`, `mcp_remove`). A writer that inferred removal from its own `data` would be catastrophic: `config/claude_allowlist.configure_plan_hooks` and the Cursor/Copilot equivalents call writers directly with a single-hook `SyncData` on every plan-mode/worktree session setup, and would wipe every other hook in the file.
- **Ownership is what crossby *wrote*, not what matched the source.** Each writer reports the identities it created **fresh** this run via `SyncResult.created` (an entry it appended, a permission/server it added — never one already present). `run_sync()` records `new_owned = (ledger_owned ∩ current) ∪ created`. This is what keeps a hand-authored entry that merely shares a `(event, command)` / pattern / server-name with the source from being claimed — and thereby later narrowed or revoked — by crossby.
- `run_sync()` records new ownership **only** for writers that did not return `error` — an error leaves the ledger untouched so the next run retries cleanly.
- The ledger is **gitignored** (via `gitignore_utils.update_managed_block`) and therefore **per-machine**: a fresh clone starts with an empty ledger and can only *add* until it catches up with what is already on disk — it never revokes an entry it has no record of writing. A missing or malformed `owned.json` degrades to "own nothing", never a crash. Persistence is best-effort too: an `OSError` while saving `owned.json` or updating `.gitignore` is logged and swallowed, so a read-only dir or full disk never discards the `SyncResult`s whose writes already succeeded (the ledger simply retries next run).
- MCP is narrower than hooks/permissions: it does **not** auto-revoke a server merely absent from the source. `mcp_remove` bounds the existing `enabled=False` deletion so crossby can never delete a same-named server a human wrote. Hooks and permissions do revoke-on-absence.
- A revocation-only row classifies as `Removed` (not `Added`) in `sync/report.py`, and `--plan`/`--doctor` count it via `PlanSummary.revoked_count`.
- **The interactive wizard revokes on *environment-wide absence*.** Each revocable concern (hooks, permissions, MCP) dispatches on *data OR ownership*: the wizard loads the ledger once and runs `run_sync()` for a concern when the merged source data is non-empty **or** any installed target still owns entries for it. Because the wizard merges discovery across *all* installed tools (no `--from` scope), "revoke" here means the concern has emptied across the **whole environment** — an entry still present on any installed tool is rediscovered as current and is **not** stripped. This is *not* entry-for-entry `--from` parity (where an entry present only on other tools would be revoked). Two consequences to know: MCP revokes only via `disabled ∩ owned`, so an owned-but-emptied MCP dispatch removes nothing and just emits `skipped`/discovery rows; and declining a still-present port while the ledger owns entries leaves `data` empty for that concern, so the ownership arm fires and revokes the owned copies (a later confirm re-adds them). Hardening "decline ≠ revoke" (e.g. also requiring `not scan.<concern>.found`) is a possible follow-up, not current behavior.

### Scene activation mechanisms

A **scene** narrows what each installed tool sees for one or more concerns
(skills, agents, MCP, hooks, permissions). The resolver
(`services/scene_resolution.py`) decides *what* is selected; the activation
engine (`scenes/`) decides *how* to make each tool honour that selection,
choosing the least-invasive mechanism available per `(tool, concern)` cell. The
matrix lives in `scenes/mechanism.py`:

- **DECLARE** — write the tool's own disable key. Non-destructive and instantly
  reversible; the user's real skill/server/agent files are left untouched.
  Claude `skillOverrides` (skills, gated on `claude >= 2.1.129`),
  `permissions.deny: ["Agent(<name>)"]` (agents), and `disabledMcpjsonServers`
  (MCP); Codex `mcp_servers.<id>.enabled = false`; Antigravity CLI
  `mcpServers.<name>.disabled = true`.
- **PROJECT** — materialise a scene-filtered source tree of relative symlinks
  under `.crossby/scene/active/{skills,agents}` (carrying the `.crossby-managed`
  marker) and re-point the existing sync writers at it, or, for hooks and
  permissions, filter the concern's list and drive `run_sync`'s revocable
  removal channel. No new per-tool path knowledge is added — the engine composes
  the registered writers.
- **UNSUPPORTED** — the tool has no per-item lever (Cursor / Copilot MCP); the
  cell is reported, never silently faked.

| Concern | Claude | Codex | Antigravity CLI | Cursor | Copilot |
|---|---|---|---|---|---|
| skills | DECLARE (`skillOverrides`) | PROJECT | PROJECT | PROJECT | PROJECT |
| agents | DECLARE (`permissions.deny`) | PROJECT | PROJECT | PROJECT | PROJECT |
| mcp | DECLARE (`disabledMcpjsonServers`) | DECLARE (`enabled=false`) | DECLARE (`disabled=true`) | UNSUPPORTED | UNSUPPORTED |
| hooks | PROJECT (removal) | PROJECT (removal) | PROJECT (removal) | PROJECT (removal) | PROJECT (removal) |
| permissions | PROJECT (removal) | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |

Rules that keep the matrix honest:

- **Shared paths make PROJECT authoritative.** Codex and Antigravity CLI share
  the literal directory `.agents/skills`. Antigravity has no skills DECLARE
  lever, so PROJECT re-points that physical directory for both — a best-effort
  Codex `[[skills.config]]` toggle could only contradict it, so the cell is
  PROJECT. The rule is general: when tools share a resolved target path and any
  lacks a DECLARE lever, PROJECT wins for the whole group (resolved *after*
  grouping by path, not per-tool in isolation).
- **The canonical source is never re-pointed onto a tree that links into it.**
  The real skills/agents source (e.g. `.claude/skills`) stays real and filters
  itself via DECLARE; only the *other* tools' directories re-point at the scene
  tree. Re-pointing crossby's own symlink always forces (a scene switch must
  replace the prior link); a real, non-crossby directory still honours `--force`
  (refused otherwise, backed up with it).
- **DECLARE provenance is ledgered.** Every DECLARE surface here is new and not
  covered by the revocable-sync ledger's additive concerns, so the five key
  types get their own `scene` section in `.crossby/owned.json` (`SceneDeclareKey`
  in `sync/ownership.py`). `clear_scene` reverts only entries crossby recorded —
  a user's own `skillOverrides`/`deny`/`disabled` value survives untouched.
- **Report honestly.** Plugin-provided skills (reachable by neither mechanism),
  Codex toggles on an untrusted project (silently ignored until trusted), and a
  Claude build below `2.1.129` are each surfaced as report rows rather than a
  quiet no-op. `dry_run` computes the full result set and never touches disk;
  `apply_scene` is idempotent and safe to re-run from a half-applied state.

### Session-scoped scene launch

`crossby launch --scene <name>` applies a scene to a **single session** without
mutating any tracked file — the session-scoped counterpart to the persistent
`scene use` above. The two share the resolver (`ResolvedScene`) but diverge on
enactment: instead of writing tool config files, each adapter renders throwaway
artefacts under `.crossby/scene/<name>/launch/` (kept out of git via
`.git/info/exclude`, never a tracked `.gitignore` edit) and returns the
flags/env that point its CLI at them. The code lives in `scenes/launch.py`
(the `SceneLaunchContext`/`SceneLaunchArgs` types plus rendering, the Codex
profile helpers, and pruning), each adapter's `scene_launch_args`, the
`scene_*` capability fields on `AIToolCapabilities`, and the `--scene` handling
in `cli/launch.py`.

| Tool | Session-scoped lever |
|---|---|
| Claude | `--mcp-config <file> --strict-mcp-config` (selected servers), a `--settings` file of `skillOverrides` (gated on `claude ≥ 2.1.129`), and `--disallowedTools "Agent(<name>)"` per deselected agent |
| Codex | `--profile <name>` layering a generated `$CODEX_HOME/<name>.config.toml` (deselected servers → `enabled = false`); gated on `codex ≥ 0.134.0` |
| Copilot | `--disable-mcp-server <name>` per deselected server (visibility layer); a profile's `--allow-tool` entries naming an excluded tool are filtered out (approval layer) before both are emitted |
| Cursor | `CURSOR_CONFIG_DIR` → a scene-materialised config dir (`mcp.json` of selected servers) |
| OpenCode | `OPENCODE_CONFIG` → a scene-rendered config file (`{"mcp": {…}}` of selected servers) |
| Antigravity CLI | none — no launch lever; falls back to persistent activation |
| VS Code / Antigravity IDE | none (GUI, override `launch()`); the CLI warns and drops the scene before dispatch |

Rules that keep this honest:

- **Every artefact is temp-file-then-atomic-rename** (`atomic_write_text`), so a
  crash mid-render never leaves a half-written file. Concurrent launches of the
  same scene race to the same paths; the atomic rename removes the torn-read
  failure mode and last-writer-wins on content is accepted rather than locked.
- **`$CODEX_HOME` is the one location exception.** `codex --profile <name>` only
  reads `$CODEX_HOME/<name>.config.toml` (usually `~/.codex`, shared across
  projects), so its profile can't live under `.crossby/scene/`. The generated
  filename is namespaced by a project-root hash
  (`crossby-<slug>-<scene>.config.toml`) so two repos' same-named scenes never
  collide, and the file carries a generated-by header (`CODEX_PROFILE_MARKER`).
- **Pruning is ownership-gated on both sides.** On the next launch,
  `prune_stale_artifacts` removes stale crossby-owned Codex profiles and stale
  local `.crossby/scene/<name>/launch/` trees for scenes no longer defined — but
  each has an ownership test it must pass first: a Codex profile only when its
  first line is the header (`is_crossby_codex_profile`), a local tree only when
  it carries the `.crossby-managed` marker crossby stamps into it. A hand-written
  profile matching the naming pattern, or a hand-made directory under
  `.crossby/scene/`, is therefore never deleted — the filename/path alone is
  never sufficient.
- **No silent no-op, no silent mutation.** A terminal tool with no launch lever
  (Antigravity CLI) or a runtime gate that failed (Codex too old) falls back to
  persistent `scene use` activation for that tool, warning that config was
  written. A GUI tool warns that a scene cannot apply and launches without it.
- **Precedence** matches the profile rule: explicit CLI flags > scene > profile
  > `ai:` defaults. A scene may name a default `profile:`; an explicit
  `--profile` (or positional profile name) overrides it. `--scene` targets
  exactly one tool and never fans out.

### Scene authoring (writing `.crossby.yml`)

`crossby init` is not the only thing that writes `.crossby.yml` — `crossby scene
create`/`add`/`remove`/`delete` do too, and they run repeatedly. Two pieces make
that safe:

- **One checked-write helper.** `config/safe_write.py:write_config_checked`
  backs up → writes atomically → re-parses (and runs an optional `validate`
  callback, used to resolve a scene's `extends` chain) → restores the previous
  file byte-for-byte on any failure. `init` passes `keep_backup=True` (a
  full-file overwrite deserves a recovery net); the scene commands leave no
  `.bak`. Both callers funnel through it rather than re-implementing the
  sequence.
- **Scoped splicing, not re-render.** `scenes/authoring.py` rewrites only the
  byte span of the single `scenes.<name>` entry being edited. The span is found
  with `yaml.compose()` node `start_mark`/`end_mark` offsets — **never**
  line-scanning for "the next top-level key", which a block scalar containing a
  `key:`-looking line would break. Everything outside that span (comments in
  `ai:`/`profiles:`/`models:`, sibling scenes) is preserved. A brand-new
  `scenes:` key is appended deterministically after the last top-level key.
  Selector edits enforce the cross-channel rule (adding a pattern to `include`
  drops it from `exclude`, and vice-versa) so the two can never contradict.

Starter scenes are bundled YAML under `src/crossby/data/scenes/` and loaded by
`scenes/starters.py:load_starter_scenes`. Keep them **self-contained** — glob
selectors only, no `extends`/`profile` — so `install-starters` can drop them
into any project and they resolve (with warnings, never errors) where the named
items are absent. Add a starter by dropping a `<name>.yml` (one `<name>: <body>`
mapping) into that directory; the parse test asserts every bundle validates
under the schema.

### Adding a scene mechanism for a new tool

When you teach crossby a new tool (see **Adding a New AI Tool**), give it a
scene cell in the matrix under **Scene activation mechanisms** for each concern:

1. Prefer **DECLARE** — a native, non-destructive disable key. Add the read side
   to the tool's config reader and the write/revert side in `scenes/declare.py`,
   record its provenance as a `SceneDeclareKey` in `sync/ownership.py`, and gate
   it on the minimum tool version in `scenes/versioning.py` if the key is new.
2. If the tool has no per-item key, fall back to **PROJECT**: the engine already
   re-points skills/agents directories and drives the revocable-sync removal
   channel for hooks/permissions — you usually add no per-tool path knowledge,
   just map the concern to PROJECT in `scenes/mechanism.py`.
3. If neither exists, mark the cell **UNSUPPORTED** so it is *reported*, never
   silently faked, and add a row to the matrix in this file and in
   `references/differences.md`.

### Symlink, copy, or translate

Writers that own file-tree concerns (rules, agents, skills) support up to three strategies via `SyncData.<concern>_strategy`:

- **`symlink`** (default): create relative symlinks. Cheapest; edits propagate everywhere; only works when source and target use the same on-disk schema.
- **`copy`**: physical copy with optional per-file rewrite (e.g. translate tool names `Bash`→`Shell` for Cursor). Used when the user wants a real file to commit, or when a marker on the source content would otherwise leak across schemas.
- **`translate`**: agent writers delegate to `subagents.api.convert(from_tool, to_tool, content)` (canonical `SubagentIR` + per-tool parsers/emitters). Skill writers parse via `SkillDefinition` and call `translate_skill_for_target`. Both attach manual-fix notes for fields the target doesn't honour, render back to the target's on-disk shape, hash-compare for idempotency, and remove stale outputs whose source disappeared.

When you add a new writer that handles one of these concerns, decide which strategies it supports, plumb each through `_sync_symlink` / `_sync_copy` / `_sync_translate` (see `agents.py` / `skills.py` for the existing pattern), and add tests for each strategy plus the no-op idempotent case.

### Adding a manual-fix path

If your writer translates a field that the target tool may not enforce or understand:

1. Build a `ManualFixNote` with a short `category` (e.g. `permissionMode`, `allowed-tools`) and a user-facing `message`.
2. Attach the note via `definition.with_notes([note])` on the canonical model.
3. The renderer (`render_markdown_skill`, `render_markdown_agent`, `render_toml_agent`) appends a `<!-- crossby:manual-fix --> … <!-- /crossby:manual-fix -->` block at the bottom — no extra plumbing needed.

Keep notes short and literal. Avoid Crossby-internal terminology in the message; users editing the file shouldn't need to know about `SubagentIR` or `SkillDefinition` to act on the note.

## Tool Reference

Crossby translates its unified CLI flags into each tool's native syntax. A dash (—) means the tool does not support that feature; crossby raises an error if you pass an explicit flag that the target tool doesn't support (e.g. `--yolo` with OpenCode).

### Launch Flags

| Crossby Flag  | Claude                             | Copilot           | Antigravity CLI                  | Codex                                      | OpenCode          | Cursor                     | VS Code | Antigravity IDE |
| ------------- | ---------------------------------- | ----------------- | --------------------------------- | ------------------------------------------ | ----------------- | -------------------------- | ------- | --------------- |
| Binary        | `claude`                           | `copilot`         | `agy`                             | `codex`                                    | `opencode`        | `agent`                    | `code`  | `antigravity`   |
| `--model`     | `--model`                          | `--model`         | `--model`                         | `--model`                                  | `--model`         | `--model`                  | —       | —               |
| `--yolo`      | `--dangerously-skip-permissions`   | `--yolo`          | `--dangerously-skip-permissions --sandbox` | `--yolo`                           | —                 | `--force`                  | —       | —               |
| `--plan`      | `--permission-mode plan`           | `--plan`          | `--mode plan`                     | —                                          | —                 | `--mode plan`              | —       | —               |
| `--effort`    | `--effort <level>`                 | —                 | model suffix (`-<level>`)          | `-c model_reasoning_effort="…"`            | `--variant <level>` | model suffix (`-thinking`) | —       | —               |
| `--prompt`    | positional                         | `-i <prompt>`     | `--prompt-interactive <prompt>`   | positional                                 | `--prompt <prompt>` | positional                | —       | —               |
| `--transcript`| `script` wrapper                   | `script` wrapper  | `script` wrapper                  | `script` wrapper                           | `script` wrapper  | `script` wrapper           | —       | —               |
| `--resume`    | `--resume <id>`                    | `--resume=<id>`   | `--conversation <id>`             | `codex resume <id>` (subcommand)           | `-s <id>`         | —                          | —       | —               |
| `--trusted-dir` | `--add-dir`                      | `--add-dir`       | `--add-dir`                       | `--sandbox workspace-write --add-dir`      | —                 | —                          | —       | —               |

### Effort Level Mapping

| Crossby Level | Claude   | Codex   | OpenCode | Cursor              | Antigravity CLI  |
| ------------- | -------- | ------- | -------- | ------------------- | ---------------- |
| `low`         | `low`    | `low`   | `low`    | —                   | `<model>-low`    |
| `medium`      | `medium` | `medium`| `medium` | —                   | `<model>-medium` |
| `high`        | `high`   | `high`  | `high`   | `<model>-thinking`  | `<model>-high`   |
| `xhigh`       | `xhigh`  | `xhigh` | `high`   | `<model>-thinking`  | `<model>-high`   |
| `max`         | `max`    | `xhigh` | `high`   | `<model>-thinking`  | `<model>-high`   |

Antigravity CLI (`agy`) bakes reasoning effort into the model ID rather than
emitting a separate `--effort` flag (which it rejects alongside a suffixed
model). Only the Gemini families encode effort — `gemini-3.6-flash` and
`gemini-3.5-flash` accept `low`/`medium`/`high`, `gemini-3.1-pro` accepts only
`low`/`high` (a requested `medium` snaps to the nearest valid tier). `xhigh`/
`max` normalize to `high`, and a Gemini model launched with no effort gets a
deterministic default (`medium`, or the nearest tier). Non-Gemini models
(`claude-*`, `gpt-oss-120b`) launch bare and ignore effort — a spurious effort
suffix on one (e.g. the retired `gpt-oss-120b-medium` catalog ID) is dropped
with a warning so `agy` is never handed a suffixed ID it rejects.

### Permission & Allowlist Configuration

Crossby stores canonical command patterns (e.g. `myapp:*`) and writes them into each tool's native config format.

Antigravity CLI has no per-project allowlist or hooks config — permissions
are mode-based launch flags (`--dangerously-skip-permissions`/`--sandbox`/
`--mode`) and it has no hook system at all, so `(ANTIGRAVITY_CLI,
PERMISSIONS)` and `(ANTIGRAVITY_CLI, HOOKS)` have no writer (same as Codex
having no permission writer).

| Feature            | Claude                      | Copilot                        | Cursor                        |
| ------------------ | --------------------------- | ------------------------------ | ----------------------------- |
| Config file        | `.claude/settings.json`     | `.github/hooks/hooks.json`     | `.cursor/cli.json`            |
| Allowlist format   | `Bash(cmd:args)`            | `shell(cmd:args)`              | `Shell(cmd:args)`             |
| Launch flag        | `--allowedTools`            | `--allow-tool`                 | — (config-file only)          |
| Hook config        | `hooks.PreToolUse`          | `hooks.preToolUse`             | `preToolUse` in `hooks.json`  |
| Hook guard matcher | `Edit\|Write\|NotebookEdit` | `Write\|Delete`                | `Write\|Delete`               |

### Session Preservation & Resume

| Feature                 | Claude                  | Copilot             | Antigravity CLI       | Codex              | OpenCode   | Cursor                   |
| ----------------------- | ----------------------- | ------------------- | --------------------- | ------------------ | ---------- | ------------------------ |
| Resume command          | `claude --resume <id>`  | `copilot --resume=<id>` | `agy --conversation <id>` | `codex resume <id>` | `opencode -s <id>` | — |
| Session data path       | `~/.claude/projects/`   | —                   | —                      | —                  | —          | `~/.cursor/projects/`    |
| Session data preserved  | Yes (worktree → main)   | —                   | —                      | —                  | —          | Yes (worktree → main)    |

Session IDs are extracted automatically from transcripts when `--transcript` is used.

### Transcript Parsing (`crossby stats`)

| Feature                  | Claude | Copilot | Codex |
| ------------------------ | ------ | ------- | ----- |
| Total tokens             | Yes    | Yes     | Yes   |
| Input / output breakdown | Yes    | Yes     | Yes   |
| Cached tokens            | Yes    | Yes     | Yes   |
| Per-model breakdown      | —      | Yes     | —     |
| Premium requests         | —      | Yes     | —     |
| Session ID extraction    | Yes    | Yes     | Yes   |

### Handoff Sources & Targets

| Tool                                  | Source (read)                                        | Target (launch) |
| ------------------------------------- | ---------------------------------------------------- | --------------- |
| Claude                                | ✓ (`~/.claude/projects/<encoded>/<id>.jsonl`)        | ✓               |
| Cursor                                | ✓ (`~/.cursor/projects/<encoded>/chat.json`)         | ✓               |
| Codex                                 | ✓ (`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`)   | ✓               |
| Copilot                               | ✓ (`~/.copilot/session-state/<id>/events.jsonl`)     | ✓               |
| Antigravity CLI, OpenCode, Antigravity IDE, VS Code| —                                        | ✓               |

### Library API (not exposed via CLI)

Available on each adapter for programmatic use:

| Feature           | Claude                                   | Copilot             | Antigravity CLI | Codex        | OpenCode           | Cursor          |
| ----------------- | ---------------------------------------- | ------------------- | ---------------- | ------------ | ------------------ | --------------- |
| Trusted dirs      | `--add-dir`                              | `--add-dir`         | `--add-dir`      | `--add-dir`  | —                  | —               |
| Structured output | `--output-format json --json-schema …`   | —                   | —                | —            | —                  | —               |
| Model format      | dashed (`claude-haiku-4-5`)              | dotted (`claude-haiku-4.5`) | as-is     | as-is        | `provider/model`   | as-is           |

## Commit Conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```

Common types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.

Examples:
```
feat(ai_tools): add Antigravity CLI adapter
fix(cli): handle missing .crossby.yml gracefully
docs: update compatibility table for Codex effort levels
```

Breaking changes: append `!` after the type (e.g. `feat!:`) and add a `BREAKING CHANGE:` footer.

## Release Process

### Every release

```bash
uv run python scripts/auto_version.py patch --push   # or minor / major
```

This bumps `pyproject.toml` and `src/crossby/__init__.py`, commits, tags
(`vX.Y.Z`), and pushes. From there:

1. `release.yml` creates a **draft GitHub Release** for the new tag with
   auto-generated notes.
2. Review the draft on GitHub and click **Publish Release**.
3. `publish.yml` builds the wheel/sdist with `uv build` and publishes to
   PyPI, authenticated via [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
   (OIDC) — no API tokens stored anywhere. If PyPI ever rejects the publish
   with `invalid-publisher`, the trusted publisher for `crossby` needs to be
   (re-)registered on pypi.org under **Publishing** settings, matching
   `ivanviragine/crossby`, workflow `publish.yml`, environment `pypi`.

If you have `./scripts/install-hooks.sh` set up (see below), you rarely need
to run the bump command manually — pushing a conventional-commit-prefixed
commit straight to `main` auto-bumps, tags, and pushes for you via a
`pre-push` hook.

**Merging a PR through GitHub's UI does *not* trigger a release on its own.**
`auto-version.yml` still bumps/commits/tags on merge, but that push is
authenticated as `GITHUB_TOKEN`, and GitHub Actions never lets a
`GITHUB_TOKEN`-authored push trigger other workflows (it's an anti-recursion
guard) — so `release.yml` never fires and the tag is left without a release.
If that happens, either re-push the tag yourself (`git push origin vX.Y.Z
--force` from a checkout with your own credentials) or create the release
manually: `gh release create vX.Y.Z --draft --generate-notes`.

### Git Hooks

```bash
./scripts/install-hooks.sh          # install into .git/hooks/
./scripts/install-hooks.sh --force  # overwrite existing hooks
```

Installs `pre-push` from `scripts/hooks/pre-push`, which detects
conventional-commit prefixes on pushes to `main`/`master` and runs the
version-bump step above automatically (skipped if the tip commit is already
a version bump, to avoid double-bumping). Because the hook pushes the tag
using your own git identity rather than `GITHUB_TOKEN`, it doesn't hit the
cascade limitation above — `release.yml` fires normally.

### Manual fallback

`./scripts/release.sh` builds and publishes the current version directly
from your machine (needs a PyPI API token via `UV_PUBLISH_TOKEN` or
`~/.pypirc` — Trusted Publishing only works from within GitHub Actions).
Use it if the CI pipeline is unavailable. `--dry-run` previews without
publishing.

### Version bump types

```bash
uv run python scripts/auto_version.py patch   # bug fixes     0.1.0 → 0.1.1
uv run python scripts/auto_version.py minor   # new features  0.1.0 → 0.2.0
uv run python scripts/auto_version.py major   # breaking      0.1.0 → 1.0.0
```

Add `--dry-run` to preview without making changes.
