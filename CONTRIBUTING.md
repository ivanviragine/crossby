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
├── ai_tools/     # Per-tool adapters (Claude, Copilot, Gemini, Codex, …)
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
Claude Code, Cursor, Gemini, Copilot, and Codex.  The architecture is a
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
Gemini's A2A `kind: remote`, Codex `max_depth`) are out of scope and
documented as not translatable.

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
- **`.crossby.yml`** is loaded by `config/loader.py` into Pydantic v2 models. **Sync does not depend on it** — it reads each tool's native config directly from standard paths. The config is only consulted by `crossby launch` for defaults.
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

| Crossby Flag  | Claude                             | Copilot           | Gemini                       | Codex                                      | OpenCode          | Cursor                     | VS Code | Antigravity |
| ------------- | ---------------------------------- | ----------------- | ---------------------------- | ------------------------------------------ | ----------------- | -------------------------- | ------- | ----------- |
| Binary        | `claude`                           | `copilot`         | `gemini`                     | `codex`                                    | `opencode`        | `agent`                    | `code`  | `antigravity` |
| `--model`     | `--model`                          | `--model`         | `--model`                    | `--model`                                  | `--model`         | `--model`                  | —       | —           |
| `--yolo`      | `--dangerously-skip-permissions`   | `--yolo`          | `--yolo`                     | `--yolo`                                   | —                 | `--force`                  | —       | —           |
| `--plan`      | `--permission-mode plan`           | `--plan`          | `--approval-mode plan`       | —                                          | —                 | `--mode plan`              | —       | —           |
| `--effort`    | `--effort <level>`                 | —                 | —                            | `-c model_reasoning_effort="…"`            | `--variant <level>` | model suffix (`-thinking`) | —       | —           |
| `--prompt`    | positional                         | `-i <prompt>`     | positional                   | positional                                 | `--prompt <prompt>` | positional                | —       | —           |
| `--transcript`| `script` wrapper                   | `script` wrapper  | `script` wrapper             | `script` wrapper                           | `script` wrapper  | `script` wrapper           | —       | —           |
| `--resume`    | `--resume <id>`                    | `--resume=<id>`   | `--resume <id>`              | `codex resume <id>` (subcommand)           | `-s <id>`         | —                          | —       | —           |
| `--trusted-dir` | `--add-dir`                      | `--add-dir`       | `--include-directories`      | `--sandbox workspace-write --add-dir`      | —                 | —                          | —       | —           |

### Effort Level Mapping

| Crossby Level | Claude   | Codex   | OpenCode | Cursor              |
| ------------- | -------- | ------- | -------- | ------------------- |
| `low`         | `low`    | `low`   | `low`    | —                   |
| `medium`      | `medium` | `medium`| `medium` | —                   |
| `high`        | `high`   | `high`  | `high`   | `<model>-thinking`  |
| `max`         | `max`    | `xhigh` | `high`   | `<model>-thinking`  |

### Permission & Allowlist Configuration

Crossby stores canonical command patterns (e.g. `myapp:*`) and writes them into each tool's native config format.

| Feature            | Claude                      | Copilot                        | Gemini                 | Cursor                        |
| ------------------ | --------------------------- | ------------------------------ | ---------------------- | ----------------------------- |
| Config file        | `.claude/settings.json`     | `.github/hooks/hooks.json`     | `.gemini/settings.json`| `.cursor/cli.json`            |
| Allowlist format   | `Bash(cmd:args)`            | `shell(cmd:args)`              | `shell(cmd:args)`      | `Shell(cmd:args)`             |
| Launch flag        | `--allowedTools`            | `--allow-tool`                 | `--allowed-tools`      | — (config-file only)          |
| Hook config        | `hooks.PreToolUse`          | `hooks.preToolUse`             | `hooks.BeforeTool`     | `preToolUse` in `hooks.json`  |
| Hook guard matcher | `Edit\|Write\|NotebookEdit` | `Write\|Delete`                | file-write tools       | `Write\|Delete`               |

### Session Preservation & Resume

| Feature                 | Claude                  | Copilot             | Gemini          | Codex              | OpenCode   | Cursor                   |
| ----------------------- | ----------------------- | ------------------- | --------------- | ------------------ | ---------- | ------------------------ |
| Resume command          | `claude --resume <id>`  | `copilot --resume=<id>` | `gemini --resume <id>` | `codex resume <id>` | `opencode -s <id>` | — |
| Session data path       | `~/.claude/projects/`   | —                   | —               | —                  | —          | `~/.cursor/projects/`    |
| Session data preserved  | Yes (worktree → main)   | —                   | —               | —                  | —          | Yes (worktree → main)    |

Session IDs are extracted automatically from transcripts when `--transcript` is used.

### Transcript Parsing (`crossby stats`)

| Feature                  | Claude | Copilot | Gemini | Codex |
| ------------------------ | ------ | ------- | ------ | ----- |
| Total tokens             | Yes    | Yes     | Yes    | Yes   |
| Input / output breakdown | Yes    | Yes     | Yes    | Yes   |
| Cached tokens            | Yes    | Yes     | Yes    | Yes   |
| Per-model breakdown      | —      | Yes     | Yes    | —     |
| Premium requests         | —      | Yes     | —      | —     |
| Session ID extraction    | Yes    | Yes     | Yes    | Yes   |

### Handoff Sources & Targets

| Tool                                  | Source (read)                                        | Target (launch) |
| ------------------------------------- | ---------------------------------------------------- | --------------- |
| Claude                                | ✓ (`~/.claude/projects/<encoded>/<id>.jsonl`)        | ✓               |
| Cursor                                | ✓ (`~/.cursor/projects/<encoded>/chat.json`)         | ✓               |
| Codex                                 | ✓ (`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`)   | ✓               |
| Copilot                               | ✓ (`~/.copilot/session-state/<id>/events.jsonl`)     | ✓               |
| Gemini, OpenCode, Antigravity, VS Code| —                                                    | ✓               |

### Library API (not exposed via CLI)

Available on each adapter for programmatic use:

| Feature           | Claude                                   | Copilot             | Gemini                   | Codex        | OpenCode           | Cursor          |
| ----------------- | ---------------------------------------- | ------------------- | ------------------------ | ------------ | ------------------ | --------------- |
| Trusted dirs      | `--add-dir`                              | `--add-dir`         | `--include-directories`  | `--add-dir`  | —                  | —               |
| Structured output | `--output-format json --json-schema …`   | —                   | `--output-format json`   | —            | —                  | —               |
| Model format      | dashed (`claude-haiku-4-5`)              | dotted (`claude-haiku-4.5`) | as-is            | as-is        | `provider/model`   | as-is           |

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
feat(sync): add allowlist conversion for Gemini
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
