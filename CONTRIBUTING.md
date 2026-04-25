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
├── services/     # High-level orchestrators (sync, launch, handoff)
├── ai_tools/     # Per-tool adapters (Claude, Copilot, Gemini, Codex, …)
├── sync/         # Sync writers — translate and write config per tool
├── handoff/      # Session readers, summarizer, prompt loader, handoff writer
├── config/       # .crossby.yml loading and Pydantic models
├── models/       # Shared data models (AIToolID, capabilities, …)
├── data/         # Static model catalog and bundled prompt presets
├── ui/           # Rich/questionary UI components
└── logging/      # structlog configuration
```

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
- **`SyncConcern`** — enumeration of what a writer handles: `RULES`, `AGENTS`, `SKILLS`, `PERMISSIONS`, `HOOKS`, `MCP`.
- **`.crossby.yml`** is loaded by `config/loader.py` into Pydantic v2 models. **Sync does not depend on it** — it reads each tool's native config directly from standard paths. The config is only consulted by `crossby launch` for defaults.
- **Symlinks are always relative** (`os.path.relpath`, `config/linker.py`) so they survive repo moves.
- **Sync is idempotent** — re-running on already-linked files is a no-op.

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
3. Must be idempotent — re-running on unchanged state should return `action="skip"` or `action="noop"`.
4. Must respect `dry_run` — compute the intended change but make no filesystem writes.
5. On write conflicts, honor `force` (backup + overwrite) vs. raising.

Register the instance in `src/crossby/sync/__init__.py` alongside the other writers. `SyncRegistry` enforces uniqueness by `(tool_id, concern)`.

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

1. Ensure `./scripts/check-all.sh` passes on `main`.
2. Update the version in `pyproject.toml`.
3. Commit: `chore: release vX.Y.Z`.
4. Tag: `git tag vX.Y.Z`.
5. Push both: `git push origin main && git push origin vX.Y.Z`.
