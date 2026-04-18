# Contributing to crossby

## Development Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ivanviragine/crossby
cd crossby
uv sync --extra dev
```

## Running Checks

| Command | What it does |
|---|---|
| `./scripts/test.sh` | Run the full test suite |
| `./scripts/check.sh` | Lint (ruff) + type check (mypy strict) |
| `./scripts/fmt.sh` | Auto-format with ruff |
| `./scripts/check-all.sh` | Tests + lint + format check + type check |

Run `./scripts/check-all.sh` before submitting a PR.

## Architecture

```
src/crossby/
├── cli/          # Typer CLI commands (entry point: cli/main.py:cli_main)
├── services/     # High-level services (AI tool resolution, prompt delivery)
├── ai_tools/     # Per-tool adapters (Claude, Copilot, Gemini, Codex, …)
├── sync/         # Sync writers — translate and write config to each tool
├── config/       # .crossby.yml loading and Pydantic models
├── models/       # Shared data models
└── ui/           # Rich/questionary UI components
```

**Request flow:**

```
CLI command
  → service (e.g. run_sync)
    → ai_tools adapter (AbstractAITool — auto-registered via __init_subclass__)
      → sync writer (AbstractSyncWriter, keyed by (tool_id, concern) in SyncRegistry)
```

**Key patterns:**

- `AIToolID` is a `StrEnum` — works as both enum and string key
- `SyncRegistry` maps `(tool_id, concern)` → writer instance; `run_sync()` orchestrates them
- Config is loaded from `.crossby.yml` into Pydantic v2 models in `config/loader.py`
- Sync does not depend on `.crossby.yml` — it reads tool configs directly from standard paths
- Symlinks are always relative (`os.path.relpath`) so they survive repo moves
- Sync is idempotent: re-running on already-linked files is a no-op

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

Breaking changes: append `!` after the type, e.g. `feat!:`, and add a `BREAKING CHANGE:` footer.

## Release Process

1. Ensure `./scripts/check-all.sh` passes on `main`
2. Update the version in `pyproject.toml`
3. Commit: `chore: release vX.Y.Z`
4. Tag: `git tag vX.Y.Z`
5. Push both: `git push origin main && git push origin vX.Y.Z`
