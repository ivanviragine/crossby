# Issue #2: feat: add `crossby sync` command to port configs across AI tools

## Complexity
very_complex

## Context / Problem

Crossby's core purpose is to be a cross-platform bridge for AI agent configs,
but today it has no way to read one tool's full configuration and port it to
other tools. Users must manually copy allowlists, rewrite system instructions,
and recreate skills when switching between Claude, Cursor, Copilot, Gemini, and
Codex — defeating the purpose of a bridge.

The project already uses manual symlinks (`.agents/skills/ → .claude/skills`,
`.cursor/skills/ → .claude/skills`, etc.) proving the concept works. This
feature automates that pattern and extends it to all config types.

## Proposed Solution

A new `crossby sync` command that reads any tool's config (allowlists,
instructions, skills) and ports it to one or more target tools, **using
symlinks by default** and only converting when formats are incompatible
(allowlists).

Two modes:
- **Wizard** (`crossby sync` with no args): interactive prompts for source,
  targets, and confirmation
- **Direct** (`crossby sync --from claude --to cursor`): scriptable one-liner

### Scope

**In scope**: Claude, Cursor, Copilot, Gemini, Codex.
**Out of scope**: VSCode, OpenCode, Antigravity — these tools have no
instructions/skills/allowlist config to sync. If selected as a target, produce
an `UNSUPPORTED` warning.

### Source model for instructions

Any tool with an existing instruction file can be the source:
- `--from claude` → reads `CLAUDE.md`
- `--from cursor` → reads `.cursorrules`
- `--from copilot` → reads `.github/copilot-instructions.md`
- `--from gemini` → reads `GEMINI.md`
- `--from codex` → reads `AGENTS.md`

If the source file doesn't exist, skip instructions sync with an info message
(not an error).

### Strategy rules

| Config type    | Strategy | Rationale |
|----------------|----------|-----------|
| Instructions   | **LINK** | All plain markdown; symlink each tool's primary file to source |
| Skills         | **LINK** | All tools use identical SKILL.md format |
| Allowlists     | **CONVERT** | Different wrapper formats (`Bash()` vs `Shell()` vs `shell()`) |
| Allowlists (copilot/gemini) | **WARN** | No persistent config file (CLI flags only) |

### Overwrite policy (for LINK strategy)

- **Wizard mode**: prompt user before replacing an existing regular file
- **Direct mode**: replace existing regular file with symlink (destructive —
  uses `os.unlink()` then `os.symlink()`)
- If symlink already points to correct target: no-op (idempotent)

Note: git tracks symlinks. When `crossby sync` creates `.cursorrules → CLAUDE.md`,
ensure `CLAUDE.md` (the source) is committed or the symlink will be broken in
clones.

### Instruction symlink map

Source file is determined by `--from` tool. Target file per tool:

| Target  | Primary file | Symlink example (source = CLAUDE.md) |
|---------|--------------|--------------------------------------|
| Claude  | `CLAUDE.md` | `CLAUDE.md → <source>` |
| Cursor  | `.cursorrules` | `.cursorrules → CLAUDE.md` |
| Copilot | `.github/copilot-instructions.md` | `.github/copilot-instructions.md → ../CLAUDE.md` |
| Gemini  | `GEMINI.md` | `GEMINI.md → CLAUDE.md` |
| Codex   | `AGENTS.md` | `AGENTS.md → CLAUDE.md` |

### Skills symlink map

Source of truth: the first real (non-symlinked) skills directory found among
`.claude/skills/`, `.gemini/skills/`, `.agents/skills/`. Guard against circular
symlinks using `Path.resolve()` with `strict=False` and checking for cycles.

| Target dir        | Symlink |
|-------------------|---------|
| `.agents/skills/` | `→ ../<source_skills>` |
| `.cursor/skills/` | `→ ../<source_skills>` |
| `.gemini/skills/` | `→ ../<source_skills>` |
| `.github/skills/` | `→ ../<source_skills>` |

### Allowlist conversion

| Source → Target | Action |
|-----------------|--------|
| Claude → Cursor | Strip `Bash()`, wrap `Shell()`, write `.cursor/cli.json` |
| Cursor → Claude | Strip `Shell()`, wrap `Bash()`, write `.claude/settings.json` |
| Any → Copilot   | Warn: no persistent config (suggest `--allow-tool` flags) |
| Any → Gemini    | Warn: no persistent config |
| Any → Codex     | Warn: Codex uses sandbox mode, no allowlist config |

## Tasks

- [ ] Create `src/crossby/models/sync.py` — define `SyncStrategy` enum (`LINK`, `CONVERT`, `WARN`, `UNSUPPORTED`), `SyncAction` dataclass (`config_type`, `strategy`, `source_path`, `target_path`, `message`), `SyncResult` dataclass (`actions: list`, `linked: int`, `converted: int`, `warnings: list[str]`)
- [ ] Create `src/crossby/config/linker.py` — `create_symlink(source, link, *, force, dry_run)` with relative path computation, `os.unlink()` + `os.symlink()` for force mode, parent dir creation, circular symlink guard
- [ ] Create `src/crossby/config/instructions.py` — `get_instructions_source(tool_id, root)` to find the source instruction file, `get_instructions_target(tool_id, root)` to get the target path per tool, instruction file path mappings
- [ ] Create `src/crossby/config/skills.py` — `detect_skills_source(root)` to find real (non-symlinked) skills dir with circular symlink guard
- [ ] Add `read_allowlist()` to `src/crossby/config/claude_allowlist.py` — read `.claude/settings.json`, strip `Bash()` wrappers, return canonical patterns; return `[]` if file missing
- [ ] Add `read_allowlist()` to `src/crossby/config/cursor_allowlist.py` — read `.cursor/cli.json`, strip `Shell()` wrappers, return canonical patterns; return `[]` if file missing
- [ ] Create `src/crossby/services/sync.py` — `sync_configs()` orchestrator: resolve strategy per (target, config_type), dispatch to linker/converter, collect results; handle missing source configs gracefully (skip with info, not crash)
- [ ] Create `src/crossby/cli/sync.py` — wizard mode (interactive prompts via `ui.prompts`), direct mode (`--from`, `--to`, `--all`, `--dry-run`, `--allowlist`, `--instructions`, `--skills` flags), sync result formatter (table output for dry-run and summary)
- [ ] Register `sync` command in `src/crossby/cli/main.py`
- [ ] Add tests: `tests/unit/test_config/test_linker.py`, `tests/unit/test_config/test_instructions.py`, `tests/unit/test_config/test_skills_reader.py`, `tests/unit/test_config/test_allowlist_reader.py`, `tests/unit/test_services/test_sync.py`, `tests/unit/test_cli/test_sync.py`

## Acceptance Criteria

- [ ] `crossby sync --from claude --to cursor` creates `.cursorrules → CLAUDE.md`, `.cursor/skills → .claude/skills`, and writes converted allowlist patterns to `.cursor/cli.json`
- [ ] `crossby sync --from claude --all` syncs to all installed tools (detected via `AbstractAITool.detect_installed()`)
- [ ] `crossby sync` (no args, TTY) launches wizard with source/target selection, sync plan preview, and confirmation prompt
- [ ] Wizard prompts before overwriting existing regular files; direct mode overwrites silently
- [ ] Running sync twice is idempotent — second run is all no-ops
- [ ] `crossby sync --dry-run` prints the sync plan without creating any files or modifying configs
- [ ] Copilot/Gemini allowlist targets produce informative warnings (not errors)
- [ ] Missing source configs (e.g., no CLAUDE.md, no allowlist) skip gracefully with info message
- [ ] VSCode/OpenCode/Antigravity as targets produce `UNSUPPORTED` warning
- [ ] All new functions have unit tests passing

URL: https://github.com/ivanviragine/crossby/issues/2
