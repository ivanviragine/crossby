# Issue #11: feat: custom agents sync across AI tools

## Complexity
complex

## Dependencies
Depends on Plan 1 (sync framework) for `AbstractSyncWriter`, `SyncResult`,
`SyncConcern`, `SyncRegistry`, and `cli/sync.py`.
Depends on Plan 2 (rules sync) for the `.gitignore` managed-block utility
(reuse, don't reimplement).

## Context / Problem

Custom agents (sub-agents) are emerging as the primary customization
abstraction for AI coding tools. All major tools now support them, but each
uses a different directory, file extension, and frontmatter schema:

| Tool | Directory | Extension | Key Frontmatter |
|------|-----------|-----------|----------------|
| Claude | `.claude/agents/` | `.md` | `name`, `description`, `tools`, `model`, `memory`, `skills` |
| Copilot | `.github/agents/` | `.agent.md` | `name`, `description`, `tools` |
| Cursor | `.cursor/agents/` | `.md` | Similar to Claude; also natively reads `.claude/agents/` |
| Gemini | `.gemini/agents/` | `.md` | Similar to Claude |
| Codex | `.agents/` | `.md` | Similar format |

The formats are converging around YAML frontmatter + Markdown body, but the
differences in directory paths, file extensions, and supported frontmatter
fields make cross-tool agent sharing manual.

Crossby currently does **not** manage custom agents. The only related pattern
in the repo is manual symlinks for skills directories.

**Out of scope:** OpenCode and VS Code. Neither has a documented custom agent
directory format. They are excluded and can be added later.

**Note on Cursor:** Cursor natively reads `.claude/agents/` in addition to its
own `.cursor/agents/`. If Claude is also a sync target, Cursor users get agents
automatically. The `CursorAgentsWriter` is still provided for completeness
(users who want explicit `.cursor/agents/` control, or who don't use Claude),
but the plan documents this overlap so implementers and users are aware.

## Proposed Solution

Add agent sync support that distributes agent definitions from a canonical
source directory to each tool's native agent directory.

### 1. Configuration

```yaml
agents:
  source: .crossby/agents       # canonical agent directory (default)
  strategy: symlink              # "symlink" | "copy" (default: symlink)
  gitignore: true                # manage .gitignore entries (default: true)
  targets:                       # which tools to sync to (default: all installed)
    claude: true                 # -> .claude/agents/
    copilot: true                # -> .github/agents/
    cursor: true                 # -> .cursor/agents/
    gemini: true                 # -> .gemini/agents/
    codex: true                  # -> .agents/
```

### 2. Canonical agent format

Agents are stored as `.md` files in the source directory with YAML frontmatter:

```markdown
---
name: code-reviewer
description: Reviews code for quality, security, and best practices
tools:
  - Read
  - Glob
  - Grep
model: sonnet
---

You are a code reviewer. Analyze code changes for...
```

Supported frontmatter fields (superset of all tools):
- `name` (string, required): Agent identifier
- `description` (string, required): What the agent does (used for auto-delegation)
- `tools` (list[str]): Available tools (canonical names)
- `model` (string): Model hint — passed through as-is to all tools. Each tool
  interprets model hints in its own way; crossby does not translate these.
- `memory` (string): Persistent memory directory path (Claude-specific, ignored by others)
- `skills` (list[str]): Skills to inject at startup (Claude-specific, ignored by others)

### 3. Sync strategies

**Symlink (default):** Create per-tool agent directories as symlinks to the
source directory. Mirrors the existing manual pattern for skills.

```
.claude/agents/  -> ../../.crossby/agents/
.github/agents/  -> ../../.crossby/agents/
.cursor/agents/  -> ../../.crossby/agents/
.gemini/agents/  -> ../../.crossby/agents/
.agents/         -> ../.crossby/agents/
```

All paths computed via `os.path.relpath()` — not hardcoded.

**Exception — Copilot:** Copilot requires `.agent.md` extension. With symlink
strategy, the Copilot writer creates **file-level symlinks** instead of a
directory symlink:
- `.github/agents/code-reviewer.agent.md` -> `../../.crossby/agents/code-reviewer.md`

The Copilot writer also runs a **stale cleanup pass**: removes `.agent.md`
symlinks in `.github/agents/` whose source `.md` file no longer exists in the
source directory. This prevents dangling symlinks when agents are deleted.

**Copy:** Copy agent files to each tool's directory. For Copilot, rename
`.md` -> `.agent.md` during copy.

**Symlink error fallback:** If symlink creation fails (e.g., Windows), fall
back to copy with a warning.

### 4. Existing target directory handling

When a target directory already exists as a **real directory** (not a symlink):

- **Default behavior**: Error with a clear message: "`.claude/agents/` exists
  as a directory. Migrate its contents to `.crossby/agents/` first, or use
  `--force` to replace it."
- **`--force` flag**: Back up the existing directory to `<dir>.bak/` and
  replace with the symlink/copy.
- **`crossby init` migration**: When discovering existing agent directories,
  offer to move their contents into `.crossby/agents/` and replace with symlinks.

### 5. Tool name mapping (copy strategy only)

When using copy strategy, agent `tools` field values can optionally be
translated per platform. The mapping covers the known differences:

| Canonical | Claude | Copilot | Cursor | Gemini | Codex |
|-----------|--------|---------|--------|--------|-------|
| `Read` | `Read` | `read` | `Read` | `Read` | `Read` |
| `Edit` | `Edit` | `edit` | `Edit` | `Edit` | `Edit` |
| `Grep` | `Grep` | `search` | `Grep` | `Grep` | `Grep` |
| `Glob` | `Glob` | `glob` | `Glob` | `Glob` | `Glob` |
| `Bash` | `Bash` | `shell` | `Shell` | `Bash` | `Bash` |
| `WebSearch` | `WebSearch` | `web_search` | `WebSearch` | `WebSearch` | `WebSearch` |
| `WebFetch` | `WebFetch` | `web_fetch` | `WebFetch` | `WebFetch` | `WebFetch` |

With symlink strategy, tool names are kept as-is (tools silently ignore
unknown names). Translation is only applied during copy.

### 6. Gitignore management

When `agents.gitignore` is `true` (default), crossby manages a block in
`.gitignore` (reusing the managed-block utility from Plan 2):

```
# >>> crossby agents sync (generated — do not edit) >>>
.claude/agents
.github/agents
.cursor/agents
.gemini/agents
.agents
# <<< crossby agents sync <<<
```

The source directory (`.crossby/agents/`) is **not** gitignored.

### 7. CLI integration

```
crossby sync agents                 # sync agents to all installed tools
crossby sync agents --dry-run       # preview changes
crossby sync agents --tool copilot  # sync only to Copilot
crossby sync agents --force         # overwrite existing target directories
```

### 8. Init integration

`crossby init` scans for existing agent directories across installed tools:
- If `.claude/agents/` has `.md` files, offer to copy them to `.crossby/agents/`
- If `.github/agents/` has `.agent.md` files, discover and merge (strip `.agent.` extension)
- Propose `agents:` config section with detected source and targets

## Tasks
- [ ] Add `AgentsConfig` model to `models/config.py` (source, strategy, targets, gitignore)
- [ ] Add `agents` section parsing to config loader
- [ ] Create `sync/agents.py` with base agent sync logic
- [ ] Implement directory symlink strategy (create tool dirs as symlinks via `os.path.relpath()`)
- [ ] Implement Copilot file-level symlink strategy (`.md` -> `.agent.md` symlinks, stale cleanup)
- [ ] Implement copy strategy with Copilot extension renaming
- [ ] Implement tool name mapping for copy strategy (canonical -> tool-specific translation table)
- [ ] Implement `ClaudeAgentsWriter` — target: `.claude/agents/`
- [ ] Implement `CopilotAgentsWriter` — target: `.github/agents/` (file-level symlinks + stale cleanup)
- [ ] Implement `CursorAgentsWriter` — target: `.cursor/agents/`
- [ ] Implement `GeminiAgentsWriter` — target: `.gemini/agents/`
- [ ] Implement `CodexAgentsWriter` — target: `.agents/`
- [ ] Register all agent writers in the sync framework registry
- [ ] Detect pre-existing target directories (error by default, `--force` to replace with backup)
- [ ] Add symlink creation error handling (fall back to copy with warning)
- [ ] Add `.gitignore` managed-block for generated agent directories (reuse Plan 2 utility)
- [ ] Integrate agent directory discovery into `crossby init` (scan, offer migration, write config)
- [ ] Add unit tests: directory symlink creation with correct `os.path.relpath` paths
- [ ] Add unit tests: Copilot file-level symlinks (creation, stale cleanup, extension renaming)
- [ ] Add unit tests: copy strategy with tool name mapping
- [ ] Add unit tests: existing directory detection (error, force with backup)
- [ ] Add unit tests: idempotency (re-running sync produces no changes)
- [ ] Add unit tests: `--dry-run` produces correct output without writing
- [ ] Add integration test: full `crossby sync agents` with multi-tool config

## Acceptance Criteria
- [ ] `crossby sync agents` distributes agents from source dir to all configured tool directories
- [ ] Symlink paths are relative, computed via `os.path.relpath()` (not hardcoded)
- [ ] Copilot gets file-level symlinks with `.agent.md` extension
- [ ] Stale Copilot symlinks (source deleted) are cleaned up on sync
- [ ] Copy strategy translates tool names per-platform
- [ ] Pre-existing real directories are not silently replaced (error with migration hint)
- [ ] `--force` replaces existing directories after creating `.bak` backup
- [ ] Sync is idempotent — running twice produces no changes on the second run
- [ ] Source directory not found produces a clear error message
- [ ] `.gitignore` is updated with generated directory entries
- [ ] Symlink creation failure falls back to copy with a warning
- [ ] `crossby init` discovers existing agent directories and proposes config
- [ ] `--dry-run` reports what would be created without writing

URL: https://github.com/ivanviragine/crossby/issues/11
