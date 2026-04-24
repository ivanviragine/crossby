# crossby

**One config. Every AI tool.**

Stop re-writing your rules, permissions, and agents for every CLI. `crossby` syncs them across Claude, Copilot, Gemini, Codex, Cursor, OpenCode, VS Code, and Antigravity — and lets you hand off a live session from one tool to another without losing context.

```
$ crossby sync --from codex

✓  rules         AGENTS.md         →  CLAUDE.md, GEMINI.md, .cursorrules, +1 more
✓  agents        .agents/          →  .claude/agents/, .cursor/agents/, +2 more
✓  skills        .agents/skills/   →  .claude/skills/, .cursor/skills/, +2 more
✓  permissions                     →  translated for Claude, Cursor, Gemini
✓  hooks                           →  written for Claude, Cursor, Copilot, Gemini
✓  mcp servers                     →  merged into Claude, Cursor, Codex, Copilot, Gemini
```

Already on Claude? `crossby sync --from claude` works the same way — any tool can be the source.

---

## Why crossby?

- **Every new tool inherits your setup.** Install a new AI CLI tomorrow and one `crossby sync` gives it your rules, agents, permissions, hooks, and MCP servers — translated into whatever format that tool expects.
- **Pick any tool as your source.** `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.cursorrules`, Copilot instructions — whatever you already write in becomes canonical. No migration, no lock-in.
- **One set of launch flags for every tool.** `--model`, `--effort`, `--yolo`, `--plan`, `--resume` — crossby handles the per-tool translation. Unsupported flags raise errors instead of silently disappearing.
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

# Sync your setup to every installed tool (replace codex with claude, cursor, gemini, copilot…)
crossby sync --from codex

# Or launch the interactive wizard
crossby sync

# Launch a saved profile (e.g. Claude + Sonnet + high effort + YOLO — see below)
crossby launch ccyolo

# …or spell it out with unified flags
crossby launch --tool claude --model claude-sonnet-4.6 --effort high --yolo

# Hand the current session off to another tool
crossby handoff --from claude --to codex

# Parse a session transcript for token usage
crossby stats /path/to/transcript.txt
```

Every command with missing arguments drops into a "Proceed / Change X" review so you can accept the resolved defaults with one keystroke or tweak any single value before it runs.

## What gets synced

| Config      | Strategy | Notes                                                                                        |
| ----------- | -------- | -------------------------------------------------------------------------------------------- |
| Rules       | Symlink  | `AGENTS.md` ↔ `CLAUDE.md` ↔ `GEMINI.md` ↔ `.cursorrules` ↔ `.github/copilot-instructions.md` |
| Agents      | Symlink  | Each tool's agents directory ↔ equivalent per tool                                           |
| Skills      | Symlink  | `.claude/skills/` ↔ `.agents/skills/` ↔ `.gemini/skills/` ↔ `.cursor/skills/` ↔ `.github/skills/` |
| Permissions | Convert  | Canonical ↔ `Bash()` / `Shell()` / `shell()` format per tool                                 |
| Hooks       | Write    | Per-tool native hook schema                                                                  |
| MCP servers | Merge    | Source tool's MCP config → each target tool's MCP config                                     |

Before writing anything, `crossby sync` scans the source tool and shows a plan — what it can port, what it can't, and why. Use `--dry-run` to preview without applying.

> Need to translate a single allowlist pattern by hand (e.g. while editing a config file)? `crossby convert "Bash(myapp:*)" --from claude --to cursor` prints the equivalent pattern for the target tool.

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

Supported sources: Claude, Cursor, Codex, Copilot. Supported targets: all of the above plus Gemini, OpenCode, Antigravity, VS Code.

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

Profiles are just named bundles of `--tool` / `--model` / `--effort` / `--yolo`. Run them by name (`crossby launch ccyolo`) or with `--profile ccyolo`. Explicit flags on the command line still override the profile.

`sync_defaults` and `handoff_defaults` feed the interactive prompts for those commands — CLI flags still win, and you always get the "Proceed / Change X" review before anything runs. `crossby sync` does **not** require this file — it reads directly from each tool's standard paths.

## Supported tools

| Tool        | Sync | Launch | Handoff (source) | Handoff (target) |
| ----------- | ---- | ------ | ---------------- | ---------------- |
| Claude      | ✓    | ✓      | ✓                | ✓                |
| Copilot     | ✓    | ✓      | ✓                | ✓                |
| Gemini      | ✓    | ✓      | —                | ✓                |
| Codex       | ✓    | ✓      | ✓                | ✓                |
| Cursor      | ✓    | ✓      | ✓                | ✓                |
| OpenCode    | ✓    | ✓      | —                | ✓                |
| VS Code     | ✓    | ✓      | —                | ✓                |
| Antigravity | ✓    | ✓      | —                | ✓                |

Per-tool flag mappings and adapter details live in [CONTRIBUTING.md](CONTRIBUTING.md#tool-reference).

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) — architecture, how to add a new tool, per-tool flag reference, release process.

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and architecture.

## License

MIT
