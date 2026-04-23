# crossby

**One config. Every AI tool** — sync rules, permissions, MCP servers, hooks, and agents across Claude, Copilot, Gemini, Codex, Cursor, and more.

## See it in action

You've set up Claude with custom instructions, agents, and permissions. Now share it everywhere:

```
$ crossby sync --from claude  # output illustrative — actual output is a Rich table

✓  CLAUDE.md         → .cursorrules              (symlinked)
✓  CLAUDE.md         → GEMINI.md                 (symlinked)
✓  CLAUDE.md         → AGENTS.md                 (symlinked)
✓  .claude/settings.json permissions → .cursor/cli.json  (converted)
```

Every tool now shares the same instructions and configs — automatically kept in sync.

## Installation

```bash
pip install crossby
# or
uv tool install crossby
# or
pipx install crossby
```

## Quick Start

```bash
# Sync configs from Claude to all installed tools
crossby sync --from claude

# Sync interactively (wizard mode — review before applying)
crossby sync

# Launch an AI tool with resolved config
crossby launch --tool claude --model claude-sonnet-4.6

# Show resolved configuration
crossby config show

# Parse session transcript for token usage
crossby stats /path/to/transcript.txt

# Convert allowlist patterns between tools
crossby convert "Bash(myapp:*)" --from claude --to cursor

# Hand off the current session from one AI CLI to another
crossby handoff --from claude --to codex
```

## Configuration

Add a `.crossby.yml` to your project root to configure defaults:

```yaml
version: 1
ai:
  default_tool: claude
  default_model: claude-sonnet-4.6
  effort: medium
  commands:
    plan:
      tool: claude
      model: claude-opus-4.6
      effort: high
    implement:
      tool: copilot
models:
  claude:
    easy: claude-haiku-4.5
    medium: claude-sonnet-4.6
    complex: claude-sonnet-4.6
    very_complex: claude-opus-4.6
```

## AI Tool Compatibility

Crossby translates its unified CLI flags into each tool's native syntax. A dash (—) means the tool does not support that feature. Crossby raises an error if you pass an explicit flag that the selected tool doesn't support (e.g. `--yolo` with OpenCode).

### Launch Flags

| Crossby Flag | Claude | Copilot | Gemini | Codex | OpenCode | Cursor | VS Code | Antigravity |
|---|---|---|---|---|---|---|---|---|
| Binary | `claude` | `copilot` | `gemini` | `codex` | `opencode` | `agent` | `code` | `antigravity` |
| `--model` | `--model` | `--model` | `--model` | `--model` | `--model` | `--model` | — | — |
| `--yolo` | `--dangerously-skip-permissions` | `--yolo` | `--yolo` | `--yolo` | — | `--force` | — | — |
| `--effort` | `--effort <level>` | — | — | `-c model_reasoning_effort="…"` | `--variant <level>` | model suffix (`-thinking`) | — | — |
| `--prompt` | positional arg | `-i <prompt>` | positional arg | positional arg | `--prompt <prompt>` | positional arg | — | — |
| `--transcript` | `script` wrapper | `script` wrapper | `script` wrapper | `script` wrapper | `script` wrapper | `script` wrapper | — | — |
| `--resume` | `--resume <id>` | `--resume=<id>` | `--resume <id>` | `codex resume <id>` (subcommand) | `-s <id>` | — | — | — |
| `--trusted-dir` | `--add-dir` | `--add-dir` | `--include-directories` | `--sandbox workspace-write --add-dir` | — | — | — | — |

### Effort Level Mapping

| Crossby Level | Claude | Codex | OpenCode | Cursor |
|---|---|---|---|---|
| `low` | `low` | `low` | `low` | — |
| `medium` | `medium` | `medium` | `medium` | — |
| `high` | `high` | `high` | `high` | `<model>-thinking` |
| `max` | `max` | `xhigh` | `high` | `<model>-thinking` |

### Permission & Allowlist Configuration

Crossby writes canonical command patterns (e.g. `myapp:*`) into each tool's native config format.

| Feature | Claude | Copilot | Gemini | Cursor |
|---|---|---|---|---|
| Config file | `.claude/settings.json` | `.github/hooks/hooks.json` | `.gemini/settings.json` | `.cursor/cli.json` |
| Allowlist format | `Bash(cmd:args)` | `shell(cmd:args)` | `shell(cmd:args)` | `Shell(cmd:args)` |
| Launch flag | `--allowedTools` | `--allow-tool` | `--allowed-tools` | — (config-file only) |
| Hook config | `hooks.PreToolUse` | `hooks.preToolUse` | `hooks.BeforeTool` | `preToolUse` in hooks.json |
| Hook guard matcher | `Edit\|Write\|NotebookEdit` | `Write\|Delete` | file-write tools | `Write\|Delete` |

Use `crossby convert` to translate patterns between formats:

```bash
crossby convert "Bash(myapp:*)" --from claude --to cursor
crossby convert "myapp:*" --from canonical --to gemini
```

### Config Sync (`crossby sync`)

Sync portable configs between AI tools — no project config required. Reads files directly from their standard locations.

```bash
# Interactive wizard — select source, targets, review plan, approve
crossby sync

# Direct mode — Claude to Cursor
crossby sync --from claude --to cursor

# Claude to all installed tools
crossby sync --from claude

# Preview without applying
crossby sync --from claude --to cursor --dry-run

# Sync only rules concern
crossby sync rules --from claude --to cursor
```

**What gets synced:**

| Config Type | Strategy | Details |
|---|---|---|
| Instructions | Symlink | `CLAUDE.md` / `.cursorrules` / `GEMINI.md` / `AGENTS.md` / `.github/copilot-instructions.md` |
| Agents | Symlink | `.claude/agents/` and equivalent per tool |
| Permissions | Convert | Claude `Bash()` ↔ Cursor `Shell()` format translation |
| Hooks | Write | Tool-native hook schema per target |
| MCP Servers | Merge | Claude `.claude/settings.json` → `mcpServers` and equivalent per-tool MCP config paths |

Before syncing, crossby scans the source tool and shows everything it found — what can be ported and what can't (with reasons).

### Session Preservation & Resume

| Feature | Claude | Copilot | Gemini | Codex | OpenCode | Cursor |
|---|---|---|---|---|---|---|
| Resume command | `claude --resume <id>` | `copilot --resume=<id>` | `gemini --resume <id>` | `codex resume <id>` | `opencode -s <id>` | — |
| Session data path | `~/.claude/projects/` | — | — | — | — | `~/.cursor/projects/` |
| Session data preserved | Yes (worktree → main) | — | — | — | — | Yes (worktree → main) |

Session IDs are extracted from transcripts automatically when `--transcript` is used.

### Transcript Parsing (`crossby stats`)

| Feature | Claude | Copilot | Gemini | Codex |
|---|---|---|---|---|
| Total tokens | Yes | Yes | Yes | Yes |
| Input / output breakdown | Yes | Yes | Yes | Yes |
| Cached tokens | Yes | Yes | Yes | Yes |
| Per-model breakdown | — | Yes | Yes | — |
| Premium requests | — | Yes | — | — |
| Session ID extraction | Yes | Yes | Yes | Yes |

### Library API

These methods are available on each adapter for programmatic use but are not exposed through the crossby CLI.

| Feature | Claude | Copilot | Gemini | Codex | OpenCode | Cursor |
|---|---|---|---|---|---|---|
| Approval mode | `--permission-mode plan` | — | `--approval-mode plan` | — | — | `--mode plan` |
| Trusted dirs | `--add-dir` | `--add-dir` | `--include-directories` | `--add-dir` | — | — |
| Structured output | `--output-format json --json-schema …` | — | `--output-format json` | — | — | — |
| Model format | dashed (`claude-haiku-4-5`) | dotted (`claude-haiku-4.5`) | as-is | as-is | `provider/model` | as-is |

## Session Handoff

Carry context from one AI CLI to another without re-explaining. `crossby handoff` reads the latest session of the source tool, asks an LLM to summarize it into a structured handoff document, writes `.crossby/handoffs/HANDOFF-<timestamp>.md`, and launches the target tool with the file **path** (not content) as the initial prompt — so the handoff fits comfortably under OS argv limits regardless of transcript size.

```bash
# Most common: latest Claude session → Codex
crossby handoff --from claude --to codex

# Write the file but skip the launch (review before switching tools)
crossby handoff --from cursor --to copilot --no-launch

# Pick a specific session by id instead of the most recent one
crossby handoff --from codex --to claude --session-id 019cb497-ec14-7453-9224
```

### Flags

| Flag | Description |
|---|---|
| `--from` | Source tool to read the session from (claude, cursor, codex, copilot). |
| `--to` | Target tool to launch with the handoff as its initial prompt. |
| `--session-id` | Override the latest-session heuristic with a specific session id. |
| `--output` | Write the handoff file to this path instead of `.crossby/handoffs/HANDOFF-<timestamp>.md`. |
| `--no-launch` | Write the handoff file but do not launch the target. |
| `--summarizer-tool` | Tool to run the summarization pass. Defaults to the source tool. |
| `--token-budget` | Approximate token budget for the transcript before truncation (default `32000`). |
| `--path` | Project root directory (default: current directory). |

Run without `--from`/`--to` to use the interactive wizard, which lists installed tools and lets you pick.

### Supported sources

| Tool | Source (read) | Target (launch) |
|---|---|---|
| Claude | ✓ (`~/.claude/projects/<encoded>/<id>.jsonl`) | ✓ |
| Cursor | ✓ (`~/.cursor/projects/<encoded>/chat.json`) | ✓ |
| Codex | ✓ (`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`) | ✓ |
| Copilot | ✓ (`~/.copilot/session-state/<id>/events.jsonl`) | ✓ |
| Gemini, OpenCode, Antigravity, VS Code | — (not yet supported) | ✓ |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, architecture overview, and how to submit changes.

## License

MIT
