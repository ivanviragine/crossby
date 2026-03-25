# CROSSBY

**Cross-platform Bridge for Your AI agents** — a Python CLI toolkit for managing AI coding tools across platforms.

CROSSBY provides:

- **AI Tool Adapters**: Unified interface for Claude, Copilot, Gemini, Codex, Cursor, OpenCode, and more
- **Configuration Resolution**: Read, merge, and translate config hierarchies across AI tools
- **Agent Launching**: Universal launcher with model selection, effort levels, and YOLO mode
- **Session Stats**: Token usage extraction from session transcripts
- **Permission Management**: Cross-tool allowlist and hook configuration

## Installation

```bash
pip install crossby
```

## Quick Start

```bash
# Sync .crossby.yml into installed tools
crossby sync

# Preview sync changes
crossby sync --dry-run

# Sync only permissions for Claude
crossby sync permissions --tool claude

# Initialize config in your project
crossby init

# Launch an AI tool with resolved config
crossby launch --tool claude --model claude-sonnet-4.6

# Show resolved configuration
crossby config show

# Parse session transcript for token usage
crossby stats /path/to/transcript.txt

# Convert allowlist patterns between tools
crossby convert "Bash(myapp:*)" --from claude --to cursor
```

## Configuration

CROSSBY reads `.crossby.yml` from your project root:

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
permissions:
  allowed_commands:
    - "myapp:*"
    - "./scripts/check.sh:*"
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

### Session Preservation & Resume

| Feature | Claude | Copilot | Gemini | Codex | OpenCode | Cursor |
|---|---|---|---|---|---|---|
| Resume command | `claude --resume <id>` | `copilot --resume=<id>` | `gemini --resume <id>` | `codex resume <id>` | `opencode -s <id>` | — |
| Session data path | `~/.claude/projects/` | — | — | — | — | `~/.cursor/projects/` |
| Session data preserved | Yes (worktree → main) | — | — | — | — | Yes (worktree → main) |

Session IDs are extracted from transcripts automatically when `--transcript` is used.

### Config Sync (`crossby sync`)

Syncs the concerns declared in `.crossby.yml` into tool-specific files.
`crossby sync` uses your project config as the source of truth; it no longer
ports files directly from one AI tool to another.

```bash
# All concerns for all installed tools
crossby sync

# Permissions only
crossby sync permissions

# All concerns for one tool
crossby sync --tool claude

# Preview without applying
crossby sync --dry-run

# Start from a nested path; crossby writes to the discovered project root
crossby sync --path ./src/feature
```

**What gets synced:**

| Concern | Strategy | Details |
|---|---|---|
| Permissions | Merge | Writes canonical `allowed_commands` into `.claude/settings.json` and `.cursor/cli.json` |
| Rules | Symlink or copy | Distributes a canonical rules file such as `AGENTS.md` into tool-specific instruction files |
| Agents | Symlink or copy | Distributes `.crossby/agents/` into each tool's agent directory |

**Concern filters and scope:**

- Use positional concerns such as `permissions`, `rules`, or `agents` to sync only one concern.
- Use `--tool` to target one installed tool explicitly.
- Use `sync.tools` in `.crossby.yml` to restrict the default multi-tool sync lane.
- `--dry-run` previews the exact files that would be created or updated without writing them.

`mcp` is reserved as a sync concern, but there are no MCP writers yet, so filtering to it currently produces no matching writers.

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

## Development

```bash
pip install -e ".[dev]"
./scripts/test.sh       # Unit + non-E2E deterministic tests
./scripts/test-e2e.sh   # Deterministic subprocess workflow tests
./scripts/test-live-ai.sh  # Opt-in real AI tool smoke tests
./scripts/test-live-probe.sh  # Opt-in installed CLI probe lane
./scripts/check.sh      # Lint + type check
./scripts/fmt.sh        # Auto-format
./scripts/probe_models.sh  # Probe installed AI CLIs and model registry drift
```

Live AI tests are opt-in and env-gated. The default `./scripts/test.sh` lane excludes
`e2e_deterministic`, `live_ai`, and `live_probe` markers so CI and local fast paths stay
deterministic, while `./scripts/test-e2e.sh` runs the subprocess contract lane separately.
When running `./scripts/test-live-ai.sh`, either set `CROSSBY_LIVE_AI_TOOLS` explicitly
or export one or more `CROSSBY_LIVE_MODEL_<TOOL>` variables so the runner can infer the
selected tools. The smoke run uses an isolated temporary Git working directory rather
than the current repo checkout. `./scripts/test-live-probe.sh` is a separate manual lane
for probing installed CLIs against CROSSBY's bundled model registry and expected help
flags, with optional selection via `CROSSBY_LIVE_PROBE_TOOLS`.

## License

MIT
