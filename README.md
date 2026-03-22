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

Crossby translates its unified CLI flags into each tool's native syntax. The table below shows what each `crossby launch` flag maps to.

### Feature Support

| Crossby Flag | Claude | Copilot | Gemini | Codex | OpenCode | Cursor | VS Code | Antigravity |
|---|---|---|---|---|---|---|---|---|
| `--tool` | claude | copilot | gemini | codex | opencode | agent | code | antigravity |
| `--model` | `--model` | `--model` | `--model` | `--model` | `--model` | `--model` | — | — |
| `--yolo` | `--dangerously-skip-permissions` | `--yolo` | `--yolo` | `--yolo` | — | `--force` | — | — |
| `--effort` | `--effort <level>` | — | — | `-c model_reasoning_effort="<level>"` | `--variant <level>` | model suffix (`-thinking`) | — | — |
| `--prompt` | positional arg | `-i <prompt>` | positional arg | positional arg | `--prompt <prompt>` | positional arg | — | — |
| `--transcript` | `script` wrapper | `script` wrapper | `script` wrapper | `script` wrapper | `script` wrapper | `script` wrapper | — | — |

### Effort Level Mapping

| Crossby Level | Claude | Codex | OpenCode | Cursor |
|---|---|---|---|---|
| `low` | `low` | `low` | `low` | — |
| `medium` | `medium` | `medium` | `medium` | — |
| `high` | `high` | `high` | `high` | `<model>-thinking` |
| `max` | `max` | `xhigh` | `high` | `<model>-thinking` |

### Library API (used by WADE, not crossby CLI)

| Feature | Claude | Copilot | Gemini | Codex | OpenCode | Cursor |
|---|---|---|---|---|---|---|
| Plan mode | `--permission-mode plan` | — | `--approval-mode plan` | — | — | `--mode plan` |
| Plan dir | `--add-dir` | `--add-dir` | `--include-directories` | `--add-dir` | — | — |
| Allowed commands | `--allowedTools Bash(cmd:args)` | `--allow-tool shell(cmd:args)` | `--allowed-tools shell(cmd:args)` | — | — | — |
| Resume | `claude --resume <id>` | `copilot --resume=<id>` | `gemini --resume <id>` | `codex resume <id>` | `opencode -s <id>` | — |
| Structured output | `--output-format json --json-schema <schema>` | — | `--output-format json` | — | — | — |

A dash (—) means the tool does not support that feature. Crossby will raise an error if you pass an explicit flag that the selected tool doesn't support (e.g. `--yolo` with OpenCode).

## Development

```bash
pip install -e ".[dev]"
./scripts/test.sh       # Run tests
./scripts/check.sh      # Lint + type check
./scripts/fmt.sh        # Auto-format
```

## License

MIT
