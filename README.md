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

## Development

```bash
pip install -e ".[dev]"
./scripts/test.sh       # Run tests
./scripts/check.sh      # Lint + type check
./scripts/fmt.sh        # Auto-format
```

## License

MIT
