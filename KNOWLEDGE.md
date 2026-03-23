# Project Knowledge

Shared learnings from AI planning and implementation sessions.
Read this at the start of every session. Add new entries via `wade knowledge add`.

---

## Config Sync: Tool Config Landscape

### Portable config types (crossby can sync today)

| Config Type | Claude | Cursor | Copilot | Gemini | Codex |
|---|---|---|---|---|---|
| Instructions | `CLAUDE.md` | `.cursorrules` | `.github/copilot-instructions.md` | `GEMINI.md` | `AGENTS.md` |
| Skills dir | `.claude/skills/` | `.cursor/skills/` | `.github/skills/` | `.gemini/skills/` | `.agents/skills/` |
| Allowlist file | `.claude/settings.json` | `.cursor/cli.json` | -- (flag only) | -- (flag only) | -- (sandbox) |
| Allowlist format | `Bash(cmd:args)` | `Shell(cmd:args)` | `--allow-tool` flag | `--allowed-tools` flag | sandbox mode |

- **Instructions**: Symlinked across tools. All tools read a single markdown file.
- **Skills**: Symlinked as a directory. Each skill is a subdirectory with `SKILL.md`.
- **Allowlist**: Converted between Claude's `Bash()` and Cursor's `Shell()` wrapper formats via canonical `cmd:args` patterns. Copilot/Gemini/Codex don't have persistent allowlist config files.

### Non-portable config types (detected but not synced)

| Config Type | Claude | Cursor | Copilot | Gemini |
|---|---|---|---|---|
| Hooks | `settings.json` `hooks.PreToolUse` | `.cursor/hooks.json` | `.github/hooks/hooks.json` | `.gemini/settings.json` `hooks[]` |
| MCP servers | `settings.json` `mcpServers{}` | -- | -- | -- |
| Custom commands | `.claude/commands/*.md` | -- | -- | -- |

- **Hooks**: Every tool uses a different schema. Claude nests hooks under event keys with matcher arrays, Cursor uses a separate JSON file, Copilot nests under `.github/hooks/`, Gemini uses a flat array. Too different to auto-convert today.
- **MCP servers**: Claude-specific. No cross-tool equivalent yet.
- **Custom commands**: Claude-specific (slash commands). No cross-tool equivalent yet.

### Unsupported tools

VS Code, OpenCode, and Antigravity have no instruction/skills/allowlist config format that crossby can read. Sync emits an `UNSUPPORTED` warning for these.

### Sync strategies

- **LINK**: Create a relative symlink from target to source (instructions, skills).
- **CONVERT**: Read source format, translate patterns, write target format (allowlists).
- **WARN**: Target exists or config can't be ported — inform user with reason.
- **UNSUPPORTED**: Target tool has no equivalent config concept.

### Design decisions

- `crossby sync` does NOT depend on `crossby init` or `.crossby.yml`. It reads files directly from their standard locations.
- Wizard mode (no args) uses interactive prompts with per-conflict overwrite approval.
- Direct mode (`--from claude --to cursor`) overwrites silently (force=True).
- Detection runs before sync to show users exactly what was found, what can be ported, and what can't (with reasons).
- Symlinks are always relative (`os.path.relpath`) so they survive repo moves.
- Idempotent: re-running sync on already-linked files is a no-op.

---

## General: Testing

- pytest is a dev dependency; run tests with `uv run --with pytest --with pytest-cov pytest`.
- CLI tests that use `Path.cwd()` must `monkeypatch.chdir(tmp_path)` to avoid testing against real filesystem.

---
