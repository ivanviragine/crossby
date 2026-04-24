# Pre-launch manual test plan

Run through each feature below before going public. Start in a scratch project (or worktree) where you can freely create/overwrite files.

```bash
# Prep: install from local source so you test the current code, not an old PyPI build
uv tool install --from . crossby --force
crossby --version   # prints crossby.__version__ (currently 0.2.3); must be kept in sync with pyproject.toml on every release
crossby --help      # should list: launch, sync, convert, stats, handoff
```

---

## 1. `crossby sync`

**TLDR.** Reads one AI tool's config (rules, agents, skills, permissions, hooks, MCP servers) and writes it into every other installed tool's native format — symlinks for files/dirs, JSON merges for config, pattern translation for allowlists.

### 1a. Interactive wizard

- [ ] `cd` into a project that already has at least one tool configured (e.g. a `CLAUDE.md` and `.claude/agents/`)
- [ ] Run `crossby sync` (no flags)
- [ ] Wizard should detect installed tools, find your source tool, show a scan summary, and ask per-concern whether to port
- [ ] Accept the defaults and confirm the plan
- [ ] Verify: target files created, `.gitignore` updated with a `crossby` managed block

### 1b. Direct mode

- [ ] `crossby sync --from claude` — should port everything Claude has to every other installed tool without prompting
- [ ] `crossby sync --from codex` — should work just as well if you have an `AGENTS.md` and `.agents/` setup
- [ ] `crossby sync --from claude --to cursor` — narrows the target to a single tool
- [ ] `crossby sync rules --from claude` — runs only the rules concern (also try `agents`, `skills`, `permissions`, `hooks`, `mcp`)

### 1c. Dry-run and idempotency

- [ ] `crossby sync --from claude --dry-run` — should print the plan without touching the filesystem; verify no new files appeared
- [ ] Run the same `crossby sync --from claude` twice — the second run should produce only `skipped` / `noop` results (nothing re-written)

### 1d. Safety

- [ ] Create a non-symlink `.cursorrules` with custom content, then run `crossby sync --from claude` — it should refuse to overwrite without `--force`
- [ ] Rerun with `--force` — verify the original is backed up (look for `.bak` or similar) and the symlink is in place

---

## 2. `crossby launch`

**TLDR.** Launches an AI CLI with unified flags (`--model`, `--effort`, `--yolo`, `--resume`, `--trusted-dir`) that crossby translates into each tool's native syntax. Supports saved profiles in `.crossby.yml`.

### 2a. Basic launch

- [ ] `crossby launch --tool claude` — should start Claude Code in the current dir
- [ ] `crossby launch --tool claude --model claude-sonnet-4.6` — verify the model shown at startup
- [ ] `crossby launch --tool claude --effort high` — verify effort shown
- [ ] `crossby launch --tool claude --yolo` — verify YOLO mode shown
- [ ] `crossby launch --tool claude --model claude-sonnet-4.6 --effort high --yolo` — all three at once

### 2b. Profiles

- [ ] Add to `.crossby.yml`:
      ```yaml
      profiles:
        ccyolo:
          tool: claude
          model: claude-sonnet-4.6
          effort: high
          yolo: true
      ```
- [ ] `crossby launch ccyolo` — positional shortcut should launch with the profile's values
- [ ] `crossby launch --profile ccyolo` — same result via explicit flag
- [ ] `crossby launch ccyolo --effort low` — explicit CLI flag should override the profile's `effort: high`
- [ ] `crossby launch nonexistent` — should error with "Unknown profile"

### 2c. Cross-tool launch

- [ ] `crossby launch --tool codex` — should start Codex
- [ ] `crossby launch --tool cursor --yolo` — verify `--yolo` is translated (Cursor uses `--force`)
- [ ] `crossby launch --tool opencode --yolo` — OpenCode doesn't support YOLO; verify you get a warning (or an error — check current behavior)

### 2d. Plan mode

- [ ] `crossby launch --tool claude --plan` — verify "Plan mode: on" shows at startup and Claude starts in plan mode (`--permission-mode plan` under the hood)
- [ ] `crossby launch --tool copilot --plan` — translated to `--plan` (Copilot's native flag, GA'd Jan 2026)
- [ ] `crossby launch --tool gemini --plan` — translated to `--approval-mode plan`
- [ ] `crossby launch --tool cursor --plan` — translated to `--mode plan`
- [ ] `crossby launch --tool codex --plan` — Codex doesn't support plan mode (its `-s read-only` is a sandbox, not a planning conversation); verify clean error "Codex does not support --plan."
- [ ] `crossby launch --tool opencode --plan` — OpenCode's plan mode is a TUI toggle, not a CLI flag; verify clean error
- [ ] `crossby launch --tool claude --plan --yolo` — YOLO should supersede plan; verify warning or clear precedence behavior

### 2d. Resume

- [ ] Find a real Claude session ID (`ls ~/.claude/projects/<encoded>/`)
- [ ] `crossby launch --tool claude --resume <id>` — should resume that session
- [ ] `crossby launch --tool gemini --resume anything` — Gemini supports resume? If not, verify clean error

---

## 3. `crossby handoff`

**TLDR.** Reads the latest session from source tool (or a specific one via `--session-id`), asks an LLM to summarize it, writes `.crossby/handoffs/HANDOFF-<timestamp>.md`, and launches the target tool with the file path as its initial prompt.

### 3a. Default flow

- [ ] Have at least one recent Claude session in this project (`~/.claude/projects/<encoded>/*.jsonl` should exist)
- [ ] `crossby handoff --from claude --to codex` — verify:
  - [ ] A summary is generated (watch for the LLM call)
  - [ ] `.crossby/handoffs/HANDOFF-<timestamp>.md` is created and readable
  - [ ] Codex launches with the handoff file path as the opening prompt (not the contents)

### 3b. Variants

- [ ] `crossby handoff --from claude --to codex --no-launch` — file written, no launch
- [ ] Pick a specific session ID from `~/.claude/projects/<encoded>/` and run `crossby handoff --from claude --to codex --session-id <id>` — verify that session (not the latest) is summarized
- [ ] `crossby handoff --from claude --to codex --session-id bogus123` — should error "No claude session with id…"

### 3c. Cross-source

- [ ] `crossby handoff --from cursor --to claude` (needs a Cursor session in `~/.cursor/projects/`)
- [ ] `crossby handoff --from codex --to claude` (needs a Codex session in `~/.codex/sessions/`)
- [ ] `crossby handoff --from copilot --to claude` (needs a Copilot session in `~/.copilot/session-state/`)
- [ ] `crossby handoff --from gemini --to claude` — Gemini isn't a supported source; verify clean error

---

## 4. `crossby convert`

**TLDR.** Translates a single allowlist pattern between tools' formats. Standalone utility — sync handles bulk translation automatically.

- [ ] `crossby convert "Bash(myapp:*)" --from claude --to cursor` — should print `Shell(myapp:*)`
- [ ] `crossby convert "Shell(git:*)" --from cursor --to claude` — reverse direction
- [ ] `crossby convert "myapp:*" --from canonical --to gemini` — canonical form input

---

## 5. `crossby stats`

**TLDR.** Parses a saved session transcript for token usage (total, input/output, cached, session ID, and — where supported — per-model breakdown and premium-request counts).

- [ ] Capture a transcript: `crossby launch --tool claude --transcript /tmp/claude.txt` and exit after a few turns
- [ ] `crossby stats /tmp/claude.txt` — verify total tokens, input/output split, cached tokens, session ID
- [ ] Repeat with `--tool copilot`, `--tool gemini`, `--tool codex` — verify per-model breakdown for Copilot/Gemini, premium requests for Copilot

---

## 6. Console output sanity

- [ ] `crossby launch --tool copilot --model claude-opus-4.7 --yolo` — the "AI tool / Model / YOLO mode" block should print **once** (with the display name "GitHub Copilot"), not twice. Regression test for a prior double-print.
- [ ] `crossby launch --tool claude` (no other flags, in a TTY) — confirmation menu should show the selection, then after "Proceed" the launch.py block prints the final state once.

## 7. Edge cases and failure modes

- [ ] Run any command in a directory with a malformed `.crossby.yml` (e.g. unclosed string) — verify a clean error, not a traceback
- [ ] `crossby sync` in an empty directory with no AI tool configs — should say there's nothing to sync, not crash
- [ ] `crossby launch --tool claude` when `claude` is not on PATH — should error with an install hint, not crash
- [ ] `crossby sync --from claude` when Claude's `.claude/settings.json` contains an MCP server without `tomli-w` installed (for Codex target) — verify behavior matches the known issue (either clean skip or clear error)

---

## 8. Packaging / install sanity

- [ ] `uv build` — wheel and sdist build without errors
- [ ] `pip install dist/crossby-*.whl` in a fresh venv — `crossby --help` works
- [ ] `pipx install crossby --force` from local `dist/` — same check
- [ ] Confirm `LICENSE` is present at repo root
- [ ] `crossby --version` prints the right version
