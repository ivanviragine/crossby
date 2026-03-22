# Review Fixes Plan

Fixes for all 6 findings from the code review. Ordered by priority.

---

## Fix 1: Explicit flags must fail fast, not silently degrade

**Problem**: When a user passes `--model gpt-4o --tool claude`, the incompatible model is silently dropped to `None`. Same for `--effort high` on tools without effort support, and `--yolo` on tools without yolo support. The resolution service logs at `info` level but the default log level is `ERROR`, so the user sees nothing.

**Root cause**: The resolution functions were designed for WADE's service layer where silent fallback is correct (config-driven, not user-driven). But when flags come from the CLI, the user's explicit intent should be respected.

**Fix**: Add `strict` mode to resolution functions. When the caller knows a value was explicitly provided by the user, incompatibility raises an error instead of silently returning `None`.

### Files to change

**`src/crossby/services/ai_resolution.py`**:

- `resolve_model()` — add `*, strict: bool = False` parameter. When `strict=True` and a model is incompatible with the tool, raise `ValueError(f"Model '{model}' is not compatible with {tool}")` instead of logging and returning `None`.

```python
def resolve_model(
    model: str | None,
    config: CrossbyConfig,
    command: str = "plan",
    *,
    tool: str | None = None,
    complexity: str | None = None,
    strict: bool = False,  # NEW
) -> str | None:
    ...
    # Compatibility gate
    if resolved and tool:
        try:
            adapter = AbstractAITool.get(AIToolID(tool))
            if not adapter.is_model_compatible(resolved):
                if strict:
                    raise ValueError(
                        f"Model '{resolved}' is not compatible with {tool}"
                    )
                logger.info("model.incompatible", model=resolved, tool=tool)
                return None
        except (ValueError, KeyError):
            if strict:
                raise
    return resolved
```

- `resolve_effort()` — add `*, strict: bool = False` parameter. When `strict=True` and the tool doesn't support effort, raise `ValueError`. Same for invalid effort level strings.

```python
def resolve_effort(
    effort: str | None,
    config: CrossbyConfig,
    command: str = "plan",
    *,
    tool: str | None = None,
    strict: bool = False,  # NEW
) -> EffortLevel | None:
    ...
    # Validate
    try:
        level = EffortLevel(resolved)
    except ValueError:
        if strict:
            raise ValueError(f"Invalid effort level: '{resolved}'")
        logger.warning("effort.invalid_level", effort=resolved)
        return None

    # Check tool support
    if tool:
        try:
            adapter = AbstractAITool.get(AIToolID(tool))
            if not adapter.capabilities().supports_effort:
                if strict:
                    raise ValueError(
                        f"{tool} does not support effort levels"
                    )
                logger.info("effort.unsupported_tool", tool=tool, effort=resolved)
                return None
        except (ValueError, KeyError):
            if strict:
                raise
    return level
```

- `resolve_yolo()` — add `*, strict: bool = False` parameter. Same pattern.

```python
def resolve_yolo(
    yolo: bool | None,
    config: CrossbyConfig,
    command: str = "plan",
    *,
    tool: str | None = None,
    strict: bool = False,  # NEW
) -> bool:
    ...
    if tool:
        try:
            adapter = AbstractAITool.get(AIToolID(tool))
            if not adapter.capabilities().supports_yolo:
                if strict:
                    raise ValueError(f"{tool} does not support YOLO mode")
                logger.warning("yolo.unsupported_tool", tool=tool)
                return False
        except (ValueError, KeyError):
            if strict:
                raise
    return True
```

**`src/crossby/cli/launch.py`**:

- Pass `strict=True` when the flag was explicitly provided by the user:

```python
    resolved_model = resolve_model(
        model, config, command or "default",
        tool=resolved_tool, complexity=complexity,
        strict=model is not None,  # fail fast if user explicitly passed --model
    )
    resolved_effort = resolve_effort(
        effort, config, command or "default",
        tool=resolved_tool,
        strict=effort is not None,
    )
    resolved_yolo = resolve_yolo(
        yolo, config, command or "default",
        tool=resolved_tool,
        strict=yolo is not None,
    )
```

- Wrap the resolution block in a try/except to catch `ValueError` and show a clean error:

```python
    try:
        resolved_model = resolve_model(...)
        resolved_effort = resolve_effort(...)
        resolved_yolo = resolve_yolo(...)
    except ValueError as e:
        console.error(str(e))
        raise typer.Exit(1) from e
```

### Tests to add

**`tests/unit/test_services/test_ai_resolution.py`** (NEW):

- `test_resolve_model_strict_incompatible` — `resolve_model("gpt-4o", config, tool="claude", strict=True)` raises `ValueError`
- `test_resolve_model_nonstrict_incompatible` — same but `strict=False` returns `None`
- `test_resolve_effort_strict_unsupported` — `resolve_effort("high", config, tool="copilot", strict=True)` raises `ValueError` (copilot doesn't support effort)
- `test_resolve_effort_strict_invalid_level` — `resolve_effort("ultra", config, strict=True)` raises `ValueError`
- `test_resolve_yolo_strict_unsupported` — `resolve_yolo(True, config, tool="opencode", strict=True)` raises `ValueError`
- `test_resolve_model_strict_compatible` — `resolve_model("claude-sonnet-4.6", config, tool="claude", strict=True)` returns the model (no error)

---

## Fix 2: Unify canonical allowlist format to colon syntax

**Problem**: The documented canonical format uses colon syntax (`myapp:*`), but `copilot.py` and `gemini.py` adapters parse with `cmd.split(None, 1)` (space-splitting), and `convert.py`'s `_to_shell()` splits on `:`. These produce different results for the same input.

**Root cause**: The copilot/gemini `allowed_commands_args()` methods were copied from WADE where the canonical format was ambiguously defined. The colon format was standardized later but these methods weren't updated.

**Fix**: Make all adapters consistently parse the colon format.

### Files to change

**`src/crossby/ai_tools/copilot.py`** — change `allowed_commands_args()`:

```python
    def allowed_commands_args(self, commands: list[str]) -> list[str]:
        """Translate canonical patterns to Copilot --allow-tool flags.

        Canonical ``"cmd:args"`` becomes ``--allow-tool "shell(cmd:args)"``.
        """
        result: list[str] = []
        for cmd in commands:
            # Canonical format uses colon separator: "binary:args"
            parts = cmd.split(":", 1)
            binary = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            pattern = f"shell({binary}:{args})" if args else f"shell({binary})"
            result.extend(["--allow-tool", pattern])
        return result
```

**`src/crossby/ai_tools/gemini.py`** — same change:

```python
    def allowed_commands_args(self, commands: list[str]) -> list[str]:
        """Translate canonical patterns to Gemini --allowed-tools flags.

        Canonical ``"cmd:args"`` becomes ``"shell(cmd:args)"``.
        """
        result: list[str] = []
        for cmd in commands:
            # Canonical format uses colon separator: "binary:args"
            parts = cmd.split(":", 1)
            binary = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            pattern = f"shell({binary}:{args})" if args else f"shell({binary})"
            result.extend(["--allowed-tools", pattern])
        return result
```

**`src/crossby/ai_tools/base.py`** — update docstring at line 239 to be explicit:

```python
    def allowed_commands_args(self, commands: list[str]) -> list[str]:
        """Get CLI args to pre-authorize a list of command patterns.

        Canonical patterns use colon-separated syntax: ``"binary:glob"``.
        Examples: ``"myapp:*"``, ``"./scripts/check.sh:*"``.
        Each adapter translates them into tool-specific flags.

        Default: no support (returns empty list). Override per tool.
        """
```

**`src/crossby/cli/convert.py`** — `_to_shell()` already uses `split(":", 1)` which is correct. No change needed.

**`README.md`** — verify examples use colon format consistently. Current `"Bash(myapp:*)"` is correct.

### Tests to update

**`tests/unit/test_ai_tools/test_allowed_commands.py`** — verify all test patterns use colon format. If any tests pass space-separated patterns, update them to colon format and verify the expected output.

---

## Fix 3: Create parent directory for `--transcript` path

**Problem**: `crossby launch --transcript /tmp/nonexistent/dir/transcript.txt` fails because `script` can't create the file in a missing directory.

**Fix**: Create parent directory before launching.

### Files to change

**`src/crossby/cli/launch.py`** — add parent dir creation before the launch call:

```python
    # Ensure transcript parent directory exists
    if transcript:
        transcript.parent.mkdir(parents=True, exist_ok=True)

    # Launch
    exit_code = adapter.launch(...)
```

### Tests to add

**`tests/unit/test_cli/test_launch.py`** (NEW) — add test that verifies transcript parent dir is created. This can be a unit test that mocks the adapter and checks `mkdir` is called.

---

## Fix 4: Catch `AttributeError` in config loader

**Problem**: `models: [bad]` or `ai: {commands: [bad]}` in `.crossby.yml` causes an `AttributeError` to leak instead of a clean `ConfigError`.

**Fix**: Add `AttributeError` to the exception tuple in `parse_config_file`, and add defensive `isinstance` checks in `_build_config`.

### Files to change

**`src/crossby/config/loader.py`**:

1. Line 89 — add `AttributeError` to the caught exceptions:

```python
    try:
        return _build_config(validated, config_path)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise ConfigError(f"Invalid config structure in {config_path}: {e}") from e
```

2. In `_build_config()` — add defensive checks for the `models` and `commands` sections:

```python
    # Parse ai section
    ai_raw = raw.get("ai", {}) or {}
    if not isinstance(ai_raw, dict):
        raise ConfigError("'ai' must be a mapping")

    commands: dict[str, CommandConfig] = {}
    commands_raw = ai_raw.get("commands", {}) or {}
    if not isinstance(commands_raw, dict):
        raise ConfigError("'ai.commands' must be a mapping")
    for cmd_name, cmd_raw in commands_raw.items():
        commands[cmd_name] = _parse_command_config(cmd_raw)

    ...

    # Parse models section
    models_raw = raw.get("models", {}) or {}
    if not isinstance(models_raw, dict):
        raise ConfigError("'models' must be a mapping")
```

### Tests to add

**`tests/unit/test_config/test_loader.py`** — add tests:

- `test_models_as_list_raises` — write `models: [bad]` to config, verify `ConfigError`
- `test_commands_as_list_raises` — write `ai: {commands: [bad]}` to config, verify `ConfigError`
- `test_ai_as_list_raises` — write `ai: [bad]` to config, verify `ConfigError`

---

## Fix 5: Fix misleading model discovery docs

**Problem**: `AIModel` docstring says "discovered at runtime by querying the tool" but the implementation reads from a bundled static JSON file. The `crossby init` help says "queries their available models" which overstates what happens.

**Fix**: Update docstrings to accurately describe the static registry.

### Files to change

**`src/crossby/models/ai.py`** — update `AIModel` docstring:

```python
class AIModel(BaseModel, frozen=True):
    """A concrete model available through an AI tool.

    Models come from a bundled static registry (``data/models.json``).
    The model ID format matches what each tool's CLI accepts.
    """
```

**`src/crossby/ai_tools/base.py`** — update `get_models()` docstring:

```python
    def get_models(self) -> list[AIModel]:
        """Return known models from the bundled static registry.

        Uses universal tier classification. Override for tools with
        special model ID formats (e.g. OpenCode's provider/model).
        Returns an empty list if no models are registered for this tool.
        """
```

**`src/crossby/data/__init__.py`** — update module docstring:

```python
"""Bundled model registry — reads models.json, the static list of known model IDs."""
```

**`src/crossby/cli/init.py`** — update docstring:

```python
def init(...) -> None:
    """Initialize CROSSBY config in a project.

    Detects installed AI tools, reads their known models from the
    bundled registry, and generates a .crossby.yml config file.
    """
```

And update the step message:

```python
    console.step("Detecting installed AI tools...")
```

(This one is already correct — no "querying" language.)

---

## Fix 6: Remove speculative unused surface area

**Problem**: `CommandConfig.mode` and `CommandConfig.enabled` are not used by any CLI command or service. The `mode` field is never read from config, and `enabled` is never checked. The loader parses them, but nothing consumes them. The headless/structured-output branches in `build_launch_command()` and the terminal launcher are library API for WADE but unused by Crossby's own CLI.

**Fix**: Remove unused fields from `CommandConfig`. Keep the library API surface (headless, structured output, terminal launcher) since those are explicitly for WADE consumption, but document them as library-only.

### Files to change

**`src/crossby/models/config.py`** — remove `mode` and `enabled` from `CommandConfig`:

```python
class CommandConfig(BaseModel):
    """Per-command AI tool and model override."""

    tool: str | None = None
    model: str | None = None
    effort: str | None = None
    yolo: bool | None = None
```

**`src/crossby/config/loader.py`** — remove `mode` and `enabled` from `_parse_command_config()`:

```python
def _parse_command_config(raw: dict[str, Any] | None) -> CommandConfig:
    """Parse a per-command AI config section."""
    if not raw or not isinstance(raw, dict):
        return CommandConfig()
    return CommandConfig(
        tool=raw.get("tool"),
        model=raw.get("model") or None,
        effort=raw.get("effort"),
        yolo=raw.get("yolo"),
    )
```

**`src/crossby/ai_tools/base.py`** — add a comment block above the library-only methods:

```python
    # ------------------------------------------------------------------
    # Library API — used by consumers (e.g. WADE), not by crossby's CLI
    # ------------------------------------------------------------------

    def plan_mode_args(self) -> list[str]:
        ...
```

### Tests to update

- Update any tests that reference `CommandConfig(mode=...)` or `CommandConfig(enabled=...)`.

---

## Bonus: Deduplicate session preservation code

**Not in the review's priority list, but flagged for "worth discussing next".**

The Claude and Cursor `preserve_session_data()` methods are near-identical (same copy logic, different path encoding). This can be refactored into a shared helper in `base.py`:

```python
# In base.py
def _preserve_session_data_generic(
    source_dir: Path,
    target_dir: Path,
    projects_base: Path,
    encode_fn: Callable[[Path], str],
    tool_name: str,
) -> bool:
    """Shared session data preservation logic."""
    ...
```

And the Copilot/Gemini `allowed_commands_args()` methods are now identical after Fix 2. Extract a shared helper:

```python
# In base.py or a new utils module
def _canonical_to_shell_args(
    commands: list[str], flag: str
) -> list[str]:
    """Translate canonical colon patterns to shell() flags."""
    result: list[str] = []
    for cmd in commands:
        parts = cmd.split(":", 1)
        binary = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        pattern = f"shell({binary}:{args})" if args else f"shell({binary})"
        result.extend([flag, pattern])
    return result
```

**This is lower priority.** Do it after Fixes 1-6 are done.

---

## Verification

After all fixes:

```bash
cd /Users/ivanviragine/Documents/workspace/crossby
./scripts/fmt.sh
./scripts/check.sh
./scripts/test.sh
```

Manual verification for Fix 1:
```bash
crossby launch --tool claude --model gpt-4o .    # Should error: "Model 'gpt-4o' is not compatible with claude"
crossby launch --tool opencode --yolo .           # Should error: "opencode does not support YOLO mode"
crossby launch --tool copilot --effort high .     # Should error: "copilot does not support effort levels"
```

Manual verification for Fix 3:
```bash
crossby launch --tool claude --transcript /tmp/crossby-test-nested/deep/transcript.txt .
# Should create /tmp/crossby-test-nested/deep/ and launch normally
```
