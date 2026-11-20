# Provider Reference & File Map

Detailed reference for Crossby's six registry-backed tools, `scripts/probe_models.py`,
and the source/test files touched during a model catalog refresh.

## Per-Provider Discovery, Deprecations & Format Rules

| Tool (`models.json` key) | Live / Doc Discovery Source | Deprecation / Retirement Source | Registry ID Format & Quirks |
| --- | --- | --- | --- |
| `claude` | Scrapes `https://platform.claude.com/docs/en/about-claude/models/overview` between `id="latest-models-comparison"` and `id="using-the-models-api"`. **Never** run `claude models` (Claude interprets it as a prompt). | `https://platform.claude.com/docs/en/about-claude/model-deprecations` (`id="model-status"` table only; stops before `id="deprecation-history"` to ignore replacements). | Dotted versions (`claude-haiku-4.5`, `claude-fable-5.2`). Dated snapshots (`-YYYYMMDD`, optional `-vN`) collapse to stable aliases. `ClaudeAdapter.normalize_model_format` dashes at launch. |
| `cursor` | Signed-in `agent --list-models` (`<id> - <display name>`). Returns empty if CLI missing or signed out. | No dedicated web deprecation URL; removals come from logged-in `agent --list-models` corroborated against upstream shutdowns, plus `_CATALOG_EXCLUSIONS["cursor"]` for pre-4.6 Claude models. | Exact CLI IDs including effort/thinking/fast suffixes (`claude-fable-5-2-high-fast`, `gemini-4-carbon-high`, `auto`). Sorted alphabetically. `CursorAdapter.resolve_effort_model` reads sibling IDs directly from `models.json`. |
| `copilot` | Merges `https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference` (`id="supported-models"` to `id="tool-availability-values"`) **and** `https://docs.github.com/en/copilot/reference/ai-models/supported-models` (`id="supported-ai-models-per-client"` rows where `Copilot CLI` is `Included`, via text or SVG `aria-label="Included"`). Avoid `copilot --model x` (newer Copilot CLI builds launch the interactive TUI and hang). | `https://docs.github.com/en/copilot/reference/ai-models/supported-models` (`id="model-retirement-history"` table; slugs display names and appends `-fast` for `(fast mode)` so retiring fast mode does not retire the base model; `probe_deprecations("copilot")` also retires `<retired-base>-fast`). | Dotted Claude versions (`claude-sonnet-5.5`), bare IDs for GPT/Gemini/Grok/Kimi/MAI (`gpt-6.2-sol`, `mai-code-2-flash`). |
| `antigravity-cli` | Signed-in `agy models` (first whitespace-delimited token per line, ANSI-stripped). | `https://ai.google.dev/gemini-api/docs/deprecations` (supports both HTML and Markdoc tables; only counts shutdown dates already passed or within `_GEMINI_SHUTDOWN_HORIZON = 180 days`). | **Base IDs only** for effort-expanded Gemini & Claude 5.5+ families (`gemini-4-neon`, `gemini-4-hydrogen`, `gemini-4-argon`, `claude-opus-5-5`); `_ANTIGRAVITY_EFFORT_RE` folds `-low`/`-medium`/`-high` (and `-xhigh` for `hydrogen|neon`). **Verbatim IDs** for bare/fixed-suffix models (`gemini-pro-orchestrator-neon`, `claude-opus-4-6-thinking`, `gpt-oss-120b-medium`). Keep `_ANTIGRAVITY_CLI_EFFORT_TIERS`, `_ANTIGRAVITY_CLI_FIXED_SUFFIX_MODELS`, and `supported_efforts` aligned in `antigravity_cli.py`. |
| `codex` | Merges published docs (`https://learn.chatgpt.com/docs/models`, `codex -m <id>`) **and** `~/.codex/models_cache.json` (`visibility == "list"`). | `https://learn.chatgpt.com/docs/models` (`id="deprecated-codex-models"` to `id="configure-your-default-local-model"`, excluding `with <replacement>` and `id="recommended-models"`). | Bare IDs (`gpt-6.1-sol`, `gpt-6.1-terra`). Effort is passed separately via `-c model_reasoning_effort=...`. |
| `opencode` | `opencode models` (lines whose first token contains `/`). | `https://models.dev/api.json` (`status == "deprecated"`), combined with Copilot retirements on `github-copilot/` (exact ID, `-fast`, and display-name match) and Gemini shutdowns on `google/`. | Exact `provider/model` IDs (`anthropic/...`, `github-copilot/...`, `google/...`, `openai/...`, `opencode/...`), sorted alphabetically. `OpenCodeAdapter.get_models()` classifies the suffix after `/`. |

## Source Files Involved

| File | Responsibility |
| --- | --- |
| `src/crossby/data/models.json` | Static bundled model catalog loaded by `src/crossby/data/__init__.py` (`MODELS`, `get_models_for_tool`). |
| `scripts/probe_models.py` | Live CLI/web discovery, deprecation scraping, `_CATALOG_EXCLUSIONS`, `_ANTIGRAVITY_EFFORT_RE`, `_EXPECTED_FLAGS`, and CLI arg verification. |
| `src/crossby/ai_tools/model_utils.py` | `classify_tier_universal()` and `has_date_suffix()`. |
| `src/crossby/ai_tools/antigravity_cli.py` | `_ANTIGRAVITY_CLI_EFFORT_TIERS`, `_ANTIGRAVITY_CLI_FIXED_SUFFIX_MODELS`, and `resolve_effort_model()`. |
| `src/crossby/ai_tools/cursor.py` | `_effort_template()` and `resolve_effort_model()` over the Cursor catalog. |
| `src/crossby/ai_tools/claude.py` | `is_model_compatible()`, `normalize_model_format()` (dot->dash), `standardize_model_id()` (dash->dot). |
| `src/crossby/ai_tools/copilot.py` | `normalize_model_format()` (dash->dot for Claude IDs). |
| `src/crossby/ai_tools/codex.py` | `is_model_compatible()` (`codex-*`, `gpt-*`, `o\d*`). |
| `src/crossby/sync/translation.py` | `MODEL_FAMILY_MAPPINGS` (`claude -> codex`) and `CODEX_TO_CLAUDE_DEFAULTS` (`codex -> claude`). |
| `src/crossby/config/defaults.py` | `TOOL_DEFAULTS` fallback mapping per tool and complexity tier. |
| `src/crossby/data/skill/references/differences.md` | Cross-tool model translation table (`## Models`, `## Agents`). |
| `CONTRIBUTING.md` | Model-catalog refresh notes (`## Running Checks`) and Antigravity CLI effort-tier docs. |
| `README.md` | User-facing translation examples and minimum CLI version caveats. |

## Test Files to Update

| Test File | What to Cover |
| --- | --- |
| `tests/unit/test_ai_tools/test_model_registry.py` | Newly added models per tool, exact subset/variant checks (Cursor families, Codex GPT-6.x), Antigravity base-only + fixed-suffix assertions, and `test_deprecated_models_are_pruned`. |
| `tests/unit/test_models/test_ai_models.py` | `TestClassifyTierUniversal` for new family/tier keywords, `TestModelCompatibility`, and `TestCrossProviderModelTranslation`. |
| `tests/unit/test_scripts/test_probe_models.py` | Scrape regex regressions, dated-alias collapsing, deprecation parser fixtures (HTML/Markdoc/replacements), `_CATALOG_EXCLUSIONS`, `normalize_antigravity_model_id`, and `test_probe_routing_matches_registry_keys_exactly`. |
| `tests/unit/test_ai_tools/test_antigravity_cli_launch.py` | Effort resolution for any new base family added to `_ANTIGRAVITY_CLI_EFFORT_TIERS` or `_ANTIGRAVITY_CLI_FIXED_SUFFIX_MODELS`, plus `supported_efforts` and bare passthrough models. |
| `tests/unit/test_ai_tools/test_plan_sessions.py` | Tool-wide `supported_efforts` rejection (`test_unsupported_effort_tier_is_rejected_before_collection`) and per-model Antigravity effort rejection (`test_antigravity_rejects_effort_its_model_cannot_encode`). |
| `tests/unit/test_ai_tools/test_cursor_resolve_effort_model.py` | Effort sibling resolution for new Cursor families/naming patterns (also enforces that no catalog model + effort yields an unregistered ID). |
| `tests/unit/test_sync/test_translation.py` | Forward (`map_model_claude_to_codex`) and reverse (`map_model_codex_to_claude`) mappings for new or retargeted models. |
| `tests/unit/test_config/test_defaults.py` | `_EXPECTED_TIERS`, `TestDefaultsRegistryGuard`, `TestDocumentedFamilyRoles` (keyword precedence), and `test_refreshed_provider_defaults`. |
