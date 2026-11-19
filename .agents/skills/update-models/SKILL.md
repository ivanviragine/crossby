---
name: update-models
description: >-
  Refresh Crossby's bundled AI model catalogs across Claude, Cursor, GitHub Copilot,
  Antigravity CLI, Codex, and OpenCode. Use whenever the user asks to update models,
  add a newly released model, prune deprecated/retired models, refresh `models.json`,
  or run and fix `scripts/probe_models.py`.
---

# Updating Provider Models in Crossby

Follow this runbook to refresh `src/crossby/data/models.json`, keep `scripts/probe_models.py`
accurate, update downstream tier/effort/translation/default logic, and verify everything cleanly.

For the detailed per-provider probe URLs, ID spelling rules, and test-file map, see
[providers-and-files.md](./references/providers-and-files.md).

## 1. Run the Live Probe First

Run the probe with network access and signed-in CLIs (`agent`, `agy`, `copilot`, `codex`, `opencode`):

```bash
uv run python scripts/probe_models.py
```

Read every section of its output carefully:

- **`DEPRECATED`**: Provider-confirmed retirements still present in `models.json` -> remove them.
- **`NEW`**: Models discovered by live CLI or official docs that are not in `models.json` -> inspect and add (subject to catalog policy below).
- **`NOT RETURNED`**: Registered models not seen in this run -> **retain by default**. Absence from a local cache, plan-gated listing, or partial doc page is not retirement evidence on its own.
- **`IGNORED`**: Models excluded by provider deprecations or `_CATALOG_EXCLUSIONS` -> no action unless policy changed.
- **`SKIPPED`**: A tool's CLI was missing, unauthenticated, or failed -> **never assume that tool is up to date**. Authenticate the CLI and re-run, or manually check its official docs and record which source verified each change.
- **`CLI Arguments` (`MISSING` / `CAPABILITY MISMATCH`)**: Flag checks against each tool's `--help`; investigate if a CLI renamed/removed a flag Crossby emits.

> [!IMPORTANT]
> Do not rely solely on the probe's interactive AI auto-correct prompt. It only rewrites `models.json` and skips `probe_models.py`, tier classification, Antigravity effort tiers, translation tables, defaults, docs, and tests.

## 2. Repair `scripts/probe_models.py` Before Trusting the Diff

Provider docs and CLI outputs change frequently. Inspect whether probe drift caused false positives or missed models:

1. **Scrape patterns (`_SCRAPE_PATTERNS`)**: Add new vendor/family prefixes (e.g. `grok|kimi`, `mai-`, `fable`) while keeping trailing guards that reject dated snapshots (`-YYYYMMDD`), `-v<n>`, and doc-slug run-ons.
2. **Antigravity effort folding (`_ANTIGRAVITY_EFFORT_RE`)**: If `agy models` introduces a new effort-expanded family (`-low`/`-medium`/`-high`), add it so `normalize_antigravity_model_id()` collapses variants to the base ID. Never fold fixed-suffix model IDs (e.g. `gpt-oss-120b-medium`, `claude-opus-4-6-thinking`) or non-`low/medium/high` suffixes.
3. **Deprecation parsers**: Check HTML anchors, Markdoc table syntax (`parse_gemini_deprecations`), replacement-exclusion rules (`parse_codex_deprecations`, `parse_claude_deprecations`), Copilot display-name slugging (`_copilot_model_slug`), and `models.dev` parsing (`parse_opencode_deprecations`).
4. **Catalog exclusions (`_CATALOG_EXCLUSIONS`)**: Add deliberately excluded models still advertised by a CLI (e.g. Cursor's pre-4.6 Claude models) so the probe reports `IGNORED` and exits `0`.
5. **Key parity**: `_MODEL_PROBES` and `_MODEL_PROBE_SOURCES` must match the non-`_` keys of `models.json` exactly.

## 3. Update `src/crossby/data/models.json`

Apply updates using each tool's exact accepted spelling and catalog policy:

- **Additive & source-backed**: Only add an ID to a tool when that specific tool's CLI or official docs confirm it. Presence in another provider is corroborating context, not proof that another tool accepts the same model or spelling.
- **Per-tool ID format**:
  - `claude`: dotted versions (`claude-haiku-4.5`, `claude-fable-5.2`) — `ClaudeAdapter` dashes on launch and dots on read.
  - `copilot`: dotted Claude versions (`claude-sonnet-5.5`), bare IDs for other vendors.
  - `codex`: bare IDs (`gpt-6.1-terra`), no effort suffixes.
  - `cursor`: exact `agent --list-models` IDs (full effort/thinking/fast matrix + `auto`), sorted alphabetically.
  - `antigravity-cli`: **base IDs only** for effort-expanded families (`gemini-4-argon`, `claude-opus-5-5`), plus **verbatim IDs** for fixed-suffix models (`claude-opus-4-6-thinking`, `gpt-oss-120b-medium`).
  - `opencode`: exact `provider/model` IDs, sorted alphabetically.
- **Pruning rules**:
  - Remove entries confirmed deprecated/retired by the provider (or shutting down within Google's 180-day horizon), or confirmed dropped by a logged-in CLI **and** absent from official docs.
  - Retain models that are merely `NOT RETURNED`.
  - **Claude 4.6 floor**: Keep Claude releases older than 4.6 out of the catalogs, **except `claude-haiku-4.5`** (still active and used for `easy` fallback defaults).

## 4. Check Tier Classification (`src/crossby/ai_tools/model_utils.py`)

Check how `classify_tier_universal()` classifies every newly added model ID (and its `provider/`-stripped suffix for OpenCode):

- `POWERFUL` (checked 1st): `opus`, `fable`, `astra`, `pro`, `sol`, `ultra`, `argon`, `carbon`, `wonder` (plus exact `gpt-5.4`).
- `FAST` (checked 2nd): `haiku`, `flash`, `spark`, `mini`, `luna`.
- `BALANCED` (checked 3rd): `sonnet`, `terra`, or unrecognized fallback.
- `max` (checked last): `POWERFUL` only when no family keyword matches.

**Precedence rules to preserve**:
- Upgrade/variant markers (`pro`, `sol`, `astra`, …) outrank family markers (`openai/gpt-5.6-luna-pro` -> `POWERFUL`).
- Family markers outrank the `max` effort suffix (`gpt-5.6-luna-max` -> `FAST`, `gpt-5.6-terra-max` -> `BALANCED`).
- Keywords match component boundaries (`-` or `.`); verify a new keyword cannot accidentally match another family name.

If you edit `classify_tier_universal()`, run a catalog-wide tier diff before/after across every ID in `models.json` to ensure no unrelated models shifted tier:

```bash
uv run python -c '
import json
from crossby.ai_tools.model_utils import classify_tier_universal
data = json.loads(open("src/crossby/data/models.json").read())
for tool, ids in data.items():
    if tool.startswith("_"):
        continue
    for mid in ids:
        raw = mid.split("/")[-1] if tool == "opencode" and "/" in mid else mid
        print(f"{tool}\t{mid}\t{classify_tier_universal(raw).value}")
'
```

## 5. Check Tool Adapters for Effort & Compatibility

1. **Antigravity CLI (`src/crossby/ai_tools/antigravity_cli.py`)**:
   - If `agy models` lists a new model only with `-low`/`-medium`/`-high` suffixes, add its base ID and exact supported `EffortLevel` tuple to `_ANTIGRAVITY_CLI_EFFORT_TIERS` (verified against `agy models` — e.g. `gemini-3.1-pro` only supports `(LOW, HIGH)`). Without this, `resolve_effort_model()` strips the suffix and launches an invalid bare ID.
   - If a new `agy models` ID has a fixed suffix that looks like an effort level (like `gpt-oss-120b-medium`), add it to `_ANTIGRAVITY_CLI_FIXED_SUFFIX_MODELS` instead.
   - Keep `_ANTIGRAVITY_CLI_EFFORT_TIERS` in sync with `_ANTIGRAVITY_EFFORT_RE` in `scripts/probe_models.py`.
2. **Cursor (`src/crossby/ai_tools/cursor.py`)**:
   - `resolve_effort_model()` derives effort siblings dynamically from `models.json`. Verify any new Cursor naming pattern resolves cleanly and does not break the exhaustive registry check in `tests/unit/test_ai_tools/test_cursor_resolve_effort_model.py`.
3. **Compatibility / normalization**:
   - Verify `is_model_compatible()` in `codex.py` (`codex-*`, `gpt-*`, `o\d*`) and `claude.py` (`claude-*`) accepts any new model prefixes.

## 6. Check Cross-Provider Translation (`src/crossby/sync/translation.py`)

- **`CODEX_TO_CLAUDE_DEFAULTS`**: Add every new Codex/GPT model ID with an appropriate current Claude target (e.g. `gpt-6.1-sol -> claude-opus-5.5`, `gpt-6.1-terra -> claude-sonnet-5`). **Keep retired Codex keys** (`gpt-5.4`, `gpt-5.5`, `gpt-5.4-mini`, …) so existing user configs continue translating.
- **`MODEL_FAMILY_MAPPINGS`**: Only retarget forward `claude -> codex` mappings when the target Codex model is generally available (not plan-gated/rolling out) or a previous target was retired.

## 7. Check Fallback Defaults (`src/crossby/config/defaults.py`)

- **Required update**: If any model in `TOOL_DEFAULTS` was removed from `models.json`, replace it. Every default (or its effort-stripped base for `antigravity-cli`) must exist in `models.json`.
- **Conservative churn**: Do not change defaults for newly announced models that are still rolling out per plan; keep existing working defaults unless a clear GA replacement is warranted.
- **Invariants**: `medium != easy` and `medium == complex` for all tools; for `claude`, `copilot`, `codex`, and `opencode`, `easy` must classify `FAST`, `medium`/`complex` `BALANCED`, and `very_complex` `POWERFUL`. Update `_EXPECTED_TIERS` in `tests/unit/test_config/test_defaults.py` whenever `TOOL_DEFAULTS` changes.

## 8. Update Docs (Only Where Affected)

Check these files and update only the sections affected by your changes:

- `src/crossby/data/skill/references/differences.md` (`## Models` and `## Agents` rows) if `MODEL_FAMILY_MAPPINGS` or `CODEX_TO_CLAUDE_DEFAULTS` changed.
- `CONTRIBUTING.md`:
  - Antigravity CLI effort-tier prose if `_ANTIGRAVITY_CLI_EFFORT_TIERS` or `_ANTIGRAVITY_CLI_FIXED_SUFFIX_MODELS` changed.
  - Model-catalog probe guidance (under `## Running Checks`) if probe behavior or catalog policy changed.
- `README.md` if translation examples, minimum CLI version notes, or example model IDs became stale.
- Leave `.wade.yml` untouched unless explicitly asked to change Wade execution defaults.

## 9. Update Tests & Validate Everything

Add/update regression tests for every changed layer (see [providers-and-files.md](./references/providers-and-files.md#test-files-to-update)), then run the full validation sequence:

1. **Re-run the live probe** — must exit `0` with no unexpected `NEW`, `DEPRECATED`, `MISSING`, or `CAPABILITY MISMATCH` entries:
   ```bash
   uv run python scripts/probe_models.py
   ```
2. **Run focused model & probe tests**:
   ```bash
   uv run pytest \
     tests/unit/test_ai_tools/test_model_registry.py \
     tests/unit/test_models/test_ai_models.py \
     tests/unit/test_config/test_defaults.py \
     tests/unit/test_sync/test_translation.py \
     tests/unit/test_scripts/test_probe_models.py \
     tests/unit/test_ai_tools/test_antigravity_cli_launch.py \
     tests/unit/test_ai_tools/test_cursor_resolve_effort_model.py
   ```
3. **Run full repo checks**:
   ```bash
   ./scripts/check-all.sh
   ```
4. **Lint and format `scripts/probe_models.py` explicitly** (`./scripts/check.sh` only checks `src/` and `tests/`):
   ```bash
   uv run ruff check scripts/probe_models.py
   uv run ruff format --check scripts/probe_models.py
   git diff --check
   ```

> [!NOTE]
> On macOS in restricted/sandboxed environments, `test_a_naturally_exited_session_is_not_reaped_before_close` can fail even on untouched `main`. Compare against `HEAD` under identical permissions before attributing PTY failures to catalog changes.

## 10. Summarize the Refresh Clearly

In your final response or PR description, report:
- Which sources were probed live vs. checked via docs (and any `SKIPPED` CLIs).
- Models added and removed per tool, with retirement evidence for removals.
- Any changes to `probe_models.py`, tier classification (with catalog-wide diff result if touched), Antigravity effort tiers, translation maps, defaults, and docs.
- Exact validation results (live probe exit status, focused/full test counts, ruff, mypy).
