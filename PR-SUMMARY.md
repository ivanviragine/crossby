# PR Summary — Issue #40: Skills Sync (Writers, Readers, Wizard)

## What

Implements the full skills-sync pipeline, mirroring the existing agents sync pattern end-to-end. Skills directories (`.claude/skills/`, `.cursor/skills/`, etc.) can now be synced across all five AI tools via directory-level symlinks.

## Changed Files

| File | Change |
|------|--------|
| `src/crossby/sync/base.py` | Added `SyncConcern.SKILLS`; added `skills_source`, `skills_strategy`, `skills_gitignore` fields to `SyncData` |
| `src/crossby/sync/skills.py` | **New** — five concrete writers (`ClaudeSkillsWriter`, `CursorSkillsWriter`, `CodexSkillsWriter`, `GeminiSkillsWriter`, `CopilotSkillsWriter`) + `_is_managed_skills_dir` + `update_skills_gitignore` |
| `src/crossby/sync/__init__.py` | Registered all five skills writers; wired `update_skills_gitignore` post-run |
| `src/crossby/sync/readers.py` | Added `detect_skills`, `suggest_skills_source`; extended `ProjectScan`/`scan_project`/`build_sync_data` |
| `src/crossby/config/skills.py` | Extended `_SCAN_ORDER` to all 5 tools; added shared `count_skills()` helper |
| `src/crossby/cli/sync.py` | Added Skills branch to wizard (source picker + confirm + per-concern execution) |
| `tests/unit/test_sync/test_skills.py` | **New** — 44 tests covering writers, managed-dir check, gitignore, run_sync |
| `tests/unit/test_sync/test_readers.py` | **New** — 27 tests covering detect, suggest, build_sync_data, scan_project |
| `tests/unit/test_cli/test_sync_cmd.py` | Added `TestSyncCommandSkills` (5 CLI integration tests) |

## Key Design Decisions

- **"Managed directory" semantics for skills differ from agents**: agents = flat `.md` files; skills = subdirectories each containing `SKILL.md`. `_is_managed_skills_dir` enforces this to avoid wrongly overwriting user directories without `--force`.
- **Circular source/target guard**: checks `not target_dir.is_symlink()` before comparing resolved paths — prevents false-positive "would link to itself" errors on idempotent re-runs.
- **`_SCAN_ORDER` as shared priority list**: both `config/skills.py` and `sync/readers.py` reference the same ordered list (imported as `_SKILLS_SCAN_ORDER`) so source-tool preference is defined once.
- **`count_skills()` in `config/skills.py`**: shared helper used by both `detection.py` and `readers.py` to count SKILL.md-bearing subdirectories.
- **No Copilot per-file variant**: unlike agents (which has a copy-fallback for Copilot), all five skills writers use the same `_BaseSkillsWriter` with symlink strategy.

## Test Coverage

912 tests total, 0 failures. mypy strict and ruff pass on all modified/new files.
