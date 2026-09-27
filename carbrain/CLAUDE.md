# carbrain

Versioned evidence database about the Brazilian car market. Official and
licensed sources flow into SQLite with full provenance; typed tools on top give
a chatbot and post generator cited numbers. Design rationale:
the "Brazil Car Data Blueprint" report (revision 2).

## Before and after every implementation step

1. **Before:** read `LEARNINGS.md` in full. Apply its rules and known traps.
2. **After:** add what you learned to `LEARNINGS.md` (source quirks, bugs and
   their root cause, decisions and why). Rewrite entries that proved wrong.

## Commands (run from `carbrain/`)

- `uv sync --all-extras` — install
- `./scripts/check.sh` — ruff, format check, mypy strict, offline tests
- `uv run pytest -m live` — smoke tests against the real sources (network)
- `uv run carbrain --help` — CLI (`init`, `sync`, `status`, `purge`, `review`, `tool`, `ask`)

## Layout

- `src/carbrain/db.py` — SQLite schema and connection
- `src/carbrain/rights.py` + `data/sources.yaml` — source registry and usage rights; every
  collect/retain/embed/AI/display/publish use goes through `Rights.require()`
- `src/carbrain/archive.py` — raw snapshots, content-hashed; unchanged files are not re-parsed
- `src/carbrain/observations.py` — facts with `as_of`, `fetched_at`, source and revisions
- `src/carbrain/catalog.py`, `resolve.py`, `data/families.yaml` — internal vehicle IDs,
  registry/FIPE name matching, review queue
- `src/carbrain/connectors/` — one module per source
- `src/carbrain/calc/` — deterministic calculators (financing, fuel, deflation, ownership cost)
- `src/carbrain/tools.py` — chatbot tools; every result carries citations and a status
- `src/carbrain/chat.py` — optional Claude loop over the tools

## Rules

- Build parsers from real files; keep trimmed real copies in `tests/fixtures/`.
- Never invent vehicle facts in seed data. Seed only what is verified; leave the rest empty.
- Tools return `no_data` or `restricted` rather than guessing; the chat layer must not
  produce numbers no tool returned.
- FIPE collection stays blocked in the rights registry until a contract exists.
