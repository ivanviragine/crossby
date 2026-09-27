#!/usr/bin/env bash
# Lint, format check, type check and offline tests. Run from carbrain/.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest
