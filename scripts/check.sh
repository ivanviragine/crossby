#!/usr/bin/env bash
set -euo pipefail

run_lint=true
run_types=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lint)  run_types=false; shift ;;
        --types) run_lint=false; shift ;;
        *) echo "Usage: $0 [--lint|--types]"; exit 1 ;;
    esac
done

if $run_lint; then
    echo "=== Lint + format check ==="
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/
fi

if $run_types; then
    echo "=== Type check ==="
    uv run mypy src/crossby/
fi
