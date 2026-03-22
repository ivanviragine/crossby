#!/usr/bin/env bash
set -euo pipefail

# Run non-E2E deterministic tests by default. Pass extra args to pytest when needed.
if [[ $# -gt 0 ]]; then
    uv run pytest -m "not live_ai and not live_probe and not e2e_deterministic" "$@"
else
    uv run pytest -m "not live_ai and not live_probe and not e2e_deterministic" tests/
fi
