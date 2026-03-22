#!/usr/bin/env bash
set -euo pipefail

# Run deterministic tests by default. Pass extra args to pytest when needed.
if [[ $# -gt 0 ]]; then
    uv run pytest -m "not live_ai and not live_probe" "$@"
else
    uv run pytest -m "not live_ai and not live_probe" tests/
fi
