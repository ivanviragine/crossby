#!/usr/bin/env bash
set -euo pipefail

export RUN_LIVE_AI_TESTS=1

if [[ $# -gt 0 ]]; then
    uv run pytest -m "live_ai" "$@"
else
    uv run pytest -m "live_ai" tests/live/
fi
