#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
    uv run pytest -m "e2e_deterministic and not live_ai and not live_probe" "$@"
else
    uv run pytest -m "e2e_deterministic and not live_ai and not live_probe" tests/e2e/
fi
