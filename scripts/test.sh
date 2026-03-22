#!/usr/bin/env bash
set -euo pipefail

# Run tests — pass extra args to pytest (e.g. ./scripts/test.sh tests/unit/)
uv run pytest "${@:-tests/}"
