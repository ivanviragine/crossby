#!/usr/bin/env bash
set -euo pipefail

uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
