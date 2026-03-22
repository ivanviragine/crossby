#!/usr/bin/env bash
set -euo pipefail

echo "=== Tests ==="
./scripts/test.sh

echo ""
echo "=== Lint + Types ==="
./scripts/check.sh
