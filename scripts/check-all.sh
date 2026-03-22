#!/usr/bin/env bash
set -euo pipefail

echo "=== Tests ==="
./scripts/test.sh

echo ""
echo "=== Deterministic E2E ==="
./scripts/test-e2e.sh

echo ""
echo "=== Lint + Types ==="
./scripts/check.sh
