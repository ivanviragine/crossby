#!/usr/bin/env bash
# Set up a new worktree with dev dependencies.
#
# This script is called automatically by the post_worktree_create hook
# when a new worktree is created via `wade implement`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Setting up worktree in ${ROOT_DIR}..."
echo ""

# Create/ensure virtual environment and install dev dependencies
echo "Installing dev dependencies..."
uv sync --all-extras
echo ""

echo "✓ Worktree setup complete!"
