#!/usr/bin/env bash
# The daily job: check every source, store what changed, fetch headlines, enforce retention.
# Schedule it with cron, a CI scheduler or a cloud job, e.g.:  15 6 * * *  /path/to/daily.sh
# Set CARBRAIN_DATA_DIR to a persistent directory; YOUTUBE_API_KEY enables YouTube.
set -uo pipefail
cd "$(dirname "$0")/.."
status=0
uv run carbrain sync || status=1
uv run carbrain sync-content || status=1
uv run carbrain purge || status=1
uv run carbrain status
exit $status
