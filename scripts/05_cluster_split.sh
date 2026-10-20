#!/usr/bin/env bash
# Stage 1e wrapper, kept for the interface the project brief describes.
# Logic lives in 05_cluster_split.py so the split guarantees are testable.
set -euo pipefail
exec python3 "$(dirname "$0")/05_cluster_split.py" "$@"
