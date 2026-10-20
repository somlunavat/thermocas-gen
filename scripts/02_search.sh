#!/usr/bin/env bash
# Stage 1b wrapper, kept for the interface the project brief describes.
#
# The logic lives in 02_search.py so that command construction and the unit
# conversions in it are covered by tests; a coverage fraction passed where a
# percentage is expected does not crash, it silently changes the corpus.
#
#   ./scripts/02_search.sh --database /path/to/uniref50.fasta
set -euo pipefail
exec python3 "$(dirname "$0")/02_search.py" "$@"
