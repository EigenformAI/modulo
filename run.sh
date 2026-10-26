#!/bin/bash
# Run a self-steering agent session.
# Usage:
#   ./run.sh                              # interactive session
#   ./run.sh --auto --max-turns 10         # non-interactive demo
#   ./run.sh --seed "Try steering yourself" --auto --max-turns 8
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec python3 -m agent.loop "$@"
