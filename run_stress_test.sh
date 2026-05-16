#!/usr/bin/env bash
# run_stress_test.sh — Activates venv, runs the PA stress-test, and compresses outputs.
#
# Usage:
#   ./run_stress_test.sh [--n 20000] [--era 2026]
#
# All extra arguments are forwarded to scripts/bulk_test.py.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_DIR/venv"

# ── Activate venv ────────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    echo "[run] Activated venv: $VENV_DIR"
else
    echo "[run] No venv found at $VENV_DIR — using system Python"
fi

cd "$REPO_DIR"

# ── Run stress test ───────────────────────────────────────────────────────────
START=$(date +%s)
python scripts/bulk_test.py "$@"
END=$(date +%s)
echo "[run] Wall-clock runtime: $((END - START))s"

# ── Compress outputs ──────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%dT%H%M%S)
ARCHIVE="outputs/stress_test_run_${TIMESTAMP}.tar.gz"
tar -czf "$ARCHIVE" outputs/receipts/ outputs/summaries/
echo "[run] Compressed → $REPO_DIR/$ARCHIVE"
