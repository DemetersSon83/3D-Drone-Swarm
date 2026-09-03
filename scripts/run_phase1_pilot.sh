#!/usr/bin/env bash
# Reproducible Phase I pilot: all supplied experiment scenarios × 20 paired seeds.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${REPO_ROOT}/scripts/run_experiments.sh" \
  --config-dir "${REPO_ROOT}/configs/experiments" \
  --seeds 0:19 \
  --jobs "${JOBS:-4}" \
  --output-root "${OUTPUT_ROOT:-${REPO_ROOT}/outputs/phase1_pilot}" \
  "$@"
