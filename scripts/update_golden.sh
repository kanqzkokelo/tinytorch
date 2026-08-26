#!/usr/bin/env bash
# Regenerate the per-position golden baseline (data/golden/*.jsonl).
#
# Use after intentionally changing engine behaviour: confirm the diff
# stats in the new baseline look reasonable, then commit. CI then runs
# `python3 tests/test_engine_golden.py verify` to ensure no further drift.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 tests/test_engine_golden.py record "$@"
