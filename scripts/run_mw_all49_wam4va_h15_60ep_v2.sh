#!/usr/bin/env bash
# Compatibility entry point; the expanded dataset is now the true MT50.
set -euo pipefail
cd "$(dirname "$0")/.."
exec scripts/run_mw_mt50_wam4va_h15_60ep_v2.sh "$@"
