#!/usr/bin/env bash
# Delegate to scripts/run_train_supervised.sh (keeps a short path at repo root).
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/run_train_supervised.sh" "$@"
