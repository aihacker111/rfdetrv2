#!/usr/bin/env bash
# Delegate to scripts/run_finetune.sh (multi-GPU torchrun launcher).
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/run_finetune.sh" "$@"
