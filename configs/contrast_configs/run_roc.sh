#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python scripts/plot_compare_roc.py --config configs/contrast_configs/roc_grid_protocols.yaml
