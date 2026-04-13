#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Two progressive-threshold ablations only:
# 1) 0.60 -> 0.45 : looser later, keeps pseudo-label branch active
# 2) 0.50 -> 0.60 : stricter later, tests the "more precise later" hypothesis
python DA/train_tscan.py --config configs/conf_configs/tscan_protocol_I_conf_prog_60_45.yaml
python DA/train_tscan.py --config configs/conf_configs/tscan_protocol_I_conf_prog_50_60.yaml
