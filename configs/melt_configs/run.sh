#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# One-factor ablations on top of the same full baseline.
# Shared confidence setting: tau 0.5 -> 0.75 in the first 50 epochs.
python DA/train_tscan.py --config configs/melt_configs/tscan_protocol_I_full.yaml
python DA/train_tscan.py --config configs/melt_configs/tscan_protocol_I_no_aug.yaml
python DA/train_tscan.py --config configs/melt_configs/tscan_protocol_I_no_pseudo.yaml
python DA/train_tscan.py --config configs/melt_configs/tscan_protocol_I_no_da.yaml
python DA/train_tscan.py --config configs/melt_configs/tscan_protocol_I_no_ema.yaml
