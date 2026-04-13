#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python DA/train_mmd_baseline.py --config configs/contrast_configs/mmd_protocol_I.yaml
python DA/train_mmd_baseline.py --config configs/contrast_configs/mmd_protocol_II.yaml
python DA/train_mmd_baseline.py --config configs/contrast_configs/mmd_protocol_III.yaml
python DA/train_mmd_baseline.py --config configs/contrast_configs/mmd_protocol_IV.yaml
