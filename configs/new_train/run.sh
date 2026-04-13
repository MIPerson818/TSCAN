#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python DA/train_tscan.py --config configs/new_train/tscan_protocol_I.yaml
python DA/train_tscan.py --config configs/new_train/tscan_protocol_II.yaml
python DA/train_tscan.py --config configs/new_train/tscan_protocol_III.yaml
python DA/train_tscan.py --config configs/new_train/tscan_protocol_IV.yaml
