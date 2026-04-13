#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# python DA/train_tscan.py --config configs/conf234_configs/tscan_protocol_II_tau_06.yaml
python DA/train_tscan.py --config configs/conf234_configs/tscan_protocol_II_tau_06_08.yaml
python DA/train_tscan.py --config configs/conf234_configs/tscan_protocol_II_tau_05_08.yaml
