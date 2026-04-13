#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python DA/train_tscan.py --config configs/conf_configs/tscan_protocol_I_conf_00.yaml
python DA/train_tscan.py --config configs/conf_configs/tscan_protocol_I_conf_04.yaml
python DA/train_tscan.py --config configs/conf_configs/tscan_protocol_I_conf_06.yaml
python DA/train_tscan.py --config configs/conf_configs/tscan_protocol_I_conf_08.yaml
