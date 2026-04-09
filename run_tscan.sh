#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/workstation/Palm/PR_20241226_copy"
PROTO="${1:-I}"

case "$PROTO" in
  I)   CONFIG="$ROOT/configs/train_configs/tscan_protocol_I.yaml" ;;
  II)  CONFIG="$ROOT/configs/train_configs/tscan_protocol_II.yaml" ;;
  III) CONFIG="$ROOT/configs/train_configs/tscan_protocol_III.yaml" ;;
  IV)  CONFIG="$ROOT/configs/train_configs/tscan_protocol_IV.yaml" ;;
  *)
    echo "Usage: $0 {I|II|III|IV}"
    exit 1
    ;;
esac

cd "$ROOT"
python DA/train_tscan.py --config "$CONFIG"
