#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

bash configs/conf234_configs/run_protocol_II.sh
bash configs/conf234_configs/run_protocol_III.sh
bash configs/conf234_configs/run_protocol_IV.sh
