#!/usr/bin/env bash
# تشغيل عامل Living Mesh مرتبط ببذرة
# الاستخدام:
#   SEED_NODE_URL=127.0.0.1:7860 NODE_ID=worker_1 PORT=7861 ./scripts/run_mesh_worker.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export NODE_ID="${NODE_ID:-worker_1}"
export PORT="${PORT:-7861}"
export SEED_NODE_URL="${SEED_NODE_URL:-127.0.0.1:7860}"
export NSM_NODE_DATA_DIR="${NSM_NODE_DATA_DIR:-$ROOT/artifacts/living_mesh/nodes/${NODE_ID}}"
mkdir -p "$NSM_NODE_DATA_DIR"
exec python ai/node_launcher.py \
  --id "$NODE_ID" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --data-dir "$NSM_NODE_DATA_DIR"
