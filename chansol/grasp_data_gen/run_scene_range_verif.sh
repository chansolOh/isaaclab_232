#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/../../.venv/bin/python}"
MAIN_PY="$SCRIPT_DIR/Direct_RL_main_verif.py"
ROOT_PATH="${ROOT_PATH:-/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen/test}"
OUTPUT_DIR="$ROOT_PATH/output_grasp_verif"

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <start_scene_num> <end_scene_num> [step]"
    echo "Example: $0 0 100"
    echo "Example: $0 50 200 5"
    exit 1
fi

START_SCENE="$1"
END_SCENE="$2"
STEP="${3:-1}"

mkdir -p "$OUTPUT_DIR"

for ((scene_num=START_SCENE; scene_num<=END_SCENE; scene_num+=STEP)); do
    scene_file="$(printf "%s/%04d.json" "$OUTPUT_DIR" "$scene_num")"

    if [[ -f "$scene_file" ]]; then
        echo "[SKIP] scene $(printf "%04d" "$scene_num") already exists: $scene_file"
        continue
    fi

    echo "[RUN ] scene $(printf "%04d" "$scene_num")"
    "$PYTHON_BIN" "$MAIN_PY" --scene-num "$scene_num"
done
