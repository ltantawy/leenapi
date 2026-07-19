#!/usr/bin/env bash
# Fetch FastSAM-s and export it to NCNN (fastest CPU format on the Pi 5 ARM core).
# Produces models/FastSAM-s_ncnn_model/ ; the app auto-prefers it over the .pt.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$HERE/models"
MODEL="${1:-FastSAM-s}"   # e.g. FastSAM-s or FastSAM-x

mkdir -p "$MODELS"

if [[ -d "$MODELS/${MODEL}_ncnn_model" ]]; then
    echo "NCNN model already present: $MODELS/${MODEL}_ncnn_model"
    exit 0
fi

# ultralytics auto-downloads ${MODEL}.pt on first use, then exports to NCNN.
# The NCNN export needs the `pnnx` converter, declared as the build-time-only
# `export` extra (see pyproject.toml). Run inside the project's uv venv so
# ultralytics/torch are available.
cd "$MODELS"
uv run --project "$HERE" --extra export yolo export "model=${MODEL}.pt" format=ncnn

echo "Done -> $MODELS/${MODEL}_ncnn_model/"
