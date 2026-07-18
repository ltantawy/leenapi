#!/usr/bin/env bash
# Download the MediaPipe DeepLabV3 image-segmentation model (~21 Pascal-VOC classes).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/models/deeplab_v3.tflite"
URL="https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/latest/deeplab_v3.tflite"

mkdir -p "$HERE/models"
if [[ -f "$DEST" ]]; then
    echo "Model already present: $DEST"
    exit 0
fi

echo "Downloading DeepLabV3 model -> $DEST"
curl -fsSL "$URL" -o "$DEST"
echo "Done ($(du -h "$DEST" | cut -f1))."
