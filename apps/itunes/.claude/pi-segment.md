# Pi Segment — build notes & usage

Live **FastSAM "segment everything"** app for the Raspberry Pi 5 camera. Captures
frames, runs FastSAM in a background thread, and blends the colorized per-object
mask overlay onto the smooth live video, served as a headless MJPEG stream.

```
rpicam-vid (MJPEG) ─► OpenCV decode ─┬─► blend latest overlay ─► HTTP MJPEG stream (:8000)
                                     └─► FastSAM (bg thread)  ─► colorized overlay
```

## What was done

1. **Started from** a MediaPipe DeepLabV3 semantic segmenter (21 Pascal-VOC
   classes). It could only color recognized classes and left the rest as the raw
   image, so it couldn't "segment every aspect of the image."
2. **Swapped to FastSAM** (ultralytics), which produces prompt-free instance
   masks for everything in one forward pass — the Segment-Anything look, feasible
   on a Pi CPU where real SAM / MobileSAM / EdgeSAM automatic mode are not.
3. **Decoupled the pipeline** (see diagram). The camera streams smooth video; a
   background thread runs FastSAM at ~1–4 FPS and publishes the latest overlay,
   which the capture loop blends onto every frame. Masks lag moving objects —
   accepted tradeoff.
4. **Reused capture + HTTP plumbing** unchanged (`src/camera.py` `rpicam-vid`
   MJPEG; `ThreadingHTTPServer` multipart stream).

## Dependencies (changed)

- Removed **mediapipe** (and its cp312-only aarch64 constraint).
- Added **ultralytics** (FastSAM; pulls **torch/torchvision** CPU wheels) and
  **ncnn** (fast CPU inference of the NCNN-exported model). Kept `opencv-python`.
- `requires-python` widened to `>=3.10` (3.13 now possible); the uv venv is still
  3.12 to avoid a rebuild. `numpy` pin relaxed to `>=1.26`.
- Torch is a heavy dep (hundreds of MB) vs. the old ~3 MB tflite — fine on the
  8 GB Pi 5. Fallback if size/install bites: FastSAM → ONNX + onnxruntime.

## How to use

```bash
cd apps/itunes

# one-time setup
uv sync                       # creates the venv, installs deps (torch is large)
bash scripts/setup_model.sh   # fetches FastSAM-s + exports to NCNN

# run
uv run python main.py         # prints: Serving segmented stream at http://<pi-ip>:8000/
```

Open `http://<pi-ip>:8000/` from any device on the network.

### Options

```bash
uv run python main.py --help
  --host 0.0.0.0 --port 8000
  --width 1280 --height 720 --framerate 15
  --alpha 0.5          # mask overlay opacity (0..1)
  --quality 80         # output JPEG quality
  --model FastSAM-s    # .pt name or NCNN model dir
  --imgsz 512          # inference size (smaller = faster, coarser)
  --conf 0.4 --iou 0.9
  --retina-masks       # full-res mask edges (slower)
```

Lower `--imgsz` for a faster mask refresh; the video stays smooth regardless.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | HTTP MJPEG server; decoupled capture / segment-worker / blend threads, dual FPS overlay |
| `src/camera.py` | `rpicam-vid` subprocess → decoded BGR frames (MJPEG SOI/EOI splitting) |
| `src/segmenter.py` | FastSAM segment-everything → `(color_bgr, alpha_map)` overlay + `blend()` |
| `models/FastSAM-s_ncnn_model/` | NCNN model (generated; gitignored) |
| `scripts/setup_model.sh` | Fetches FastSAM + exports to NCNN |
| `pyproject.toml` / `.python-version` / `uv.lock` | uv project (venv on Python 3.12) |

## Troubleshooting

- **`failed to acquire camera … Device or resource busy`** — another process holds
  the camera (commonly a script running in **Thonny**, or another
  `rpicam`/`picamera2` program). Close it. Find the holder:
  `fuser /dev/media0 /dev/video0`.
- **Model won't load** — run `bash scripts/setup_model.sh`; ensure `models/` has
  `FastSAM-s_ncnn_model/` or `FastSAM-s.pt`.
- **Masks lag / refresh slowly** — expected on CPU (segmentation ~1–4 FPS). Lower
  `--imgsz` or the capture resolution. Only the mask overlay lags; video is smooth.
- **Rebuilding the venv** — `uv venv --python 3.12 --clear && uv sync`.

## Key constraints (why it's built this way)

- Real SAM / MobileSAM / EdgeSAM automatic "everything" mode is multi-pass and too
  slow on a Pi CPU; FastSAM's single-pass mask generation is the practical choice.
- The Pi camera is a **single-holder** device — only one process at a time.
