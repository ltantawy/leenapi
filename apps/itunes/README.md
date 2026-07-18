# Pi Segment — live FastSAM "segment everything" on the Raspberry Pi 5 camera

Captures the Pi camera and runs **FastSAM** (a CNN Segment-Anything variant) in
"segment everything" mode, drawing every detected object/region in its own color
over the live scene — the [Segment Anything](https://github.com/facebookresearch/segment-anything)
look, on the Pi 5 CPU. Served as a headless **MJPEG stream** you can open in any
browser.

```
rpicam-vid (MJPEG) ─► OpenCV decode ─┬─► blend latest overlay ─► HTTP MJPEG stream
                                     └─► FastSAM (bg thread)  ─► colorized overlay
```

## How it works (important)

Real Segment Anything is far too heavy for a Pi CPU, and even lightweight SAM
variants' automatic "everything" mode is multi-pass. **FastSAM** gets all masks
in a single forward pass, but still only reaches ~1–4 FPS on the Pi 5 CPU. So the
pipeline is **decoupled**:

- The camera streams **smooth live video** at the camera framerate.
- **FastSAM runs in a background thread** (~1–4 FPS); its most recent colorized
  mask overlay is blended onto every live frame.
- Consequence: masks **lag moving objects** by a fraction of a second. Intentional
  tradeoff for smooth video.

## Prerequisites

- Raspberry Pi 5 (8 GB recommended — torch/ultralytics are heavy), Pi camera
  (tested: Camera Module 3 NoIR / `imx708`)
- `rpicam-apps` (`rpicam-vid` on PATH — already present on Raspberry Pi OS)
- [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Setup

```bash
uv sync                        # creates the venv and installs deps (torch is large)
bash scripts/setup_model.sh    # fetches FastSAM-s + exports it to NCNN (fast CPU path)
```

The first run needs internet once to download the model; afterwards it's local.

## Run

```bash
uv run python main.py
# -> Serving segmented stream at http://<pi-ip>:8000/
```

Open `http://<pi-ip>:8000/` from any device on the network.

### Options

```bash
uv run python main.py --help
  --host 0.0.0.0 --port 8000
  --width 1280 --height 720 --framerate 15
  --alpha 0.5          # mask overlay opacity (0..1)
  --quality 80         # output JPEG quality
  --model FastSAM-s    # FastSAM .pt name or NCNN model dir
  --imgsz 512          # FastSAM inference size (smaller = faster, coarser)
  --conf 0.4           # confidence threshold (how many masks appear)
  --iou 0.9            # NMS IoU threshold
  --retina-masks       # full-resolution mask edges (slower, cleaner)
```

Segmentation is the bottleneck: lower `--imgsz` (e.g. 320) for a faster mask
refresh, raise it (e.g. 640) for sharper masks. Video stays smooth regardless.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | HTTP MJPEG server; decoupled capture / segment / blend threads |
| `src/camera.py` | `rpicam-vid` subprocess → decoded BGR frames |
| `src/segmenter.py` | FastSAM segment-everything → colorized overlay |
| `models/FastSAM-s_ncnn_model/` | NCNN-exported model (generated; gitignored) |
| `scripts/setup_model.sh` | Fetches FastSAM + exports to NCNN |

## Troubleshooting

- **`failed to acquire camera … Device or resource busy`** — another process holds
  the camera (e.g. a script running in Thonny, or another `rpicam`/`picamera2`
  program). Close it: `fuser /dev/media0 /dev/video0` shows the holding PID.
- **Model won't load** — run `bash scripts/setup_model.sh`; check `models/` has
  either `FastSAM-s_ncnn_model/` or `FastSAM-s.pt`.
- **Masks refresh slowly / lag a lot** — expected on CPU. Lower `--imgsz`, or drop
  `--width/--height`. The video itself stays smooth; only the mask overlay lags.
