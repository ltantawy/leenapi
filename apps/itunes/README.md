# Pi Segment — live FastSAM "segment everything" on the Raspberry Pi 5 camera

Captures the Pi camera and runs **FastSAM** (a CNN Segment-Anything variant) in
"segment everything" mode, painting every detected object/region in its own solid
color — the [Segment Anything](https://github.com/facebookresearch/segment-anything)
look, on the Pi 5 CPU. Served as a headless **MJPEG stream** you can open in any
browser.

By default it shows **solid segmented blocks** (no live video underneath): every
pixel is a solid color, so the background fill is unambiguous. Pass `--overlay` to
instead blend the colors translucently over the live video.

```
rpicam-vid (MJPEG) ─► OpenCV decode ─┬─► crossfade + blend ─► HTTP MJPEG stream
                                     └─► FastSAM (bg thread) ─► color-stable blocks
```

## How it works (important)

Real Segment Anything is far too heavy for a Pi CPU, and even lightweight SAM
variants' automatic "everything" mode is multi-pass. **FastSAM** gets all masks
in a single forward pass, but still only reaches ~1–10 FPS on the Pi 5 CPU. So the
pipeline is **decoupled**:

- The camera streams frames at the camera framerate; capture always jumps to the
  **newest** frame, dropping any that piled up while busy — this bounds the
  real-life-to-web latency to about one frame instead of letting it grow.
- **FastSAM runs in a background thread** (~1–10 FPS); its most recent colorized
  result is the display.
- **Stable colors:** a lightweight tracker matches each pass's masks to recent
  ones by centroid + area and carries each region's color forward (surviving brief
  FastSAM dropouts), so colors follow object identity instead of flashing a new
  color every pass.
- **Smooth playback:** because segmentation only updates a few times a second, the
  display **crossfades** from the previous result to the newest one across the
  live frames in between — a smooth morph at the full stream frame rate rather than
  a ~5 FPS stutter. Disable with `--no-smooth`.
- Consequence: the blocks **lag moving objects** by a fraction of a second (the
  segmentation interval). Intentional tradeoff for a stable, smooth picture.

## Prerequisites

- Raspberry Pi 5 (4 GB works; torch/ultralytics are heavy — see the memory note below), Pi camera
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
  --overlay            # blend colors over live video instead of solid blocks
  --no-smooth          # disable the crossfade between segmentation passes
  --alpha 0.5          # instance-mask opacity in --overlay mode (0..1)
  --bg-color 255,255,255  # B,G,R fill for pixels no mask covers
  --bg-alpha 0.85      # background block opacity in --overlay mode
  --stability 0.5      # hold regions through FastSAM dropouts (0=crisp, 1=static)
  --no-vector          # upscale the low-res mask grid instead of smooth polygons
  --edge-smooth 0.5    # how much to round the vector block outlines (0..1)
  --quality 80         # output JPEG quality
  --model FastSAM-s    # FastSAM .pt name or NCNN model dir
  --imgsz 320          # FastSAM inference size (smaller = faster, coarser)
  --conf 0.4           # confidence threshold (how many masks appear)
  --iou 0.9            # NMS IoU threshold
  --retina-masks       # full-resolution mask edges (slower, cleaner)
```

**Default (blocks) mode:** every pixel is a solid color — instance masks get
distinct, temporally-stable colors and all remaining pixels get `--bg-color`.
Nothing is see-through, so full coverage is obvious.

**`--overlay` mode:** the colors are blended over the live video — instances at
`--alpha`, background at `--bg-alpha` (near-solid so it still reads as a filled
block). `--bg-color`, `--bg-alpha` and `--alpha` only affect this mode.

**Smooth (vector) edges:** FastSAM produces masks at `--imgsz` (320), so painting
them by upscaling that grid to a 1280x720 frame puts a hard 4-pixel staircase on
every boundary. Instead the mask outlines are traced as polygons, simplified and
corner-rounded in point space, then filled at full frame resolution with subpixel
anti-aliased edges — the blocks scale up as *shapes*, not as pixels. Raise
`--edge-smooth` for softer, blobbier outlines or drop it to 0 to follow the mask
faithfully (still anti-aliased, just not rounded). `--no-vector` restores the old
pixelated upscale. On this Pi the vector path costs ~19 ms/pass over the old one.

Segmentation is the bottleneck: the **NCNN** model path is by far the fastest on
the Pi 5 CPU (~10 FPS at `--imgsz 320` vs ~1.3 FPS on the torch `.pt` path) — run
`scripts/setup_model.sh` so `models/FastSAM-s_ncnn_model/` exists. Lower `--imgsz`
(e.g. 256) for an even faster refresh, raise it (e.g. 448) for sharper masks.
Video stays smooth regardless.

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
- **Killed / `MemoryError` / OOM on a 4 GB Pi** — torch + FastSAM peaks around ~1–1.5 GB.
  Prefer the **NCNN** model path (`scripts/setup_model.sh`; much lower runtime RAM than
  the `.pt` torch path), keep `--imgsz` modest (≤512), close other apps, and make sure
  swap is on (`sudo dphys-swapfile` — 2 GB is plenty of headroom). If it still won't fit,
  the fallback is FastSAM → ONNX + onnxruntime (drops the torch runtime entirely).
