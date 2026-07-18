# Pi Segment — build notes & usage

Live semantic-segmentation app for the Raspberry Pi 5 camera. Captures frames,
runs MediaPipe DeepLabV3 on each one, and serves the color-coded overlay as an
MJPEG stream viewable in a browser (fully headless).

```
rpicam-vid (MJPEG) ──▶ OpenCV decode ──▶ MediaPipe DeepLabV3 ──▶ overlay ──▶ HTTP MJPEG stream (:8000)
```

## What was done

1. **Installed `uv`** (`~/.local/bin/uv`, v0.11.29).
2. **Created a uv project + Python 3.12 venv** in `apps/itunes/`.
   - System Python is 3.13, but MediaPipe's last Linux **aarch64** wheel is
     **0.10.18 / cp312**, so the app runs on a uv-managed **Python 3.12**.
   - `pyproject.toml` pins `requires-python >=3.10,<3.13`, `mediapipe==0.10.18`,
     `opencv-python`, `numpy<2`. `.python-version` pins 3.12.
3. **Chose `rpicam-vid` over `picamera2`** for capture. `picamera2` is a system
   apt package bound to Python 3.13 and unusable from the 3.12 venv; piping MJPEG
   from `rpicam-vid` decouples capture from the Python version entirely.
4. **Wrote the app** (see Layout below) and **downloaded** `deeplab_v3.tflite`.
5. **Verified end-to-end:** deps import on 3.12; real 720p frames decode;
   segmentation ~90–110 ms/frame (≈10 FPS); `/` serves the page and `/stream`
   delivers valid multipart JPEG frames.

## How to use

```bash
cd apps/itunes

# one-time setup
uv sync                         # creates the 3.12 venv, installs deps
bash scripts/download_model.sh  # downloads models/deeplab_v3.tflite (~2.7 MB)

# run
uv run python main.py           # prints: Serving segmented stream at http://<pi-ip>:8000/
```

Then open `http://<pi-ip>:8000/` from any device on the network.

### Options

```bash
uv run python main.py --help
  --host 0.0.0.0 --port 8000
  --width 1280 --height 720 --framerate 15
  --alpha 0.5     # mask opacity (0..1)
  --quality 80    # output JPEG quality
```

Lower `--width/--height` for higher FPS. DeepLabV3 knows 21 classes
(person, chair, bottle, car, cat, dog, sofa, tv, …); unknown objects fall into
`background`.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | HTTP MJPEG server; wires camera → segmenter → overlay, FPS overlay, graceful shutdown |
| `src/camera.py` | `rpicam-vid` subprocess → decoded BGR frames (MJPEG SOI/EOI splitting) |
| `src/segmenter.py` | MediaPipe DeepLabV3 wrapper + colorized mask blend |
| `models/deeplab_v3.tflite` | Model (downloaded; gitignored) |
| `scripts/download_model.sh` | Fetches the model |
| `pyproject.toml` / `.python-version` / `uv.lock` | uv project pinned to Python 3.12 |

## Troubleshooting

- **`failed to acquire camera … Device or resource busy`** — another process holds
  the camera (commonly a script running in **Thonny**, or another
  `rpicam`/`picamera2` program). Close it. Find the holder:
  `fuser /dev/media0 /dev/video0`.
- **`Model not found`** — run `bash scripts/download_model.sh`.
- **Only `background`/one class shows** — expected; DeepLabV3 only knows its 21
  classes. Point the camera at people/furniture/vehicles/pets.
- **Rebuilding the venv** — `uv venv --python 3.12 --clear && uv sync`.

## Key constraints (why it's built this way)

- MediaPipe aarch64 wheels stop at **0.10.18 (cp312)** — no cp313, nothing newer
  for ARM Linux. Hence Python 3.12, not the system 3.13.
- The Pi camera is a **single-holder** device — only one process at a time.
