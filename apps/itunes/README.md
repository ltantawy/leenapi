# Pi Segment — live semantic segmentation on the Raspberry Pi 5 camera

Captures the Pi camera, runs **MediaPipe Image Segmenter (DeepLabV3, ~21
Pascal-VOC classes)** on every frame, and serves the color-coded overlay as a
live **MJPEG stream** you can open in any browser — fully headless.

```
rpicam-vid (MJPEG) ──▶ OpenCV decode ──▶ MediaPipe DeepLabV3 ──▶ overlay ──▶ HTTP MJPEG stream
```

## Why this setup (important)

- MediaPipe's last Linux **aarch64** wheel is **0.10.18 / cp312**. This Pi's
  system Python is 3.13, so the app runs in a **uv-managed Python 3.12** venv.
- `picamera2` is a system package bound to Python 3.13, so it can't be used from
  the 3.12 venv. Instead frames are piped from **`rpicam-vid`** — no `picamera2`
  dependency, no Python-version coupling.

## Prerequisites

- Raspberry Pi 5, Pi camera (tested: Camera Module 3 NoIR / `imx708`)
- `rpicam-apps` (`rpicam-vid` on PATH — already present on Raspberry Pi OS)
- [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Setup

```bash
uv sync                       # creates the 3.12 venv and installs deps
bash scripts/download_model.sh   # downloads models/deeplab_v3.tflite (~2.7 MB)
```

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
  --alpha 0.5        # mask opacity (0..1)
  --quality 80       # output JPEG quality
```

On the Pi 5 CPU, DeepLabV3 runs ~90–110 ms/frame (≈10 FPS). Lower `--width/--height`
for higher FPS.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | HTTP MJPEG server; wires camera → segmenter → overlay |
| `src/camera.py` | `rpicam-vid` subprocess → decoded BGR frames |
| `src/segmenter.py` | MediaPipe DeepLabV3 wrapper + colorized overlay |
| `models/deeplab_v3.tflite` | Model (downloaded; gitignored) |
| `scripts/download_model.sh` | Fetches the model |

## Troubleshooting

- **`failed to acquire camera … Device or resource busy`** — another process holds
  the camera (e.g. a script running in Thonny, or another `rpicam`/`picamera2`
  program). Close it: `fuser /dev/media0 /dev/video0` shows the holding PID.
- **Model not found** — run `bash scripts/download_model.sh`.
- **Only `person`/`background` show up** — DeepLabV3 knows 21 classes; point it at
  people, chairs, bottles, cars, pets, etc. Unknown objects fall into `background`.
