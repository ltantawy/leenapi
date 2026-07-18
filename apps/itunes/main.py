"""Live semantic-segmentation MJPEG server for the Raspberry Pi camera.

Pipeline: rpicam-vid (MJPEG) -> OpenCV decode -> MediaPipe DeepLabV3 -> colorized
overlay -> multipart/x-mixed-replace MJPEG stream over HTTP.

Run:   uv run python main.py
View:  http://<pi-ip>:8000/
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from src.camera import CameraConfig, RpicamMjpegCamera
from src.segmenter import Segmenter

_INDEX_HTML = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pi Segmentation</title>
<style>
  body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center}
  h1{font-size:1rem;font-weight:600;padding:.6rem;margin:0}
  img{max-width:100%;height:auto}
</style></head>
<body><h1>Raspberry Pi 5 &mdash; live DeepLabV3 segmentation</h1>
<img src="/stream" alt="segmented stream"></body></html>
"""


class Broadcaster:
    """Runs capture+segmentation in one thread; serves the latest JPEG to clients."""

    def __init__(self, camera: RpicamMjpegCamera, segmenter: Segmenter, jpeg_quality: int = 80):
        self._camera = camera
        self._segmenter = segmenter
        self._jpeg_quality = jpeg_quality
        self._latest: bytes | None = None
        self._lock = threading.Condition()
        self._running = False
        self._fps = 0.0

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, name="capture", daemon=True).start()

    def _loop(self) -> None:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        last = time.monotonic()
        for frame in self._camera.frames():
            if not self._running:
                break
            overlay = self._segmenter.overlay(frame)

            now = time.monotonic()
            dt = now - last
            last = now
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            cv2.putText(
                overlay, f"{self._fps:4.1f} FPS", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
            )

            ok, jpeg = cv2.imencode(".jpg", overlay, encode_params)
            if not ok:
                continue
            with self._lock:
                self._latest = jpeg.tobytes()
                self._lock.notify_all()

    def snapshot(self, prev_id: int) -> tuple[bytes, int]:
        """Block until a frame newer than prev_id is available."""
        with self._lock:
            while self._latest is None or id(self._latest) == prev_id:
                if not self._running:
                    return b"", prev_id
                self._lock.wait(timeout=5)
            return self._latest, id(self._latest)

    def stop(self) -> None:
        self._running = False
        with self._lock:
            self._lock.notify_all()


def make_handler(broadcaster: Broadcaster):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quieter logs
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_INDEX_HTML)))
                self.end_headers()
                self.wfile.write(_INDEX_HTML)
            elif self.path == "/stream":
                self._stream()
            else:
                self.send_error(404)

        def _stream(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            prev = 0
            try:
                while True:
                    jpeg, prev = broadcaster.snapshot(prev)
                    if not jpeg:
                        break
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # client disconnected

    return Handler


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--framerate", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.5, help="mask opacity 0..1")
    parser.add_argument("--quality", type=int, default=80, help="output JPEG quality")
    args = parser.parse_args()

    camera = RpicamMjpegCamera(
        CameraConfig(width=args.width, height=args.height, framerate=args.framerate)
    )
    try:
        segmenter = Segmenter(alpha=args.alpha)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    broadcaster = Broadcaster(camera, segmenter, jpeg_quality=args.quality)
    broadcaster.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(broadcaster))
    url = f"http://{_lan_ip()}:{args.port}/"
    print(f"Serving segmented stream at {url}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        broadcaster.stop()
        server.shutdown()
        camera.close()
        segmenter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
