"""Live FastSAM "segment-everything" MJPEG server for the Raspberry Pi camera.

Decoupled pipeline: capture always jumps to the newest camera frame (dropping any
backlog, so latency stays low) while FastSAM runs in a background thread (~1-10 FPS
on the Pi 5 CPU). By default the display is the segmentation itself — solid color
blocks with temporally-stable colors — crossfaded from one pass to the next so it
plays smoothly at the full stream frame rate. Pass --overlay to blend the colors
over the live video instead. The blocks lag moving objects by the segmentation
interval: an intentional tradeoff for a stable, smooth picture.

    rpicam-vid (MJPEG) ─► newest frame ─┬─► crossfade + blend ─► MJPEG stream (HTTP)
                                        └─► FastSAM (bg thread) ─► color-stable blocks

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
import numpy as np

from src.camera import CameraConfig, RpicamMjpegCamera
from src.segmenter import Segmenter

_INDEX_HTML = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pi Segmentation</title>
<style>
  body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center}
  h1{font-size:1rem;font-weight:600;padding:.6rem;margin:0}
  img{max-width:100%;height:auto}
</style></head>
<body><h1>Raspberry Pi 5 &mdash; live FastSAM segment-everything blocks</h1>
<img src="/stream" alt="segmented stream"></body></html>
"""


class Pipeline:
    """Capture + blend on the fast path; FastSAM segmentation on a slow bg thread.

    Three threads share state: the capture loop publishes each blended frame to
    HTTP clients and hands the raw frame to the segment worker; the segment
    worker produces the colorized overlay reused across many frames.
    """

    def __init__(
        self,
        camera: RpicamMjpegCamera,
        segmenter: Segmenter,
        jpeg_quality: int = 80,
        smooth: bool = True,
    ):
        self._camera = camera
        self._segmenter = segmenter
        self._jpeg_quality = jpeg_quality
        self._smooth = smooth
        self._running = False

        # Latest raw frame handed from capture -> segment worker.
        self._raw = None
        self._raw_id = 0
        self._raw_cond = threading.Condition()

        # Segmentation only updates a few times a second, so the color blocks
        # would visibly pop from one pass to the next. To raise the *displayed*
        # frame rate we crossfade from the previous overlay (``_ov_from``) to the
        # newest one (``_ov_to``) across the many live frames captured in between,
        # over ``_ov_dur`` seconds. The duration tracks the measured segmentation
        # interval so each fade finishes just as the next pass lands — a smooth
        # morph at the full stream rate instead of a ~5 FPS stutter.
        self._ov_from = None  # (color_bgr, alpha_map)
        self._ov_to = None  # (color_bgr, alpha_map)
        self._ov_start = 0.0
        self._ov_dur = 0.25
        self._overlay_lock = threading.Lock()

        # Latest encoded JPEG served to clients.
        self._latest: bytes | None = None
        self._out_cond = threading.Condition()

        self._video_fps = 0.0
        self._seg_fps = 0.0

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._capture_loop, name="capture", daemon=True).start()
        threading.Thread(target=self._segment_loop, name="segment", daemon=True).start()

    def _capture_loop(self) -> None:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        last = time.monotonic()
        for frame in self._camera.frames():
            if not self._running:
                break

            # Hand the newest raw frame to the segment worker.
            with self._raw_cond:
                self._raw = frame
                self._raw_id += 1
                self._raw_cond.notify_all()

            # Blend the most recent overlay (if any) onto this live frame,
            # crossfading between the previous and newest segmentation for a
            # smooth, high-frame-rate morph rather than a hard pop each pass.
            overlay = self._current_overlay(time.monotonic())
            if overlay is not None:
                display = self._segmenter.blend(frame, overlay[0], overlay[1])
            else:
                display = frame

            now = time.monotonic()
            dt = now - last
            last = now
            if dt > 0:
                self._video_fps = 0.9 * self._video_fps + 0.1 * (1.0 / dt)
            cv2.putText(
                display, f"{self._video_fps:4.1f} FPS video  {self._seg_fps:4.1f} FPS seg",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )

            ok, jpeg = cv2.imencode(".jpg", display, encode_params)
            if not ok:
                continue
            with self._out_cond:
                self._latest = jpeg.tobytes()
                self._out_cond.notify_all()

    def _segment_loop(self) -> None:
        processed_id = 0
        last = time.monotonic()
        while self._running:
            with self._raw_cond:
                while self._running and (self._raw is None or self._raw_id == processed_id):
                    self._raw_cond.wait(timeout=1.0)
                if not self._running:
                    break
                frame = self._raw
                processed_id = self._raw_id

            color, alpha = self._segmenter.segment(frame)

            now = time.monotonic()
            dt = now - last
            last = now
            if dt > 0:
                self._seg_fps = 0.7 * self._seg_fps + 0.3 * (1.0 / dt)

            with self._overlay_lock:
                # Start a fresh crossfade: fade from whatever is on screen now
                # (the last target, or this new one on the very first pass) to
                # the new overlay. Aim the fade to finish about when the next
                # pass lands by tracking the measured segmentation interval.
                self._ov_from = self._ov_to if self._ov_to is not None else (color, alpha)
                self._ov_to = (color, alpha)
                self._ov_start = now
                if self._smooth and dt > 0:
                    self._ov_dur = 0.6 * self._ov_dur + 0.4 * min(max(dt, 0.08), 0.8)
                else:
                    self._ov_dur = 0.0

    def _current_overlay(self, now: float):
        """The overlay to display now, crossfaded between the last two passes.

        Returns ``(color_bgr, alpha_map)`` or ``None`` before the first pass.
        In steady state (fade complete) it returns the newest overlay directly
        with no per-frame math; only mid-fade does it interpolate.
        """
        with self._overlay_lock:
            ov_from, ov_to = self._ov_from, self._ov_to
            start, dur = self._ov_start, self._ov_dur
        if ov_to is None:
            return None
        if ov_from is None or dur <= 0.0:
            return ov_to
        t = (now - start) / dur
        if t >= 1.0:
            return ov_to
        if t <= 0.0:
            return ov_from
        # Linear crossfade. With temporally-stable colors most pixels are
        # unchanged between passes, so the fade is clean; only regions that
        # actually changed morph across the intermediate frames.
        cf = ov_from[0].astype(np.float32)
        ct = ov_to[0].astype(np.float32)
        color = (cf + (ct - cf) * t).astype(np.uint8)
        af = ov_from[1]
        at = ov_to[1]
        alpha = af + (at - af) * t
        return color, alpha

    def snapshot(self, prev_id: int) -> tuple[bytes, int]:
        """Block until a frame newer than prev_id is available."""
        with self._out_cond:
            while self._latest is None or id(self._latest) == prev_id:
                if not self._running:
                    return b"", prev_id
                self._out_cond.wait(timeout=5)
            return self._latest, id(self._latest)

    def stop(self) -> None:
        self._running = False
        with self._raw_cond:
            self._raw_cond.notify_all()
        with self._out_cond:
            self._out_cond.notify_all()


def make_handler(pipeline: Pipeline):
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
                    jpeg, prev = pipeline.snapshot(prev)
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
    parser.add_argument("--alpha", type=float, default=0.5, help="mask overlay opacity 0..1")
    parser.add_argument("--quality", type=int, default=80, help="output JPEG quality")
    parser.add_argument("--model", default="FastSAM-s", help="FastSAM .pt or NCNN model dir")
    parser.add_argument("--imgsz", type=int, default=320, help="FastSAM inference size (smaller=faster)")
    parser.add_argument("--conf", type=float, default=0.4, help="FastSAM confidence threshold")
    parser.add_argument("--iou", type=float, default=0.9, help="FastSAM NMS IoU threshold")
    parser.add_argument("--retina-masks", action="store_true", help="full-resolution mask edges (slower)")
    parser.add_argument(
        "--bg-color", default="40,40,40",
        help="B,G,R color for pixels no mask covers (whole frame stays color-blocked)",
    )
    parser.add_argument(
        "--bg-alpha", type=float, default=0.85,
        help="overlay-mode opacity of the background block (higher=more solid fill)",
    )
    parser.add_argument(
        "--overlay", action="store_true",
        help="blend colors over the live video instead of showing solid blocks "
             "(default: solid segmented blocks, no video underneath)",
    )
    parser.add_argument(
        "--smooth", action=argparse.BooleanOptionalAction, default=True,
        help="crossfade between segmentation passes for a higher displayed frame "
             "rate (default: on; --no-smooth to disable)",
    )
    args = parser.parse_args()

    try:
        bg_color = tuple(int(c) for c in args.bg_color.split(","))
        if len(bg_color) != 3:
            raise ValueError
    except ValueError:
        print("--bg-color must be three comma-separated ints, e.g. 40,40,40", file=sys.stderr)
        return 1

    camera = RpicamMjpegCamera(
        CameraConfig(width=args.width, height=args.height, framerate=args.framerate)
    )
    try:
        segmenter = Segmenter(
            model=args.model,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            alpha=args.alpha,
            retina_masks=args.retina_masks,
            bg_color=bg_color,
            bg_alpha=args.bg_alpha,
            overlay=args.overlay,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1

    pipeline = Pipeline(camera, segmenter, jpeg_quality=args.quality, smooth=args.smooth)
    pipeline.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(pipeline))
    url = f"http://{_lan_ip()}:{args.port}/"
    print(f"Serving segmented stream at {url}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pipeline.stop()
        server.shutdown()
        camera.close()
        segmenter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
