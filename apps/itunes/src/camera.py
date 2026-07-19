"""Raspberry Pi camera capture via an ``rpicam-vid`` subprocess.

We deliberately avoid ``picamera2``: it is a system (apt) package tied to the
system Python 3.13, while this app runs on a uv-managed Python 3.12 venv (the
last MediaPipe with an aarch64 wheel is cp312). Piping MJPEG from ``rpicam-vid``
decouples capture from the Python version entirely.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

# JPEG stream markers.
_SOI = b"\xff\xd8"  # Start Of Image
_EOI = b"\xff\xd9"  # End Of Image


@dataclass
class CameraConfig:
    width: int = 1280
    height: int = 720
    framerate: int = 15
    # Passed through to rpicam-vid; -t 0 means run forever.
    extra_args: tuple[str, ...] = ()


class RpicamMjpegCamera:
    """Yields decoded BGR frames from ``rpicam-vid --codec mjpeg``."""

    def __init__(self, config: CameraConfig | None = None):
        self.config = config or CameraConfig()
        self._proc: subprocess.Popen[bytes] | None = None

    def _binary(self) -> str:
        for name in ("rpicam-vid", "libcamera-vid"):
            path = shutil.which(name)
            if path:
                return path
        raise RuntimeError(
            "Neither 'rpicam-vid' nor 'libcamera-vid' found on PATH. "
            "Install the Raspberry Pi camera apps (rpicam-apps)."
        )

    def start(self) -> None:
        cmd = [
            self._binary(),
            "-t", "0",
            "--codec", "mjpeg",
            "--width", str(self.config.width),
            "--height", str(self.config.height),
            "--framerate", str(self.config.framerate),
            "--inline",
            "--nopreview",
            "--flush",
            "-o", "-",
            *self.config.extra_args,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def frames(self) -> Iterator[np.ndarray]:
        """Generator of BGR frames (H, W, 3) uint8."""
        if self._proc is None:
            self.start()
        assert self._proc is not None and self._proc.stdout is not None

        buf = bytearray()
        read = self._proc.stdout.read
        while True:
            chunk = read(65536)
            if not chunk:
                break  # subprocess ended
            buf += chunk

            # Drain the buffer to the *newest* complete JPEG, discarding any
            # older complete frames without decoding them. This is the key
            # anti-lag step: while a slow consumer (blend + JPEG re-encode) is
            # busy, rpicam keeps producing and frames pile up in this buffer and
            # the OS pipe. Decoding and yielding every one of them would make us
            # forever process stale frames, so end-to-end latency would grow and
            # stay high. Skipping straight to the latest keeps the displayed
            # image fresh — bounded to about one frame of lag regardless of how
            # far behind we fall.
            latest_jpeg = None
            while True:
                start = buf.find(_SOI)
                if start < 0:
                    # No frame start yet; keep only a tail in case a marker
                    # straddles the next chunk.
                    if len(buf) > 1:
                        del buf[:-1]
                    break
                end = buf.find(_EOI, start + 2)
                if end < 0:
                    # Incomplete frame; drop bytes before the start marker and
                    # wait for the rest on the next read.
                    if start > 0:
                        del buf[:start]
                    break
                end += 2  # include EOI
                latest_jpeg = bytes(buf[start:end])
                del buf[:end]
                # Loop again: if another complete frame is already buffered we
                # overwrite latest_jpeg with it and drop this (older) one.

            if latest_jpeg is not None:
                frame = cv2.imdecode(
                    np.frombuffer(latest_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    yield frame

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def __enter__(self) -> "RpicamMjpegCamera":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def frames(config: CameraConfig | None = None) -> Iterator[np.ndarray]:
    """Convenience generator that manages the camera lifecycle."""
    cam = RpicamMjpegCamera(config)
    try:
        yield from cam.frames()
    finally:
        cam.close()
