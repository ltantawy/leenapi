"""Wall dashboard: todo list on the monitor, editable from a phone.

Run:      uv run python main.py
Monitor:  http://localhost:8001/display
Phone:    http://<pi-ip>:8001/phone
"""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

from src.app import create_app
from src.store import Store


def _lan_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path.home() / ".local/share/dashboard/dashboard.db",
        help="SQLite database path",
    )
    args = parser.parse_args()

    app = create_app(Store(args.db))
    print(f"Monitor:  http://localhost:{args.port}/display")
    print(f"Phone:    http://{_lan_ip()}:{args.port}/phone")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
