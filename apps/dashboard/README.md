# Dashboard — todo list on the wall

A Raspberry Pi 5 shows a fullscreen todo list on an attached monitor. Add and
check off items from your phone, on the same wifi. Everything is self-hosted;
no accounts, no internet needed.

```
phone browser ──POST──▶ Flask ──▶ SQLite
                          │
monitor (Chromium kiosk) ─┘  polls GET /api/todos every 3s
```

This is phase 1 of the smart display described in
`docs/superpowers/specs/2026-07-18-smart-display-design.md`. Calendar events,
travel times, and leave-by alerts come in phases 2 and 3.

## Prerequisites

- Raspberry Pi 5 with a monitor attached, on the same wifi as your phone
- [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Setup

```bash
uv sync
```

## Run

```bash
uv run python main.py
```

Then open the printed URLs — `/display` on the monitor, `/phone` from your
phone.

### Options

```bash
uv run python main.py --help
  --host 0.0.0.0 --port 8001
  --db ~/.local/share/dashboard/dashboard.db
```

## Run on boot

Two pieces: the server (systemd user service) and the browser (compositor
autostart).

```bash
mkdir -p ~/.config/systemd/user
cp scripts/dashboard.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now dashboard
systemctl --user status dashboard    # confirm it is running
```

`dashboard.service` has `WorkingDirectory` set to this repo's location on the
Pi (`~/Documents/niall/leenapi/apps/dashboard`). If you clone the repo
somewhere else, edit that line to match before enabling the service.

For the browser, check your compositor with `echo $XDG_CURRENT_DESKTOP`, then
add `scripts/kiosk.sh` to its autostart — see the comments at the top of that
file. This Pi runs **labwc**, so add a line to `~/.config/labwc/autostart`:

```bash
mkdir -p ~/.config/labwc
echo "$HOME/Documents/niall/leenapi/apps/dashboard/scripts/kiosk.sh &" >> ~/.config/labwc/autostart
```

Reboot to confirm both come up.

## Tests

```bash
uv run pytest
```

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Entry point: CLI flags, wires Store into the Flask app |
| `src/store.py` | SQLite persistence for todos. No Flask dependency. |
| `src/app.py` | Flask factory, JSON API, page routes |
| `templates/display.html` | Monitor view — large type, polls every 3s |
| `templates/phone.html` | Phone view — add, check off, delete |
| `scripts/dashboard.service` | systemd user service for the server |
| `scripts/kiosk.sh` | Launches Chromium fullscreen at login |

## Notes

**There is no authentication.** Anyone on your wifi can edit the list. This is
deliberate for a home wall display; do not expose the port to the internet.

## Troubleshooting

- **Phone can't reach it** — confirm the phone is on the same wifi (not
  cellular, not a guest network), and use the LAN IP the server prints, not
  `localhost`.
- **Display is blank or shows a Chromium error** — the browser started before
  the server. `systemctl --user status dashboard` shows whether the server is
  up; `kiosk.sh` already waits up to 30s for it.
- **Screen blanks after a few minutes** — disable the screensaver in
  Raspberry Pi Configuration → Display, or install `xdg-utils` and run
  `xset s off -dpms` from the autostart.
- **Todos vanished** — the database lives at
  `~/.local/share/dashboard/dashboard.db`, not in the repo. Check the `--db`
  path the service is using.
