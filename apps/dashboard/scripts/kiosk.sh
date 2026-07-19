#!/usr/bin/env bash
# Launches Chromium fullscreen on the dashboard, for autostart at login.
#
# Wire it up by adding this line to the compositor's autostart file:
#   labwc:   ~/.config/labwc/autostart
#   wayfire: ~/.config/wayfire.ini  (under [autostart], as: dashboard = /path/to/kiosk.sh)
# Check which one you have: echo $XDG_CURRENT_DESKTOP
# This Pi runs labwc, so the target is ~/.config/labwc/autostart.
set -euo pipefail

URL="http://localhost:8001/display"

# The browser binary is `chromium` on this Pi's Raspberry Pi OS; other distros
# ship it as `chromium-browser`. Prefer whichever exists.
CHROMIUM="$(command -v chromium || command -v chromium-browser)"

# Wait for the server to accept connections; Chromium shows an error page if
# it starts first, and never retries on its own.
for _ in $(seq 1 30); do
  if curl -sf -o /dev/null "$URL"; then break; fi
  sleep 1
done

# Clear crash flags so no "Restore pages?" bubble covers the display after an
# unclean shutdown.
PREFS="$HOME/.config/chromium/Default/Preferences"
if [ -f "$PREFS" ]; then
  sed -i 's/"exited_cleanly":false/"exited_cleanly":true/; s/"exit_type":"Crashed"/"exit_type":"Normal"/' "$PREFS"
fi

# This Pi runs a Wayland session (labwc). Without --ozone-platform=wayland
# Chromium defaults to the X11 backend and dies with "Missing X server or
# $DISPLAY". The autostart inherits WAYLAND_DISPLAY from the labwc session.
exec "$CHROMIUM" \
  --ozone-platform=wayland \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --check-for-update-interval=31536000 \
  "$URL"
