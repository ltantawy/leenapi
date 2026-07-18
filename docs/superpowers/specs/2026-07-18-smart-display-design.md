# Pi Smart Display — Todo List + Daily Calendar

**Date:** 2026-07-18
**Status:** Approved, not yet implemented

## Goal

A Raspberry Pi 5 drives a monitor showing a wall display: today's todo list
alongside today's calendar events. The todo list is editable from a phone on
the same network. Events with a location get a computed "leave by" time, and
the display alerts — visually and audibly — when it's time to leave.

## Context

- Raspberry Pi 5, plugged into an HDMI monitor. No camera set up yet.
- Development happens on a Mac; the Pi is the deployment target.
- Long-term, this is one app among several, switched by holding numbers up to
  a camera. That launcher is explicitly **out of scope** here. This app is
  built standalone and proves the web-page-per-app model works.

### Hardware constraint: audio

The Pi 5 has no 3.5mm audio jack (removed relative to the Pi 4). Sound must
come from one of:

- HDMI audio to the monitor's built-in speakers, if it has them
- A USB speaker or USB audio dongle (preferred if the monitor has no speakers)
- Bluetooth — not recommended; pairing at boot on a headless Pi is unreliable

**Open item:** confirm whether the monitor has speakers before phase 3.

## Repo conventions

This app lives at `apps/dashboard/`, following the layout established by
`apps/itunes/`:

- Self-contained directory under `apps/`, with its own `pyproject.toml`,
  `.python-version`, and `uv.lock`. Managed with `uv`.
- Entry point at `main.py`; supporting modules under `src/`.
- README with Prerequisites / Setup / Run / Layout / Troubleshooting sections.
- Headless — the UI is served over HTTP and opened from any device on the LAN.

**Deviation from `apps/itunes/`:** that app uses the stdlib `http.server` with
no web framework. This app uses **Flask**. The itunes constraint was a video
stream with minimal dependencies; this app has form posts, JSON endpoints, and
two rendered pages, where hand-rolled routing in `BaseHTTPRequestHandler` is
tedium without benefit. Decision confirmed with the repo owner.

There is no test suite anywhere in the repo yet; this app introduces one.

## Architecture

A single Python service on the Pi owns storage and serves two pages:

- `/display` — the monitor view. Large type, readable across a room. Todos on
  one side, today's schedule on the other. Read-only.
- `/phone` — the editing view, opened from a phone on the same wifi.

Chromium boots fullscreen (kiosk) into `/display`. The display polls the
server every few seconds for updates. WebSockets are deliberately not used —
at this scale polling is simpler and the latency difference is imperceptible.

Storage is SQLite. The list is self-hosted and works without internet;
calendar data is a cached mirror of Google Calendar.

### Modules

Separated along the seams most likely to break or be replaced:

| Module | Responsibility | Depends on |
|---|---|---|
| `store` | SQLite: todos, cached events, fired-notification state | — |
| `calendar_sync` | Poll Google Calendar, normalize events into local shape | `store` |
| `travel` | Destination → travel minutes, via Maps API, cached | — |
| `notify` | Pure logic: given events + current time, what is due? | — |
| `server` | HTTP routes, serves both pages | all |

`calendar_sync` is the only module that knows Google's data model. `travel` is
the only module that knows Maps. Both can be faked, so the display is
developable on the Mac with no network and no credentials.

`notify` performs no I/O, which makes the alerting logic directly testable.

### Leave-by computation

```
leave_by = event_start − travel_minutes − buffer
```

- Trip origin (home address) lives in a config file.
- `buffer` is configurable; default 5 minutes.
- Events without a location get no leave-by and display as normal entries.
- Travel results are cached aggressively — travel time to a given destination
  does not change meaningfully minute to minute, and every call is billable.

### Notification behavior

When `now >= leave_by`, the display shows a full-screen takeover with the
event name and "LEAVE NOW" in large type, and plays a sound.

- Fires **once** per event. The fired state is persisted, so a reboot does not
  re-trigger alerts for events earlier in the day.
- The banner dismisses itself after a few minutes, or when the event starts.

## Failure handling

| Failure | Behavior |
|---|---|
| Internet down | Show last-cached events with a subtle stale indicator. Never show an empty screen. |
| Maps API fails | Show the event without a leave-by. Do not crash. |
| OAuth expired | Visible banner on the display. Silent sync failure is the worst outcome. |

## Build phases

Each phase ends with something usable on the monitor.

**Phase 1 — Display + self-hosted todos**
Monitor shows the todo list; phone adds, checks off, and deletes items.
SQLite persistence, Chromium kiosk on boot. No Google, no Maps, no network
dependencies beyond the local wifi.

**Phase 2 — Calendar, read-only**
Today's Google Calendar events appear beside the todos. OAuth is set up in
this phase.

**Phase 3 — Leave-by and alerts**
Maps lookup for events with locations, leave-by computation, on-screen
takeover, and sound.

## Known setup hazards

- **Google OAuth refresh tokens expire after 7 days** while the Cloud project
  is in "Testing" status. Sync will silently stop weekly until the project is
  moved to "In production" — a form, not a review, for personal use with a
  single account. Address this during phase 2.
- **The Maps API requires billing enabled** — a real card on file, though the
  free tier covers this usage volume comfortably. If that is unacceptable, the
  fallback is a manually entered "travel minutes" field per event, which
  changes only the `travel` module.

## Testing

- Real unit tests for the leave-by math and the fire-once notification logic.
  These are where bugs actually cause harm (a missed or repeated alert).
- `calendar_sync` and `travel` are faked at their interfaces.
- No browser-level tests for the display pages.

## Explicitly out of scope

- The app launcher and camera-based number detection.
- Access from outside the home network.
- Authentication — the phone UI is unauthenticated on the local network.
- Multiple lists, multi-day calendar views, recurring todos.
