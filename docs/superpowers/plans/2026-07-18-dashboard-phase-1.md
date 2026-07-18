# Smart Display Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Raspberry Pi 5 serves a fullscreen todo list on an attached monitor, editable from a phone on the same wifi.

**Architecture:** A Flask app at `apps/dashboard/` owns a SQLite database of todo items and serves two pages: `/display` (large read-only view for the monitor, polls for updates every 3 seconds) and `/phone` (add, check off, delete). Chromium launches fullscreen into `/display` at boot. Storage logic lives in `src/store.py` with no Flask dependency, so it is testable without a server.

**Tech Stack:** Python 3.12, Flask, SQLite (stdlib `sqlite3`), pytest, `uv` for dependency management.

## Global Constraints

- App lives at `apps/dashboard/`, self-contained, following `apps/itunes/` layout: own `pyproject.toml`, `.python-version`, `uv.lock`; entry point `main.py`; modules under `src/`.
- Python 3.12 (`.python-version` contains `3.12`).
- Code style follows `apps/itunes/src/`: `from __future__ import annotations` at the top of every module, type hints on all public functions, dataclasses for records, module docstrings that explain *why* not just *what*.
- No authentication. The phone UI is unauthenticated on the LAN — this is a deliberate, documented decision.
- Server binds `0.0.0.0` so the phone can reach it. Default port `8001` (`apps/itunes` already uses 8000).
- Out of scope for phase 1: Google Calendar, Maps, travel times, notifications, sound. Do not build them.

---

### Task 1: Project scaffold and todo storage

**Files:**
- Create: `apps/dashboard/.python-version`
- Create: `apps/dashboard/.gitignore`
- Create: `apps/dashboard/pyproject.toml`
- Create: `apps/dashboard/src/__init__.py`
- Create: `apps/dashboard/src/store.py`
- Test: `apps/dashboard/tests/test_store.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `Todo` frozen dataclass with fields `id: int`, `text: str`, `done: bool`, `created_at: str`. `Store(db_path: Path)` with methods `add(text: str) -> Todo`, `list() -> list[Todo]`.

- [ ] **Step 1: Create the project scaffold**

`apps/dashboard/.python-version`:

```
3.12
```

`apps/dashboard/.gitignore`:

```
.venv/
__pycache__/
*.pyc
*.db
```

`apps/dashboard/pyproject.toml`:

```toml
[project]
name = "dashboard"
version = "0.1.0"
description = "Wall display for a Raspberry Pi 5: todo list on the monitor, editable from a phone"
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
    "flask>=3.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
# Without this, pytest puts tests/ on sys.path but not the project root, and
# `from src.store import Store` fails with ModuleNotFoundError.
pythonpath = ["."]
```

`apps/dashboard/src/__init__.py` is empty.

Then run:

```bash
cd apps/dashboard && uv sync
```

Expected: creates `.venv/` and `uv.lock`, installs Flask and pytest.

- [ ] **Step 2: Write the failing test**

`apps/dashboard/tests/test_store.py`:

```python
from __future__ import annotations

import pytest

from src.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


def test_add_returns_the_created_todo(store):
    todo = store.add("buy milk")

    assert todo.id > 0
    assert todo.text == "buy milk"
    assert todo.done is False


def test_list_returns_added_todos_oldest_first(store):
    store.add("first")
    store.add("second")

    texts = [t.text for t in store.list()]

    assert texts == ["first", "second"]


def test_list_is_empty_for_a_fresh_store(store):
    assert store.list() == []


def test_data_survives_reopening_the_database(tmp_path):
    path = tmp_path / "persist.db"
    Store(path).add("remember me")

    reopened = Store(path)

    assert [t.text for t in reopened.list()] == ["remember me"]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd apps/dashboard && uv run pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.store'`

- [ ] **Step 4: Write the implementation**

`apps/dashboard/src/store.py`:

```python
"""SQLite storage for todo items.

Deliberately free of any Flask import: the storage layer is exercised directly
in tests without starting a server, and later phases (calendar events, fired
notifications) add tables here without touching the web layer.

A connection is opened per operation rather than held open. Flask serves
requests on multiple threads, and SQLite connections cannot be shared across
threads; at this volume (a handful of writes a day) the cost is irrelevant.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS todos (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT    NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL
);
"""


@dataclass(frozen=True)
class Todo:
    id: int
    text: str
    done: bool
    created_at: str


class Store:
    """Todo persistence. Creates the database file and schema on first use."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, text: str) -> Todo:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO todos (text, done, created_at) VALUES (?, 0, ?)",
                (text, created_at),
            )
            return Todo(
                id=cursor.lastrowid,
                text=text,
                done=False,
                created_at=created_at,
            )

    def list(self) -> list[Todo]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, text, done, created_at FROM todos ORDER BY id"
            ).fetchall()
        return [
            Todo(
                id=row["id"],
                text=row["text"],
                done=bool(row["done"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd apps/dashboard && uv run pytest tests/test_store.py -v`
Expected: PASS, 4 passed

- [ ] **Step 6: Commit**

```bash
git add apps/dashboard
git commit -m "feat(dashboard): add project scaffold and todo storage"
```

---

### Task 2: Toggling and deleting todos

**Files:**
- Modify: `apps/dashboard/src/store.py`
- Modify: `apps/dashboard/tests/test_store.py`

**Interfaces:**
- Consumes: `Store`, `Todo` from Task 1.
- Produces: `Store.toggle(todo_id: int) -> None` flips the `done` flag. `Store.delete(todo_id: int) -> None` removes the row. Both are silent no-ops for an unknown id.

- [ ] **Step 1: Write the failing tests**

Append to `apps/dashboard/tests/test_store.py`:

```python
def test_toggle_marks_a_todo_done(store):
    todo = store.add("water plants")

    store.toggle(todo.id)

    assert store.list()[0].done is True


def test_toggle_twice_returns_to_not_done(store):
    todo = store.add("water plants")

    store.toggle(todo.id)
    store.toggle(todo.id)

    assert store.list()[0].done is False


def test_delete_removes_the_todo(store):
    todo = store.add("obsolete")

    store.delete(todo.id)

    assert store.list() == []


def test_toggle_unknown_id_is_a_no_op(store):
    store.add("untouched")

    store.toggle(9999)

    assert store.list()[0].done is False


def test_delete_unknown_id_is_a_no_op(store):
    store.add("untouched")

    store.delete(9999)

    assert len(store.list()) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd apps/dashboard && uv run pytest tests/test_store.py -v`
Expected: FAIL — `AttributeError: 'Store' object has no attribute 'toggle'`

- [ ] **Step 3: Write the implementation**

Add these two methods to the `Store` class in `apps/dashboard/src/store.py`, after `list`:

```python
    def toggle(self, todo_id: int) -> None:
        """Flip the done flag. Unknown ids are ignored — a stale phone tab
        deleting then toggling should not 500."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE todos SET done = 1 - done WHERE id = ?", (todo_id,)
            )

    def delete(self, todo_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd apps/dashboard && uv run pytest tests/test_store.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
git add apps/dashboard
git commit -m "feat(dashboard): add toggle and delete to todo storage"
```

---

### Task 3: JSON API

**Files:**
- Create: `apps/dashboard/src/app.py`
- Create: `apps/dashboard/tests/test_api.py`

**Interfaces:**
- Consumes: `Store`, `Todo` from Tasks 1-2.
- Produces: `create_app(store: Store) -> flask.Flask`. Routes: `GET /api/todos` → `{"todos": [{"id", "text", "done"}]}`; `POST /api/todos` with JSON `{"text": str}` → 201 and the created todo; `POST /api/todos/<int:todo_id>/toggle` → 204; `DELETE /api/todos/<int:todo_id>` → 204. Empty or whitespace-only text → 400.

The app factory takes a `Store` so tests can inject a temporary database.

- [ ] **Step 1: Write the failing tests**

`apps/dashboard/tests/test_api.py`:

```python
from __future__ import annotations

import pytest

from src.app import create_app
from src.store import Store


@pytest.fixture
def client(tmp_path):
    app = create_app(Store(tmp_path / "test.db"))
    app.config["TESTING"] = True
    return app.test_client()


def test_get_todos_is_empty_initially(client):
    response = client.get("/api/todos")

    assert response.status_code == 200
    assert response.get_json() == {"todos": []}


def test_post_creates_a_todo(client):
    response = client.post("/api/todos", json={"text": "buy milk"})

    assert response.status_code == 201
    assert response.get_json()["text"] == "buy milk"
    assert response.get_json()["done"] is False


def test_posted_todo_appears_in_the_list(client):
    client.post("/api/todos", json={"text": "buy milk"})

    todos = client.get("/api/todos").get_json()["todos"]

    assert [t["text"] for t in todos] == ["buy milk"]


def test_post_rejects_empty_text(client):
    response = client.post("/api/todos", json={"text": "   "})

    assert response.status_code == 400
    assert client.get("/api/todos").get_json() == {"todos": []}


def test_post_rejects_missing_text(client):
    response = client.post("/api/todos", json={})

    assert response.status_code == 400


def test_toggle_marks_done(client):
    todo_id = client.post("/api/todos", json={"text": "x"}).get_json()["id"]

    response = client.post(f"/api/todos/{todo_id}/toggle")

    assert response.status_code == 204
    assert client.get("/api/todos").get_json()["todos"][0]["done"] is True


def test_delete_removes_the_todo(client):
    todo_id = client.post("/api/todos", json={"text": "x"}).get_json()["id"]

    response = client.delete(f"/api/todos/{todo_id}")

    assert response.status_code == 204
    assert client.get("/api/todos").get_json() == {"todos": []}


def test_text_is_trimmed(client):
    response = client.post("/api/todos", json={"text": "  spaced  "})

    assert response.get_json()["text"] == "spaced"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd apps/dashboard && uv run pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app'`

- [ ] **Step 3: Write the implementation**

`apps/dashboard/src/app.py`:

```python
"""Flask application factory and routes.

The factory takes a Store rather than building one, so tests inject a
temporary database and the entry point supplies the real one.

The display and phone views share one JSON API. The display polls
GET /api/todos; the phone posts mutations and re-renders from the same
response shape, so there is exactly one description of a todo on the wire.
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from src.store import Store, Todo


def _serialize(todo: Todo) -> dict:
    return {"id": todo.id, "text": todo.text, "done": todo.done}


def create_app(store: Store) -> Flask:
    app = Flask(__name__)

    @app.get("/api/todos")
    def list_todos():
        return jsonify(todos=[_serialize(t) for t in store.list()])

    @app.post("/api/todos")
    def create_todo():
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        if not text:
            return jsonify(error="text is required"), 400
        return jsonify(_serialize(store.add(text))), 201

    @app.post("/api/todos/<int:todo_id>/toggle")
    def toggle_todo(todo_id: int):
        store.toggle(todo_id)
        return "", 204

    @app.delete("/api/todos/<int:todo_id>")
    def delete_todo(todo_id: int):
        store.delete(todo_id)
        return "", 204

    return app
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd apps/dashboard && uv run pytest -v`
Expected: PASS, 17 passed

- [ ] **Step 5: Commit**

```bash
git add apps/dashboard
git commit -m "feat(dashboard): add JSON API for todos"
```

---

### Task 4: Phone editing page

**Files:**
- Create: `apps/dashboard/templates/phone.html`
- Modify: `apps/dashboard/src/app.py`
- Modify: `apps/dashboard/tests/test_api.py`

**Interfaces:**
- Consumes: `create_app` and the JSON API from Task 3.
- Produces: `GET /phone` → 200, HTML. No new Python functions.

The page is one self-contained template — no build step, no framework, no external assets. It talks to the JSON API with `fetch`.

- [ ] **Step 1: Write the failing test**

Append to `apps/dashboard/tests/test_api.py`:

```python
def test_phone_page_renders(client):
    response = client.get("/phone")

    assert response.status_code == 200
    assert b"<form" in response.data
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd apps/dashboard && uv run pytest tests/test_api.py::test_phone_page_renders -v`
Expected: FAIL — 404

- [ ] **Step 3: Add the route**

Add to `create_app` in `apps/dashboard/src/app.py`, before `return app`:

```python
    @app.get("/phone")
    def phone():
        return render_template("phone.html")
```

- [ ] **Step 4: Write the template**

`apps/dashboard/templates/phone.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Todo</title>
  <style>
    :root { color-scheme: dark; }
    body {
      margin: 0; padding: 1rem;
      background: #111; color: #eee;
      font-family: system-ui, sans-serif;
    }
    form { display: flex; gap: .5rem; margin-bottom: 1rem; }
    input[type=text] {
      flex: 1; padding: .75rem; font-size: 1.1rem;
      border: 1px solid #444; border-radius: .4rem;
      background: #1c1c1c; color: #eee;
    }
    button {
      padding: .75rem 1rem; font-size: 1.1rem;
      border: 0; border-radius: .4rem;
      background: #2b6; color: #012; font-weight: 600;
    }
    ul { list-style: none; padding: 0; margin: 0; }
    li {
      display: flex; align-items: center; gap: .75rem;
      padding: .75rem .25rem; border-bottom: 1px solid #262626;
    }
    li .text { flex: 1; font-size: 1.1rem; }
    li.done .text { text-decoration: line-through; color: #777; }
    input[type=checkbox] { width: 1.4rem; height: 1.4rem; }
    .del {
      background: none; color: #a55; font-size: 1.4rem;
      padding: 0 .5rem;
    }
  </style>
</head>
<body>
  <form id="add">
    <input type="text" id="text" placeholder="Add a todo" autocomplete="off" required>
    <button type="submit">Add</button>
  </form>
  <ul id="list"></ul>

  <script>
    const list = document.getElementById("list");

    async function refresh() {
      const response = await fetch("/api/todos");
      const { todos } = await response.json();
      list.replaceChildren(...todos.map(render));
    }

    function render(todo) {
      const li = document.createElement("li");
      if (todo.done) li.classList.add("done");

      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = todo.done;
      box.addEventListener("change", async () => {
        await fetch(`/api/todos/${todo.id}/toggle`, { method: "POST" });
        refresh();
      });

      const span = document.createElement("span");
      span.className = "text";
      span.textContent = todo.text;

      const del = document.createElement("button");
      del.className = "del";
      del.textContent = "×";
      del.addEventListener("click", async () => {
        await fetch(`/api/todos/${todo.id}`, { method: "DELETE" });
        refresh();
      });

      li.append(box, span, del);
      return li;
    }

    document.getElementById("add").addEventListener("submit", async (event) => {
      event.preventDefault();
      const input = document.getElementById("text");
      const text = input.value.trim();
      if (!text) return;
      await fetch("/api/todos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      input.value = "";
      refresh();
    });

    refresh();
  </script>
</body>
</html>
```

Note: `textContent` is used rather than `innerHTML` throughout, so a todo containing `<script>` renders as text.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd apps/dashboard && uv run pytest -v`
Expected: PASS, 18 passed

- [ ] **Step 6: Commit**

```bash
git add apps/dashboard
git commit -m "feat(dashboard): add phone editing page"
```

---

### Task 5: Monitor display page and entry point

**Files:**
- Create: `apps/dashboard/templates/display.html`
- Create: `apps/dashboard/main.py`
- Modify: `apps/dashboard/src/app.py`
- Modify: `apps/dashboard/tests/test_api.py`

**Interfaces:**
- Consumes: `create_app` from Task 3, `Store` from Task 1.
- Produces: `GET /display` → 200, HTML. `GET /` redirects to `/display`. `main.py` runnable via `uv run python main.py` with `--host`, `--port`, `--db` flags.

- [ ] **Step 1: Write the failing tests**

Append to `apps/dashboard/tests/test_api.py`:

```python
def test_display_page_renders(client):
    response = client.get("/display")

    assert response.status_code == 200
    assert b"todo" in response.data.lower()


def test_root_redirects_to_display(client):
    response = client.get("/")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/display")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd apps/dashboard && uv run pytest tests/test_api.py -k "display or redirect" -v`
Expected: FAIL — 404

- [ ] **Step 3: Add the routes**

Add to `create_app` in `apps/dashboard/src/app.py`, before `return app`:

```python
    @app.get("/")
    def index():
        return redirect(url_for("display"))

    @app.get("/display")
    def display():
        return render_template("display.html")
```

Update the Flask import at the top of the file to:

```python
from flask import Flask, jsonify, redirect, render_template, request, url_for
```

- [ ] **Step 4: Write the display template**

`apps/dashboard/templates/display.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dashboard</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0; height: 100vh; overflow: hidden;
      background: #0b0b0d; color: #f2f2f2;
      font-family: system-ui, sans-serif;
      display: flex; flex-direction: column;
      padding: 3vh 4vw;
      cursor: none;
    }
    header { margin-bottom: 3vh; }
    #date { font-size: 4vh; font-weight: 600; }
    #clock { font-size: 9vh; font-weight: 200; line-height: 1; }
    h2 {
      font-size: 3vh; font-weight: 600; text-transform: uppercase;
      letter-spacing: .1em; color: #888; margin: 0 0 2vh;
    }
    ul { list-style: none; padding: 0; margin: 0; overflow: hidden; }
    li {
      font-size: 4.5vh; padding: 1.2vh 0;
      border-bottom: 1px solid #1e1e22;
    }
    li.done { color: #555; text-decoration: line-through; }
    #empty { font-size: 3.5vh; color: #555; }
  </style>
</head>
<body>
  <header>
    <div id="clock"></div>
    <div id="date"></div>
  </header>
  <h2>Todo</h2>
  <ul id="list"></ul>
  <div id="empty" hidden>Nothing on the list.</div>

  <script>
    const list = document.getElementById("list");
    const empty = document.getElementById("empty");

    function tick() {
      const now = new Date();
      document.getElementById("clock").textContent =
        now.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
      document.getElementById("date").textContent =
        now.toLocaleDateString([], {
          weekday: "long", month: "long", day: "numeric",
        });
    }

    async function refresh() {
      try {
        const response = await fetch("/api/todos");
        const { todos } = await response.json();
        empty.hidden = todos.length > 0;
        list.replaceChildren(...todos.map((todo) => {
          const li = document.createElement("li");
          if (todo.done) li.classList.add("done");
          li.textContent = todo.text;
          return li;
        }));
      } catch (err) {
        // Server restarting or briefly unreachable — keep showing the last
        // known list rather than blanking the screen.
      }
    }

    tick();
    refresh();
    setInterval(tick, 1000);
    setInterval(refresh, 3000);
  </script>
</body>
</html>
```

Sizes are in `vh` so the layout scales to whatever monitor is attached. The clock is client-side, so it stays correct between polls.

- [ ] **Step 5: Write the entry point**

`apps/dashboard/main.py`:

```python
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
```

The database defaults to `~/.local/share/dashboard/` rather than the repo, so a `git clean` never destroys the list. `Store.__init__` creates the directory.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd apps/dashboard && uv run pytest -v`
Expected: PASS, 20 passed

- [ ] **Step 7: Verify it runs end to end**

```bash
cd apps/dashboard && uv run python main.py
```

Open `http://localhost:8001/phone` in a browser, add a todo, then open
`http://localhost:8001/display` in a second tab. The item appears within
3 seconds. Check it off from the phone tab; the display shows it struck
through. Stop with Ctrl-C.

- [ ] **Step 8: Commit**

```bash
git add apps/dashboard
git commit -m "feat(dashboard): add monitor display page and entry point"
```

---

### Task 6: Run on boot, and document it

**Files:**
- Create: `apps/dashboard/scripts/dashboard.service`
- Create: `apps/dashboard/scripts/kiosk.sh`
- Create: `apps/dashboard/README.md`

**Interfaces:**
- Consumes: `main.py` from Task 5.
- Produces: no code interfaces. Deployment artifacts and documentation.

These files are installed on the Pi, not run from the repo checkout. They are not covered by tests; Step 4 verifies them on the actual Pi.

- [ ] **Step 1: Write the systemd unit**

`apps/dashboard/scripts/dashboard.service`:

```ini
# Runs the dashboard server as a user service.
# Install:
#   mkdir -p ~/.config/systemd/user
#   cp scripts/dashboard.service ~/.config/systemd/user/
#   systemctl --user daemon-reload
#   systemctl --user enable --now dashboard
# A user service only runs while the user is logged in. The Pi boots to a
# desktop session automatically, so this holds. If you later switch the Pi to
# console-only, run `sudo loginctl enable-linger $USER`.

[Unit]
Description=Wall dashboard server
After=network-online.target

[Service]
WorkingDirectory=%h/raspbpi/leenapi/apps/dashboard
ExecStart=/usr/bin/env uv run python main.py
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
```

- [ ] **Step 2: Write the kiosk launcher**

`apps/dashboard/scripts/kiosk.sh`:

```bash
#!/usr/bin/env bash
# Launches Chromium fullscreen on the dashboard, for autostart at login.
#
# Wire it up by adding this line to the compositor's autostart file:
#   labwc:   ~/.config/labwc/autostart
#   wayfire: ~/.config/wayfire.ini  (under [autostart], as: dashboard = /path/to/kiosk.sh)
# Check which one you have: echo $XDG_CURRENT_DESKTOP
set -euo pipefail

URL="http://localhost:8001/display"

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

exec chromium-browser \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --check-for-update-interval=31536000 \
  "$URL"
```

Then: `chmod +x apps/dashboard/scripts/kiosk.sh`

- [ ] **Step 3: Write the README**

`apps/dashboard/README.md`:

````markdown
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

For the browser, check your compositor with `echo $XDG_CURRENT_DESKTOP`, then
add `scripts/kiosk.sh` to its autostart — see the comments at the top of that
file. Reboot to confirm both come up.

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
````

- [ ] **Step 4: Verify on the Pi**

Copy the repo to the Pi if you are developing on the Mac, then run the setup
and boot steps from the README. Reboot the Pi. Expected: the monitor comes up
in fullscreen showing the clock and the list, with no visible browser
chrome, and the phone can still add items.

- [ ] **Step 5: Commit**

```bash
git add apps/dashboard
git commit -m "feat(dashboard): add boot scripts and README"
```

---

## Done when

- `uv run pytest` passes in `apps/dashboard/`.
- Adding a todo on the phone appears on the monitor within 3 seconds.
- Rebooting the Pi brings the display back with no manual intervention.
- The todo list survives a reboot.
