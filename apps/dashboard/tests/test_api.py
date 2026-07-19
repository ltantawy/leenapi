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


def test_phone_page_renders(client):
    response = client.get("/phone")

    assert response.status_code == 200
    assert b"<form" in response.data
