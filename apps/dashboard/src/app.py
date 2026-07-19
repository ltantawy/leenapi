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
