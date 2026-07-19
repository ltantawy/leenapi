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
