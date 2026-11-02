import json

import pytest

from memory.adapters.migrate_faiss import migrate_faiss
from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.entities import MemoryKind


@pytest.fixture
def store(clock):
    s = SqliteMemoryStore(":memory:", clock)
    yield s
    s.close()


ROWS = [
    {"text": "what time is it", "intent": "quick", "timestamp": 100.0},
    {"text": "design a scalable architecture", "intent": "think", "timestamp": 200.0},
    {"text": "fix the bug in router.py", "intent": "code", "timestamp": 300.0},
]


def test_imports_rows_as_route_examples(store, embedder, tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(ROWS))
    assert migrate_faiss(store, embedder, str(path)) == 3
    rows = store.list(kinds=[MemoryKind.ROUTE_EXAMPLE])
    assert [(m.content, m.label, m.created_at) for m in rows] == [
        (r["text"], r["intent"], r["timestamp"]) for r in ROWS
    ]


def test_second_call_is_noop(store, embedder, tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(ROWS))
    migrate_faiss(store, embedder, str(path))
    assert migrate_faiss(store, embedder, str(path)) == 0
    assert len(store.list(kinds=[MemoryKind.ROUTE_EXAMPLE])) == 3


def test_missing_file_returns_zero(store, embedder, tmp_path):
    assert migrate_faiss(store, embedder, str(tmp_path / "nope.json")) == 0
