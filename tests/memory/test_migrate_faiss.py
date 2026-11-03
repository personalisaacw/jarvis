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


def test_truncated_json_leaves_store_and_file_untouched(store, embedder, tmp_path, capsys):
    path = tmp_path / "metadata.json"
    text = json.dumps(ROWS)[:-20]
    path.write_text(text)
    assert migrate_faiss(store, embedder, str(path)) == 0
    assert store.list(None) == []
    assert path.read_text() == text
    out = capsys.readouterr().out
    assert "[Memory] WARNING: FAISS migration failed" in out
    assert "leaving .vectorstore untouched" in out


def test_row_missing_timestamp_imports_nothing(store, embedder, tmp_path, capsys):
    rows = [dict(r) for r in ROWS]
    del rows[2]["timestamp"]
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(rows))
    assert migrate_faiss(store, embedder, str(path)) == 0
    assert store.list(None) == []
    assert "FAISS migration failed" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [{"intent": "music"}, {"text": ""}, {"timestamp": "soon"}])
def test_invalid_row_imports_nothing(store, embedder, tmp_path, bad):
    rows = [dict(r) for r in ROWS]
    rows[1].update(bad)
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(rows))
    assert migrate_faiss(store, embedder, str(path)) == 0
    assert store.list(None) == []


def test_insert_failure_midway_rolls_back(store, embedder, tmp_path, monkeypatch):
    import memory.adapters.sqlite_store as mod

    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(ROWS))
    real = mod._memory_to_row
    calls = []

    def flaky(m):
        calls.append(m.id)
        if len(calls) == 3:
            raise RuntimeError("disk full")
        return real(m)

    monkeypatch.setattr(mod, "_memory_to_row", flaky)
    assert migrate_faiss(store, embedder, str(path)) == 0
    monkeypatch.setattr(mod, "_memory_to_row", real)
    assert store.list(None) == []
