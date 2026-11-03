import threading
from typing import Optional

import numpy as np
import pytest

from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.entities import (
    AccessKind,
    EventLabel,
    Memory,
    MemoryEvent,
    MemoryKind,
    Status,
    Tier,
    Turn,
)
from memory.domain.params import Params
from tests.memory.conftest import vec


@pytest.fixture
def store(clock):
    s = SqliteMemoryStore(":memory:", clock)
    yield s
    s.close()


_counter = [0]


def _m(text, embedder, **overrides) -> Memory:
    _counter[0] += 1
    fields = dict(
        id="m%d" % _counter[0],
        kind=MemoryKind.FACT,
        scope="global",
        content=text,
        embedding=vec(text),
        importance=0.5,
        status=Status.CURRENT,
        tier=Tier.ACTIVE,
        created_at=1_000_000.0,
    )
    fields.update(overrides)
    return Memory(**fields)


def _turn(i, session="s1", at=None, extracted: Optional[float] = None) -> Turn:
    return Turn(
        id="t%s" % i,
        session_id=session,
        at=float(i) if at is None else at,
        project_scope="project:/x",
        mode="quick",
        user_text="u%s" % i,
        assistant_text="a%s" % i,
        embedding=vec("turn%s" % i),
        salience=0.5,
        recalled_ids=["a", "b"],
        extracted_at=extracted,
    )


def test_round_trip_all_fields(store, embedder):
    m = _m(
        "hello", embedder, label="quick", completed_at=5.0, supersedes_id="x",
        superseded_by_id="y", feature_id="f", kind=MemoryKind.FEATURE,
        scope="project:/p", importance=0.9, tier=Tier.DORMANT,
    )
    store.add(m)
    got = store.get(m.id)
    assert got is not None
    for f in ("id", "kind", "scope", "content", "importance", "status", "tier",
              "created_at", "label", "completed_at", "supersedes_id",
              "superseded_by_id", "feature_id"):
        assert getattr(got, f) == getattr(m, f)
    assert np.allclose(got.embedding, m.embedding)
    assert got.embedding.dtype == np.float32
    assert len(got.accesses) == 1
    assert got.accesses[0].kind == AccessKind.CREATED
    assert got.accesses[0].at == m.created_at
    assert store.get("nope") is None


def test_update_changes_mutable_columns(store, embedder):
    m = _m("hello", embedder)
    store.add(m)
    m.status = Status.SUPERSEDED
    m.tier = Tier.ARCHIVED
    m.content = "changed"
    m.completed_at = 7.0
    m.superseded_by_id = "z"
    store.update(m)
    got = store.get(m.id)
    assert (got.status, got.tier, got.content, got.completed_at, got.superseded_by_id) == (
        Status.SUPERSEDED, Tier.ARCHIVED, "changed", 7.0, "z")


def test_list_filters_status_kind_scope(store, embedder):
    a = _m("a", embedder, kind=MemoryKind.FACT, scope="global")
    b = _m("b", embedder, kind=MemoryKind.DECISION, scope="project:/p")
    c = _m("c", embedder, kind=MemoryKind.FACT, scope="project:/p", status=Status.FORGOTTEN)
    for m in (a, b, c):
        store.add(m)
    assert {m.id for m in store.list()} == {a.id, b.id}
    assert {m.id for m in store.list(status=None)} == {a.id, b.id, c.id}
    assert {m.id for m in store.list(status=Status.FORGOTTEN)} == {c.id}
    assert {m.id for m in store.list(kinds=[MemoryKind.DECISION])} == {b.id}
    assert {m.id for m in store.list(scope="global")} == {a.id}
    assert {m.id for m in store.list(status=None, kinds=[MemoryKind.FACT], scope="project:/p")} == {c.id}


def test_nearest_orders_by_cosine_and_respects_scope(store, embedder):
    a = _m("alpha", embedder, scope="global")
    b = _m("beta", embedder, scope="project:/p", kind=MemoryKind.DECISION)
    c = _m("gamma", embedder, scope="global")
    gone = _m("alpha2", embedder, status=Status.FORGOTTEN)
    for m in (a, b, c, gone):
        store.add(m)
    res = store.nearest(vec("alpha"), k=3)
    assert [m.id for m, _ in res][0] == a.id
    assert res[0][1] == pytest.approx(1.0, abs=1e-5)
    sims = [s for _, s in res]
    assert sims == sorted(sims, reverse=True)
    assert gone.id not in [m.id for m, _ in res]
    assert len(store.nearest(vec("alpha"), k=2)) == 2
    in_global = store.nearest(vec("beta"), k=5, scope="global")
    assert b.id not in [m.id for m, _ in in_global]
    only_dec = store.nearest(vec("alpha"), k=5, kinds=[MemoryKind.DECISION])
    assert [m.id for m, _ in only_dec] == [b.id]
    forgotten = store.nearest(vec("alpha2"), k=1, status=Status.FORGOTTEN)
    assert forgotten[0][0].id == gone.id
    assert store.nearest(vec("alpha"), k=3, scope="nowhere") == []


def test_cache_invalidates_on_write(store, embedder):
    assert store.max_cosine(vec("hello")) == 0.0
    m = _m("hello", embedder)
    store.add(m)
    assert store.max_cosine(vec("hello")) == pytest.approx(1.0)
    m.status = Status.FORGOTTEN
    store.update(m)
    assert store.max_cosine(vec("hello")) == 0.0
    assert store.max_cosine(vec("hello"), kinds=[MemoryKind.FACT], scope="global") == 0.0


def test_accesses_move_and_sorted(store, embedder):
    a, b = _m("a", embedder), _m("b", embedder)
    store.add(a)
    store.add(b)
    store.add_access(a.id, 2_000_000.0, AccessKind.RECALLED)
    store.add_access(a.id, 1_500_000.0, AccessKind.REINFORCED)
    got = store.get(a.id)
    assert [x.at for x in got.accesses] == sorted(x.at for x in got.accesses)
    assert got.last_accessed_at == 2_000_000.0
    store.move_accesses(a.id, b.id)
    assert len(store.get(a.id).accesses) == 0
    assert len(store.get(b.id).accesses) == 4


def test_events_since(store, embedder):
    store.add_event(MemoryEvent("e1", 10.0, EventLabel.HIT, "m1", "t1", {"similarity": 0.8}))
    store.add_event(MemoryEvent("e2", 20.0, EventLabel.MISS, None, None, {}))
    got = store.events_since(15.0)
    assert [e.id for e in got] == ["e2"]
    assert got[0].label == EventLabel.MISS and got[0].memory_id is None
    first = store.events_since(0.0)[0]
    assert first.context == {"similarity": 0.8} and first.turn_id == "t1"


def test_delete_events_removes_only_named_events(store):
    store.add_event(MemoryEvent("e1", 10.0, EventLabel.HIT, "m1", None, {}))
    store.add_event(MemoryEvent("e2", 20.0, EventLabel.MISS, None, None, {}))
    assert store.delete_events(["e1", "nope"]) == 1
    assert [e.id for e in store.events_since(0.0)] == ["e2"]
    assert store.delete_events([]) == 0


def test_params_default_then_saved(store):
    assert store.load_params().get("recall_k") == 6
    store.save_params(
        Params.defaults().with_value("recall_k", 4),
        [("recall_k", 6, 4, 10.0, 8.0, "replay")],
    )
    assert store.load_params().get("recall_k") == 4
    h = store.params_history()[0]
    assert h["reason"] == "replay" and h["name"] == "recall_k"
    assert h["old_value"] == 6 and h["new_value"] == 4
    assert h["old_loss"] == 10.0 and h["new_loss"] == 8.0


def test_meta(store):
    assert store.get_meta("schema_version") == "1"
    assert store.get_meta("last_sleep_at") is None
    store.set_meta("last_sleep_at", "5")
    store.set_meta("last_sleep_at", "6")
    assert store.get_meta("last_sleep_at") == "6"


def test_turns_pending_and_mark_extracted(store, clock):
    for i in (3, 1, 2):
        store.add_turn(_turn(i))
    pending = store.pending_turns()
    assert [t.id for t in pending] == ["t1", "t2", "t3"]
    assert pending[0].recalled_ids == ["a", "b"]
    assert np.allclose(pending[0].embedding, vec("turn1"))
    store.mark_extracted(["t1", "t2"], clock.now())
    assert [t.id for t in store.pending_turns()] == ["t3"]
    store.mark_extracted([], clock.now())


def test_recent_turns_newest_last(store):
    for i in range(1, 6):
        store.add_turn(_turn(i))
    store.add_turn(_turn(9, session="other"))
    got = store.recent_turns("s1", 3)
    assert [t.id for t in got] == ["t3", "t4", "t5"]


def test_delete_turns_before(store, clock):
    for i in range(1, 6):
        store.add_turn(_turn(i, extracted=100.0))
    store.add_turn(_turn(6, extracted=None))
    store.add_turn(_turn(7, session="other", extracted=100.0))
    n = store.delete_turns_before(at=100.0, keep_session="s1", keep_n=2)
    assert n == 5  # t1-t4 of s1 + t7 of other; newest two of s1 (t5, t6) kept
    left = {t.id for t in store.recent_turns("s1", 10)}
    assert left == {"t5", "t6"}


def test_purge_removes_archived_and_forgotten_with_children(store, clock, embedder):
    old_arch = _m("oa", embedder, tier=Tier.ARCHIVED, created_at=100.0)
    old_forg = _m("of", embedder, status=Status.FORGOTTEN, created_at=100.0)
    recent_arch = _m("ra", embedder, tier=Tier.ARCHIVED, created_at=100.0)
    old_active = _m("ac", embedder, created_at=100.0)
    for m in (old_arch, old_forg, recent_arch, old_active):
        store.add(m)
    store.add_access(recent_arch.id, 900.0, AccessKind.RECALLED)
    store.add_event(MemoryEvent("e1", 1.0, EventLabel.HIT, old_arch.id, None, {}))
    store.add_event(MemoryEvent("e2", 1.0, EventLabel.HIT, old_active.id, None, {}))
    n = store.purge(before=500.0)
    assert n == 2
    assert store.get(old_arch.id) is None and store.get(old_forg.id) is None
    assert store.get(recent_arch.id) is not None and store.get(old_active.id) is not None
    assert [e.id for e in store.events_since(0.0)] == ["e2"]
    assert store._conn.execute(
        "SELECT COUNT(*) FROM accesses WHERE memory_id IN (?, ?)", (old_arch.id, old_forg.id)
    ).fetchone()[0] == 0


def test_count_by(store, embedder):
    store.add(_m("a", embedder, kind=MemoryKind.FACT, tier=Tier.ACTIVE, scope="global"))
    store.add(_m("b", embedder, kind=MemoryKind.FACT, tier=Tier.ACTIVE, scope="global"))
    store.add(_m("c", embedder, kind=MemoryKind.DECISION, tier=Tier.DORMANT, scope="project:/p"))
    store.add(_m("d", embedder, status=Status.FORGOTTEN))
    assert sorted(store.count_by()) == [
        ("decision", "dormant", "project:/p", 1),
        ("fact", "active", "global", 2),
    ]


def test_file_database_creates_parent_dir(tmp_path, clock, embedder):
    path = str(tmp_path / "sub" / "dir" / "memory.sqlite")
    s = SqliteMemoryStore(path, clock)
    s.add(_m("hello", embedder))
    s.close()
    s2 = SqliteMemoryStore(path, clock)
    assert len(s2.list()) == 1
    assert s2.get_meta("schema_version") == "1"
    s2.close()


def test_concurrent_writer_thread(store, embedder):
    errors = []

    def writer():
        try:
            for i in range(200):
                store.add(_m("w%d" % i, embedder))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    t = threading.Thread(target=writer)
    t.start()
    for _ in range(200):
        store.nearest(vec("q"), k=3)
    t.join()
    assert errors == []
    assert len(store.list()) == 200



def _healthy_db(path, clock, embedder):
    s = SqliteMemoryStore(path, clock)
    s.add(_m("keep me", embedder))
    s.close()


def _corrupt_siblings(path):
    import glob
    return sorted(glob.glob(path + ".corrupt-*"))


def test_locked_db_is_not_renamed(tmp_path, clock, embedder):
    import os
    import sqlite3
    path = str(tmp_path / "memory.sqlite")
    _healthy_db(path, clock, embedder)
    other = sqlite3.connect(path, timeout=0, isolation_level=None)
    try:
        other.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.OperationalError):
            SqliteMemoryStore(path, clock, busy_timeout=0.1)
        assert os.path.exists(path)
        assert _corrupt_siblings(path) == []
    finally:
        other.execute("ROLLBACK")
        other.close()
    s = SqliteMemoryStore(path, clock)
    assert len(s.list()) == 1
    s.close()


def test_corrupt_db_is_renamed_and_fresh_store_opens(tmp_path, clock, capsys):
    import os
    path = str(tmp_path / "memory.sqlite")
    with open(path, "wb") as f:
        f.write(b"not a database" * 100)
    s = SqliteMemoryStore(path, clock)
    assert s.list() == []
    s.close()
    moved = _corrupt_siblings(path)
    assert len(moved) == 1 and os.path.exists(path)
    assert "WARNING" in capsys.readouterr().out


def test_corrupt_rename_never_overwrites_existing(tmp_path, clock):
    path = str(tmp_path / "memory.sqlite")
    earlier = "%s.corrupt-%d" % (path, int(clock.now()))
    with open(earlier, "wb") as f:
        f.write(b"earlier evidence")
    with open(path, "wb") as f:
        f.write(b"not a database" * 100)
    s = SqliteMemoryStore(path, clock)
    s.close()
    with open(earlier, "rb") as f:
        assert f.read() == b"earlier evidence"
    assert _corrupt_siblings(path) == [earlier, earlier + "-1"]
    with open(earlier + "-1", "rb") as f:
        assert f.read() == b"not a database" * 100


def test_unopenable_path_propagates(tmp_path, clock):
    import os
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"i am a file")
    with pytest.raises(OSError):
        SqliteMemoryStore(str(blocker / "sub" / "memory.sqlite"), clock)
    assert blocker.read_bytes() == b"i am a file"
    assert os.listdir(str(tmp_path)) == ["blocker"]


def test_directory_as_db_path_propagates_untouched(tmp_path, clock):
    import os
    import sqlite3
    d = tmp_path / "memory.sqlite"
    d.mkdir()
    with pytest.raises(sqlite3.DatabaseError):
        SqliteMemoryStore(str(d), clock)
    assert os.path.isdir(str(d))
    assert _corrupt_siblings(str(d)) == []
