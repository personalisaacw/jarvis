from typing import List, Optional

import pytest

from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.entities import (
    GLOBAL_SCOPE,
    AccessKind,
    EventLabel,
    ExplicitOp,
    Extraction,
    Memory,
    MemoryKind,
    MemoryOp,
    NewMemory,
    Status,
    Supersede,
    Tier,
    Turn,
    project_scope,
)
from memory.domain.params import Params
from memory.use_cases.capture import CaptureBatchUseCase
from tests.memory.conftest import FakeExtractor, vec

SCOPE = project_scope("/p")


@pytest.fixture
def store(clock):
    s = SqliteMemoryStore(":memory:", clock)
    yield s
    s.close()


def make_capture(store, embedder, clock, extraction=None, **kw):
    ext = FakeExtractor([extraction] if extraction is not None else None)
    return CaptureBatchUseCase(
        store, ext, embedder, lambda: Params.defaults(), clock, **kw
    ), ext


def run(store, embedder, clock, extraction, turns, **kw):
    uc, _ = make_capture(store, embedder, clock, extraction, **kw)
    return uc.execute(turns, SCOPE, 1.0)


def turn(clock, tid="t1", text="hello there", recalled: Optional[List[str]] = None, mode="think"):
    return Turn(
        id=tid, session_id="s", at=clock.now(), project_scope=SCOPE, mode=mode,
        user_text=text, assistant_text="ok", embedding=vec(text),
        recalled_ids=list(recalled or []),
    )


def put(store, clock, mid, text, kind=MemoryKind.FACT, scope=SCOPE, tier=Tier.ACTIVE,
        importance=0.5):
    m = Memory(
        id=mid, kind=kind, scope=scope, content=text, embedding=vec(text),
        importance=importance, status=Status.CURRENT, tier=tier, created_at=clock.now(),
    )
    store.add(m)
    return m


def events(store, label=None):
    return [e for e in store.events_since(0) if label is None or e.label == label]


def test_new_inserts_with_project_scope_resolved(store, embedder, clock):
    ex = Extraction(new=[
        NewMemory(MemoryKind.DECISION, "project", "use postgres", 0.7),
        NewMemory(MemoryKind.PREFERENCE, "global", "likes tabs", 5.0),
        NewMemory(MemoryKind.FACT, "weird", "odd scope", 0.4),
    ])
    t = turn(clock)
    store.add_turn(t)
    run(store, embedder, clock, ex, [t])
    by = {m.content: m for m in store.list(Status.CURRENT)}
    assert by["use postgres"].scope == SCOPE
    assert by["likes tabs"].scope == GLOBAL_SCOPE
    assert by["likes tabs"].importance == 1.0
    assert by["odd scope"].scope == GLOBAL_SCOPE
    assert store.pending_turns() == []


def test_new_dedups_onto_existing_and_reinforces(store, embedder, clock):
    put(store, clock, "m1", "use postgres")
    ex = Extraction(new=[NewMemory(MemoryKind.DECISION, "project", "use postgres", 0.7)])
    run(store, embedder, clock, ex, [turn(clock)])
    assert len(store.list(Status.CURRENT)) == 1
    kinds = [a.kind for a in store.get("m1").accesses]
    assert kinds.count(AccessKind.REINFORCED) == 1
    assert events(store, EventLabel.MISS) == []


def test_new_dedup_onto_dormant_logs_miss(store, embedder, clock):
    put(store, clock, "m1", "use postgres", tier=Tier.DORMANT)
    ex = Extraction(new=[NewMemory(MemoryKind.DECISION, "project", "use postgres", 0.7)])
    run(store, embedder, clock, ex, [turn(clock)])
    miss = events(store, EventLabel.MISS)
    assert len(miss) == 1 and miss[0].memory_id == "m1"
    assert miss[0].context["tier_at_time"] == "dormant"
    assert miss[0].context["rank_at_time"] == 0
    assert miss[0].context["mode"] == "think"
    assert miss[0].context["similarity"] > 0.99


def test_apply_new_explicit_inserts_with_importance_one_and_explicit_access(
    store, embedder, clock
):
    uc, _ = make_capture(store, embedder, clock)
    m = uc.apply_new(NewMemory(MemoryKind.FACT, "project", "my cat is Tom", 0.2), SCOPE, True, "t1")
    stored = store.get(m.id)
    assert stored.importance == 1.0
    assert AccessKind.EXPLICIT in [a.kind for a in stored.accesses]


def test_apply_new_explicit_dedup_pins_importance(store, embedder, clock):
    put(store, clock, "m1", "my cat is Tom", importance=0.2)
    uc, _ = make_capture(store, embedder, clock)
    m = uc.apply_new(NewMemory(MemoryKind.FACT, "project", "my cat is Tom", 0.2), SCOPE, True, "t1")
    stored = store.get("m1")
    assert m.id == "m1" and stored.importance == 1.0
    assert AccessKind.EXPLICIT in [a.kind for a in stored.accesses]
    assert len(store.list(Status.CURRENT)) == 1


def test_public_apply_new_fires_profile_callback(store, embedder, clock):
    calls = []
    uc, _ = make_capture(store, embedder, clock, on_profile_change=lambda: calls.append(1))
    uc.apply_new(NewMemory(MemoryKind.FACT, "project", "plain", 0.5), SCOPE, False, None)
    assert calls == []
    uc.apply_new(NewMemory(MemoryKind.PREFERENCE, "global", "likes tabs", 0.5), SCOPE, True, None)
    assert calls == [1]


def test_supersede_skips_non_current_old(store, embedder, clock):
    old = put(store, clock, "old", "use mysql")
    old.status = Status.FORGOTTEN
    store.update(old)
    run(store, embedder, clock,
        Extraction(supersede=[Supersede("old", "x", MemoryKind.FACT, 0.5)]),
        [turn(clock)])
    assert store.list(Status.CURRENT) == []
    assert store.get("old").status == Status.FORGOTTEN


def test_reinforce_logs_hit_and_irrelevant_logs_false_recall(store, embedder, clock):
    put(store, clock, "a", "alpha fact")
    put(store, clock, "b", "beta fact")
    t = turn(clock, recalled=["a", "b"], mode="quick")
    ex = Extraction(reinforce=["a", "ghost"], irrelevant=["b", "ghost"])
    run(store, embedder, clock, ex, [t])
    hit = events(store, EventLabel.HIT)
    fr = events(store, EventLabel.FALSE_RECALL)
    assert [e.memory_id for e in hit] == ["a"] and [e.memory_id for e in fr] == ["b"]
    assert AccessKind.REINFORCED in [a.kind for a in store.get("a").accesses]
    assert AccessKind.REINFORCED not in [a.kind for a in store.get("b").accesses]
    c = fr[0].context
    assert c["rank_at_time"] == 1 and c["mode"] == "quick" and c["query_scope"] == SCOPE
    assert c["tier_at_time"] == "active" and c["scope"] == SCOPE and "similarity" in c
    assert fr[0].turn_id == "t1"


def test_supersede_links_both_ways_and_logs_stale_when_recalled(store, embedder, clock):
    put(store, clock, "old", "use mysql")
    ex = Extraction(supersede=[
        Supersede("old", "use postgres", MemoryKind.DECISION, 0.8),
        Supersede("ghost", "nothing", MemoryKind.FACT, 0.5),
    ])
    run(store, embedder, clock, ex, [turn(clock, recalled=["old"])])
    old = store.get("old")
    assert old.status == Status.SUPERSEDED and old.superseded_by_id
    new = store.get(old.superseded_by_id)
    assert new.supersedes_id == "old" and new.content == "use postgres"
    assert new.scope == SCOPE and new.status == Status.CURRENT
    assert [e.memory_id for e in events(store, EventLabel.STALE)] == ["old"]
    assert len(store.list(Status.CURRENT)) == 1


def test_complete_sets_completed_at(store, embedder, clock):
    put(store, clock, "f", "ship login", kind=MemoryKind.FEATURE)
    put(store, clock, "x", "some fact")
    clock.advance(100)
    run(store, embedder, clock, Extraction(complete=["f", "x", "ghost"]), [turn(clock)])
    assert store.get("f").completed_at == clock.now()
    assert store.get("x").completed_at is None
    assert events(store, EventLabel.EMPTY_EXTRACTION) == []
    assert len(events(store)) == 0


def test_forget_marks_forgotten_and_logs_false_recall_if_recalled(store, embedder, clock):
    put(store, clock, "a", "alpha")
    put(store, clock, "b", "beta")
    run(store, embedder, clock, Extraction(forget=["a", "b", "ghost"]),
        [turn(clock, recalled=["a"])])
    assert store.get("a").status == Status.FORGOTTEN
    assert store.get("b").status == Status.FORGOTTEN
    assert [e.memory_id for e in events(store, EventLabel.FALSE_RECALL)] == ["a"]


def test_explicit_ops_add_prototype_and_log_missed_op(store, embedder, clock):
    t = turn(clock, text="please remember my cat is called Tom")
    ex = Extraction(explicit_ops=[ExplicitOp("t1", MemoryOp.STORE)])
    run(store, embedder, clock, ex, [t], classifier_score_for=lambda tid: 0.31)
    protos = store.list(Status.CURRENT, [MemoryKind.MEMORY_OP_EXAMPLE])
    assert len(protos) == 1
    assert protos[0].label == "store" and protos[0].scope == GLOBAL_SCOPE
    assert protos[0].importance == 0.5
    assert protos[0].content == "please remember my cat is called Tom"
    missed = events(store, EventLabel.MISSED_OP)
    assert len(missed) == 1 and missed[0].turn_id == "t1"
    assert missed[0].context == {"classifier_score": 0.31}
    # same text again: no duplicate prototype, and a None score omits the key
    run(store, embedder, clock, ex, [t])
    assert len(store.list(Status.CURRENT, [MemoryKind.MEMORY_OP_EXAMPLE])) == 1
    assert events(store, EventLabel.MISSED_OP)[1].context == {}


def test_empty_extraction_logged_with_salience(store, embedder, clock):
    uc, _ = make_capture(store, embedder, clock, Extraction())
    uc.execute([turn(clock, "t1"), turn(clock, "t2")], SCOPE, 0.42)
    e = events(store, EventLabel.EMPTY_EXTRACTION)
    assert len(e) == 1 and e[0].memory_id is None and e[0].turn_id == "t2"
    assert e[0].context == {"buffer_salience": 0.42}


def test_profile_change_callback_fires_only_for_preference_or_feature(store, embedder, clock):
    calls = []

    def go(ex):
        run(store, embedder, clock, ex, [turn(clock)], on_profile_change=lambda: calls.append(1))

    go(Extraction(new=[NewMemory(MemoryKind.FACT, "project", "plain fact", 0.5)]))
    assert calls == []
    go(Extraction(new=[
        NewMemory(MemoryKind.PREFERENCE, "global", "likes tabs", 0.5),
        NewMemory(MemoryKind.FEATURE, "project", "build search", 0.5),
    ]))
    assert calls == [1]  # once per batch
    # dedup reinforce does not count
    go(Extraction(new=[NewMemory(MemoryKind.PREFERENCE, "global", "likes tabs", 0.5)]))
    assert calls == [1]
    feat = store.list(Status.CURRENT, [MemoryKind.FEATURE])[0]
    go(Extraction(complete=[feat.id]))
    assert calls == [1, 1]


def test_failure_retries_once_then_drops(store, embedder, clock):
    ext = FakeExtractor()
    ext.raise_next = 2
    uc = CaptureBatchUseCase(store, ext, embedder, lambda: Params.defaults(), clock)
    t = turn(clock)
    store.add_turn(t)
    assert uc.execute([t], SCOPE, 1.0) is None
    assert store.pending_turns()
    assert uc.execute([t], SCOPE, 1.0) is None
    assert not store.pending_turns()
    assert not store.get_meta("retry:t1")
