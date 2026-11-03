"""The tuner on events produced by the live recall -> capture pipeline (not synthetic events).

Regression for the final-review probe: the RECALLED access that put a memory in front of the
model was counted when replaying the event it produced, so every recalled memory looked
active under any decay and the loss was identical at decay.decision 0.05 / 0.30 / 1.00.
"""
import dataclasses
import uuid

import pytest

from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.entities import (
    AccessKind,
    EventLabel,
    Extraction,
    Memory,
    MemoryKind,
    Status,
    Supersede,
    Tier,
)
from memory.domain.params import FIXED, Params
from memory.service import MemoryService
from memory.use_cases.tune import TuneParamsUseCase
from tests.memory.conftest import FakeEmbedder, FakeExtractor, vec

DAY = 86400.0
# importance 0.42, one access 20 days old: 0.42 + 21**-0.3 = 0.821 (active at 0.8), and one
# step decay.decision 0.30 -> 0.35 gives 0.42 + 21**-0.35 = 0.765 (dormant). (At importance
# 0.35 the memory is already dormant under defaults, so STALE would cost nothing to begin with.)
IMPORTANCE = 0.42


@pytest.fixture
def store(clock):
    s = SqliteMemoryStore(":memory:", clock)
    yield s
    s.close()


def _decision(store, scope, i, at):
    content = "Decision %d about the build" % i
    m = Memory(id=str(uuid.uuid4()), kind=MemoryKind.DECISION, scope=scope, content=content,
               embedding=vec(content), importance=IMPORTANCE, status=Status.CURRENT,
               tier=Tier.ACTIVE, created_at=at)
    store.add(m)  # the CREATED access, 20 days old
    return m


def _live_stale_events(store, clock):
    extractor = FakeExtractor()
    service = MemoryService(FakeEmbedder(), "/p", extractor=extractor, clock=clock, store=store,
                            start_threads=False)
    decisions = [_decision(store, service.project_scope, i, clock.now() - 20 * DAY)
                 for i in range(5)]
    for m in decisions:
        q = service.embed(m.content)
        rr = service.recall_for_prompt(q, "think")
        assert m.id in [r.id for r in rr.memories]
        assert [a.kind for a in store.get(m.id).accesses].count(AccessKind.RECALLED) == 1
        clock.advance(5)  # the answer streams before the turn is recorded
        extractor.responses = [Extraction(supersede=[
            Supersede(m.id, m.content + " (revised)", MemoryKind.DECISION, 0.5)])]
        service.record_turn(m.content, "ok", "think", q, rr, False)
        service.buffer.flush("idle")
        clock.advance(60)
    stale = [e for e in store.events_since(0) if e.label == EventLabel.STALE]
    assert sorted(e.memory_id for e in stale) == sorted(m.id for m in decisions)
    return stale


def test_live_stale_events_move_decision_decay(store, clock):
    stale = _live_stale_events(store, clock)
    # Replicate the real events (same context, as capture writes them) past min_events.
    for i in range(FIXED["min_events"] + 10 - len(stale)):
        store.add_event(dataclasses.replace(stale[i % len(stale)], id=str(uuid.uuid4())))
    tuner = TuneParamsUseCase(store, clock)
    events = [e for e in store.events_since(0) if e.label == EventLabel.STALE]
    losses = {d: tuner.loss(Params.defaults().with_value("decay.decision", d), events,
                            clock.now()) for d in (0.05, 0.30, 1.00)}
    assert losses[0.05] > losses[1.00]  # the decay rate now matters to the replay
    r = tuner.run()
    assert r["status"] == "tuned"
    assert any(m["name"] == "decay.decision" and m["new"] > m["old"] for m in r["moves"])
    assert store.load_params().get("decay.decision") > 0.3
