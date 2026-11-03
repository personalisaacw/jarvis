"""One peer-memory lifecycle through MemoryService with fakes (spec section 11, end-to-end row)."""
import uuid

import pytest

from memory.adapters.memory_op_seeds import SEEDS
from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.entities import (
    AccessKind,
    EventLabel,
    Extraction,
    Memory,
    MemoryKind,
    MemoryOp,
    NewMemory,
    OpResolution,
    Status,
    Tier,
)
from memory.domain.params import FIXED, Params
from memory.service import MemoryService
from tests.memory.conftest import FakeEmbedder, FakeExtractor, FrozenClock, vec
from tests.memory.helpers import event

DAY = 86400.0
FAISS = "We use FAISS for routing"
SQLITE = "We use SQLite for routing"


@pytest.fixture
def clock():
    return FrozenClock(1_800_000_000.0)


@pytest.fixture
def store(clock):
    s = SqliteMemoryStore(":memory:", clock)
    yield s
    s.close()


@pytest.fixture
def extractor():
    return FakeExtractor()


@pytest.fixture
def service(extractor, clock, store):
    # FakeEmbedder vectors are random, so a verbatim seed's nearest neighbours are noise that
    # dilutes its Net 1 vote to 0.69 (ask band at defaults). A real embedder's neighbours are
    # the other revise seeds; lower both thresholds so the seed executes as it would there.
    store.save_params(Params.defaults().with_value("memory_op_threshold", 0.5)
                      .with_value("memory_op_ask_threshold", 0.3), [])
    return MemoryService(FakeEmbedder(), "/p", extractor=extractor, clock=clock, store=store,
                         start_threads=False)


def _spy_flushes(service):
    reasons = []
    real = service.buffer._on_flush

    def spy(turns, reason):
        reasons.append(reason)
        real(turns, reason)

    service.buffer._on_flush = spy
    return reasons


def _extract_calls(extractor):
    return [c for c in extractor.calls if c[0] == "extract"]


def _events(store, label):
    return [e for e in store.events_since(0) if e.label == label]


def _by_content(store, content):
    return [m for m in store.list(None) if m.content == content]


def _recalled_accesses(store, memory_id):
    return [a for a in store.get(memory_id).accesses if a.kind == AccessKind.RECALLED]


def _old_memory(store, content, kind, scope, importance, at):
    m = Memory(id=str(uuid.uuid4()), kind=kind, scope=scope, content=content,
               embedding=vec(content), importance=importance, status=Status.CURRENT,
               tier=Tier.ACTIVE, created_at=at)
    store.add(m)  # adds the CREATED access at `at`
    return m


def test_peer_memory_lifecycle(service, store, extractor, clock):
    flushes = _spy_flushes(service)
    scope = service.project_scope

    # 1. Salience-gated extraction (spec 7.1): two novel think-mode turns cross
    #    extract_salience_threshold on the second turn and flush as one batch.
    extractor.responses = [Extraction(new=[NewMemory(MemoryKind.DECISION, "project", FAISS, 0.8)])]
    for text in ("I keep thinking about how routing should pick between quick answers",
                 "Maybe vector search over past utterances beats a fixed list of examples"):
        service.record_turn(text, "...", "think", service.embed(text), None, False)
    assert flushes == ["salience"]
    assert store.pending_turns() == []
    [(_, (batch, recalled))] = _extract_calls(extractor)
    assert len(batch) == 2 and recalled == []

    # 2. Capture (spec 7.1): the batch's extraction becomes a current, active project decision.
    [faiss] = _by_content(store, FAISS)
    assert (faiss.kind, faiss.scope, faiss.status, faiss.tier) == (
        MemoryKind.DECISION, scope, Status.CURRENT, Tier.ACTIVE)

    # 3. Gated recall and the session working set (spec 7.2): a related query brings the
    #    decision in once; a repeat recall keeps it without appending another access.
    query = service.embed(FAISS)
    rr = service.recall_for_prompt(query, "think")
    assert rr.memories[0].content == FAISS
    assert [m.id for m in rr.newly_added] == [faiss.id]
    rr = service.recall_for_prompt(query, "think")
    assert rr.newly_added == [] and [m.id for m in rr.memories] == [faiss.id]
    assert len(_recalled_accesses(store, faiss.id)) == 1

    # 4. Net 1 revise (spec 7.3): a seeded phrasing executes without asking; the old decision
    #    is superseded by the replacement and, having been recalled, is logged STALE.
    utterance = SEEDS[MemoryOp.REVISE][0]
    assert utterance == "we're not using faiss anymore"
    extractor.resolution = OpResolution(
        target_ids=[faiss.id], new=NewMemory(MemoryKind.DECISION, "project", SQLITE, 0.8))
    reply = service.try_memory_op(utterance, service.embed(utterance))
    assert reply is not None and "SQLite" in reply
    [sqlite] = _by_content(store, SQLITE)
    old = store.get(faiss.id)
    assert old.status == Status.SUPERSEDED and old.superseded_by_id == sqlite.id
    assert sqlite.status == Status.CURRENT and sqlite.supersedes_id == faiss.id
    assert [e.memory_id for e in _events(store, EventLabel.STALE)] == [faiss.id]

    # 5. Supersede is respected by recall (spec 7.2 + 7.3): the superseded decision leaves the
    #    working set at once (it does not linger for working_set_patience turns), and the
    #    replacement is what recall brings in. FakeEmbedder vectors carry no meaning, so the
    #    replacement is reached with its own text rather than the old query.
    rr = service.recall_for_prompt(query, "think")
    assert faiss.id not in [m.id for m in rr.memories]
    rr = service.recall_for_prompt(service.embed(SQLITE), "think")
    assert [m.content for m in rr.memories] == [SQLITE]

    # 6. Debounced idle flush (spec 7.1): quick turns that only restate what is already known
    #    (max_cosine 1.0 against the current decision) stay below the salience threshold and
    #    are held until extract_idle_seconds pass; a batch that yields nothing is logged as
    #    EMPTY_EXTRACTION (cadence too eager). Novelty is measured against stored memories,
    #    not earlier turns, so three FakeEmbedder "ok"s (random vectors) would each score ~0.5
    #    and cross the 1.5 threshold on the third turn.
    extractor.responses = [Extraction()]
    for _ in range(3):
        service.record_turn(SQLITE, "...", "quick", service.embed(SQLITE), None, False)
    assert service.buffer.total < store.load_params().get("extract_salience_threshold")
    assert len(store.pending_turns()) == 3
    assert len(_extract_calls(extractor)) == 1
    clock.advance(91)
    service.buffer.tick()
    assert flushes == ["salience", "idle"]
    assert store.pending_turns() == []
    assert len(_extract_calls(extractor)) == 2
    assert len(_events(store, EventLabel.EMPTY_EXTRACTION)) == 1

    # 7. Sleep consolidation (spec 8): past episode_summary_age_days, a project's episodes fold
    #    into one summary that stays active while the episodes are archived and linked to it;
    #    the tuner stays silent below FIXED["min_events"] labelled events (spec 12, step 2).
    clock.advance(8 * DAY)
    ten_days_ago = clock.now() - 10 * DAY
    episodes = [_old_memory(store, text, MemoryKind.EPISODE, scope, 0.5, ten_days_ago)
                for text in ("Sketched the diff reviewer layout",
                             "Wired the diff reviewer to the agy session",
                             "Fixed hunk ordering in the diff reviewer")]
    for ep in episodes:
        store.add_access(ep.id, ten_days_ago, AccessKind.RECALLED)
    extractor.summaries = [NewMemory(MemoryKind.FEATURE, "project", "Built the diff reviewer", 0.8)]
    report = service.sleep.run()
    assert report.summarised == 1
    [summary] = _by_content(store, "Built the diff reviewer")
    assert (summary.status, summary.tier, summary.scope) == (Status.CURRENT, Tier.ACTIVE, scope)
    for ep in episodes:
        folded = store.get(ep.id)
        assert folded.tier == Tier.ARCHIVED and folded.feature_id == summary.id
    assert report.tune["status"] == "insufficient_data"

    # 8. Self-tuning (spec 9.3): once enough labelled events exist, counterfactual replay moves
    #    at most max_moves_per_pass params; decisions repeatedly missed while dormant make
    #    decisions fade more slowly. Importance 0.35 with one 20-day-old access is dormant
    #    under defaults (0.35 + 21**-0.3 = 0.751 < 0.8) and active after one decay.decision
    #    step (0.35 + 21**-0.25 = 0.817).
    clock.advance(DAY)
    now = clock.now()
    decisions = [_old_memory(store, "Decision %d about the build" % i, MemoryKind.DECISION,
                             scope, 0.35, now - 20 * DAY) for i in range(5)]
    for i in range(FIXED["min_events"] + 10):
        store.add_event(event(EventLabel.MISS, decisions[i % 5], at=now, tier_at_time="dormant",
                              rank_at_time=0, similarity=0.9, mode="think"))
    report2 = service.sleep.run()
    assert report2.tune["status"] == "tuned"
    assert 1 <= len(report2.tune["moves"]) <= FIXED["max_moves_per_pass"]
    assert store.load_params().get("decay.decision") < 0.3
