import uuid
from typing import Optional

import numpy as np
import pytest

from memory.adapters.memory_op_seeds import SEEDS, seed_memory_ops
from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.params import Params
from memory.domain.entities import (
    GLOBAL_SCOPE,
    AccessKind,
    EventLabel,
    Memory,
    MemoryKind,
    MemoryOp,
    NewMemory,
    OpResolution,
    Status,
    Tier,
    Turn,
    project_scope,
)
from memory.use_cases.capture import CaptureBatchUseCase
from memory.use_cases.memory_ops import (
    ClassifyMemoryOpUseCase,
    HandleMemoryOpUseCase,
    OpOutcome,
    PendingAsk,
)
from memory.use_cases.recall import RecallUseCase
from tests.memory.conftest import FakeExtractor, vec

SCOPE = project_scope("/p")

FORGET_TEXT = SEEDS[MemoryOp.FORGET][0]      # "scrap the diff-reviewer idea"
REVISE_TEXT = SEEDS[MemoryOp.REVISE][0]
COMPLETE_TEXT = SEEDS[MemoryOp.COMPLETE][0]
STORE_TEXT = SEEDS[MemoryOp.STORE][0]
ASK_TEXT = SEEDS[MemoryOp.ASK][0]
UNDO_TEXT = "undo that"
MEDIUM_TEXT = "hang on to the client's name"


# ---- fixtures and helpers ----

@pytest.fixture
def store(clock):
    s = SqliteMemoryStore(":memory:", clock)
    yield s
    s.close()


@pytest.fixture
def seeded(store, embedder, clock):
    return seed_memory_ops(store, embedder, clock)


@pytest.fixture
def classifier(store, clock, params):
    return ClassifyMemoryOpUseCase(store, lambda: params, clock)


@pytest.fixture
def extractor():
    return FakeExtractor()


@pytest.fixture
def recall(store, clock, params):
    return RecallUseCase(store, lambda: params, clock, "s")


@pytest.fixture
def handler(store, extractor, recall, embedder, clock, params):
    capture = CaptureBatchUseCase(store, extractor, embedder, lambda: params, clock)
    for op, text in [
        (MemoryOp.STORE, STORE_TEXT), (MemoryOp.FORGET, FORGET_TEXT),
        (MemoryOp.REVISE, REVISE_TEXT), (MemoryOp.COMPLETE, COMPLETE_TEXT),
        (MemoryOp.ASK, ASK_TEXT), (MemoryOp.STATUS, SEEDS[MemoryOp.STATUS][0]),
        (MemoryOp.UNDO, UNDO_TEXT), (MemoryOp.NONE, SEEDS[MemoryOp.NONE][0]),
    ]:
        proto(store, clock, text, op)
    return HandleMemoryOpUseCase(
        store, extractor, recall, capture, embedder, lambda: params, clock,
        project_scope_provider=lambda: SCOPE, status_line_provider=lambda: "status ok",
    )


def proto(store, clock, text, op, embedding: Optional[np.ndarray] = None) -> Memory:
    m = Memory(
        id=str(uuid.uuid4()), kind=MemoryKind.MEMORY_OP_EXAMPLE, scope=GLOBAL_SCOPE,
        content=text, embedding=vec(text) if embedding is None else embedding,
        importance=0.5, status=Status.CURRENT, tier=Tier.ACTIVE, created_at=clock.now(),
        label=op.value,
    )
    store.add(m)
    return m


def put(store, clock, text, kind=MemoryKind.FACT, tier=Tier.ACTIVE,
        embedding: Optional[np.ndarray] = None) -> Memory:
    m = Memory(
        id=str(uuid.uuid4()), kind=kind, scope=SCOPE, content=text,
        embedding=vec(text) if embedding is None else embedding, importance=0.5,
        status=Status.CURRENT, tier=tier, created_at=clock.now(),
    )
    store.add(m)
    return m


def near(text: str, similarity: float) -> np.ndarray:
    """A unit vector whose cosine to vec(text) is `similarity` (float32 rounding aside)."""
    u, w = vec(text), vec("orthogonal filler for %s at %s" % (text, similarity))
    w = w - np.dot(w, u) * u
    w = w / np.linalg.norm(w)
    v = similarity * u + np.sqrt(1 - similarity ** 2) * w
    return (v / np.linalg.norm(v)).astype(np.float32)


def say(handler, text) -> OpOutcome:
    return handler.handle(text, vec(text), [])


def events(store, label):
    return [e for e in store.events_since(0) if e.label == label]


def examples(store, text):
    return [m for m in store.list(Status.CURRENT, [MemoryKind.MEMORY_OP_EXAMPLE])
            if m.content == text]


def ask_medium(handler, store, clock, op=MemoryOp.STORE) -> OpOutcome:
    """A prototype at cosine 0.8 to MEDIUM_TEXT: score ~0.6, between the two thresholds."""
    proto(store, clock, "keep the client's name around", op, embedding=near(MEDIUM_TEXT, 0.8))
    return say(handler, MEDIUM_TEXT)


# ---- seeds ----

def test_seeds_every_op_has_at_least_twenty(store, embedder, clock):
    for op in MemoryOp:
        assert len(set(SEEDS[op])) >= 20, op
    total = sum(len(v) for v in SEEDS.values())
    assert seed_memory_ops(store, embedder, clock) == total
    assert seed_memory_ops(store, embedder, clock) == 0
    stored = store.list(Status.CURRENT, [MemoryKind.MEMORY_OP_EXAMPLE])
    assert len(stored) == total
    assert {m.label for m in stored} == {op.value for op in MemoryOp}
    assert all(m.scope == GLOBAL_SCOPE and m.importance == 0.5 and m.tier == Tier.ACTIVE
               for m in stored)


# ---- classifier ----

def test_classifier_empty_store_falls_through(classifier):
    assert classifier.execute(vec("remember this")).action == "fallthrough"


def test_seeded_forget_phrase_does_not_fall_through(seeded, classifier):
    assert seeded == sum(len(v) for v in SEEDS.values())
    d = classifier.execute(vec(FORGET_TEXT))
    assert d.op == MemoryOp.FORGET and d.action in {"execute", "ask"}


def test_classifier_executes_above_threshold(store, clock, classifier):
    # equal strengths; weights 1.0 (forget) and 0.25 (store): score = 1.0/1.25 * 1.0 = 0.8
    q = "drop the faiss note"
    proto(store, clock, "forget proto", MemoryOp.FORGET, embedding=near(q, 1.0))
    proto(store, clock, "store proto", MemoryOp.STORE, embedding=near(q, 0.25))
    d = classifier.execute(vec(q))
    assert d.op == MemoryOp.FORGET and d.action == "execute"
    assert d.score == pytest.approx(0.8, abs=1e-3)


def test_classifier_none_label_falls_through(seeded, classifier):
    d = classifier.execute(vec(SEEDS[MemoryOp.NONE][0]))
    assert d.op == MemoryOp.NONE and d.action == "fallthrough"
    assert classifier.execute(vec("an utterance nobody ever seeded")).action == "fallthrough"


def test_classifier_medium_score_asks(store, clock, classifier):
    # weights 0.9 (forget) and 0.6 (store): score = 0.9/1.5 * 0.9 = 0.54, between 0.5 and 0.7
    q = "drop the faiss note"
    proto(store, clock, "forget proto", MemoryOp.FORGET, embedding=near(q, 0.9))
    proto(store, clock, "store proto", MemoryOp.STORE, embedding=near(q, 0.6))
    d = classifier.execute(vec(q))
    assert d.op == MemoryOp.FORGET and d.action == "ask"
    assert d.score == pytest.approx(0.54, abs=1e-3)


def test_classifier_split_vote_prefers_op_over_none_and_only_restricts(store, clock, classifier):
    proto(store, clock, "same words", MemoryOp.STORE)
    proto(store, clock, "same words", MemoryOp.NONE)
    d = classifier.execute(vec("same words"))
    assert d.op == MemoryOp.STORE and d.score == pytest.approx(0.5, abs=1e-3)
    assert classifier.execute(vec("same words"), only=[MemoryOp.UNDO]).action == "fallthrough"


# ---- operations ----

def test_store_op_writes_importance_one_and_explicit_access(handler, store, extractor):
    extractor.resolution = OpResolution([], NewMemory(MemoryKind.FACT, "project",
                                                      "the demo is on friday", 0.3))
    out = say(handler, STORE_TEXT)
    assert out.consumed and out.spoken == "Noted." and out.pending_question is None
    m = next(m for m in store.list(Status.CURRENT, [MemoryKind.FACT])
             if m.content == "the demo is on friday")
    assert m.importance == 1.0 and m.scope == SCOPE
    assert AccessKind.EXPLICIT in [a.kind for a in m.accesses]
    op, utterance, _, candidates = extractor.calls[-1][1]
    assert op == MemoryOp.STORE and utterance == STORE_TEXT and candidates == []


def test_forget_resolves_target_and_reads_back(handler, store, clock, extractor, recall):
    m = put(store, clock, "we plan a diff reviewer")
    recall.execute(m.embedding, SCOPE, "think")          # recalled this turn
    extractor.resolution = OpResolution([m.id])
    out = say(handler, FORGET_TEXT)
    assert out.consumed and out.spoken == "Forgotten: we plan a diff reviewer"
    assert store.get(m.id).status == Status.FORGOTTEN
    candidates = extractor.calls[-1][1][3]
    assert m.id in [c.id for c in candidates]
    assert all(c.kind != MemoryKind.MEMORY_OP_EXAMPLE for c in candidates)
    assert [e.memory_id for e in events(store, EventLabel.FALSE_RECALL)] == [m.id]


def test_unresolved_target_writes_nothing(handler, store, clock, extractor):
    m = put(store, clock, "we plan a diff reviewer")
    extractor.resolution = OpResolution(["no-such-id"])
    out = say(handler, FORGET_TEXT)
    assert out.consumed and out.spoken.startswith("I couldn't work out what to change")
    assert store.get(m.id).status == Status.CURRENT
    assert store.events_since(0) == []


def test_revise_supersedes_and_logs_stale_when_recalled(handler, store, clock, extractor, recall):
    old = put(store, clock, "we use faiss", kind=MemoryKind.DECISION)
    recall.execute(old.embedding, SCOPE, "think")
    extractor.resolution = OpResolution(
        [old.id], NewMemory(MemoryKind.DECISION, "project", "we use sqlite vectors", 0.6))
    out = say(handler, REVISE_TEXT)
    assert out.spoken == "Updated: we use sqlite vectors"
    old = store.get(old.id)
    new = store.get(old.superseded_by_id)
    assert old.status == Status.SUPERSEDED and new.status == Status.CURRENT
    assert new.supersedes_id == old.id and new.importance == 1.0
    assert [e.memory_id for e in events(store, EventLabel.STALE)] == [old.id]


def test_complete_sets_completed_at(handler, store, clock, extractor):
    f = put(store, clock, "build the diff reviewer", kind=MemoryKind.FEATURE)
    fact = put(store, clock, "reviewer uses groq")
    extractor.resolution = OpResolution([f.id, fact.id])
    clock.advance(60)
    out = say(handler, COMPLETE_TEXT)
    assert out.spoken == "Marked done: build the diff reviewer"
    assert store.get(f.id).completed_at == clock.now()
    assert store.get(fact.id).completed_at is None


def test_ask_speaks_top_and_logs_miss_on_dormant(handler, store, clock):
    m = put(store, clock, "we decided on postgres", tier=Tier.DORMANT, embedding=vec(ASK_TEXT))
    out = say(handler, ASK_TEXT)
    assert out.consumed and out.spoken == "we decided on postgres"
    (miss,) = events(store, EventLabel.MISS)
    assert miss.memory_id == m.id and miss.context["tier_at_time"] == "dormant"
    assert miss.context["rank_at_time"] == 0 and miss.context["mode"] == "think"
    assert miss.context["similarity"] == pytest.approx(1.0, abs=1e-3)
    assert events(store, EventLabel.LATE_EXTRACTION) == []


def test_ask_logs_late_extraction_when_answer_is_still_pending(handler, store, clock):
    for tid, text, emb, sal in [("t1", "we picked postgres", vec(ASK_TEXT), 1.2),
                                ("t2", "unrelated chatter", vec("unrelated chatter"), 0.5)]:
        store.add_turn(Turn(id=tid, session_id="s", at=clock.now(), project_scope=SCOPE,
                            mode="think", user_text=text, assistant_text="ok",
                            embedding=emb, salience=sal))
    out = say(handler, ASK_TEXT)
    assert out.spoken == "I don't remember anything about that yet."
    assert len(events(store, EventLabel.MISS)) == 1
    (late,) = events(store, EventLabel.LATE_EXTRACTION)
    assert late.context["buffer_salience"] == pytest.approx(1.7)


def test_status_speaks_status_line(handler):
    assert say(handler, SEEDS[MemoryOp.STATUS][0]).spoken == "status ok"


# ---- confirmation and learning ----

def test_ask_flow_yes_learns_positive_and_executes(handler, store, clock, extractor, params):
    extractor.resolution = OpResolution([], NewMemory(MemoryKind.FACT, "project",
                                                      "the client is acme", 0.4))
    out = ask_medium(handler, store, clock)
    assert out.consumed and out.spoken == "Should I remember that?"
    assert isinstance(out.pending_question, PendingAsk)
    assert out.pending_question.op == MemoryOp.STORE
    assert 0.5 <= out.pending_question.score < 0.7
    assert store.list(Status.CURRENT, [MemoryKind.FACT]) == []
    answer = say(handler, "yes please")
    assert answer.consumed and answer.spoken == "Noted."
    assert [m.content for m in store.list(Status.CURRENT, [MemoryKind.FACT])] == ["the client is acme"]
    assert [m.label for m in examples(store, MEDIUM_TEXT)] == ["store"]
    d = ClassifyMemoryOpUseCase(store, lambda: params, clock).execute(vec(MEDIUM_TEXT))
    assert d.op == MemoryOp.STORE and d.action == "execute"


def test_ask_flow_no_learns_none_and_logs_false_op(handler, store, clock):
    out = ask_medium(handler, store, clock)
    score = out.pending_question.score
    answer = say(handler, "nope")
    assert answer.consumed and answer.pending_question is None
    assert store.list(Status.CURRENT, [MemoryKind.FACT]) == []
    assert [m.label for m in examples(store, MEDIUM_TEXT)] == ["none"]
    (fo,) = events(store, EventLabel.FALSE_OP)
    assert fo.context["classifier_score"] == pytest.approx(score)
    assert say(handler, MEDIUM_TEXT).consumed is False


def test_confirmation_turn_is_consumed_even_if_it_looks_like_a_command(handler, store, clock,
                                                                      extractor):
    m = put(store, clock, "the client is acme")
    proto(store, clock, "no, forget it", MemoryOp.FORGET)
    extractor.resolution = OpResolution([m.id])
    ask_medium(handler, store, clock)
    out = say(handler, "no, forget it")
    assert out.consumed
    assert store.get(m.id).status == Status.CURRENT
    assert len(events(store, EventLabel.FALSE_OP)) == 1


# ---- undo ----

def test_undo_after_forget_restores_and_logs_false_op(handler, store, clock, extractor):
    m = put(store, clock, "we plan a diff reviewer")
    extractor.resolution = OpResolution([m.id])
    say(handler, FORGET_TEXT)
    out = say(handler, UNDO_TEXT)
    assert out.consumed and "we plan a diff reviewer" in out.spoken
    assert store.get(m.id).status == Status.CURRENT
    (fo,) = events(store, EventLabel.FALSE_OP)
    assert fo.context["classifier_score"] >= 0.7
    assert sorted(e.label for e in examples(store, FORGET_TEXT)) == ["forget", "none"]


def test_undo_after_store_and_revise_reverts(handler, store, clock, extractor):
    extractor.resolution = OpResolution([], NewMemory(MemoryKind.FACT, "project", "x is y", 0.3))
    say(handler, STORE_TEXT)
    say(handler, UNDO_TEXT)
    assert store.list(Status.CURRENT, [MemoryKind.FACT]) == []
    old = put(store, clock, "we use faiss", kind=MemoryKind.DECISION)
    extractor.resolution = OpResolution(
        [old.id], NewMemory(MemoryKind.DECISION, "project", "we use sqlite", 0.6))
    say(handler, REVISE_TEXT)
    new_id = store.get(old.id).superseded_by_id
    say(handler, UNDO_TEXT)
    old, new = store.get(old.id), store.get(new_id)
    assert old.status == Status.CURRENT and old.superseded_by_id is None
    assert new.status == Status.FORGOTTEN and new.supersedes_id is None


def test_undo_window_closes_after_one_turn(handler, store, clock, extractor):
    m = put(store, clock, "we plan a diff reviewer")
    extractor.resolution = OpResolution([m.id])
    say(handler, FORGET_TEXT)
    handler.note_turn_passed()
    out = say(handler, UNDO_TEXT)
    assert out.consumed is False
    assert store.get(m.id).status == Status.FORGOTTEN
    assert events(store, EventLabel.FALSE_OP) == []


def test_fallthrough_not_consumed(handler):
    assert handler.handle("what time is it", vec("what time is it"), []).consumed is False



@pytest.mark.parametrize("answer", ["please don't", "yeah no", "sure, not now", "nope",
                                    "correct me later", "No.", "nah", "don\u2019t"])
def test_negated_or_mixed_answers_are_no(handler, store, clock, extractor, answer):
    extractor.resolution = OpResolution([], NewMemory(MemoryKind.FACT, "project", "acme", 0.4))
    ask_medium(handler, store, clock)
    out = say(handler, answer)
    assert out.consumed and store.list(Status.CURRENT, [MemoryKind.FACT]) == []
    assert len(events(store, EventLabel.FALSE_OP)) == 1


@pytest.mark.parametrize("answer", ["yes", "Yes please!", "do it", "sure", "please", "Okay.",
                                    "OK", "Yup", "Absolutely", "Yes, remember that.",
                                    "Yeah, sure.", "Yes please do", "Sure thing", "go ahead",
                                    "correct", "yes thanks"])
def test_plain_yes_answers_execute(handler, store, clock, extractor, answer):
    extractor.resolution = OpResolution([], NewMemory(MemoryKind.FACT, "project", "acme", 0.4))
    ask_medium(handler, store, clock)
    assert say(handler, answer).spoken == "Noted."
    assert [m.content for m in store.list(Status.CURRENT, [MemoryKind.FACT])] == ["acme"]


def test_undo_after_deduplicated_store_keeps_existing_memory(handler, store, clock, extractor):
    existing = put(store, clock, "the demo is on friday")
    extractor.resolution = OpResolution([], NewMemory(MemoryKind.FACT, "project",
                                                      "the demo is on friday", 0.3))
    say(handler, STORE_TEXT)
    out = say(handler, UNDO_TEXT)
    assert out.consumed and out.spoken == "Nothing to undo."
    assert store.get(existing.id).status == Status.CURRENT


def test_undo_after_deduplicated_revise_restores_old_only(handler, store, clock, extractor):
    old = put(store, clock, "we use faiss", kind=MemoryKind.DECISION)
    other = put(store, clock, "we use sqlite", kind=MemoryKind.DECISION)
    extractor.resolution = OpResolution(
        [old.id], NewMemory(MemoryKind.DECISION, "project", "we use sqlite", 0.6))
    say(handler, REVISE_TEXT)
    assert store.get(old.id).superseded_by_id == other.id
    say(handler, UNDO_TEXT)
    old, other = store.get(old.id), store.get(other.id)
    assert old.status == Status.CURRENT and old.superseded_by_id is None
    assert other.status == Status.CURRENT and other.supersedes_id is None


def test_revise_onto_itself_writes_nothing(handler, store, clock, extractor):
    old = put(store, clock, "we use faiss", kind=MemoryKind.DECISION)
    extractor.resolution = OpResolution(
        [old.id], NewMemory(MemoryKind.DECISION, "project", "we use faiss", 0.6))
    assert say(handler, REVISE_TEXT).spoken == "I already have that."
    assert store.get(old.id).status == Status.CURRENT
    assert events(store, EventLabel.STALE) == []


def test_undo_withdraws_the_ops_events(handler, store, clock, extractor, recall):
    m = put(store, clock, "we plan a diff reviewer")
    recall.execute(m.embedding, SCOPE, "think")
    extractor.resolution = OpResolution([m.id])
    say(handler, FORGET_TEXT)
    assert len(events(store, EventLabel.FALSE_RECALL)) == 1
    say(handler, UNDO_TEXT)
    assert events(store, EventLabel.FALSE_RECALL) == []
    assert len(events(store, EventLabel.FALSE_OP)) == 1


def test_fallthrough_outcome_carries_classifier_score(handler, store, clock):
    q = "maybe note the build flag"
    proto(store, clock, "weak store proto", MemoryOp.STORE, embedding=near(q, 0.4))
    out = say(handler, q)
    assert out.consumed is False
    expected = ClassifyMemoryOpUseCase(store, lambda: Params.defaults(), clock).execute(vec(q))
    assert expected.action == "fallthrough" and expected.score > 0.0
    assert out.score == pytest.approx(expected.score)


# ---- review fixes: unclear answers and expiry ----

def test_unclear_answer_drops_the_question_and_flows_on(handler, store, clock, extractor):
    extractor.resolution = OpResolution([], NewMemory(MemoryKind.FACT, "project", "acme", 0.4))
    ask_medium(handler, store, clock)
    before = len(store.list(Status.CURRENT, [MemoryKind.MEMORY_OP_EXAMPLE]))
    out = say(handler, "what's the weather")
    assert (out.spoken, out.consumed, out.pending_question, out.score) == ("", False, None, 0.0)
    assert handler._pending is None
    assert events(store, EventLabel.FALSE_OP) == []
    assert store.list(Status.CURRENT, [MemoryKind.FACT]) == []
    assert len(store.list(Status.CURRENT, [MemoryKind.MEMORY_OP_EXAMPLE])) == before
    assert say(handler, "yes").consumed is False  # the question is gone, "yes" is just a turn


def test_pending_question_expires_after_sixty_seconds(handler, store, clock, extractor):
    m = put(store, clock, "we plan a diff reviewer")
    extractor.resolution = OpResolution([m.id])
    ask_medium(handler, store, clock)
    clock.advance(61)
    out = say(handler, FORGET_TEXT)  # classified normally, not taken as the answer
    assert out.consumed and out.spoken.startswith("Forgotten:")
    assert store.get(m.id).status == Status.FORGOTTEN
    assert events(store, EventLabel.FALSE_OP) == []
    assert examples(store, MEDIUM_TEXT) == []


def test_pending_question_within_sixty_seconds_still_answers(handler, store, clock,
                                                             extractor):
    extractor.resolution = OpResolution([], NewMemory(MemoryKind.FACT, "project", "acme", 0.4))
    ask_medium(handler, store, clock)
    clock.advance(59)
    assert say(handler, "yes").spoken == "Noted."


def test_undo_not_offered_after_sixty_one_seconds(handler, store, clock, extractor):
    m = put(store, clock, "we plan a diff reviewer")
    extractor.resolution = OpResolution([m.id])
    say(handler, FORGET_TEXT)
    clock.advance(61)
    out = say(handler, UNDO_TEXT)
    assert out.consumed is False
    assert store.get(m.id).status == Status.FORGOTTEN
    assert events(store, EventLabel.FALSE_OP) == []
