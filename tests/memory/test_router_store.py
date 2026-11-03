import pytest

import memory.adapters.sqlite_store as sqlite_store
from domain.entities import Intent, Utterance
from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.entities import MemoryKind, Status
from tests.memory.conftest import vec
from use_cases.routing import LearnFromFeedbackUseCase, RouteCommandUseCase


@pytest.fixture
def store(clock):
    s = SqliteMemoryStore(":memory:", clock)
    yield s
    s.close()


def test_empty_store_routes_quick_with_zero_confidence(store, embedder):
    r = RouteCommandUseCase(store, embedder).execute("anything")
    assert r.intent == Intent.QUICK and r.confidence_score == 0.0


def test_learn_then_route_same_text(store, embedder):
    LearnFromFeedbackUseCase(store, embedder).execute("fix the bug in router.py", "code")
    r = RouteCommandUseCase(store, embedder).execute("fix the bug in router.py")
    assert r.intent == Intent.CODE and r.confidence_score == pytest.approx(1.0, abs=1e-5)


def test_execute_accepts_precomputed_vector(store, embedder):
    LearnFromFeedbackUseCase(store, embedder).execute("build a flask api", "code")

    class Boom:
        def embed_text(self, text):
            raise AssertionError("must not embed when a vector is supplied")

    r = RouteCommandUseCase(store, Boom()).execute(
        "ignored", vector=vec("build a flask api").tolist()
    )
    assert r.intent == Intent.CODE and r.confidence_score == pytest.approx(1.0, abs=1e-5)


def test_search_similar_returns_utterance_and_distance(store, embedder):
    store.add_utterance(Utterance("hello", Intent.QUICK, 42.0), embedder.embed_text("hello"))
    ((u, dist),) = store.search_similar(embedder.embed_text("hello"), limit=3)
    assert (u.text, u.intent, u.timestamp) == ("hello", Intent.QUICK, 42.0)
    assert dist == pytest.approx(0.0, abs=1e-5)


def test_search_similar_only_current_route_examples(store, embedder):
    LearnFromFeedbackUseCase(store, embedder).execute("a", "quick")
    m = store.list(kinds=[MemoryKind.ROUTE_EXAMPLE])[0]
    m.status = Status.SUPERSEDED
    store.update(m)
    assert store.search_similar(embedder.embed_text("a")) == []


def test_duplicate_utterance_reinforces_instead_of_inserting(store, embedder):
    uc = LearnFromFeedbackUseCase(store, embedder)
    uc.execute("fix the bug in router.py", "code")
    uc.execute("fix the bug in router.py", "code")
    rows = store.list(kinds=[MemoryKind.ROUTE_EXAMPLE])
    assert len(rows) == 1
    # (created_at comes from the use case's wall clock, the reinforcement from the store clock)
    assert sorted(a.kind.value for a in rows[0].accesses) == ["created", "reinforced"]


def test_same_text_other_label_is_not_a_duplicate(store, embedder):
    uc = LearnFromFeedbackUseCase(store, embedder)
    uc.execute("ambiguous", "code")
    uc.execute("ambiguous", "think")
    assert len(store.list(kinds=[MemoryKind.ROUTE_EXAMPLE])) == 2


def test_distill_merges_near_duplicates(store, embedder, monkeypatch):
    uc = LearnFromFeedbackUseCase(store, embedder)
    uc.execute("write a python script", "code")
    uc.execute("write some python script", "code")
    uc.execute("what time is it", "quick")
    assert len(store.list(kinds=[MemoryKind.ROUTE_EXAMPLE])) == 3
    # FakeEmbedder gives distinct texts near-orthogonal vectors, so force a high similarity
    # in the dedup pass by patching the `cosine` that sqlite_store.dedup reads.
    monkeypatch.setattr(sqlite_store, "cosine", lambda a, b: 0.95)
    store.distill_utterances()
    current = store.list(kinds=[MemoryKind.ROUTE_EXAMPLE])
    assert sorted(m.label for m in current) == ["code", "quick"]
    old = store.list(status=Status.SUPERSEDED)
    assert len(old) == 1
    keeper = next(m for m in current if m.label == "code")
    assert old[0].superseded_by_id == keeper.id and keeper.supersedes_id == old[0].id
    assert len(keeper.accesses) == 2  # the loser's CREATED access moved over


def test_dedup_returns_merge_count_and_is_idempotent(store, embedder, monkeypatch):
    uc = LearnFromFeedbackUseCase(store, embedder)
    uc.execute("one", "code")
    uc.execute("two", "code")
    monkeypatch.setattr(sqlite_store, "cosine", lambda a, b: 0.95)
    assert store.dedup([MemoryKind.ROUTE_EXAMPLE], 0.9, 1_000_000.0) == 1
    assert store.dedup([MemoryKind.ROUTE_EXAMPLE], 0.9, 1_000_000.0) == 0
