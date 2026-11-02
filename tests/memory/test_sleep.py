import numpy as np
import pytest

from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.entities import (
    GLOBAL_SCOPE, AccessKind, Memory, MemoryKind, NewMemory, Status, Tier,
)
from memory.domain.params import Params
from memory.use_cases.consolidate import SleepReport, SleepUseCase
from tests.memory.conftest import FakeExtractor, vec

DAY = 86400.0
S = "project:/p"


class FakeTuner:
    def __init__(self):
        self.calls = 0

    def run(self):
        self.calls += 1
        return {"status": "insufficient_data", "events": 0, "moves": []}


class BufferSpy:
    def __init__(self):
        self.reasons = []

    def flush(self, reason):
        self.reasons.append(reason)


@pytest.fixture
def store(clock):
    return SqliteMemoryStore(":memory:", clock)


@pytest.fixture
def extractor():
    e = FakeExtractor()
    e.summaries = [NewMemory(MemoryKind.FEATURE, "project", "built the reviewer", 0.8)]
    return e


@pytest.fixture
def buffer_spy():
    return BufferSpy()


@pytest.fixture
def tuner():
    return FakeTuner()


@pytest.fixture
def done():
    return []


@pytest.fixture
def sleep(store, extractor, buffer_spy, tuner, clock, done):
    return SleepUseCase(
        store, extractor, buffer_spy, tuner, lambda: Params.defaults(), clock,
        on_done=lambda: done.append(1),
    )


def mem(clock, text, kind=MemoryKind.EPISODE, scope=S, age_days=0.0, importance=0.5, id=None):
    return Memory(
        id=id or text, kind=kind, scope=scope, content=text, embedding=vec(text),
        importance=importance, status=Status.CURRENT, tier=Tier.ACTIVE,
        created_at=clock.now() - age_days * DAY,
    )


def ep_id(scope, i):
    return "ep%s-%d" % (scope, i)


def add_episodes(store, clock, n=3, scope=S):
    for i in range(n):
        store.add(mem(clock, ep_id(scope, i), scope=scope, age_days=10))
        store.add_access(ep_id(scope, i), clock.now() - (9 - i) * DAY, AccessKind.RECALLED)


def test_is_due_when_never_slept_and_after_interval(sleep, store, clock):
    assert sleep.is_due()
    store.set_meta("last_sleep_at", str(clock.now()))
    assert not sleep.is_due()
    clock.advance(24 * 3600 + 1)
    assert sleep.is_due()


def test_pending_buffer_flushed_first(sleep, buffer_spy):
    sleep.run()
    assert buffer_spy.reasons == ["sleep"]


def test_rescore_moves_old_episode_to_archived(sleep, store, clock):
    store.add(mem(clock, "old", age_days=50, importance=0.0))
    store.add(mem(clock, "fresh", age_days=0, importance=0.9))
    report = sleep.run()
    assert report.rescored == 1
    assert store.get("old").tier == Tier.ARCHIVED
    assert store.get("fresh").tier == Tier.ACTIVE


def test_episodes_fold_into_summary_inheriting_accesses(sleep, store, clock):
    add_episodes(store, clock)
    newest = store.get(ep_id(S, 2))
    newest_at = max(a.at for a in newest.accesses)
    report = sleep.run()
    assert report.summarised == 1
    feats = store.list(Status.CURRENT, [MemoryKind.FEATURE], S)
    assert len(feats) == 1 and feats[0].content == "built the reviewer"
    assert feats[0].tier == Tier.ACTIVE
    assert newest_at in [a.at for a in feats[0].accesses]
    assert len(feats[0].accesses) >= len(newest.accesses)
    for i in range(3):
        ep = store.get(ep_id(S, i))
        assert ep.tier == Tier.ARCHIVED and ep.feature_id == feats[0].id


def test_fewer_than_three_episodes_not_summarised(sleep, store, clock, extractor):
    add_episodes(store, clock, n=2)
    report = sleep.run()
    assert report.summarised == 0
    assert not [c for c in extractor.calls if c[0] == "summarise"]
    assert store.list(Status.CURRENT, [MemoryKind.FEATURE]) == []


def test_global_scope_episodes_not_summarised(sleep, store, clock):
    add_episodes(store, clock, scope=GLOBAL_SCOPE)
    assert sleep.run().summarised == 0


def test_summarise_failure_skips_scope(sleep, store, clock, extractor):
    add_episodes(store, clock)

    def boom(episodes):
        raise ValueError("bad")

    extractor.summarise = boom
    assert sleep.run().summarised == 0
    assert store.list(Status.CURRENT, [MemoryKind.FEATURE]) == []
    assert store.get(ep_id(S, 0)).feature_id is None


def test_dedup_keeps_stronger_across_kinds(sleep, store, clock):
    store.add(mem(clock, "same", kind=MemoryKind.FACT, id="weak", importance=0.4))
    store.add(mem(clock, "same", kind=MemoryKind.FACT, id="strong", importance=0.9))
    report = sleep.run()
    assert report.deduped == 1
    assert store.get("strong").status == Status.CURRENT
    assert store.get("weak").status == Status.SUPERSEDED


def test_purge_respects_retention(sleep, store, clock):
    store.add(mem(clock, "ancient", age_days=100, importance=0.0))
    store.add(mem(clock, "recent-archived", age_days=30, importance=0.0))
    report = sleep.run()
    assert report.purged == 1
    assert store.get("ancient") is None
    assert store.get("recent-archived") is not None


def test_profile_invalidated_and_last_sleep_written(sleep, store, done, clock, tuner):
    report = sleep.run()
    assert done == [1]
    assert store.get_meta("last_sleep_at") == str(clock.now())
    assert tuner.calls == 1 and report.tune["status"] == "insufficient_data"
    assert report.dry_run is False and isinstance(report, SleepReport)


def test_dry_run_writes_nothing(sleep, store, done, buffer_spy, tuner, clock):
    add_episodes(store, clock)
    store.add(mem(clock, "ancient", age_days=100, importance=0.0))
    before = sorted((m.id, m.tier, m.status) for m in store.list(None))
    report = sleep.run(dry_run=True)
    assert report.dry_run is True and report.summarised == 1 and report.purged == 1
    assert sorted((m.id, m.tier, m.status) for m in store.list(None)) == before
    assert store.get_meta("last_sleep_at") is None
    assert done == [] and buffer_spy.reasons == []


def test_exception_leaves_last_sleep_unset(sleep, store, clock, done, tuner):
    add_episodes(store, clock)

    def boom():
        raise RuntimeError("tuner down")

    tuner.run = boom
    with pytest.raises(RuntimeError):
        sleep.run()
    assert store.get_meta("last_sleep_at") is None
    assert done == []


def test_summary_embedding_is_embedding_of_summary_text(
    store, extractor, buffer_spy, tuner, clock, embedder
):
    add_episodes(store, clock)
    uc = SleepUseCase(
        store, extractor, buffer_spy, tuner, lambda: Params.defaults(), clock, embedder=embedder
    )
    uc.run()
    feat = store.list(Status.CURRENT, [MemoryKind.FEATURE], S)[0]
    assert np.allclose(feat.embedding, vec("built the reviewer"), atol=1e-6)


def test_dry_run_default_does_not_tune_real_store(sleep, store, clock):
    report = sleep.run(dry_run=True)
    assert report.tune["status"] == "skipped_dry_run"
    assert store.params_history() == []


def test_dry_run_tunes_the_copy_when_factory_given(
    store, extractor, tuner, clock
):
    seen = []

    def factory(s):
        seen.append(s)
        return tuner

    uc = SleepUseCase(
        store, extractor, None, tuner, lambda: Params.defaults(), clock, tuner_for_store=factory
    )
    report = uc.run(dry_run=True)
    assert len(seen) == 1 and seen[0] is not store
    assert report.tune["status"] == "insufficient_data"
