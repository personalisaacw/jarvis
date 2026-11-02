import pytest

from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.entities import (
    GLOBAL_SCOPE,
    AccessKind,
    Memory,
    MemoryKind,
    Status,
    Tier,
    Turn,
    project_scope,
)
from memory.domain.params import Params
from memory.use_cases.recall import RecallUseCase, SessionWorkingSet
from tests.memory.conftest import vec

SCOPE = project_scope("/p")
OTHER = project_scope("/other")
_n = [0]


def _m(text, **kw) -> Memory:
    _n[0] += 1
    fields = dict(
        id="m%d" % _n[0],
        kind=MemoryKind.FACT,
        scope=SCOPE,
        content=text,
        embedding=vec(text),
        importance=1.0,
        status=Status.CURRENT,
        tier=Tier.ACTIVE,
        created_at=1_000_000.0,
    )
    fields.update(kw)
    return Memory(**fields)


@pytest.fixture
def store(clock):
    s = SqliteMemoryStore(":memory:", clock)
    yield s
    s.close()


@pytest.fixture
def recall(store, clock):
    return RecallUseCase(store, lambda: Params.defaults(), clock, session_id="s")


def _recalled(store, mid):
    return [a for a in store.get(mid).accesses if a.kind == AccessKind.RECALLED]


def _ids(result):
    return [m.id for m in result.memories]


def test_empty_store_returns_nothing(recall):
    r = recall.execute(vec("x"), SCOPE, "think")
    assert r.memories == [] and r.profile.features == []


def test_top_k_from_active_only_and_archived_never(recall, store):
    for i in range(8):
        store.add(_m("same", id="a%d" % i, importance=1.0 + i * 0.1))
    store.add(_m("same", id="arch", tier=Tier.ARCHIVED, importance=9.0))
    r = recall.execute(vec("same"), SCOPE, "think")
    assert len(r.memories) == 6
    assert "arch" not in _ids(r)
    assert _ids(r)[0] == "a7"
    assert "a0" not in _ids(r) and "a1" not in _ids(r)


def test_dormant_only_above_similarity_bar(recall, store):
    store.add(_m("near", id="d_near", tier=Tier.DORMANT))
    store.add(_m("far", id="d_far", tier=Tier.DORMANT))
    store.add(_m("near", id="act"))
    r = recall.execute(vec("near"), SCOPE, "think")
    assert set(_ids(r)) == {"d_near", "act"}


def test_scope_ordering_at_equal_similarity(recall, store):
    store.add(_m("t", id="other", scope=OTHER))
    store.add(_m("t", id="glob", scope=GLOBAL_SCOPE))
    store.add(_m("t", id="proj", scope=SCOPE))
    r = recall.execute(vec("t"), SCOPE, "think")
    assert _ids(r) == ["proj", "glob", "other"]


def test_gate_blocks_below_floor(recall, store):
    store.add(_m("alpha", id="a"))
    r = recall.execute(vec("beta"), SCOPE, "think")
    assert r.memories == [] and r.newly_added == []
    assert _recalled(store, "a") == []


def test_budget_quick_is_two(recall, store):
    for i in range(5):
        store.add(_m("same", id="a%d" % i, importance=1.0 + i * 0.1))
    r = recall.execute(vec("same"), SCOPE, "quick")
    assert _ids(r) == ["a4", "a3"]


def test_unknown_mode_behaves_like_think(recall, store):
    for i in range(8):
        store.add(_m("same", id="a%d" % i))
    assert len(recall.execute(vec("same"), SCOPE, "weird").memories) == 6


def test_working_set_keeps_order_appends_and_evicts():
    ws = SessionWorkingSet()
    a, b, c = _m("a"), _m("b"), _m("c")
    assert ws.merge([a, b], 2) == [a, b]
    assert ws.merge([b, c], 2) == [c]
    assert [m.id for m in ws.members()] == [a.id, b.id, c.id]
    ws.merge([b, c], 2)
    assert [m.id for m in ws.members()] == [b.id, c.id]


def test_working_set_miss_counter_resets_on_return():
    ws = SessionWorkingSet()
    a, b = _m("a"), _m("b")
    ws.merge([a], 2)
    ws.merge([b], 2)
    ws.merge([a], 2)  # a returns: counter reset
    ws.merge([b], 2)
    assert a.id in [m.id for m in ws.members()]
    ws.reset()
    assert ws.members() == []


def test_working_set_stable_across_turns(recall, store):
    store.add(_m("one", id="one"))
    assert _ids(recall.execute(vec("one"), SCOPE, "quick")) == ["one"]
    store.add(_m("three", id="three", importance=5.0))
    r = recall.execute(vec("three"), SCOPE, "quick")
    assert _ids(r) == ["one", "three"]
    assert [m.id for m in r.newly_added] == ["three"]


def test_access_appended_only_on_first_entry(recall, store, clock):
    store.add(_m("x", id="x"))
    r1 = recall.execute(vec("x"), SCOPE, "think")
    assert [m.id for m in r1.newly_added] == ["x"]
    r2 = recall.execute(vec("x"), SCOPE, "think")
    assert r2.newly_added == [] and _ids(r2) == ["x"]
    accesses = _recalled(store, "x")
    assert len(accesses) == 1
    assert accesses[0].at == clock.now()
    assert recall.last_recalled_ids() == ["x"]


def test_profile_excluded_from_recall_and_cached_until_invalidated(recall, store):
    store.add(_m("pref", id="p1", kind=MemoryKind.PREFERENCE, scope=GLOBAL_SCOPE))
    store.add(_m("feat", id="f1", kind=MemoryKind.FEATURE))
    store.add(_m("done", id="f2", kind=MemoryKind.FEATURE, completed_at=5.0))
    store.add(_m("pref", id="fact"))
    prof = recall.profile(SCOPE)
    assert [m.id for m in prof.preferences] == ["p1"]
    assert [m.id for m in prof.features] == ["f1"]
    r = recall.execute(vec("pref"), SCOPE, "think")
    assert "p1" not in _ids(r) and "fact" in _ids(r)
    store.add(_m("pref2", id="p2", kind=MemoryKind.PREFERENCE, scope=GLOBAL_SCOPE))
    assert recall.profile(SCOPE) is prof
    v = prof.version
    recall.invalidate_profile()
    fresh = recall.profile(SCOPE)
    assert fresh.version == v + 1
    assert {m.id for m in fresh.preferences} == {"p1", "p2"}


def test_profile_preferences_capped_and_strength_ordered(store, clock):
    p = Params.defaults().with_value("profile_preferences", 2)
    rc = RecallUseCase(store, lambda: p, clock, session_id="s")
    for i in range(4):
        store.add(_m("p%d" % i, id="p%d" % i, kind=MemoryKind.PREFERENCE,
                     scope=GLOBAL_SCOPE, importance=1.0 + i))
    assert [m.id for m in rc.profile(SCOPE).preferences] == ["p3", "p2"]


def test_code_mode_includes_all_project_features_and_decisions(recall, store):
    store.add(_m("dec", id="dec", kind=MemoryKind.DECISION))
    store.add(_m("q", id="fact"))
    r = recall.execute(vec("q"), SCOPE, "code")
    assert _ids(r)[:2] == ["dec", "fact"]  # project decision is forced to the front


def test_code_mode_features_returned_even_when_nothing_similar(recall, store):
    store.add(_m("dec", id="dec", kind=MemoryKind.DECISION))
    r = recall.execute(vec("unrelated"), SCOPE, "code")
    assert _ids(r) == ["dec"]


def test_include_dormant_returns_tiers_without_gate(recall, store):
    store.add(_m("a", id="act"))
    store.add(_m("b", id="dor", tier=Tier.DORMANT))
    store.add(_m("c", id="arch", tier=Tier.ARCHIVED))
    r = recall.execute(vec("zzz"), SCOPE, "think", include_dormant=True)
    assert set(_ids(r)) == {"act", "dor"}
    assert {m.tier for m in r.memories} == {Tier.ACTIVE, Tier.DORMANT}
    assert r.newly_added == []
    assert _recalled(store, "act") == [] and recall.working_set.members() == []


def test_working_turns_returned(recall, store):
    for i in range(8):
        store.add_turn(Turn(id="t%d" % i, session_id="s", at=float(i), project_scope=SCOPE,
                            mode="quick", user_text="u", assistant_text="a",
                            embedding=vec("t%d" % i)))
    assert len(recall.execute(vec("x"), SCOPE, "think").working_turns) == 6
