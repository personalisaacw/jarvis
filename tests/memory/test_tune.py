import random

import pytest

from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.entities import EventLabel, Memory, MemoryKind, Status, Tier
from memory.domain.params import FIXED, PARAM_SPECS, Params
from memory.domain.scoring import strength, tier_for
from memory.use_cases.tune import TuneParamsUseCase
from tests.memory.conftest import vec
from tests.memory.helpers import event

DAY = 86400.0
RECALL_CTX = dict(similarity=0.9, rank_at_time=0, mode="think")


@pytest.fixture
def store(clock):
    s = SqliteMemoryStore(":memory:", clock)
    yield s
    s.close()


@pytest.fixture
def tuner(store, clock):
    return TuneParamsUseCase(store, clock)


def _memory(store, id, kind, importance, created_at) -> Memory:
    m = Memory(
        id=id, kind=kind, scope="global", content=id, embedding=vec(id),
        importance=importance, status=Status.CURRENT, tier=Tier.ACTIVE, created_at=created_at,
    )
    store.add(m)  # adds the CREATED access at created_at
    return store.get(id)


def _add(store, events):
    for e in events:
        store.add_event(e)


def _missed_decisions(store, now):
    # importance 0.35, only access 20 days old: 0.35 + 21**-0.3 = 0.751 (dormant at 0.8);
    # one step decay.decision 0.3 -> 0.25 gives 0.35 + 21**-0.25 = 0.817 (active).
    mems = [_memory(store, "d%d" % i, MemoryKind.DECISION, 0.35, now - 20 * DAY) for i in range(5)]
    _add(store, [event(EventLabel.MISS, mems[i % 5], at=now, tier_at_time="dormant", **RECALL_CTX)
                 for i in range(50)])
    return mems


def _scores_events(label, scores, repeat):
    return [event(label, classifier_score=s) for s in scores for _ in range(repeat)]


def test_insufficient_data_below_min_events(tuner, store, clock):
    now = clock.now()
    _add(store, [event(EventLabel.EMPTY_EXTRACTION, at=now, buffer_salience=1.5)
                 for _ in range(FIXED["min_events"] - 1)])
    # outside the window: does not count
    _add(store, [event(EventLabel.EMPTY_EXTRACTION, at=now - 31 * DAY, buffer_salience=1.5)
                 for _ in range(10)])
    r = tuner.run()
    assert r == {"status": "insufficient_data", "events": FIXED["min_events"] - 1, "moves": []}
    assert store.params_history() == []


def test_loss_cost_table_known_values(tuner, store, clock):
    now = clock.now()
    p = Params.defaults()
    active = _memory(store, "a", MemoryKind.FACT, 1.0, now)          # strength 2.0, active
    dormant = _memory(store, "z", MemoryKind.DECISION, 0.35, now - 20 * DAY)
    events = [
        event(EventLabel.HIT, active, **RECALL_CTX),                               # 0
        event(EventLabel.HIT, active, similarity=0.9, rank_at_time=6, mode="think"),  # 1 (rank)
        event(EventLabel.HIT, active, similarity=0.9, rank_at_time=2, mode="quick"),  # 1 (budget)
        event(EventLabel.HIT, active, similarity=0.9, rank_at_time=9, budget_rank=1,
              mode="quick"),                                                       # 0
        event(EventLabel.FALSE_RECALL, active, **RECALL_CTX),                      # 0.5
        event(EventLabel.FALSE_RECALL, active, similarity=0.4, rank_at_time=0,
              mode="think"),                                                       # 0 (floor)
        event(EventLabel.MISS, dormant, **RECALL_CTX),                             # 1
        event(EventLabel.MISS, active, **RECALL_CTX),                              # 0
        event(EventLabel.STALE, active),                                           # 1.0
        event(EventLabel.STALE, dormant),                                          # 0
        event(EventLabel.EMPTY_EXTRACTION, buffer_salience=1.5),                   # 0.3
        event(EventLabel.EMPTY_EXTRACTION, buffer_salience=1.4),                   # 0
        event(EventLabel.LATE_EXTRACTION, buffer_salience=1.4),                    # 1.0
        event(EventLabel.LATE_EXTRACTION, buffer_salience=1.5),                    # 0
        event(EventLabel.FALSE_OP, classifier_score=0.7),                          # 1.0
        event(EventLabel.FALSE_OP, classifier_score=0.69),                         # 0
        event(EventLabel.MISSED_OP, classifier_score=0.49),                        # 0.7
        event(EventLabel.MISSED_OP, classifier_score=0.5),                         # 0
        event(EventLabel.MISS, "missing-id", **RECALL_CTX),                        # 0 (no memory)
        event(EventLabel.FALSE_OP),                                                # 0 (no score)
    ]
    expected = 1 + 1 + 0.5 + 1 + 1.0 + 0.3 + 1.0 + 1.0 + 0.7
    assert tuner.loss(p, events, now) == pytest.approx(expected)


def test_loss_ignores_accesses_after_event_time(tuner, store, clock):
    now = clock.now()
    m = _memory(store, "d", MemoryKind.DECISION, 0.35, now - 20 * DAY)
    store.add_access("d", now, "recalled")  # happens after the event below
    e = event(EventLabel.MISS, m, at=now - DAY, **RECALL_CTX)
    assert tuner.loss(Params.defaults(), [e], now) == 1


def test_missed_decisions_lower_decision_decay(tuner, store, clock):
    now = clock.now()
    mems = _missed_decisions(store, now)
    p = Params.defaults()
    assert tier_for(strength(mems[0], now, p), p) == Tier.DORMANT
    p2 = p.with_value("decay.decision", 0.25)
    assert tier_for(strength(mems[0], now, p2), p2) == Tier.ACTIVE
    r = tuner.run()
    assert r["status"] == "tuned" and r["events"] == 50
    assert len(r["moves"]) <= 3
    assert any(m["name"] == "decay.decision" and m["new"] < m["old"] for m in r["moves"])


def test_irrelevant_episodes_raise_episode_decay_or_tighten_recall(tuner, store, clock):
    now = clock.now()
    # importance 0.3, accessed 1.1 days ago: 0.3 + 2.1**-0.9 = 0.813 (active at 0.8);
    # decay.episode 0.9 -> 1.0 gives 0.3 + 2.1**-1.0 = 0.776 (dormant).
    mems = [_memory(store, "e%d" % i, MemoryKind.EPISODE, 0.3, now - 1.1 * DAY) for i in range(5)]
    p = Params.defaults()
    assert tier_for(strength(mems[0], now, p), p) == Tier.ACTIVE
    _add(store, [event(EventLabel.FALSE_RECALL, mems[i % 5], **RECALL_CTX) for i in range(50)])
    r = tuner.run()
    assert r["status"] == "tuned"
    tightened = {"decay.episode": 1, "recall_similarity_floor": 1, "recall_k": -1}
    assert any(
        m["name"] in tightened and (m["new"] - m["old"]) * tightened[m["name"]] > 0
        for m in r["moves"]
    )


def test_empty_extractions_raise_salience_threshold(tuner, store):
    _add(store, [event(EventLabel.EMPTY_EXTRACTION, buffer_salience=1.5) for _ in range(50)])
    r = tuner.run()
    assert [(m["name"], m["old"], m["new"]) for m in r["moves"]] == [
        ("extract_salience_threshold", 1.5, pytest.approx(1.6))
    ]


def test_late_extractions_lower_salience_threshold(tuner, store):
    _add(store, [event(EventLabel.LATE_EXTRACTION, buffer_salience=1.45) for _ in range(50)])
    r = tuner.run()
    assert [(m["name"], m["old"], m["new"]) for m in r["moves"]] == [
        ("extract_salience_threshold", 1.5, pytest.approx(1.4))
    ]


def test_false_ops_raise_op_threshold(tuner, store):
    _add(store, _scores_events(EventLabel.FALSE_OP, [0.7], 50))
    r = tuner.run()
    assert [(m["name"], m["old"], m["new"]) for m in r["moves"]] == [
        ("memory_op_threshold", 0.7, pytest.approx(0.72))
    ]


def test_missed_ops_lower_ask_threshold(tuner, store):
    _add(store, _scores_events(EventLabel.MISSED_OP, [0.49], 50))
    r = tuner.run()
    assert [(m["name"], m["old"], m["new"]) for m in r["moves"]] == [
        ("memory_op_ask_threshold", 0.5, pytest.approx(0.48))
    ]


def _op_scores_above(store, now):
    # 14 buckets 0.70..0.96: every +0.02 step removes >= 1/14 of the loss, up to the 0.95 max
    scores = [round(0.70 + 0.02 * i, 2) for i in range(14)]
    _add(store, _scores_events(EventLabel.FALSE_OP, scores, 4))
    return "memory_op_threshold", 0.95


def _saliences_below(store, now):
    # 14 buckets 0.1..1.4: every -0.1 step frees one bucket, down to the 0.3 min
    _add(store, [event(EventLabel.LATE_EXTRACTION, buffer_salience=round(0.1 * i, 1))
                 for i in range(1, 15) for _ in range(4)])
    return "extract_salience_threshold", 0.3


def _ask_scores_below(store, now):
    # 12 buckets 0.27..0.49: every -0.02 step frees one bucket, down to the 0.3 min
    scores = [round(0.27 + 0.02 * i, 2) for i in range(12)]
    _add(store, _scores_events(EventLabel.MISSED_OP, scores, 4))
    return "memory_op_ask_threshold", 0.3


def _weak_misses(store, now):
    # fact memories fading at 150..600 days: pulls decay and active_threshold down
    weak = [_memory(store, "w%d" % i, MemoryKind.FACT, 0.0, now - (150 + 50 * i) * DAY)
            for i in range(10)]
    _add(store, [event(EventLabel.MISS, m, **RECALL_CTX) for m in weak for _ in range(5)])
    return None, None


@pytest.mark.parametrize(
    "scenario", [_op_scores_above, _saliences_below, _ask_scores_below, _weak_misses]
)
def test_bounds_and_invariants_hold_after_many_passes(tuner, store, clock, scenario):
    name, limit = scenario(store, clock.now())
    for _ in range(10):
        tuner.run()
    p = store.load_params()
    for n, spec in PARAM_SPECS.items():
        assert spec.min <= p.get(n) <= spec.max, n
    assert p.is_valid()
    if name is not None:
        assert p.get(name) == pytest.approx(limit)


def test_margin_prevents_flip_flop(tuner, store):
    # 30 empty at 1.5 (cost 0.3 each) and 20 late at 1.45 (cost 1 each): 29.0 at default.
    # 1.4 costs 9.0; from there 1.3 also costs 9.0 and 1.5 costs 29.0, so it settles.
    _add(store, [event(EventLabel.EMPTY_EXTRACTION, buffer_salience=1.5) for _ in range(30)])
    _add(store, [event(EventLabel.LATE_EXTRACTION, buffer_salience=1.45) for _ in range(20)])
    first = tuner.run()
    assert first["status"] == "tuned"
    settled = store.load_params().as_dict()
    second = tuner.run()
    assert second == {"status": "no_improvement", "events": 50, "moves": []}
    assert store.load_params().as_dict() == settled


def test_improvement_below_margin_is_not_applied(tuner, store):
    # raising memory_op_threshold removes 1 of 50 costly events: 2% < 5% margin
    _add(store, _scores_events(EventLabel.FALSE_OP, [0.99], 49) + _scores_events(
        EventLabel.FALSE_OP, [0.7], 1))
    assert tuner.run() == {"status": "no_improvement", "events": 50, "moves": []}
    assert store.params_history() == []


def test_at_most_three_moves_per_pass(tuner, store):
    scores = [round(0.70 + 0.02 * i, 2) for i in range(10)]
    _add(store, _scores_events(EventLabel.FALSE_OP, scores, 5))
    r = tuner.run()
    assert r["status"] == "tuned"
    assert len(r["moves"]) == FIXED["max_moves_per_pass"]
    assert [m["new"] for m in r["moves"]] == pytest.approx([0.72, 0.74, 0.76])


def test_search_matches_brute_force_full_loss(tuner, store, clock):
    # mixed events on a seeded RNG; the first move must equal an exhaustive full-loss search
    now = clock.now()
    rng = random.Random(7)
    kinds = list(MemoryKind)
    mems = [_memory(store, "r%d" % i, kinds[i % len(kinds)], rng.random(),
                    now - rng.random() * 25 * DAY) for i in range(40)]
    for m in mems:
        store.add_access(m.id, now - rng.random() * 20 * DAY, "recalled")
    labels = list(EventLabel) + [EventLabel.MISS] * 4  # misses dominate, so a move pays off
    events = [
        event(labels[i % len(labels)], mems[rng.randrange(40)], at=now - rng.random() * 20 * DAY,
              similarity=0.4 + 0.6 * rng.random(), rank_at_time=rng.randrange(4),
              mode=rng.choice(["quick", "think"]), buffer_salience=rng.random() * 3,
              classifier_score=rng.random())
        for i in range(400)
    ]
    _add(store, events)
    events = store.events_since(now - FIXED["tune_window_days"] * DAY)  # times may be < 0
    assert len(events) == 400
    p = Params.defaults()
    base = tuner.loss(p, events, now)
    best = None
    for name, spec in PARAM_SPECS.items():
        for target in (p.get(name) - spec.step, p.get(name) + spec.step):
            cand = p.with_value(name, round(target, 9))
            if cand.get(name) == p.get(name) or not cand.is_valid():
                continue
            cand_loss = tuner.loss(cand, events, now)
            if cand_loss < base * (1 - FIXED["improvement_margin"]) and (
                best is None or cand_loss < best[2]
            ):
                best = (name, cand.get(name), cand_loss)
    assert best is not None
    first = tuner.run()["moves"][0]
    assert (first["name"], first["new"]) == (best[0], pytest.approx(best[1]))
    assert first["old_loss"] == pytest.approx(base)
    assert first["new_loss"] == pytest.approx(best[2])


def test_history_written_with_losses(tuner, store, clock):
    _missed_decisions(store, clock.now())
    r = tuner.run()
    rows = list(reversed(store.params_history(limit=10)))
    assert [(h["name"], h["old_value"], h["new_value"], h["old_loss"], h["new_loss"], h["reason"])
            for h in rows] == [
        (m["name"], m["old"], m["new"], m["old_loss"], m["new_loss"], "replay") for m in r["moves"]
    ]
    assert rows[0]["old_loss"] == 50 and rows[-1]["new_loss"] < rows[0]["old_loss"]
    assert store.load_params().get("decay.decision") == pytest.approx(0.25)
