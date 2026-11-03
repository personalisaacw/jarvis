from typing import List, Tuple
from uuid import uuid4

import pytest

from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.entities import Turn
from memory.domain.params import Params
from memory.use_cases.extraction_buffer import ExtractionBuffer
from tests.memory.conftest import vec


@pytest.fixture
def flushed() -> List[Tuple[List[Turn], str]]:
    return []


@pytest.fixture
def buffer(clock, flushed):
    store = SqliteMemoryStore(":memory:", clock)
    return ExtractionBuffer(
        store,
        params_provider=lambda: Params.defaults(),
        clock=clock,
        on_flush=lambda turns, reason: flushed.append((list(turns), reason)),
    )


@pytest.fixture
def make_turn(clock):
    def _turn(text, mode="think"):
        return Turn(
            id=str(uuid4()),
            session_id="s",
            at=clock.now(),
            project_scope="project:/p",
            mode=mode,
            user_text=text,
            assistant_text="",
            embedding=vec(text),
        )

    return _turn


def _low(buffer, make_turn, text="ok"):
    return buffer.add_turn(make_turn(text, mode="quick"), max_cosine=0.95, op_fired=False)


def test_flushes_when_salience_crosses_threshold(buffer, flushed, make_turn):
    s1 = buffer.add_turn(make_turn("novel idea one"), max_cosine=0.0, op_fired=False)
    assert s1 == pytest.approx(1.0225)
    assert flushed == []
    buffer.add_turn(make_turn("novel idea two"), max_cosine=0.0, op_fired=False)
    assert flushed[0][1] == "salience" and len(flushed[0][0]) == 2
    assert len(flushed) == 1


def test_add_turn_sets_and_persists_salience(buffer, make_turn):
    t = make_turn("novel idea one")
    s = buffer.add_turn(t, max_cosine=0.0, op_fired=False)
    assert t.salience == s
    assert buffer.pending_count() == 1


def test_low_salience_turns_wait_for_idle(buffer, flushed, clock, make_turn):
    for _ in range(3):
        _low(buffer, make_turn)
    buffer.tick()
    assert flushed == []
    clock.advance(91)
    buffer.tick()
    assert flushed[0][1] == "idle" and len(flushed[0][0]) == 3


def test_new_turn_resets_idle_timer(buffer, flushed, clock, make_turn):
    _low(buffer, make_turn, "a")
    clock.advance(60)
    _low(buffer, make_turn, "b")
    clock.advance(60)
    buffer.tick()
    assert flushed == []
    clock.advance(31)
    buffer.tick()
    assert flushed[0][1] == "idle"


def test_tick_with_nothing_pending_is_noop(buffer, flushed, clock):
    clock.advance(1000)
    buffer.tick()
    assert flushed == []


def test_max_turns_cap(buffer, flushed, make_turn):
    for i in range(9):
        _low(buffer, make_turn, "dup %d" % i)
    assert flushed == []
    _low(buffer, make_turn, "dup 9")
    assert len(flushed) == 1
    assert flushed[0][1] == "max_turns" and len(flushed[0][0]) == 10


def test_flush_empty_is_noop(buffer, flushed):
    buffer.flush("sleep")
    assert flushed == []


def test_total_resets_after_flush(buffer, flushed, make_turn):
    _low(buffer, make_turn)
    assert buffer.total > 0
    buffer.flush("session_end")
    assert buffer.total == 0.0
    assert flushed[0][1] == "session_end"


def test_flush_while_flushing_is_skipped(clock, make_turn):
    store = SqliteMemoryStore(":memory:", clock)
    calls = []

    def on_flush(turns, reason):
        calls.append(reason)
        buf.flush("sleep")  # re-entrant: pending turns are not yet marked extracted

    buf = ExtractionBuffer(store, lambda: Params.defaults(), clock, on_flush)
    _low(buf, make_turn)
    buf.flush("idle")
    assert calls == ["idle"]
    buf.flush("idle")  # guard cleared in finally
    assert calls == ["idle", "idle"]
