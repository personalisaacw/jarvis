import json

import numpy as np
import pytest

from memory.adapters.memory_op_seeds import SEEDS
from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.entities import (
    EventLabel,
    ExplicitOp,
    Extraction,
    MemoryKind,
    MemoryOp,
    Status,
    project_scope,
)
from memory.domain.params import Params
from memory.service import MemoryService
from memory.use_cases.memory_ops import ClassifyMemoryOpUseCase
from tests.memory.conftest import FakeExtractor
from tests.memory.helpers import event


@pytest.fixture
def store(clock):
    s = SqliteMemoryStore(":memory:", clock)
    yield s
    s.close()


@pytest.fixture
def extractor():
    return FakeExtractor()


def build(tmp_path, embedder, extractor, clock, store) -> MemoryService:
    return MemoryService(embedder, str(tmp_path), extractor=extractor, clock=clock, store=store,
                         start_threads=False)


@pytest.fixture
def service(tmp_path, embedder, extractor, clock, store):
    return build(tmp_path, embedder, extractor, clock, store)


def record(service, text, mode="quick", op_fired=False):
    v = service.embed(text)
    service.record_turn(text, "ok", mode, v, service.recall_for_prompt(v, mode), op_fired)


# ---- startup ----

def test_first_start_seeds_ops_and_migrates(tmp_path, embedder, extractor, clock, store):
    (tmp_path / ".vectorstore").mkdir()
    (tmp_path / ".vectorstore" / "metadata.json").write_text(json.dumps(
        [{"text": "play some jazz", "intent": "quick", "timestamp": 5.0}]))
    build(tmp_path, embedder, extractor, clock, store)
    build(tmp_path, embedder, extractor, clock, store)  # second start is a no-op
    routes = store.list(None, [MemoryKind.ROUTE_EXAMPLE])
    assert [(m.content, m.label) for m in routes] == [("play some jazz", "quick")]
    ops = store.list(Status.CURRENT, [MemoryKind.MEMORY_OP_EXAMPLE])
    assert len(ops) == sum(len(v) for v in SEEDS.values())


def test_no_embedder_skips_migration_seeding_and_capture(tmp_path, extractor, clock, store,
                                                         capsys):
    (tmp_path / ".vectorstore").mkdir()
    (tmp_path / ".vectorstore" / "metadata.json").write_text(json.dumps(
        [{"text": "play some jazz", "intent": "quick", "timestamp": 5.0}]))
    service = MemoryService(None, str(tmp_path), extractor=extractor, clock=clock, store=store,
                            start_threads=False)
    assert store.list(None) == []
    with pytest.raises(RuntimeError, match="embedder unavailable in CLI"):
        service.embed("x")
    t = service._turn("quick", "hi", "ok", np.zeros(384, dtype=np.float32), "global", [])
    store.add_turn(t)
    service.buffer.flush("sleep")
    assert len(store.pending_turns()) == 1 and extractor.calls == []
    assert "capture skipped" in capsys.readouterr().out


def test_embed_returns_float32_array(service, embedder):
    v = service.embed("hello there")
    assert isinstance(v, np.ndarray) and v.dtype == np.float32
    assert np.allclose(v, embedder.embed_text("hello there"))


# ---- Net 1 ----

def test_try_memory_op_returns_none_for_ordinary_text(service):
    assert service.try_memory_op("tell me about penguins", service.embed("tell me about penguins")) is None


def test_try_memory_op_returns_spoken_reply_when_consumed(service):
    text = SEEDS[MemoryOp.STATUS][0]
    assert service.try_memory_op(text, service.embed(text)) == service.status_line()


def test_missed_op_carries_the_classifier_score_of_its_turn(tmp_path, embedder, clock, store):
    class ExplicitStoreExtractor(FakeExtractor):
        def extract(self, turns, recalled):
            return Extraction(explicit_ops=[ExplicitOp(turns[0].id, MemoryOp.STORE)])

    service = build(tmp_path, embedder, ExplicitStoreExtractor(), clock, store)
    text = "hang on to the staging url"
    v = service.embed(text)
    expected = ClassifyMemoryOpUseCase(store, lambda: Params.defaults(), clock).execute(v).score
    assert service.try_memory_op(text, v) is None
    service.record_turn(text, "ok", "quick", v, service.recall_for_prompt(v, "quick"), False)
    service.buffer.flush("idle")
    missed = [e for e in store.events_since(0) if e.label == EventLabel.MISSED_OP]
    assert len(missed) == 1
    assert missed[0].context["classifier_score"] == pytest.approx(expected)


# ---- capture and recall ----

def test_record_turn_then_recall_shows_it_after_flush(service, store, extractor):
    extractor.responses = [Extraction.from_dict({"new": [
        {"kind": "decision", "scope": "project", "content": "We use SQLite for memory",
         "importance": 0.8}]})]
    record(service, "let's keep memory in sqlite", mode="think")
    service.buffer.flush("idle")
    assert store.pending_turns() == []
    result = service.recall_for_prompt(service.embed("We use SQLite for memory"), "think")
    assert result.memories[0].content == "We use SQLite for memory"
    assert result.memories[0].scope == service.project_scope


def test_record_turn_builds_turn_with_recalled_ids(service, store):
    record(service, "hi")
    turns = store.pending_turns()
    assert len(turns) == 1
    t = turns[0]
    assert (t.session_id, t.project_scope, t.mode, t.user_text, t.assistant_text) == (
        service.session_id, service.project_scope, "quick", "hi", "ok")
    assert t.recalled_ids == []


def test_record_turn_without_recall_result_records_op_turn(service, store):
    text = "remember that I prefer tabs"
    service.record_turn(text, "Got it.", "think", service.embed(text), None, op_fired=True)
    turns = store.recent_turns(service.session_id, 10)
    assert len(turns) == 1
    t = turns[0]
    assert (t.mode, t.user_text, t.assistant_text) == ("think", text, "Got it.")
    assert t.recalled_ids == []
    assert t.salience >= 1.0


def test_record_coding_turn_flushes_session_end(service, store, extractor, tmp_path):
    cwd = str(tmp_path / "sub")
    service.record_coding_turn("add a retry", 2, 1, ["a.py", "b.py"], cwd)
    assert store.pending_turns() == []
    (turns, _), = [args for name, args in extractor.calls if name == "extract"]
    assert len(turns) == 1
    t = turns[0]
    assert t.mode == "code" and t.user_text == "add a retry"
    assert t.assistant_text == "accepted 2 hunks, rejected 1 in files a.py, b.py"
    assert t.project_scope == project_scope(cwd)


def test_mode_switch_to_code_flushes_pending_turns(service, store, extractor):
    record(service, "hi")
    assert service.buffer.pending_count() == 1
    service.on_mode_switch_to_code()
    assert store.pending_turns() == []
    assert [n for n, _ in extractor.calls].count("extract") == 1


def test_capture_failure_is_logged_not_raised(service, store, extractor, capsys):
    class Boom(Exception):
        pass

    def explode(*a, **k):
        raise Boom("disk on fire")

    service.capture.execute = explode
    record(service, "hi")
    service.buffer.flush("idle")
    assert "[Memory] capture failed: disk on fire" in capsys.readouterr().out


def test_system_prompt_and_code_prefix_render_recalled_memories(service, extractor):
    extractor.responses = [Extraction.from_dict({"new": [
        {"kind": "decision", "scope": "project", "content": "We use SQLite for memory",
         "importance": 0.8}]})]
    record(service, "let's keep memory in sqlite", mode="think")
    service.buffer.flush("idle")
    result = service.recall_for_prompt(service.embed("We use SQLite for memory"), "think")
    assert service.system_prompt("BASE", result).startswith("BASE")
    assert "We use SQLite for memory" in service.system_prompt("BASE", result)
    assert "We use SQLite for memory" in service.code_prefix(result)


# ---- visibility ----

def test_status_line_mentions_counts(service):
    line = service.status_line()
    assert "memories" in line
    assert "have no recall data yet" in line


def test_status_line_reports_rates_and_last_move(service, store, clock):
    for label in [EventLabel.HIT, EventLabel.HIT, EventLabel.HIT, EventLabel.FALSE_RECALL,
                  EventLabel.MISS, EventLabel.MISS, EventLabel.EMPTY_EXTRACTION]:
        store.add_event(event(label, turn_id="t1"))  # one batch with many recall events
    for _ in range(3):
        store.add_event(event(EventLabel.EXTRACTION, turn_id="t1", turns=2, buffer_salience=1.0))
    store.add_event(event(EventLabel.EXTRACTION, at=clock.now() - 31 * 86400, turns=1,
                          buffer_salience=1.0))  # outside the window
    store.save_params(Params.defaults().with_value("decay.decision", 0.25),
                      [("decay.decision", 0.3, 0.25, 2.0, 1.0, "tune")])
    line = service.status_line()
    assert "I hold 0 memories, 0 active." in line
    assert "Over the last month I recalled correctly 75 percent of the time" in line
    assert "missed 2 times and ran 3 extractions." in line
    assert line.endswith("Last night I slowed how fast decisions fade.")


def test_status_line_says_sped_up_and_other_params(service, store):
    store.save_params(Params.defaults(), [("decay.fact", 0.5, 0.55, 2.0, 1.0, "tune")])
    assert service.status_line().endswith("Last night I sped up how fast facts fade.")
    store.save_params(Params.defaults(), [("recall_k", 6, 7, 2.0, 1.0, "tune")])
    assert service.status_line().endswith("Last night I adjusted recall_k.")


def test_report_sections_in_order_with_moved_params_flagged(service, store):
    store.save_params(Params.defaults().with_value("recall_k", 8),
                      [("recall_k", 6, 8, 2.0, 1.5, "tune")])
    store.add_event(event(EventLabel.FALSE_OP))
    out = service.report()
    headings = ["Counts", "Recall rates", "Extraction", "Classifier", "Params", "History",
                "Sleep"]
    positions = [out.index(h) for h in headings]
    assert positions == sorted(positions)
    assert "* recall_k = 8 (default 6)" in out
    assert "  importance_weight = 1\n" in out
    assert "false_op 1" in out
    assert "last_sleep_at: never" in out
    assert "batches 0, mean turns per batch 0.0" in out and "(rate n/a)" in out


def test_report_extraction_stats_come_from_extraction_events(service, store, clock):
    for n, sal in [(2, 1.0), (2, 2.0), (3, 1.5), (1, 0.5)]:
        store.add_event(event(EventLabel.EXTRACTION, turns=n, buffer_salience=sal))
    store.add_event(event(EventLabel.EXTRACTION, at=clock.now() - 31 * 86400, turns=9,
                          buffer_salience=9.0))  # outside the window
    store.add_event(event(EventLabel.EMPTY_EXTRACTION, buffer_salience=0.5))
    store.add_event(event(EventLabel.LATE_EXTRACTION))
    out = service.report()
    assert "batches 4, mean turns per batch 2.0, mean salience at flush 1.25" in out
    assert "empty_extraction 1 (rate 25%), late_extraction 1" in out


def test_try_memory_op_clears_a_stale_score_first(service):
    service._pending_score = 0.42  # left over from a turn that was never recorded

    def boom(*a, **k):
        raise RuntimeError("handler down")

    service.handler.handle = boom
    with pytest.raises(RuntimeError):
        service.try_memory_op("hi", service.embed("hi"))
    assert service._pending_score is None


def test_scores_are_pruned_after_their_batch_is_captured(service, store):
    v = service.embed("tell me about penguins")
    assert service.try_memory_op("tell me about penguins", v) is None
    service.record_turn("tell me about penguins", "ok", "quick", v,
                        service.recall_for_prompt(v, "quick"), False)
    assert len(service._scores) == 1
    service.buffer.flush("idle")
    assert service._scores == {}


def test_reset_params_restores_defaults_with_history(service, store):
    store.save_params(Params.defaults().with_value("recall_k", 8), [])
    service.reset_params()
    assert store.load_params().as_dict() == Params.defaults().as_dict()
    (row,) = store.params_history(10)
    assert (row["name"], row["old_value"], row["new_value"], row["reason"]) == (
        "recall_k", 8, 6, "manual reset")


# ---- review fixes ----

def test_repeated_quick_turn_is_less_salient_than_its_first_occurrence(service, store):
    v = service.embed("ok")
    service.record_turn("ok", "ok", "quick", v, None, False)
    service.record_turn("ok", "ok", "quick", v, None, False)
    first, second = store.recent_turns(service.session_id, 10)
    assert second.salience < first.salience
    assert second.salience == pytest.approx(
        Params.defaults().get("salience_length_bonus") / 40.0, abs=1e-5)


def test_corrupt_db_is_moved_aside_and_service_starts_fresh(tmp_path, embedder, extractor,
                                                             clock, capsys):
    db = tmp_path / "memory.sqlite"
    db.write_bytes(b"this is not a sqlite database " * 50)
    service = MemoryService(embedder, str(tmp_path), str(db), extractor=extractor, clock=clock,
                            start_threads=False)
    try:
        assert service.store.list(None, [MemoryKind.MEMORY_OP_EXAMPLE])  # seeded afresh
        moved = [p for p in tmp_path.iterdir() if p.name.startswith("memory.sqlite.corrupt-")]
        assert len(moved) == 1
        assert moved[0].read_bytes().startswith(b"this is not a sqlite database")
        assert moved[0].name == "memory.sqlite.corrupt-%d" % int(clock.now())
        out = capsys.readouterr().out
        assert ("[Memory] WARNING: memory database was unreadable; moved to %s and started "
                "fresh" % moved[0]) in out
    finally:
        service.close()


def test_startup_survives_migration_and_seed_failures(tmp_path, embedder, extractor, clock,
                                                       store, monkeypatch, capsys):
    import memory.service as svc

    def boom(*a, **k):
        raise RuntimeError("kaput")

    monkeypatch.setattr(svc, "migrate_faiss", boom)
    monkeypatch.setattr(svc, "seed_memory_ops", boom)
    build(tmp_path, embedder, extractor, clock, store)
    out = capsys.readouterr().out
    assert "[Memory] WARNING: FAISS migration failed (kaput)" in out
    assert "[Memory] WARNING: seeding memory-op examples failed (kaput)" in out


def test_explicit_op_on_a_turn_where_net1_fired_is_not_a_missed_op(tmp_path, embedder, clock,
                                                                   store):
    class ExplicitStoreExtractor(FakeExtractor):
        def extract(self, turns, recalled):
            return Extraction(explicit_ops=[ExplicitOp(t.id, MemoryOp.STORE) for t in turns])

    service = build(tmp_path, embedder, ExplicitStoreExtractor(), clock, store)
    protos_before = len(store.list(Status.CURRENT, [MemoryKind.MEMORY_OP_EXAMPLE]))
    fired, missed = "remember that I prefer tabs", "hang on to the staging url"
    service.record_turn(fired, "Noted.", "think", service.embed(fired), None, op_fired=True)
    service.record_turn(missed, "ok", "think", service.embed(missed), None, op_fired=False)
    service.buffer.flush("idle")
    (missed_op,) = [e for e in store.events_since(0) if e.label == EventLabel.MISSED_OP]
    [missed_turn] = [t for t in store.recent_turns(service.session_id, 10)
                     if t.user_text == missed]
    assert missed_op.turn_id == missed_turn.id
    protos = store.list(Status.CURRENT, [MemoryKind.MEMORY_OP_EXAMPLE])
    assert len(protos) == protos_before + 1 and fired not in [m.content for m in protos]
    assert service._op_fired_turns == set()  # pruned after the batch


def test_op_fired_mark_survives_a_failed_extraction_for_the_retry(service, store, extractor):
    text = "remember that I prefer tabs"
    extractor.raise_next = 1  # the op turn's salience flushes it at once; that call fails
    service.record_turn(text, "Noted.", "quick", service.embed(text), None, op_fired=True)
    assert len(store.pending_turns()) == 1 and len(service._op_fired_turns) == 1
    service.buffer.flush("idle")
    assert store.pending_turns() == [] and service._op_fired_turns == set()
