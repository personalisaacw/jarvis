import os
import threading
import time
import uuid
from collections import Counter
from typing import Callable, Dict, List, Optional

import numpy as np

from domain.ports import IEmbeddingEngine
from memory.adapters.extractor_factory import build_extractor
from memory.adapters.memory_op_seeds import seed_memory_ops
from memory.adapters.migrate_faiss import migrate_faiss
from memory.adapters.prompt_renderer import render_code_prefix, render_system_prompt
from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.entities import EventLabel, MemoryEvent, MemoryKind, Tier, Turn, project_scope
from memory.domain.params import FIXED, PARAM_SPECS, Params
from memory.ports.clock import IClock, SystemClock
from memory.ports.extractor import IMemoryExtractor
from memory.use_cases.capture import CaptureBatchUseCase
from memory.use_cases.consolidate import SleepUseCase
from memory.use_cases.extraction_buffer import ExtractionBuffer
from memory.use_cases.memory_ops import HandleMemoryOpUseCase
from memory.use_cases.recall import RecallResult, RecallUseCase
from memory.use_cases.tune import TuneParamsUseCase

_EXAMPLE_KINDS = (MemoryKind.ROUTE_EXAMPLE, MemoryKind.MEMORY_OP_EXAMPLE)
_RECALLABLE_KINDS = [k for k in MemoryKind if k not in _EXAMPLE_KINDS]
_RECALL_LABELS = (EventLabel.HIT, EventLabel.FALSE_RECALL, EventLabel.MISS, EventLabel.STALE)
_DAY = 86400.0
_TICK_SECONDS = 5.0


def _fmt_time(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


class MemoryService:
    """The single memory object jarvis.py talks to: wires every memory use case together.

    `embedder=None` is the store-only mode the CLI uses: no migration or seeding, captures
    are skipped (turns stay pending for the next live session) and sleep summaries fall back
    to the centroid of the folded episodes.
    """

    def __init__(
        self,
        embedder: Optional[IEmbeddingEngine],
        project_dir: str,
        db_path: str = ".jarvis/memory.sqlite",
        extractor: Optional[IMemoryExtractor] = None,
        clock: IClock = SystemClock(),
        store: Optional[SqliteMemoryStore] = None,
        start_threads: bool = True,
    ):
        self.embedder = embedder
        self.clock = clock
        self.project_scope = project_scope(project_dir)
        self.session_id = str(uuid.uuid4())
        self.store = store if store is not None else SqliteMemoryStore(db_path, clock)
        self.extractor = extractor if extractor is not None else build_extractor()
        self._threaded = start_threads
        self._params = self.store.load_params()
        self._pending_score = None  # type: Optional[float]
        self._scores: Dict[str, float] = {}
        self._capture_lock = threading.Lock()
        self._buffer_lock = threading.RLock()
        self._stop = threading.Event()

        params: Callable[[], Params] = lambda: self._params
        self.recall = RecallUseCase(self.store, params, clock, self.session_id)
        self.capture = CaptureBatchUseCase(
            self.store, self.extractor, embedder, params, clock,
            on_profile_change=self.recall.invalidate_profile,
            classifier_score_for=lambda turn_id: self._scores.pop(turn_id, None),
        )
        self.buffer = ExtractionBuffer(self.store, params, clock, self._on_flush)
        self.handler = HandleMemoryOpUseCase(
            self.store, self.extractor, self.recall, self.capture, embedder, params, clock,
            project_scope_provider=lambda: self.project_scope,
            status_line_provider=self.status_line,
        )
        self.tuner = TuneParamsUseCase(self.store, clock)
        self.sleep = SleepUseCase(
            self.store, self.extractor, self.buffer, self.tuner, params, clock,
            on_done=self._after_sleep, embedder=embedder,
            tuner_for_store=lambda s: TuneParamsUseCase(s, clock),
        )

        if embedder is not None:
            migrate_faiss(self.store, embedder,
                          os.path.join(project_dir, ".vectorstore", "metadata.json"))
            seed_memory_ops(self.store, embedder, clock)
        if start_threads:
            self._spawn(self._tick_loop)
            if self.sleep.is_due():
                self._spawn(self._run_sleep)

    # ---- hot path ----

    def embed(self, text: str) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError("embedder unavailable in CLI")
        return np.asarray(self.embedder.embed_text(text), dtype=np.float32)

    def try_memory_op(self, transcript: str, vec: np.ndarray) -> Optional[str]:
        # handle() closes the one-turn undo window itself, so no note_turn_passed() here.
        self._pending_score = None
        outcome = self.handler.handle(transcript, vec, self._working_turns())
        if outcome.consumed:
            return outcome.spoken
        self._pending_score = outcome.score
        return None

    def recall_for_prompt(self, vec: np.ndarray, mode: str) -> RecallResult:
        return self.recall.execute(vec, self.project_scope, mode)

    def system_prompt(self, base: str, result: RecallResult) -> str:
        return render_system_prompt(base, result.profile, result.memories, result.working_turns,
                                    self.project_scope)

    def code_prefix(self, result: RecallResult) -> str:
        return render_code_prefix(result.profile, result.memories, self.project_scope)

    def record_turn(
        self, user_text: str, assistant_text: str, mode: str, vec: np.ndarray,
        recalled: Optional[RecallResult], op_fired: bool,
    ) -> None:
        recalled_ids = [m.id for m in recalled.memories] if recalled is not None else []
        turn = self._turn(mode, user_text, assistant_text, vec, self.project_scope, recalled_ids)
        if self._pending_score is not None:
            self._scores[turn.id] = self._pending_score
            self._pending_score = None
        max_cos = self.store.max_cosine(vec, kinds=_RECALLABLE_KINDS)
        with self._buffer_lock:
            self.buffer.add_turn(turn, max_cos, op_fired)

    def record_coding_turn(
        self, prompt: str, accepted: int, rejected: int, files: List[str], cwd: str
    ) -> None:
        summary = "accepted %d hunks, rejected %d in files %s" % (accepted, rejected,
                                                                 ", ".join(files))
        vec = self.embed(prompt)
        turn = self._turn("code", prompt, summary, vec, project_scope(cwd), [])
        max_cos = self.store.max_cosine(vec, kinds=_RECALLABLE_KINDS)
        with self._buffer_lock:
            self.buffer.add_turn(turn, max_cos, False)
            self.buffer.flush("session_end")

    def on_mode_switch_to_code(self) -> None:
        with self._buffer_lock:
            self.buffer.flush("mode_switch")

    def reset_params(self) -> None:
        old, new = self.store.load_params().as_dict(), Params.defaults()
        history = [(n, old[n], v, 0.0, 0.0, "manual reset")
                   for n, v in new.as_dict().items() if old[n] != v]
        self.store.save_params(new, history)
        self._after_sleep()

    def close(self) -> None:
        self._stop.set()
        self.store.close()

    # ---- visibility ----

    def status_line(self) -> str:
        counts = [c for c in self.store.count_by() if MemoryKind(c[0]) not in _EXAMPLE_KINDS]
        total = sum(c[3] for c in counts)
        active = sum(c[3] for c in counts if c[1] == Tier.ACTIVE.value)
        events = self._window_events()
        n = Counter(e.label for e in events)
        judged = n[EventLabel.HIT] + n[EventLabel.FALSE_RECALL]
        recall = ("recalled correctly %d percent of the time"
                  % round(100.0 * n[EventLabel.HIT] / judged) if judged
                  else "have no recall data yet")
        line = ("I hold %d memories, %d active. Over the last month I %s, missed %d times "
                "and ran %d extractions." % (total, active, recall, n[EventLabel.MISS],
                                             n[EventLabel.EXTRACTION]))
        last = self.store.params_history(1)
        return line + (" " + self._move_sentence(last[0]) if last else "")

    def report(self) -> str:
        events = self._window_events()
        days = FIXED["tune_window_days"]
        sections = [
            ["Counts (kind / tier / scope)"] + self._report_counts(),
            ["Recall rates (last %d days)" % days] + self._report_rates(events),
            ["Extraction (last %d days)" % days] + self._report_extraction(events),
            ["Classifier (last %d days)" % days,
             "  false_op %d, missed_op %d" % (
                 sum(e.label == EventLabel.FALSE_OP for e in events),
                 sum(e.label == EventLabel.MISSED_OP for e in events))],
            ["Params (* = moved from default)"] + self._report_params(),
            ["History (last 10)"] + self._report_history(),
            ["Sleep"] + self._report_sleep(),
        ]
        return "\n\n".join("\n".join(s) for s in sections) + "\n"

    # ---- private: wiring ----

    def _turn(self, mode: str, user_text: str, assistant_text: str, vec: np.ndarray,
              scope: str, recalled_ids: List[str]) -> Turn:
        return Turn(id=str(uuid.uuid4()), session_id=self.session_id, at=self.clock.now(),
                    project_scope=scope, mode=mode, user_text=user_text,
                    assistant_text=assistant_text, embedding=vec, recalled_ids=recalled_ids)

    def _working_turns(self) -> List[Turn]:
        return self.store.recent_turns(self.session_id, int(self._params.get("working_turns")))

    def _on_flush(self, turns: List[Turn], reason: str) -> None:
        if self.embedder is None:
            print("[Memory] capture skipped (no embedder): %d turns left pending" % len(turns))
            return
        if not self._capture_lock.acquire(blocking=False):
            return  # a capture is still running; these turns stay pending for the next flush
        if self._threaded:
            self._spawn(self._capture, turns)
        else:
            self._capture(turns)

    def _capture(self, turns: List[Turn]) -> None:
        try:
            self.capture.execute(turns, self.project_scope, sum(t.salience for t in turns))
            for t in turns:  # scores not claimed by a missed_op are no longer needed
                self._scores.pop(t.id, None)
            self.store.delete_turns_before(self.clock.now() - _DAY, self.session_id,
                                           int(self._params.get("working_turns")))
        except Exception as e:
            print("[Memory] capture failed: %s" % e)
        finally:
            self._capture_lock.release()

    def _after_sleep(self) -> None:
        self._params = self.store.load_params()
        self.recall.invalidate_profile()

    def _spawn(self, target: Callable, *args) -> None:
        threading.Thread(target=target, args=args, daemon=True).start()

    def _tick_loop(self) -> None:
        while not self._stop.wait(_TICK_SECONDS):
            try:
                with self._buffer_lock:
                    self.buffer.tick()
            except Exception as e:
                print("[Memory] idle tick failed: %s" % e)

    def _run_sleep(self) -> None:
        try:
            self.sleep.run()
        except Exception as e:
            print("[Memory] sleep failed: %s" % e)

    # ---- private: report helpers ----

    def _window_events(self) -> List[MemoryEvent]:
        return self.store.events_since(self.clock.now() - FIXED["tune_window_days"] * _DAY)

    @staticmethod
    def _move_sentence(row: dict) -> str:
        name = row["name"]
        if name.startswith("decay."):
            verb = "slowed" if row["new_value"] < row["old_value"] else "sped up"
            kind = name[len("decay."):].replace("_", " ") + "s"
            return "Last night I %s how fast %s fade." % (verb, kind)
        return "Last night I adjusted %s." % name

    def _report_counts(self) -> List[str]:
        rows = sorted(self.store.count_by())
        return ["  %-18s %-9s %s  %d" % r for r in rows] or ["  (no memories)"]

    def _report_rates(self, events: List[MemoryEvent]) -> List[str]:
        by_kind: Dict[str, Counter] = {}
        for e in events:
            if e.label in _RECALL_LABELS:
                m = self.store.get(e.memory_id) if e.memory_id else None
                by_kind.setdefault(m.kind.value if m else "unknown", Counter())[e.label] += 1
        lines = []
        for kind in sorted(by_kind):
            c, total = by_kind[kind], sum(by_kind[kind].values())
            lines.append("  %s: " % kind + ", ".join(
                "%s %d (%d%%)" % (lb.value, c[lb], round(100.0 * c[lb] / total))
                for lb in _RECALL_LABELS))
        return lines or ["  (no recall events)"]

    def _report_extraction(self, events: List[MemoryEvent]) -> List[str]:
        batches = [e.context for e in events if e.label == EventLabel.EXTRACTION]
        n = len(batches)
        empty = sum(e.label == EventLabel.EMPTY_EXTRACTION for e in events)
        late = sum(e.label == EventLabel.LATE_EXTRACTION for e in events)
        mean_turns = sum(c.get("turns", 0) for c in batches) / n if n else 0.0
        mean_sal = sum(c.get("buffer_salience", 0.0) for c in batches) / n if n else 0.0
        rate = "%d%%" % round(100.0 * empty / n) if n else "n/a"
        return ["  batches %d, mean turns per batch %.1f, "
                "mean salience at flush %.2f" % (n, mean_turns, mean_sal),
                "  empty_extraction %d (rate %s), late_extraction %d" % (empty, rate, late)]

    def _report_params(self) -> List[str]:
        current = self.store.load_params()
        lines = []
        for name, spec in PARAM_SPECS.items():
            v = current.get(name)
            if v != spec.default:
                lines.append("* %s = %g (default %g)" % (name, v, spec.default))
            else:
                lines.append("  %s = %g" % (name, v))
        return lines

    def _report_history(self) -> List[str]:
        rows = self.store.params_history(10)
        return ["  %s %s %g -> %g, loss %.3f -> %.3f (%s)" % (
            _fmt_time(r["at"]), r["name"], r["old_value"], r["new_value"], r["old_loss"],
            r["new_loss"], r["reason"]) for r in rows] or ["  (none)"]

    def _report_sleep(self) -> List[str]:
        last = self.store.get_meta("last_sleep_at")
        if last is None:
            return ["  last_sleep_at: never", "  next due: now"]
        due = float(last) + self._params.get("sleep_interval_hours") * 3600
        return ["  last_sleep_at: %s" % _fmt_time(float(last)), "  next due: %s" % _fmt_time(due)]
