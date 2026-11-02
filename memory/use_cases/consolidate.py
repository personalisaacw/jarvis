import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from domain.ports import IEmbeddingEngine
from memory.domain.entities import (
    GLOBAL_SCOPE, Memory, MemoryKind, NewMemory, Status, Tier,
)
from memory.domain.params import Params
from memory.domain.scoring import strength, tier_for
from memory.ports.clock import IClock
from memory.ports.extractor import IMemoryExtractor
from memory.ports.store import IMemoryStore
from memory.use_cases.extraction_buffer import ExtractionBuffer

MIN_EPISODES_TO_SUMMARISE = 3


class _SkippedTuner:
    def run(self) -> dict:
        return {"status": "skipped_dry_run", "events": 0, "moves": []}


@dataclass
class SleepReport:
    rescored: int
    summarised: int
    deduped: int
    purged: int
    tune: dict
    dry_run: bool


class SleepUseCase:
    def __init__(
        self,
        store: IMemoryStore,
        extractor: IMemoryExtractor,
        buffer: Optional[ExtractionBuffer],
        tuner,
        params_provider: Callable[[], Params],
        clock: IClock,
        on_done: Optional[Callable[[], None]] = None,
        embedder: Optional[IEmbeddingEngine] = None,
        tuner_for_store: Optional[Callable[[IMemoryStore], object]] = None,
    ):
        self._store = store
        self._extractor = extractor
        self._buffer = buffer
        self._tuner = tuner
        self._params = params_provider
        self._clock = clock
        self._on_done = on_done
        self._embedder = embedder
        self._tuner_for_store = tuner_for_store

    def is_due(self) -> bool:
        last = self._store.get_meta("last_sleep_at")
        if last is None:
            return True
        interval = self._params().get("sleep_interval_hours") * 3600
        return self._clock.now() - float(last) > interval

    def run(self, dry_run: bool = False) -> SleepReport:
        if dry_run:
            copy = self._store.copy_in_memory()
            try:
                tuner = self._tuner_for_store(copy) if self._tuner_for_store else _SkippedTuner()
                trial = SleepUseCase(
                    copy, self._extractor, None, tuner, self._params, self._clock,
                    embedder=self._embedder,
                )
                report = trial._run_steps(dry_run=True)
            except Exception as e:
                print("[Memory/Sleep] failed: %s" % e)
                raise
            finally:
                copy.close()
            return report
        try:
            if self._buffer is not None:
                self._buffer.flush("sleep")
            report = self._run_steps(dry_run=False)
            self._store.set_meta("last_sleep_at", str(self._clock.now()))
        except Exception as e:
            print("[Memory/Sleep] failed: %s" % e)
            raise
        if self._on_done:
            self._on_done()
        return report

    def _run_steps(self, dry_run: bool) -> SleepReport:
        now, p = self._clock.now(), self._params()
        rescored = self._rescore(now, p)
        summarised = self._summarise(now, p)
        deduped = self._store.dedup(list(MemoryKind), p.get("dedup_threshold"), now)
        purged = self._store.purge(now - p.get("retention_days") * 86400)
        tune = self._tuner.run()
        return SleepReport(rescored, summarised, deduped, purged, tune, dry_run)

    def _rescore(self, now: float, p: Params) -> int:
        changed = 0
        for m in self._store.list(Status.CURRENT):
            if m.feature_id is not None:  # folded into a summary: tier pinned at ARCHIVED
                continue
            tier = tier_for(strength(m, now, p), p)
            if tier != m.tier:
                m.tier = tier
                self._store.update(m)
                changed += 1
        return changed

    def _summarise(self, now: float, p: Params) -> int:
        cutoff = now - p.get("episode_summary_age_days") * 86400
        by_scope: Dict[str, List[Memory]] = {}
        for m in self._store.list(Status.CURRENT, [MemoryKind.EPISODE]):
            if m.scope != GLOBAL_SCOPE and m.created_at < cutoff and m.feature_id is None:
                by_scope.setdefault(m.scope, []).append(m)
        created = 0
        for scope, episodes in by_scope.items():
            if len(episodes) < MIN_EPISODES_TO_SUMMARISE:
                continue
            try:
                new = self._extractor.summarise(episodes)
            except Exception as e:
                print("[Memory/Sleep] warning: summarise failed for %s: %s" % (scope, e))
                continue
            summaries = [self._add_summary(n, scope, episodes, now, p) for n in new]
            if not summaries:
                continue
            for ep in episodes:
                ep.tier, ep.feature_id = Tier.ARCHIVED, summaries[0].id
                self._store.update(ep)
            created += len(summaries)
        return created

    def _add_summary(
        self, n: NewMemory, scope: str, episodes: List[Memory], now: float, p: Params
    ) -> Memory:
        newest = max(episodes, key=lambda m: m.last_accessed_at)
        if self._embedder is not None:
            vec = np.asarray(self._embedder.embed_text(n.content), dtype=np.float32)
        else:  # no embedder wired: fall back to the centroid of the folded episodes
            vec = np.mean([m.embedding for m in episodes], axis=0)
        norm = float(np.linalg.norm(vec))
        s = Memory(
            id=str(uuid.uuid4()), kind=n.kind, scope=scope if n.scope == "project" else GLOBAL_SCOPE,
            content=n.content, embedding=(vec / norm if norm else vec).astype(np.float32),
            importance=min(1.0, max(0.0, n.importance)), status=Status.CURRENT, tier=Tier.ACTIVE,
            created_at=now,
        )
        self._store.add(s)
        for a in newest.accesses:
            self._store.add_access(s.id, a.at, a.kind)
        s = self._store.get(s.id)
        s.tier = tier_for(strength(s, now, p), p)
        self._store.update(s)
        return s
