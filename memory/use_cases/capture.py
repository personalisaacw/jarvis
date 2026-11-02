import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from memory.domain.entities import (
    GLOBAL_SCOPE,
    AccessKind,
    EventLabel,
    ExplicitOp,
    Extraction,
    Memory,
    MemoryEvent,
    MemoryKind,
    NewMemory,
    Status,
    Supersede,
    Tier,
    Turn,
)
from memory.domain.params import Params
from memory.domain.scoring import cosine, strength, tier_for
from memory.ports.clock import IClock

_EXAMPLE_KINDS = (MemoryKind.ROUTE_EXAMPLE, MemoryKind.MEMORY_OP_EXAMPLE)
_RECALLABLE_KINDS = [k for k in MemoryKind if k not in _EXAMPLE_KINDS]
_PROFILE_KINDS = (MemoryKind.PREFERENCE, MemoryKind.FEATURE)


def _new_id() -> str:
    return str(uuid.uuid4())


class CaptureBatchUseCase:
    """Applies one Net 2 extraction to the store and logs the labelled events."""

    def __init__(
        self,
        store,
        extractor,
        embedder,
        params_provider: Callable[[], Params],
        clock: IClock,
        on_profile_change: Optional[Callable[[], None]] = None,
        classifier_score_for: Optional[Callable[[str], Optional[float]]] = None,
        id_factory: Callable[[], str] = _new_id,
    ):
        self._store = store
        self._extractor = extractor
        self._embedder = embedder
        self._params = params_provider
        self._clock = clock
        self._on_profile_change = on_profile_change
        self._classifier_score_for = classifier_score_for or (lambda turn_id: None)
        self._new_id = id_factory

    # ---- entry point ----

    def execute(
        self, turns: List[Turn], project_scope: str, buffer_salience: float
    ) -> Optional[Extraction]:
        if not turns:
            return None
        turn_ids = [t.id for t in turns]
        retry_key = "retry:" + turns[0].id
        seen: List[str] = []
        for t in turns:
            for rid in t.recalled_ids:
                if rid not in seen:
                    seen.append(rid)
        recalled = [m for m in (self._store.get(i) for i in seen) if m is not None]
        try:
            extraction = self._extractor.extract(turns, recalled)
        except Exception as e:
            print("[Memory/Capture] extraction failed: %s" % e)
            if self._store.get_meta(retry_key):
                self._store.mark_extracted(turn_ids, self._clock.now())
                self._store.set_meta(retry_key, "")
            else:
                self._store.set_meta(retry_key, "1")
            return None

        touched = False
        for nm in extraction.new:
            _, t = self._apply_new(nm, project_scope, False, turns[-1].id, turns[-1].mode)
            touched = touched or t
        for mid in extraction.reinforce:
            self._apply_reinforce(mid, turns, project_scope)
        for mid in extraction.irrelevant:
            self._log_recalled(EventLabel.FALSE_RECALL, mid, turns, project_scope)
        for s in extraction.supersede:
            touched = self._apply_supersede(s, turns, project_scope) or touched
        for mid in extraction.complete:
            touched = self._apply_complete(mid) or touched
        for mid in extraction.forget:
            touched = self._apply_forget(mid, turns, project_scope) or touched
        for op in extraction.explicit_ops:
            self._apply_missed_op(op, turns)
        if extraction.is_empty():
            self._log(EventLabel.EMPTY_EXTRACTION, None, turns[-1].id,
                      {"buffer_salience": buffer_salience})
        self._log(EventLabel.EXTRACTION, None, turns[-1].id,
                  {"turns": len(turns), "buffer_salience": buffer_salience})

        self._store.mark_extracted(turn_ids, self._clock.now())
        if self._store.get_meta(retry_key):
            self._store.set_meta(retry_key, "")
        if touched and self._on_profile_change is not None:
            self._on_profile_change()
        return extraction

    # ---- new memories ----

    def apply_new(
        self, nm: NewMemory, project_scope: str, explicit: bool, turn_id: Optional[str],
        mode: Optional[str] = None,
    ) -> Memory:
        memory, touched = self._apply_new(nm, project_scope, explicit, turn_id, mode)
        if touched and self._on_profile_change is not None:
            self._on_profile_change()
        return memory

    def _apply_new(
        self, nm: NewMemory, project_scope: str, explicit: bool, turn_id: Optional[str],
        mode: Optional[str],
    ) -> Tuple[Memory, bool]:
        now = self._clock.now()
        scope = project_scope if nm.scope == "project" else GLOBAL_SCOPE
        vector = self._embed(nm.content)
        near = self._store.nearest(vector, 1, kinds=_RECALLABLE_KINDS, scope=scope)
        if near and near[0][1] >= self._params().get("dedup_threshold"):
            existing, sim = near[0]
            self._store.add_access(
                existing.id, now, AccessKind.EXPLICIT if explicit else AccessKind.REINFORCED
            )
            if explicit and existing.importance < 1.0:
                existing.importance = 1.0
                self._store.update(existing)
            if existing.tier != Tier.ACTIVE:
                context = {
                    "tier_at_time": existing.tier.value, "similarity": sim, "rank_at_time": 0,
                    "scope": existing.scope, "query_scope": project_scope,
                }
                if mode is not None:
                    context["mode"] = mode
                self._log(EventLabel.MISS, existing.id, turn_id, context)
            return existing, False
        importance = 1.0 if explicit else min(1.0, max(0.0, float(nm.importance)))
        memory = self._insert(nm.kind, scope, nm.content, vector, importance, None, explicit)
        return memory, memory.kind in _PROFILE_KINDS

    def _insert(
        self, kind: MemoryKind, scope: str, content: str, vector: np.ndarray,
        importance: float, label: Optional[str], explicit: bool = False,
    ) -> Memory:
        m = Memory(
            id=self._new_id(), kind=kind, scope=scope, content=content, embedding=vector,
            importance=importance, status=Status.CURRENT, tier=Tier.ACTIVE,
            created_at=self._clock.now(), label=label,
        )
        self._store.add(m)
        if explicit:
            self._store.add_access(m.id, self._clock.now(), AccessKind.EXPLICIT)
        stored = self._store.get(m.id)
        p = self._params()
        tier = tier_for(strength(stored, self._clock.now(), p), p)
        if tier != stored.tier:
            stored.tier = tier
            self._store.update(stored)
        return stored

    # ---- existing memories ----

    def _apply_reinforce(self, memory_id: str, turns: List[Turn], project_scope: str) -> None:
        m = self._store.get(memory_id)
        if m is None:
            return
        self._store.add_access(m.id, self._clock.now(), AccessKind.REINFORCED)
        self._log_recalled(EventLabel.HIT, memory_id, turns, project_scope)

    def _apply_supersede(self, s: Supersede, turns: List[Turn], project_scope: str) -> bool:
        old = self._store.get(s.old_id)
        if old is None or old.status != Status.CURRENT:
            print("[Memory/Capture] supersede skipped, unknown or non-current id %s" % s.old_id)
            return False
        new = self._insert(s.kind, old.scope, s.content, self._embed(s.content),
                           min(1.0, max(0.0, float(s.importance))), None)
        new.supersedes_id = old.id
        self._store.update(new)
        old.status = Status.SUPERSEDED
        old.superseded_by_id = new.id
        self._store.update(old)
        self._log_recalled(EventLabel.STALE, old.id, turns, project_scope, old, True)
        return old.kind in _PROFILE_KINDS or new.kind in _PROFILE_KINDS

    def _apply_complete(self, memory_id: str) -> bool:
        m = self._store.get(memory_id)
        if m is None or m.kind != MemoryKind.FEATURE:
            return False
        m.completed_at = self._clock.now()
        self._store.update(m)
        return True

    def _apply_forget(self, memory_id: str, turns: List[Turn], project_scope: str) -> bool:
        m = self._store.get(memory_id)
        if m is None:
            return False
        m.status = Status.FORGOTTEN
        self._store.update(m)
        self._log_recalled(EventLabel.FALSE_RECALL, memory_id, turns, project_scope, m, True)
        return m.kind in _PROFILE_KINDS

    def _apply_missed_op(self, op: ExplicitOp, turns: List[Turn]) -> None:
        context: Dict[str, Any] = {}
        score = self._classifier_score_for(op.turn_id)
        if score is not None:
            context["classifier_score"] = score
        self._log(EventLabel.MISSED_OP, None, op.turn_id, context)
        turn = next((t for t in turns if t.id == op.turn_id), None)
        if turn is None:
            return
        known = self._store.list(Status.CURRENT, [MemoryKind.MEMORY_OP_EXAMPLE], GLOBAL_SCOPE)
        if any(m.content == turn.user_text for m in known):
            return
        self._insert(MemoryKind.MEMORY_OP_EXAMPLE, GLOBAL_SCOPE, turn.user_text,
                     self._embed(turn.user_text), 0.5, op.op.value)

    # ---- events ----

    def _embed(self, text: str) -> np.ndarray:
        return np.asarray(self._embedder.embed_text(text), dtype=np.float32)

    def _log(
        self, label: EventLabel, memory_id: Optional[str], turn_id: Optional[str],
        context: Dict[str, Any],
    ) -> None:
        self._store.add_event(
            MemoryEvent(self._new_id(), self._clock.now(), label, memory_id, turn_id, context)
        )

    def _log_recalled(
        self, label: EventLabel, memory_id: str, turns: List[Turn], project_scope: str,
        memory: Optional[Memory] = None, require_recalled: bool = False,
    ) -> None:
        memory = memory or self._store.get(memory_id)
        if memory is None:
            return
        turn_id, context = self._event_context_for(memory, turns, project_scope)
        if turn_id is None and require_recalled:
            return
        self._log(label, memory.id, turn_id, context)

    def _event_context_for(
        self, memory: Memory, turns: List[Turn], project_scope: str
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        turn = next((t for t in turns if memory.id in t.recalled_ids), None)
        if turn is None:
            return None, {"tier_at_time": memory.tier.value, "scope": memory.scope,
                          "query_scope": project_scope}
        return turn.id, {
            "tier_at_time": memory.tier.value,
            "rank_at_time": turn.recalled_ids.index(memory.id),
            "similarity": cosine(memory.embedding, turn.embedding),
            "scope": memory.scope,
            "query_scope": project_scope,
            "mode": turn.mode,
        }
