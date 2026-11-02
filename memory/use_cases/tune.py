"""Counterfactual replay tuner (spec section 9.3): deterministic, no LLM."""
import dataclasses
from typing import Callable, Dict, List, Optional, Tuple

from memory.domain.entities import EventLabel, Memory, MemoryEvent, Tier
from memory.domain.params import FIXED, PARAM_SPECS, Params
from memory.domain.scoring import effective_kind, strength, tier_for
from memory.ports.clock import IClock
from memory.ports.store import IMemoryStore

_DAY = 86400.0

# Params each label's cost reads (keep in step with the _cost_* methods); "decay.*" = any decay.
_STRENGTH_READS = ("decay.*", "importance_weight", "active_threshold")
_RECALL_READS = _STRENGTH_READS + ("recall_similarity_floor", "budget_quick", "recall_k")
_LABEL_READS = {
    EventLabel.HIT: _RECALL_READS,
    EventLabel.FALSE_RECALL: _RECALL_READS,
    EventLabel.MISS: _RECALL_READS,
    EventLabel.STALE: _STRENGTH_READS,
    EventLabel.EMPTY_EXTRACTION: ("extract_salience_threshold",),
    EventLabel.LATE_EXTRACTION: ("extract_salience_threshold",),
    EventLabel.FALSE_OP: ("memory_op_threshold",),
    EventLabel.MISSED_OP: ("memory_op_ask_threshold",),
}


def _reads(label: EventLabel, name: str) -> bool:
    reads = _LABEL_READS[label]
    return name in reads or (name.startswith("decay.") and "decay.*" in reads)


class TuneParamsUseCase:
    def __init__(self, store: IMemoryStore, clock: IClock):
        self._store = store
        self._clock = clock
        self._reset_caches()
        self._costs: Dict[EventLabel, Callable[[MemoryEvent, Params, float], float]] = {
            EventLabel.HIT: self._cost_hit,
            EventLabel.FALSE_RECALL: self._cost_false_recall,
            EventLabel.MISS: self._cost_miss,
            EventLabel.STALE: self._cost_stale,
            EventLabel.EMPTY_EXTRACTION: self._cost_empty_extraction,
            EventLabel.LATE_EXTRACTION: self._cost_late_extraction,
            EventLabel.FALSE_OP: self._cost_false_op,
            EventLabel.MISSED_OP: self._cost_missed_op,
        }

    # ---- public ----

    def loss(self, p: Params, events: List[MemoryEvent], now: float) -> float:
        return sum(self._costs[EventLabel(e.label)](e, p, min(e.at, now)) for e in events)

    def run(self) -> dict:
        now = self._clock.now()
        self._reset_caches()
        events = self._store.events_since(now - FIXED["tune_window_days"] * _DAY)
        n = len(events)
        if n < FIXED["min_events"]:
            return {"status": "insufficient_data", "events": n, "moves": []}

        by_label: Dict[EventLabel, List[MemoryEvent]] = {}
        for e in events:
            by_label.setdefault(EventLabel(e.label), []).append(e)
        current = self._store.load_params()
        label_loss = {lb: self.loss(current, evs, now) for lb, evs in by_label.items()}
        base = sum(label_loss.values())
        moves: List[dict] = []
        while len(moves) < FIXED["max_moves_per_pass"] and base > 0:
            best = self._best_move(current, by_label, label_loss, now, base)
            if best is None:
                break
            name, candidate, label_loss = best
            new_loss = sum(label_loss.values())
            moves.append({
                "name": name, "old": current.get(name), "new": candidate.get(name),
                "old_loss": base, "new_loss": new_loss,
            })
            current, base = candidate, new_loss

        if not moves:
            return {"status": "no_improvement", "events": n, "moves": []}
        history = [
            (m["name"], m["old"], m["new"], m["old_loss"], m["new_loss"], "replay") for m in moves
        ]
        self._store.save_params(current, history)
        return {"status": "tuned", "events": n, "moves": moves}

    # ---- search ----

    def _best_move(
        self,
        current: Params,
        by_label: Dict[EventLabel, List[MemoryEvent]],
        label_loss: Dict[EventLabel, float],
        now: float,
        base: float,
    ) -> Optional[Tuple[str, Params, Dict[EventLabel, float]]]:
        """Best single-step move beating the margin; re-scores only labels the param affects."""
        best: Optional[Tuple[str, Params, Dict[EventLabel, float]]] = None
        best_loss = base * (1.0 - FIXED["improvement_margin"])
        for name, spec in PARAM_SPECS.items():
            affected = [lb for lb in by_label if _reads(lb, name)]
            if not affected:
                continue
            value = current.get(name)
            for target in (value - spec.step, value + spec.step):
                # rounding keeps repeated steps on the grid (no 0.9400000000000002 drift)
                candidate = current.with_value(name, round(target, 9))
                if candidate.get(name) == value or not candidate.is_valid():
                    continue
                trial = dict(label_loss)
                for lb in affected:
                    trial[lb] = self.loss(candidate, by_label[lb], now)
                cand_loss = sum(trial.values())
                if cand_loss < best_loss:
                    best, best_loss = (name, candidate, trial), cand_loss
        return best

    # ---- counterfactual predicates ----

    def _reset_caches(self) -> None:
        self._memories: Dict[str, Optional[Memory]] = {}
        self._as_of: Dict[Tuple[str, float], Optional[Memory]] = {}
        self._strengths: Dict[Tuple[str, float, float, float], float] = {}

    def _memory_at(self, memory_id: Optional[str], t: float) -> Optional[Memory]:
        """The memory as it was at time t: only accesses that had happened by then."""
        if memory_id is None:
            return None
        key = (memory_id, t)
        if key not in self._as_of:
            if memory_id not in self._memories:
                self._memories[memory_id] = self._store.get(memory_id)
            m = self._memories[memory_id]
            self._as_of[key] = None if m is None else dataclasses.replace(
                m, accesses=[a for a in m.accesses if a.at <= t]
            )
        return self._as_of[key]

    def _tier_at(self, e: MemoryEvent, p: Params, t: float) -> Optional[Tier]:
        m = self._memory_at(e.memory_id, t)
        if m is None:
            return None
        # strength depends on p only through these two values; most candidates share them
        key = (m.id, t, p.decay(effective_kind(m)), p.get("importance_weight"))
        if key not in self._strengths:
            self._strengths[key] = strength(m, t, p)
        return tier_for(self._strengths[key], p)

    def _recalled(self, e: MemoryEvent, p: Params, t: float) -> Optional[bool]:
        """Would the memory still be recalled under p? None when not applicable."""
        ctx = e.context
        rank = ctx.get("budget_rank", ctx.get("rank_at_time"))
        similarity = ctx.get("similarity")
        tier = self._tier_at(e, p, t)
        if rank is None or similarity is None or tier is None:
            return None
        budget = p.get("budget_quick") if ctx.get("mode") == "quick" else p.get("recall_k")
        return (
            tier == Tier.ACTIVE
            and similarity >= p.get("recall_similarity_floor")
            and rank < budget
        )

    # ---- cost per label (spec section 9.3 table) ----

    def _cost_hit(self, e: MemoryEvent, p: Params, t: float) -> float:
        return 1.0 if self._recalled(e, p, t) is False else 0.0

    def _cost_false_recall(self, e: MemoryEvent, p: Params, t: float) -> float:
        return FIXED["w_false_recall"] if self._recalled(e, p, t) else 0.0

    def _cost_miss(self, e: MemoryEvent, p: Params, t: float) -> float:
        return 1.0 if self._recalled(e, p, t) is False else 0.0

    def _cost_stale(self, e: MemoryEvent, p: Params, t: float) -> float:
        return FIXED["w_stale"] if self._tier_at(e, p, t) == Tier.ACTIVE else 0.0

    def _cost_empty_extraction(self, e: MemoryEvent, p: Params, t: float) -> float:
        s = e.context.get("buffer_salience")
        flushed = s is not None and s >= p.get("extract_salience_threshold")
        return FIXED["w_empty_extraction"] if flushed else 0.0

    def _cost_late_extraction(self, e: MemoryEvent, p: Params, t: float) -> float:
        s = e.context.get("buffer_salience")
        held = s is not None and s < p.get("extract_salience_threshold")
        return FIXED["w_late_extraction"] if held else 0.0

    def _cost_false_op(self, e: MemoryEvent, p: Params, t: float) -> float:
        s = e.context.get("classifier_score")
        fired = s is not None and s >= p.get("memory_op_threshold")
        return FIXED["w_false_op"] if fired else 0.0

    def _cost_missed_op(self, e: MemoryEvent, p: Params, t: float) -> float:
        s = e.context.get("classifier_score")
        shy = s is not None and s < p.get("memory_op_ask_threshold")
        return FIXED["w_missed_op"] if shy else 0.0
