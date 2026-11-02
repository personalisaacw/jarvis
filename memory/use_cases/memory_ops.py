import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from memory.domain.entities import (
    GLOBAL_SCOPE,
    EventLabel,
    Memory,
    MemoryEvent,
    MemoryKind,
    MemoryOp,
    Status,
    Tier,
    Turn,
)
from memory.domain.params import Params
from memory.domain.scoring import cosine, strength
from memory.ports.clock import IClock

_EXAMPLE_KINDS = (MemoryKind.ROUTE_EXAMPLE, MemoryKind.MEMORY_OP_EXAMPLE)
_RECALLABLE_KINDS = [k for k in MemoryKind if k not in _EXAMPLE_KINDS]
_OP_ORDER = {op: i for i, op in enumerate(MemoryOp)}
_NO_ASK = (MemoryOp.ASK, MemoryOp.STATUS)
_YES_WORDS = {"yes", "yeah", "yep", "sure", "do it", "please", "correct"}
_QUESTIONS = {
    MemoryOp.STORE: "Should I remember that?",
    MemoryOp.FORGET: "Should I forget that?",
    MemoryOp.REVISE: "Should I update that?",
    MemoryOp.COMPLETE: "Should I mark that as done?",
}
_UNRESOLVED = "I couldn't work out what to change, could you say it another way?"
_NOTHING = "I don't remember anything about that yet."
_ASK_MODE = "think"


@dataclass
class OpDecision:
    op: MemoryOp
    score: float
    action: str  # "execute" | "ask" | "fallthrough"


@dataclass
class PendingAsk:
    op: MemoryOp
    utterance: str
    vec: np.ndarray
    candidates: List[Memory] = field(default_factory=list)
    score: float = 0.0


@dataclass
class OpOutcome:
    spoken: str
    consumed: bool
    pending_question: Optional[PendingAsk]


class ClassifyMemoryOpUseCase:
    """Net 1: nearest memory_op_example prototypes vote by label (spec §7.3)."""

    def __init__(self, store, params_provider: Callable[[], Params], clock: IClock):
        self._store = store
        self._params = params_provider
        self._clock = clock

    def execute(self, vec: np.ndarray, only: Optional[List[MemoryOp]] = None) -> OpDecision:
        p, now = self._params(), self._clock.now()
        k = int(p.get("memory_op_k"))
        if only is None:
            near = self._store.nearest(vec, k, kinds=[MemoryKind.MEMORY_OP_EXAMPLE])
        else:
            allowed = {op.value for op in only}
            pool = [(m, cosine(m.embedding, vec))
                    for m in self._store.list(Status.CURRENT, [MemoryKind.MEMORY_OP_EXAMPLE])
                    if m.label in allowed]
            near = sorted(pool, key=lambda s: s[1], reverse=True)[:k]
        totals: Dict[MemoryOp, float] = {}
        best: Dict[MemoryOp, float] = {}
        for m, sim in near:
            try:
                op = MemoryOp(m.label)
            except ValueError:
                continue
            totals[op] = totals.get(op, 0.0) + max(sim, 0.0) * strength(m, now, p)
            best[op] = max(best.get(op, sim), sim)
        grand = sum(totals.values())
        if grand <= 0.0:
            return OpDecision(MemoryOp.NONE, 0.0, "fallthrough")
        winner = max(totals, key=lambda op: (totals[op], op != MemoryOp.NONE, -_OP_ORDER[op]))
        score = totals[winner] / grand * best[winner]
        return OpDecision(winner, score, self._action(winner, score, p))

    @staticmethod
    def _action(op: MemoryOp, score: float, p: Params) -> str:
        if op == MemoryOp.NONE:
            return "fallthrough"
        if score >= p.get("memory_op_threshold"):
            return "execute"
        if score >= p.get("memory_op_ask_threshold"):
            return "execute" if op in _NO_ASK else "ask"
        return "fallthrough"


class HandleMemoryOpUseCase:
    """Runs Net 1 decisions: confirmation, execution, one-turn undo and prototype learning."""

    def __init__(
        self,
        store,
        extractor,
        recall,
        capture,
        embedder,
        params_provider: Callable[[], Params],
        clock: IClock,
        project_scope_provider: Callable[[], str],
        status_line_provider: Callable[[], str],
    ):
        self._store = store
        self._extractor = extractor
        self._recall = recall
        self._capture = capture
        self._embedder = embedder
        self._params = params_provider
        self._clock = clock
        self._scope = project_scope_provider
        self._status_line = status_line_provider
        self._classifier = ClassifyMemoryOpUseCase(store, params_provider, clock)
        self._pending: Optional[PendingAsk] = None
        # (op, affected ids, utterance, vec, classifier score) of the op run on the last turn
        self._last_executed: Optional[Tuple[MemoryOp, List[str], str, np.ndarray, float]] = None

    # ---- entry points ----

    def handle(self, utterance: str, vec: np.ndarray, working_turns: List[Turn]) -> OpOutcome:
        last, self._last_executed = self._last_executed, None
        if self._pending is not None:
            pending, self._pending = self._pending, None
            if _is_yes(utterance):
                self.learn(pending.utterance, pending.vec, pending.op, True, None, pending.score)
                spoken = self._run(pending.op, pending.utterance, pending.vec, working_turns,
                                   pending.score, pending.candidates)
            else:
                self.learn(pending.utterance, pending.vec, pending.op, False,
                           EventLabel.FALSE_OP, pending.score)
                spoken = "Okay, I'll leave it."
            return OpOutcome(spoken, True, None)
        if last is not None:
            if self._classifier.execute(vec, only=[MemoryOp.UNDO]).action == "execute":
                op, ids, old_utterance, old_vec, score = last
                spoken = self._revert(op, ids)
                self.learn(old_utterance, old_vec, op, False, EventLabel.FALSE_OP, score)
                return OpOutcome(spoken, True, None)
        d = self._classifier.execute(vec)
        if d.op == MemoryOp.UNDO or d.action == "fallthrough":
            return OpOutcome("", False, None)
        if d.action == "ask":
            self._pending = PendingAsk(d.op, utterance, vec, self._candidates(d.op, vec), d.score)
            return OpOutcome(_QUESTIONS[d.op], True, self._pending)
        return OpOutcome(self._run(d.op, utterance, vec, working_turns, d.score), True, None)

    def note_turn_passed(self) -> None:
        self._last_executed = None

    def learn(
        self, utterance: str, vec: np.ndarray, op: MemoryOp, positive: bool,
        label: Optional[EventLabel], score: Optional[float] = None,
    ) -> None:
        proto_label = op.value if positive else MemoryOp.NONE.value
        known = self._store.list(Status.CURRENT, [MemoryKind.MEMORY_OP_EXAMPLE])
        if not any(m.content == utterance and m.label == proto_label for m in known):
            self._store.add(Memory(
                id=str(uuid.uuid4()), kind=MemoryKind.MEMORY_OP_EXAMPLE, scope=GLOBAL_SCOPE,
                content=utterance, embedding=np.asarray(vec, dtype=np.float32), importance=0.5,
                status=Status.CURRENT, tier=Tier.ACTIVE, created_at=self._clock.now(),
                label=proto_label,
            ))
        if label is not None:
            self._log(label, None, {} if score is None else {"classifier_score": score})

    # ---- operations ----

    def _run(
        self, op: MemoryOp, utterance: str, vec: np.ndarray, working_turns: List[Turn],
        score: float = 0.0, candidates: Optional[List[Memory]] = None,
    ) -> str:
        if op == MemoryOp.ASK:
            return self._run_ask(vec)
        if op == MemoryOp.STATUS:
            return self._status_line()
        if op == MemoryOp.STORE:
            spoken, ids = self._run_store(utterance, vec, working_turns)
        else:
            if candidates is None:
                candidates = self._candidates(op, vec)
            try:
                res = self._extractor.resolve_op(op, utterance, working_turns, candidates)
            except Exception as e:
                print("[Memory/Ops] resolve_op failed: %s" % e)
                return _UNRESOLVED
            targets = [m for m in (self._store.get(i) for i in res.target_ids)
                       if m is not None and m.status == Status.CURRENT]
            run = {MemoryOp.FORGET: self._run_forget, MemoryOp.REVISE: self._run_revise,
                   MemoryOp.COMPLETE: self._run_complete}[op]
            spoken, ids = run(targets, res, vec) if targets else (_UNRESOLVED, [])
        if ids:
            self._last_executed = (op, ids, utterance, vec, score)
            self._recall.invalidate_profile()
        return spoken

    def _run_store(self, utterance: str, vec: np.ndarray,
                   working_turns: List[Turn]) -> Tuple[str, List[str]]:
        try:
            res = self._extractor.resolve_op(MemoryOp.STORE, utterance, working_turns, [])
        except Exception as e:
            print("[Memory/Ops] resolve_op failed: %s" % e)
            return _UNRESOLVED, []
        if res.new is None:
            return _UNRESOLVED, []
        now = self._clock.now()
        m = self._capture.apply_new(res.new, self._scope(), explicit=True, turn_id=None,
                                    mode=_ASK_MODE)
        if m.created_at < now and m.tier != Tier.ACTIVE:  # capture logged a dedup miss
            self._log_late_extraction(vec)
        return "Noted.", [m.id]

    def _run_forget(self, targets: List[Memory], res, vec: np.ndarray) -> Tuple[str, List[str]]:
        recalled = self._recall.last_recalled_ids()
        for m in targets:
            m.status = Status.FORGOTTEN
            self._store.update(m)
            if m.id in recalled:
                self._log(EventLabel.FALSE_RECALL, m.id, self._context(m, vec, recalled))
        return "Forgotten: " + "; ".join(m.content for m in targets), [m.id for m in targets]

    def _run_revise(self, targets: List[Memory], res, vec: np.ndarray) -> Tuple[str, List[str]]:
        if res.new is None:
            return _UNRESOLVED, []
        old = targets[0]
        recalled = self._recall.last_recalled_ids()
        new = self._capture.apply_new(res.new, self._scope(), explicit=True, turn_id=None,
                                      mode=_ASK_MODE)
        if new.id == old.id:
            return "Updated: " + new.content, []
        new.supersedes_id = old.id
        self._store.update(new)
        old.status = Status.SUPERSEDED
        old.superseded_by_id = new.id
        self._store.update(old)
        if old.id in recalled:
            self._log(EventLabel.STALE, old.id, self._context(old, vec, recalled))
        return "Updated: " + new.content, [old.id, new.id]

    def _run_complete(self, targets: List[Memory], res, vec: np.ndarray) -> Tuple[str, List[str]]:
        features = [m for m in targets if m.kind == MemoryKind.FEATURE]
        if not features:
            return _UNRESOLVED, []
        for m in features:
            m.completed_at = self._clock.now()
            self._store.update(m)
        return "Marked done: " + "; ".join(m.content for m in features), [m.id for m in features]

    def _run_ask(self, vec: np.ndarray) -> str:
        p, scope = self._params(), self._scope()
        found = self._recall.execute(vec, scope, _ASK_MODE, include_dormant=True).memories
        floor = p.get("recall_similarity_floor")
        best = found[0] if found else None
        sim = cosine(vec, best.embedding) if best is not None else 0.0
        if best is None or sim < floor or best.tier != Tier.ACTIVE:
            context: Dict[str, Any] = {"similarity": sim, "rank_at_time": 0, "mode": _ASK_MODE,
                                       "query_scope": scope}
            if best is not None:
                context.update(tier_at_time=best.tier.value, scope=best.scope)
            self._log(EventLabel.MISS, best.id if best is not None else None, context)
            self._log_late_extraction(vec)
        spoken = [m.content for m in found[:3] if cosine(vec, m.embedding) >= floor]
        return " ".join(spoken) if spoken else _NOTHING

    # ---- undo ----

    def _revert(self, op: MemoryOp, ids: List[str]) -> str:
        mems = [m for m in (self._store.get(i) for i in ids) if m is not None]
        if op == MemoryOp.FORGET:
            for m in mems:
                m.status = Status.CURRENT
            spoken = "Okay, I've kept: "
        elif op == MemoryOp.STORE:
            for m in mems:
                m.status = Status.FORGOTTEN
            spoken = "Okay, I won't keep: "
        elif op == MemoryOp.REVISE:
            old, new = mems[0], mems[1]
            old.status, old.superseded_by_id = Status.CURRENT, None
            new.status, new.supersedes_id = Status.FORGOTTEN, None
            mems = [old, new]
            spoken = "Okay, back to: "
        else:
            for m in mems:
                m.completed_at = None
            spoken = "Okay, not done yet: "
        for m in mems:
            self._store.update(m)
        self._recall.invalidate_profile()
        shown = mems[:1] if op == MemoryOp.REVISE else mems
        return spoken + "; ".join(m.content for m in shown)

    # ---- helpers ----

    def _candidates(self, op: MemoryOp, vec: np.ndarray) -> List[Memory]:
        if op not in (MemoryOp.FORGET, MemoryOp.REVISE, MemoryOp.COMPLETE):
            return []
        ids = list(self._recall.last_recalled_ids())
        ids += [m.id for m in self._recall.working_set.members()]
        ids += [m.id for m, _ in self._store.nearest(vec, 5, kinds=_RECALLABLE_KINDS)]
        out: List[Memory] = []
        for i in dict.fromkeys(ids):
            m = self._store.get(i)
            if m is not None and m.status == Status.CURRENT:
                out.append(m)
        return out

    def _context(self, m: Memory, vec: np.ndarray, recalled: List[str]) -> Dict[str, Any]:
        return {"tier_at_time": m.tier.value, "rank_at_time": recalled.index(m.id),
                "similarity": cosine(m.embedding, vec), "scope": m.scope,
                "query_scope": self._scope(), "mode": _ASK_MODE}

    def _log_late_extraction(self, vec: np.ndarray) -> None:
        pending = self._store.pending_turns()
        floor = self._params().get("recall_similarity_floor")
        if any(cosine(t.embedding, vec) >= floor for t in pending):
            self._log(EventLabel.LATE_EXTRACTION, None,
                      {"buffer_salience": sum(t.salience for t in pending)})

    def _log(self, label: EventLabel, memory_id: Optional[str], context: Dict[str, Any]) -> None:
        self._store.add_event(MemoryEvent(
            str(uuid.uuid4()), self._clock.now(), label, memory_id, None, context))


def _is_yes(utterance: str) -> bool:
    words = re.sub(r"[^a-z' ]", " ", utterance.lower()).split()
    text = " ".join(words)
    return any(text == w or text.startswith(w + " ") for w in _YES_WORDS)
