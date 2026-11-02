from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from memory.domain.entities import (
    GLOBAL_SCOPE,
    AccessKind,
    Memory,
    MemoryKind,
    Status,
    Tier,
    Turn,
)
from memory.domain.params import Params
from memory.domain.scoring import cosine, recall_score, strength
from memory.ports.clock import IClock

_EXAMPLE_KINDS = (MemoryKind.ROUTE_EXAMPLE, MemoryKind.MEMORY_OP_EXAMPLE)
_RECALLABLE_KINDS = [k for k in MemoryKind if k not in _EXAMPLE_KINDS]

# (memory, recall score, cosine to the query)
_Scored = Tuple[Memory, float, float]


@dataclass
class Profile:
    preferences: List[Memory]
    features: List[Memory]
    version: int


@dataclass
class RecallResult:
    memories: List[Memory]
    newly_added: List[Memory]
    working_turns: List[Turn]
    profile: Profile


class SessionWorkingSet:
    """Memories shown this session, in insertion order, evicted after repeated misses."""

    def __init__(self) -> None:
        self._members: List[Memory] = []
        self._misses: Dict[str, int] = {}

    def members(self) -> List[Memory]:
        return list(self._members)

    def merge(self, candidates: List[Memory], patience: int) -> List[Memory]:
        present = {m.id for m in candidates}
        known = {m.id for m in self._members}
        kept: List[Memory] = []
        for m in self._members:
            if m.id in present:
                self._misses[m.id] = 0
            else:
                self._misses[m.id] = self._misses.get(m.id, 0) + 1
            if self._misses[m.id] >= patience:
                del self._misses[m.id]
            else:
                kept.append(m)
        added: List[Memory] = []
        for m in candidates:
            if m.id not in known and m.id not in {k.id for k in added}:
                added.append(m)
                self._misses[m.id] = 0
        self._members = kept + added
        return added

    def reset(self) -> None:
        self._members = []
        self._misses = {}


class RecallUseCase:
    def __init__(
        self,
        store,
        params_provider: Callable[[], Params],
        clock: IClock,
        session_id: str,
    ):
        self._store = store
        self._params = params_provider
        self._clock = clock
        self._session_id = session_id
        self._profiles: Dict[str, Profile] = {}
        self._version = 0
        self._last_ids: List[str] = []
        self.working_set = SessionWorkingSet()

    # ---- profile ----

    def profile(self, project_scope: str) -> Profile:
        cached = self._profiles.get(project_scope)
        if cached is not None:
            return cached
        p, now = self._params(), self._clock.now()

        def by_strength(ms: List[Memory]) -> List[Memory]:
            return sorted(ms, key=lambda m: strength(m, now, p), reverse=True)

        prefs = by_strength(
            self._store.list(Status.CURRENT, [MemoryKind.PREFERENCE], GLOBAL_SCOPE)
        )[: int(p.get("profile_preferences"))]
        feats = by_strength(
            [
                m
                for m in self._store.list(Status.CURRENT, [MemoryKind.FEATURE], project_scope)
                if m.completed_at is None
            ]
        )
        prof = Profile(prefs, feats, self._version)
        self._profiles[project_scope] = prof
        return prof

    def invalidate_profile(self) -> None:
        self._profiles = {}
        self._version += 1

    def last_recalled_ids(self) -> List[str]:
        return list(self._last_ids)

    # ---- recall ----

    def execute(
        self,
        query_vec: np.ndarray,
        project_scope: str,
        mode: str,
        include_dormant: bool = False,
    ) -> RecallResult:
        p, now = self._params(), self._clock.now()
        profile = self.profile(project_scope)
        turns = self._store.recent_turns(self._session_id, int(p.get("working_turns")))
        scored = self._score_all(query_vec, project_scope, now, p)

        if include_dormant:
            pool = [s for s in scored if s[0].tier != Tier.ARCHIVED]
            top = sorted(pool, key=lambda s: s[1], reverse=True)[: int(p.get("recall_k"))]
            return self._result([s[0] for s in top], [], turns, profile)

        in_profile = {m.id for m in profile.preferences + profile.features}
        candidates = self._candidates([s for s in scored if s[0].id not in in_profile], p)
        chosen = self._gate_and_budget(candidates, scored, in_profile, project_scope, mode, p, now)
        if chosen is None:
            return self._result([], [], turns, profile)

        added = self.working_set.merge(chosen, int(p.get("working_set_patience")))
        for m in added:
            self._store.add_access(m.id, now, AccessKind.RECALLED)
        return self._result(self.working_set.members(), added, turns, profile)

    def _result(
        self, memories: List[Memory], added: List[Memory], turns: List[Turn], profile: Profile
    ) -> RecallResult:
        self._last_ids = [m.id for m in memories]
        return RecallResult(memories, added, turns, profile)

    def _score_all(
        self, query_vec: np.ndarray, project_scope: str, now: float, p: Params
    ) -> List[_Scored]:
        mems = self._store.list(Status.CURRENT, _RECALLABLE_KINDS)
        return [
            (m, recall_score(m, query_vec, project_scope, now, p), cosine(m.embedding, query_vec))
            for m in mems
        ]

    @staticmethod
    def _candidates(scored: List[_Scored], p: Params) -> List[_Scored]:
        """Top recall_k active plus every dormant memory similar enough; archived never."""
        ranked = sorted(scored, key=lambda s: s[1], reverse=True)
        active = [s for s in ranked if s[0].tier == Tier.ACTIVE][: int(p.get("recall_k"))]
        bar = p.get("dormant_recall_similarity")
        dormant = [s for s in ranked if s[0].tier == Tier.DORMANT and s[2] >= bar]
        return sorted(active + dormant, key=lambda s: s[1], reverse=True)

    def _gate_and_budget(
        self,
        candidates: List[_Scored],
        scored: List[_Scored],
        in_profile: set,
        project_scope: str,
        mode: str,
        p: Params,
        now: float,
    ) -> Optional[List[Memory]]:
        """The memories to merge this turn, or None when the gate blocks everything."""
        passes = any(s[2] >= p.get("recall_similarity_floor") for s in candidates)
        k = int(p.get("recall_k"))
        if mode == "code":
            fixed = sorted(
                (
                    s[0]
                    for s in scored
                    if s[0].scope == project_scope
                    and s[0].kind in (MemoryKind.FEATURE, MemoryKind.DECISION)
                    and s[0].id not in in_profile
                ),
                key=lambda m: strength(m, now, p),
                reverse=True,
            )
            fixed_ids = {m.id for m in fixed}
            others = [s[0] for s in candidates if s[0].id not in fixed_ids] if passes else []
            chosen = fixed + others[:k]
            return chosen or None
        if not passes:
            return None
        budget = int(p.get("budget_quick")) if mode == "quick" else k
        return [s[0] for s in candidates[:budget]]
