import numpy as np

from memory.domain.entities import GLOBAL_SCOPE, Memory, MemoryKind, Tier
from memory.domain.params import Params


def effective_kind(m: Memory) -> MemoryKind:
    """A completed feature decays like a fact."""
    if m.kind == MemoryKind.FEATURE and m.completed_at:
        return MemoryKind.FACT
    return m.kind


def strength(m: Memory, now: float, p: Params) -> float:
    rate = p.decay(effective_kind(m))
    total = p.get("importance_weight") * m.importance
    for a in m.accesses:
        base = max(0.0, (now - a.at) / 86400.0) + 1.0
        total += base ** (-rate)
    return total


def tier_for(s: float, p: Params) -> Tier:
    if s >= p.get("active_threshold"):
        return Tier.ACTIVE
    if s >= p.get("dormant_threshold"):
        return Tier.DORMANT
    return Tier.ARCHIVED


def scope_weight(memory_scope: str, query_scope: str, p: Params) -> float:
    if memory_scope == query_scope:
        return 1.0
    if memory_scope == GLOBAL_SCOPE:
        return p.get("scope_global")
    return p.get("scope_other")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of unit vectors; 0.0 if either is a zero vector."""
    if not np.any(a) or not np.any(b):
        return 0.0
    return float(np.dot(a, b))


def recall_score(
    m: Memory, query_vec: np.ndarray, query_scope: str, now: float, p: Params
) -> float:
    return (
        cosine(m.embedding, query_vec)
        * strength(m, now, p)
        * scope_weight(m.scope, query_scope, p)
    )


def salience(max_cosine: float, mode: str, word_count: int, op_fired: bool, p: Params) -> float:
    return (
        (1.0 - max_cosine) * p.mode_prior(mode)
        + p.get("salience_length_bonus") * min(1.0, word_count / 40.0)
        + (1.0 if op_fired else 0.0)
    )
