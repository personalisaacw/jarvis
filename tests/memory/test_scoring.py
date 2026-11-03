import math
from typing import List, Optional

import numpy as np
import pytest

from memory.domain.entities import (
    GLOBAL_SCOPE,
    Access,
    AccessKind,
    Memory,
    MemoryKind,
    Status,
    Tier,
    project_scope,
)
from memory.domain.scoring import (
    cosine,
    effective_kind,
    recall_score,
    salience,
    scope_weight,
    strength,
    tier_for,
)
from tests.memory.conftest import vec

NOW = 1_000_000.0


def _mem(
    kind: MemoryKind,
    accesses_days_ago: List[float],
    importance: float = 0.5,
    completed: Optional[float] = None,
    scope: str = GLOBAL_SCOPE,
    embedding: Optional[np.ndarray] = None,
) -> Memory:
    return Memory(
        id="m1",
        kind=kind,
        scope=scope,
        content="x",
        embedding=vec("x") if embedding is None else embedding,
        importance=importance,
        status=Status.CURRENT,
        tier=Tier.ACTIVE,
        created_at=NOW - max(accesses_days_ago) * 86400,
        completed_at=completed,
        accesses=[
            Access(memory_id="m1", at=NOW - d * 86400, kind=AccessKind.RECALLED)
            for d in accesses_days_ago
        ],
    )


def test_episode_fades_before_feature(params):
    ep, ft = _mem(MemoryKind.EPISODE, [10]), _mem(MemoryKind.FEATURE, [10])
    assert strength(ep, NOW, params) < strength(ft, NOW, params)


def test_reinforcement_raises_strength(params):
    assert strength(_mem(MemoryKind.FACT, [5, 1]), NOW, params) > strength(
        _mem(MemoryKind.FACT, [5]), NOW, params
    )


def test_tier_boundaries(params):
    assert tier_for(0.8, params) == Tier.ACTIVE
    assert tier_for(0.79, params) == Tier.DORMANT
    assert tier_for(0.3, params) == Tier.DORMANT
    assert tier_for(0.29, params) == Tier.ARCHIVED


def test_future_access_does_not_explode(params):
    m = _mem(MemoryKind.FACT, [-3])
    s = strength(m, NOW, params)
    assert math.isfinite(s) and s == pytest.approx(params.get("importance_weight") * 0.5 + 1.0)


def test_completed_feature_decays_like_fact(params):
    done = _mem(MemoryKind.FEATURE, [30], completed=NOW - 86400)
    assert effective_kind(done) == MemoryKind.FACT
    assert strength(done, NOW, params) == pytest.approx(
        strength(_mem(MemoryKind.FACT, [30]), NOW, params)
    )


def test_open_feature_keeps_feature_kind():
    assert effective_kind(_mem(MemoryKind.FEATURE, [1])) == MemoryKind.FEATURE


def test_scope_weight_order(params):
    q = project_scope("/a")
    assert (
        scope_weight(q, q, params)
        > scope_weight(GLOBAL_SCOPE, q, params)
        > scope_weight(project_scope("/b"), q, params)
    )


def test_cosine_unit_and_zero_vectors():
    a = vec("a")
    assert cosine(a, a) == pytest.approx(1.0, abs=1e-5)
    assert cosine(a, np.zeros(384, dtype=np.float32)) == 0.0


def test_recall_score_is_similarity_times_strength_times_scope(params):
    q = project_scope("/a")
    m = _mem(MemoryKind.FACT, [1], scope=GLOBAL_SCOPE)
    qv = vec("query")
    expected = cosine(m.embedding, qv) * strength(m, NOW, params) * params.get("scope_global")
    assert recall_score(m, qv, q, NOW, params) == pytest.approx(expected)


def test_salience_novel_beats_duplicate_and_scales_with_mode(params):
    assert salience(0.1, "think", 20, False, params) > salience(0.95, "think", 20, False, params)
    assert salience(0.1, "think", 20, False, params) > salience(0.1, "quick", 20, False, params)
    assert salience(0.5, "quick", 5, True, params) >= 1.0
