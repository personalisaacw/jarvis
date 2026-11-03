import hashlib
from typing import List, Optional, Tuple

import numpy as np
import pytest

from memory.domain.entities import Extraction, NewMemory, OpResolution
from memory.domain.params import Params
from memory.ports.clock import IClock
from memory.ports.extractor import IMemoryExtractor


def vec(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeEmbedder:
    def embed_text(self, text: str) -> List[float]:
        return vec(text).tolist()


class FrozenClock(IClock):
    def __init__(self, t: float = 1_000_000.0):
        self._t = t

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds

    def set(self, t: float) -> None:
        self._t = t


class FakeExtractor(IMemoryExtractor):
    def __init__(self, responses: Optional[List[Extraction]] = None):
        self.responses = list(responses) if responses else [Extraction()]
        self.resolution: Optional[OpResolution] = None
        self.summaries: List[NewMemory] = []
        self.raise_next: int = 0
        self.calls: List[Tuple[str, tuple]] = []

    def extract(self, turns, recalled) -> Extraction:
        self.calls.append(("extract", (turns, recalled)))
        if self.raise_next > 0:
            self.raise_next -= 1
            raise ValueError("fake failure")
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    def resolve_op(self, op, utterance, working_turns, candidates) -> OpResolution:
        self.calls.append(("resolve_op", (op, utterance, working_turns, candidates)))
        if self.resolution is not None:
            return self.resolution
        return OpResolution(target_ids=[candidates[0].id] if candidates else [], new=None)

    def summarise(self, episodes) -> List[NewMemory]:
        self.calls.append(("summarise", (episodes,)))
        return self.summaries

    def is_available(self) -> bool:
        return True


@pytest.fixture
def clock():
    return FrozenClock()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def params():
    return Params.defaults()
