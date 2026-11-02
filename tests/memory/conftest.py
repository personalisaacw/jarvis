from typing import List

import numpy as np
import pytest

from memory.domain.entities import Extraction
from memory.domain.params import Params
from memory.ports.clock import IClock


def vec(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2 ** 32)
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


class FakeExtractor:
    def extract(self, *args, **kwargs) -> Extraction:
        return Extraction.from_dict({})


@pytest.fixture
def clock():
    return FrozenClock()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def params():
    return Params.defaults()
