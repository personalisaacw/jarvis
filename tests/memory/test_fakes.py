import hashlib

import numpy as np

from tests.memory.conftest import FakeEmbedder, vec


def _expected(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")
    v = np.random.RandomState(seed).randn(384).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_fake_embedding_seed_is_pure_function_of_text():
    expected = _expected("hello")
    assert np.array_equal(vec("hello"), expected)
    assert np.array_equal(np.array(FakeEmbedder().embed_text("hello"), dtype=np.float32), expected)


def test_fake_embedding_is_normalised_384d():
    v = vec("anything")
    assert v.shape == (384,) and v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
