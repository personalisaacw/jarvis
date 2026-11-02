import pytest

from domain.entities import Intent
from memory.adapters.sqlite_store import SqliteMemoryStore
from tests.memory.conftest import FakeEmbedder
from use_cases.routing import LearnFromFeedbackUseCase, RouteCommandUseCase

CASES = [
    ("design a scalable architecture for microservices", Intent.THINK),
    ("write a python script to list files", Intent.CODE),
    ("what time is it in Tokyo", Intent.QUICK),
]


@pytest.mark.parametrize("text,intent", CASES)
def test_routes_seeded_sentence(text, intent):
    store, embedder = SqliteMemoryStore(":memory:"), FakeEmbedder()
    for t, i in CASES:
        LearnFromFeedbackUseCase(store, embedder).execute(t, i.value)
    assert RouteCommandUseCase(store, embedder).execute(text).intent == intent
