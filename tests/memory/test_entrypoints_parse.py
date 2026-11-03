"""The entry-point scripts load models and servers at import time, so they are only parsed here."""
import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENTRYPOINTS = ["jarvis.py", "ui_server.py", "router.py", "seed_db.py"]


def _source(name: str) -> str:
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("name", ENTRYPOINTS)
def test_entrypoint_parses(name):
    ast.parse(_source(name), filename=name)


@pytest.mark.parametrize("name", ENTRYPOINTS)
def test_entrypoint_does_not_use_the_deleted_faiss_adapter(name):
    assert "vector_store import FaissAdapter" not in _source(name)
