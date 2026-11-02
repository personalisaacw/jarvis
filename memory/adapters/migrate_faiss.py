import json
import os
import uuid

import numpy as np

from domain.ports import IEmbeddingEngine
from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.domain.entities import GLOBAL_SCOPE, Memory, MemoryKind, Status, Tier


def migrate_faiss(
    store: SqliteMemoryStore,
    embedder: IEmbeddingEngine,
    metadata_path: str = ".vectorstore/metadata.json",
) -> int:
    """Import the old FAISS router metadata as ROUTE_EXAMPLE memories; return rows imported.

    No-op (0) if the file is absent or the store already holds route examples. Texts are
    re-embedded because the FAISS index is not read.
    """
    if not os.path.exists(metadata_path) or store.list(None, [MemoryKind.ROUTE_EXAMPLE]):
        return 0
    with open(metadata_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        store.add(Memory(
            id=uuid.uuid4().hex, kind=MemoryKind.ROUTE_EXAMPLE, scope=GLOBAL_SCOPE,
            content=r["text"], embedding=np.asarray(embedder.embed_text(r["text"]), dtype=np.float32),
            importance=0.5, status=Status.CURRENT, tier=Tier.ACTIVE,
            created_at=float(r["timestamp"]), label=r["intent"],
        ))
    return len(rows)
