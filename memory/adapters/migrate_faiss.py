import json
import os
import uuid

import numpy as np

from domain.entities import Intent
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
    re-embedded because the FAISS index is not read. All or nothing: every row is validated
    before any insert and the inserts share one transaction; on any failure the store and
    the .vectorstore files are left untouched and 0 is returned.
    """
    if not os.path.exists(metadata_path) or store.list(None, [MemoryKind.ROUTE_EXAMPLE]):
        return 0
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError("expected a JSON list, got %s" % type(rows).__name__)
        parsed = []
        for i, r in enumerate(rows):
            text = r["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("row %d has empty text" % i)
            parsed.append((text, Intent(r["intent"]).value, float(r["timestamp"])))
        memories = [
            Memory(
                id=uuid.uuid4().hex, kind=MemoryKind.ROUTE_EXAMPLE, scope=GLOBAL_SCOPE,
                content=text, embedding=np.asarray(embedder.embed_text(text), dtype=np.float32),
                importance=0.5, status=Status.CURRENT, tier=Tier.ACTIVE, created_at=at,
                label=intent,
            )
            for text, intent, at in parsed
        ]
        store.add_many(memories)
    except Exception as e:
        print("[Memory] WARNING: FAISS migration failed (%s); leaving .vectorstore untouched" % e)
        return 0
    return len(memories)
