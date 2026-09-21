import os
import json
import time
from typing import List, Tuple
import faiss
import numpy as np
from domain.entities import Intent, Utterance
from domain.ports import IVectorStore

class FaissAdapter(IVectorStore):
    def __init__(self, storage_dir: str = ".vectorstore"):
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, "index.faiss")
        self.metadata_path = os.path.join(storage_dir, "metadata.json")
        self.dim = 384  # MiniLM-L6-v2 dimension
        
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
            
        self._load_or_create()
        
    def _load_or_create(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            # Using IndexFlatIP for Cosine Similarity (vectors must be normalized)
            # or IndexFlatL2 for standard L2 distance.
            # HuggingFaceEncoder provides unnormalized vectors by default for some models,
            # but sentence-transformers usually normalizes them.
            # We'll use L2 distance for safety, as lower distance = more similar.
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata = []
            
    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def search_similar(self, vector: List[float], limit: int = 3) -> List[Tuple[Utterance, float]]:
        if self.index.ntotal == 0:
            return []
            
        # Convert to numpy array of shape (1, dim) and float32
        vec_np = np.array([vector], dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(vec_np, limit)
        
        output = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
                
            meta = self.metadata[idx]
            utterance = Utterance(
                text=meta["text"],
                intent=Intent(meta["intent"]),
                timestamp=meta["timestamp"]
            )
            # FAISS FlatL2 returns squared L2 distance.
            # To simulate a confidence score, we can map distance. 
            # If distance is 0, they are identical.
            # Let's return the distance.
            output.append((utterance, float(dist)))
            
        return output

    def add_utterance(self, utterance: Utterance, vector: List[float]) -> None:
        vec_np = np.array([vector], dtype=np.float32)
        self.index.add(vec_np)
        
        self.metadata.append({
            "text": utterance.text,
            "intent": utterance.intent.value,
            "timestamp": utterance.timestamp
        })
        self._save()

    def distill_utterances(self) -> None:
        # Future deduplication logic
        pass
