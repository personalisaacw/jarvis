from typing import List
from semantic_router.encoders import HuggingFaceEncoder
from domain.ports import IEmbeddingEngine

class HuggingFaceAdapter(IEmbeddingEngine):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = HuggingFaceEncoder(name=model_name)
    
    def embed_text(self, text: str) -> List[float]:
        # encoder returns a list of lists (batch size of 1 here)
        return self.encoder([text])[0]
