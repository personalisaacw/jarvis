from abc import ABC, abstractmethod
from typing import List, Tuple
from domain.entities import Intent, Utterance

class IVectorStore(ABC):
    @abstractmethod
    def search_similar(self, vector: List[float], limit: int = 3) -> List[Tuple[Utterance, float]]:
        pass
    
    @abstractmethod
    def add_utterance(self, utterance: Utterance, vector: List[float]) -> None:
        pass
    
    @abstractmethod
    def distill_utterances(self) -> None:
        pass

class IEmbeddingEngine(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass
