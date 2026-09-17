import time
from typing import List, Tuple
from domain.entities import Intent, Utterance, RoutingResult
from domain.ports import IVectorStore, IEmbeddingEngine

class RouteCommandUseCase:
    def __init__(self, vector_store: IVectorStore, embedding_engine: IEmbeddingEngine, fallback_threshold: float = 0.30):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.fallback_threshold = fallback_threshold

    def execute(self, text: str) -> RoutingResult:
        # 1. Embed text
        vector = self.embedding_engine.embed_text(text)
        
        # 2. Query nearest matches
        results = self.vector_store.search_similar(vector, limit=3)
        
        if not results:
            # Empty database, fallback to QUICK
            return RoutingResult(
                intent=Intent.QUICK,
                confidence_score=0.0,
                requires_clarification=False
            )
            
        # 3. Best match
        best_match_utterance, best_distance = results[0]
        
        # FAISS FlatL2 returns squared L2 distance.
        # For normalized vectors, Cosine Similarity = 1 - (L2_sq / 2)
        confidence_score = 1.0 - (best_distance / 2.0)
        
        # Clamp to [0, 1] just in case of float imprecision
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        # 4. Check confidence
        if confidence_score < self.fallback_threshold:
            return RoutingResult(
                intent=best_match_utterance.intent,
                confidence_score=confidence_score,
                requires_clarification=True
            )
            
        return RoutingResult(
            intent=best_match_utterance.intent,
            confidence_score=confidence_score,
            requires_clarification=False
        )


class LearnFromFeedbackUseCase:
    def __init__(self, vector_store: IVectorStore, embedding_engine: IEmbeddingEngine):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine

    def execute(self, text: str, intent_str: str) -> None:
        try:
            intent = Intent(intent_str)
        except ValueError:
            # Invalid intent, ignore or log
            print(f"[LearnFromFeedback] Invalid intent: {intent_str}")
            return
            
        vector = self.embedding_engine.embed_text(text)
        utterance = Utterance(
            text=text,
            intent=intent,
            timestamp=time.time()
        )
        self.vector_store.add_utterance(utterance, vector)
        print(f"[LearnFromFeedback] Learned new mapping: '{text}' -> {intent.value}")
