from abc import ABC, abstractmethod
from typing import List

from memory.domain.entities import Extraction, Memory, MemoryOp, NewMemory, OpResolution, Turn


class IMemoryExtractor(ABC):
    """LLM-facing port. Methods raise ValueError when the model's output is malformed."""

    @abstractmethod
    def extract(self, turns: List[Turn], recalled: List[Memory]) -> Extraction:
        """Turn a batch of conversation turns into memory operations."""

    @abstractmethod
    def resolve_op(
        self, op: MemoryOp, utterance: str, working_turns: List[Turn], candidates: List[Memory]
    ) -> OpResolution:
        """Resolve one explicit memory operation to target ids and/or a new memory."""

    @abstractmethod
    def summarise(self, episodes: List[Memory]) -> List[NewMemory]:
        """Condense episode memories into longer-lived memories."""

    @abstractmethod
    def is_available(self) -> bool:
        """True when the backing model can be called."""
