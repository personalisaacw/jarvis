from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from memory.domain.entities import (
    AccessKind,
    Memory,
    MemoryEvent,
    MemoryKind,
    Status,
    Turn,
)
from memory.domain.params import Params


class IMemoryStore(ABC):
    # memories
    @abstractmethod
    def add(self, m: Memory) -> None: ...

    @abstractmethod
    def get(self, id: str) -> Optional[Memory]: ...

    @abstractmethod
    def update(self, m: Memory) -> None: ...

    @abstractmethod
    def list(
        self,
        status: Optional[Status] = Status.CURRENT,
        kinds: Optional[List[MemoryKind]] = None,
        scope: Optional[str] = None,
    ) -> List[Memory]: ...

    @abstractmethod
    def nearest(
        self,
        vec: np.ndarray,
        k: int,
        kinds: Optional[List[MemoryKind]] = None,
        scope: Optional[str] = None,
        status: Status = Status.CURRENT,
    ) -> List[Tuple[Memory, float]]: ...

    @abstractmethod
    def max_cosine(
        self,
        vec: np.ndarray,
        kinds: Optional[List[MemoryKind]] = None,
        scope: Optional[str] = None,
    ) -> float: ...

    # accesses
    @abstractmethod
    def add_access(self, memory_id: str, at: float, kind: AccessKind) -> None: ...

    @abstractmethod
    def move_accesses(self, from_id: str, to_id: str) -> None: ...

    # events
    @abstractmethod
    def add_event(self, e: MemoryEvent) -> None: ...

    @abstractmethod
    def events_since(self, at: float) -> List[MemoryEvent]: ...

    # params
    @abstractmethod
    def load_params(self) -> Params: ...

    @abstractmethod
    def save_params(
        self, p: Params, history: List[Tuple[str, float, float, float, float, str]]
    ) -> None: ...

    @abstractmethod
    def params_history(self, limit: int = 10) -> List[dict]: ...

    # meta
    @abstractmethod
    def get_meta(self, key: str) -> Optional[str]: ...

    @abstractmethod
    def set_meta(self, key: str, value: str) -> None: ...

    # turns
    @abstractmethod
    def add_turn(self, t: Turn) -> None: ...

    @abstractmethod
    def pending_turns(self) -> List[Turn]: ...

    @abstractmethod
    def mark_extracted(self, turn_ids: List[str], at: float) -> None: ...

    @abstractmethod
    def recent_turns(self, session_id: str, n: int) -> List[Turn]: ...

    @abstractmethod
    def delete_turns_before(self, at: float, keep_session: str, keep_n: int) -> int: ...

    # maintenance / reporting
    @abstractmethod
    def purge(self, before: float) -> int: ...

    @abstractmethod
    def count_by(self, kind_tier_scope: bool = True) -> List[Tuple[str, str, str, int]]: ...
