from typing import Callable, List, Optional

from memory.domain.entities import Turn
from memory.domain.params import Params
from memory.domain.scoring import salience
from memory.ports.clock import IClock
from memory.ports.store import IMemoryStore


class ExtractionBuffer:
    """Accumulates turn salience and decides when to batch turns into one extraction.

    Single-threaded: the caller owns timers and calls tick() periodically.
    """

    def __init__(
        self,
        store: IMemoryStore,
        params_provider: Callable[[], Params],
        clock: IClock,
        on_flush: Callable[[List[Turn], str], None],
    ):
        self._store = store
        self._params = params_provider
        self._clock = clock
        self._on_flush = on_flush
        self._flushing = False
        self.total = 0.0
        self.last_turn_at = None  # type: Optional[float]

    def add_turn(self, turn: Turn, max_cosine: float, op_fired: bool) -> float:
        p = self._params()
        turn.salience = salience(max_cosine, turn.mode, len(turn.user_text.split()), op_fired, p)
        self._store.add_turn(turn)
        self.total += turn.salience
        self.last_turn_at = self._clock.now()
        if self.total >= p.get("extract_salience_threshold"):
            self.flush("salience")
        elif self.pending_count() >= p.get("extract_max_turns"):
            self.flush("max_turns")
        return turn.salience

    def tick(self) -> None:
        if self.last_turn_at is None or self.pending_count() == 0:
            return
        if self._clock.now() - self.last_turn_at >= self._params().get("extract_idle_seconds"):
            self.flush("idle")

    def flush(self, reason: str) -> None:
        """Hand all pending turns to on_flush and reset the salience total.

        Turns are not marked extracted here (the capture use case does that after a
        successful extraction), so a flush while another is in progress is skipped
        to avoid sending the same turns twice.
        """
        if self._flushing:
            return
        pending = self._store.pending_turns()
        if not pending:
            return
        self._flushing = True
        try:
            self._on_flush(pending, reason)
            self.total = 0.0
        finally:
            self._flushing = False

    def pending_count(self) -> int:
        return len(self._store.pending_turns())
