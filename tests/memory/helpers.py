import uuid
from typing import Optional, Union

from memory.domain.entities import EventLabel, Memory, MemoryEvent

DEFAULT_AT = 1_000_000.0  # FrozenClock's default time


def event(
    label: EventLabel,
    memory: Optional[Union[Memory, str]] = None,
    at: Optional[float] = None,
    turn_id: Optional[str] = None,
    **context
) -> MemoryEvent:
    """Build a labelled MemoryEvent; `memory` may be a Memory or a memory id."""
    memory_id = memory.id if isinstance(memory, Memory) else memory
    return MemoryEvent(
        id=uuid.uuid4().hex,
        at=DEFAULT_AT if at is None else at,
        label=EventLabel(label),
        memory_id=memory_id,
        turn_id=turn_id,
        context=dict(context),
    )
