import uuid
from typing import Optional, Union

from memory.domain.entities import EventLabel, Memory, MemoryEvent

DEFAULT_AT = 1_000_000.0  # FrozenClock's default time


def event(
    label: EventLabel,
    memory: Optional[Union[Memory, str]] = None,
    at: Optional[float] = None,
    turn_id: Optional[str] = None,
    decided_at: Optional[float] = None,
    **context
) -> MemoryEvent:
    """Build a labelled MemoryEvent; `memory` may be a Memory or a memory id.

    `decided_at` (the recall/dedup moment) goes into the context as capture writes it.
    """
    memory_id = memory.id if isinstance(memory, Memory) else memory
    if decided_at is not None:
        context["decided_at"] = decided_at
    return MemoryEvent(
        id=uuid.uuid4().hex,
        at=DEFAULT_AT if at is None else at,
        label=EventLabel(label),
        memory_id=memory_id,
        turn_id=turn_id,
        context=dict(context),
    )
