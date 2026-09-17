from dataclasses import dataclass, field
import time
from typing import Dict, Any, Optional
from .enums import InputModality, DeviceType
from .entities import ReviewSession, DiffHunk

@dataclass
class Event:
    event_id: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class HunkResolvedEvent(Event):
    session_id: str
    hunk: DiffHunk
    decision: str  # "accepted", "rejected", "skipped"
    modality: InputModality
    device: DeviceType

@dataclass
class SessionStateChangedEvent(Event):
    session_id: str
    old_state: str
    new_state: str
    session_snapshot: ReviewSession

@dataclass
class SessionCompletedEvent(Event):
    session_id: str
    accepted_count: int
    rejected_count: int
