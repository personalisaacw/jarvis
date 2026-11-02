import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

GLOBAL_SCOPE = "global"


def project_scope(path: str) -> str:
    return "project:" + os.path.abspath(path)


class MemoryKind(str, Enum):
    EPISODE = "episode"
    FACT = "fact"
    DECISION = "decision"
    FEATURE = "feature"
    PREFERENCE = "preference"
    ROUTE_EXAMPLE = "route_example"
    MEMORY_OP_EXAMPLE = "memory_op_example"


class Status(str, Enum):
    CURRENT = "current"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"


class Tier(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    ARCHIVED = "archived"


class AccessKind(str, Enum):
    CREATED = "created"
    RECALLED = "recalled"
    REINFORCED = "reinforced"
    EXPLICIT = "explicit"


class EventLabel(str, Enum):
    HIT = "hit"
    FALSE_RECALL = "false_recall"
    MISS = "miss"
    STALE = "stale"
    EMPTY_EXTRACTION = "empty_extraction"
    LATE_EXTRACTION = "late_extraction"
    FALSE_OP = "false_op"
    MISSED_OP = "missed_op"


class MemoryOp(str, Enum):
    STORE = "store"
    FORGET = "forget"
    REVISE = "revise"
    COMPLETE = "complete"
    ASK = "ask"
    STATUS = "status"
    UNDO = "undo"
    NONE = "none"


@dataclass
class Access:
    memory_id: str
    at: float
    kind: AccessKind


@dataclass
class Memory:
    id: str
    kind: MemoryKind
    scope: str
    content: str
    embedding: np.ndarray
    importance: float
    status: Status
    tier: Tier
    created_at: float
    label: Optional[str] = None
    completed_at: Optional[float] = None
    supersedes_id: Optional[str] = None
    superseded_by_id: Optional[str] = None
    feature_id: Optional[str] = None
    accesses: List[Access] = field(default_factory=list)

    @property
    def last_accessed_at(self) -> float:
        if self.accesses:
            return max(a.at for a in self.accesses)
        return self.created_at


@dataclass
class MemoryEvent:
    id: str
    at: float
    label: EventLabel
    memory_id: Optional[str]
    turn_id: Optional[str]
    context: Dict[str, Any]


@dataclass
class Turn:
    id: str
    session_id: str
    at: float
    project_scope: str
    mode: str
    user_text: str
    assistant_text: str
    embedding: np.ndarray
    salience: float = 0.0
    recalled_ids: List[str] = field(default_factory=list)
    extracted_at: Optional[float] = None


@dataclass
class NewMemory:
    kind: MemoryKind
    scope: str  # "project" or "global", as the extractor emits it
    content: str
    importance: float


@dataclass
class Supersede:
    old_id: str
    content: str
    kind: MemoryKind
    importance: float


@dataclass
class ExplicitOp:
    turn_id: str
    op: MemoryOp


@dataclass
class Extraction:
    new: List[NewMemory] = field(default_factory=list)
    reinforce: List[str] = field(default_factory=list)
    irrelevant: List[str] = field(default_factory=list)
    supersede: List[Supersede] = field(default_factory=list)
    complete: List[str] = field(default_factory=list)
    forget: List[str] = field(default_factory=list)
    explicit_ops: List[ExplicitOp] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Extraction":
        return cls(
            new=[
                NewMemory(MemoryKind(n["kind"]), n["scope"], n["content"], float(n["importance"]))
                for n in d.get("new", [])
            ],
            reinforce=list(d.get("reinforce", [])),
            irrelevant=list(d.get("irrelevant", [])),
            supersede=[
                Supersede(s["old_id"], s["content"], MemoryKind(s["kind"]), float(s["importance"]))
                for s in d.get("supersede", [])
            ],
            complete=list(d.get("complete", [])),
            forget=list(d.get("forget", [])),
            explicit_ops=[
                ExplicitOp(o["turn_id"], MemoryOp(o["op"])) for o in d.get("explicit_ops", [])
            ],
        )

    def is_empty(self) -> bool:
        return not (self.new or self.supersede or self.complete or self.forget)


@dataclass
class OpResolution:
    target_ids: List[str]
    new: Optional[NewMemory] = None
