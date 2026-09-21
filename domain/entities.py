from enum import Enum
from dataclasses import dataclass

class Intent(str, Enum):
    QUICK = "quick"
    THINK = "think"
    CODE = "code"

@dataclass
class Utterance:
    text: str
    intent: Intent
    timestamp: float

@dataclass
class RoutingResult:
    intent: Intent
    confidence_score: float
    requires_clarification: bool
