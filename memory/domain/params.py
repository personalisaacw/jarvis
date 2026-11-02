from dataclasses import dataclass
from typing import Dict

from memory.domain.entities import MemoryKind


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: float
    min: float
    max: float
    step: float


_ROWS = [
    # decay
    ("decay.episode", 0.9, 0.3, 2.0, 0.1),
    ("decay.route_example", 0.6, 0.2, 1.5, 0.1),
    ("decay.memory_op_example", 0.6, 0.2, 1.5, 0.1),
    ("decay.fact", 0.5, 0.1, 1.5, 0.05),
    ("decay.decision", 0.3, 0.05, 1.0, 0.05),
    ("decay.feature", 0.2, 0.05, 1.0, 0.05),
    ("decay.preference", 0.2, 0.05, 1.0, 0.05),
    # strength
    ("importance_weight", 1.0, 0.2, 3.0, 0.1),
    ("active_threshold", 0.8, 0.2, 3.0, 0.1),
    ("dormant_threshold", 0.3, 0.05, 1.5, 0.05),
    # recall
    ("scope_global", 0.8, 0.3, 1.0, 0.05),
    ("scope_other", 0.4, 0.0, 1.0, 0.05),
    ("recall_k", 6, 2, 15, 1),
    ("budget_quick", 2, 0, 6, 1),
    ("dormant_recall_similarity", 0.75, 0.5, 0.95, 0.02),
    ("recall_similarity_floor", 0.45, 0.2, 0.8, 0.02),
    ("working_set_patience", 3, 1, 10, 1),
    ("working_turns", 6, 2, 20, 1),
    ("profile_preferences", 5, 0, 15, 1),
    # extraction
    ("extract_salience_threshold", 1.5, 0.3, 6.0, 0.1),
    ("extract_idle_seconds", 90, 20, 600, 10),
    ("extract_max_turns", 10, 2, 30, 1),
    ("mode_prior.quick", 0.5, 0.1, 1.5, 0.1),
    ("mode_prior.think", 1.0, 0.1, 2.0, 0.1),
    ("salience_length_bonus", 0.3, 0.0, 1.0, 0.05),
    ("dedup_threshold", 0.9, 0.8, 0.98, 0.01),
    # memory ops
    ("memory_op_threshold", 0.7, 0.4, 0.95, 0.02),
    ("memory_op_ask_threshold", 0.5, 0.3, 0.9, 0.02),
    ("memory_op_k", 5, 1, 15, 1),
    # sleep
    ("episode_summary_age_days", 7, 2, 30, 1),
    ("sleep_interval_hours", 24, 6, 168, 6),
    ("retention_days", 60, 14, 365, 7),
]

PARAM_SPECS: Dict[str, ParamSpec] = {r[0]: ParamSpec(*r) for r in _ROWS}

FIXED = {
    "tune_window_days": 30,
    "min_events": 40,
    "improvement_margin": 0.05,
    "max_moves_per_pass": 3,
    "w_false_recall": 0.5,
    "w_stale": 1.0,
    "w_empty_extraction": 0.3,
    "w_late_extraction": 1.0,
    "w_false_op": 1.0,
    "w_missed_op": 0.7,
}


class Params:
    def __init__(self, values: Dict[str, float]):
        self._values = dict(values)

    @classmethod
    def defaults(cls) -> "Params":
        return cls({n: s.default for n, s in PARAM_SPECS.items()})

    def get(self, name: str) -> float:
        return self._values[name]

    def decay(self, kind: MemoryKind) -> float:
        return self._values["decay." + MemoryKind(kind).value]

    def mode_prior(self, mode: str) -> float:
        return self._values.get("mode_prior." + mode, self._values["mode_prior.quick"])

    def with_value(self, name: str, value: float) -> "Params":
        spec = PARAM_SPECS[name]
        new = dict(self._values)
        new[name] = max(spec.min, min(spec.max, value))
        return Params(new)

    def is_valid(self) -> bool:
        v = self._values
        return (
            v["dormant_threshold"] < v["active_threshold"]
            and v["memory_op_ask_threshold"] < v["memory_op_threshold"]
        )

    def as_dict(self) -> Dict[str, float]:
        return dict(self._values)
