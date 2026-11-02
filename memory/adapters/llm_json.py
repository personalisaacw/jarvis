import json
from abc import abstractmethod
from typing import List

from memory.adapters.extractor_prompts import (
    EXTRACT_SYSTEM,
    RESOLVE_SYSTEM,
    SUMMARISE_SYSTEM,
    render_extract_user,
    render_resolve_user,
    render_summarise_user,
)
from memory.domain.entities import (
    Extraction,
    Memory,
    MemoryKind,
    MemoryOp,
    NewMemory,
    OpResolution,
    Turn,
)
from memory.ports.extractor import IMemoryExtractor

_MALFORMED = (KeyError, TypeError, ValueError)


def parse_json_object(text: str) -> dict:
    """Parse the JSON object in an LLM reply, ignoring ``` fences and surrounding prose."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in model output")
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError("invalid JSON in model output: %s" % e)
    if not isinstance(obj, dict):
        raise ValueError("model output is not a JSON object")
    return obj


def _new_memory(d: dict) -> NewMemory:
    return NewMemory(MemoryKind(d["kind"]), d["scope"], d["content"], float(d["importance"]))


class JsonLLMExtractor(IMemoryExtractor):
    """Shared prompt building and parsing; subclasses only supply `_complete`."""

    @abstractmethod
    def _complete(self, system: str, user: str) -> str:
        """Send one JSON-mode chat request and return the raw reply text."""

    def _ask(self, system: str, user: str) -> dict:
        return parse_json_object(self._complete(system, user))

    def extract(self, turns: List[Turn], recalled: List[Memory]) -> Extraction:
        data = self._ask(EXTRACT_SYSTEM, render_extract_user(turns, recalled))
        try:
            return Extraction.from_dict(data)
        except _MALFORMED as e:
            raise ValueError("malformed extraction: %r" % (e,))

    def resolve_op(
        self, op: MemoryOp, utterance: str, working_turns: List[Turn], candidates: List[Memory]
    ) -> OpResolution:
        data = self._ask(RESOLVE_SYSTEM, render_resolve_user(op, utterance, working_turns, candidates))
        try:
            new = data.get("new")
            return OpResolution(
                target_ids=[str(i) for i in data.get("target_ids", [])],
                new=_new_memory(new) if new else None,
            )
        except _MALFORMED as e:
            raise ValueError("malformed resolution: %r" % (e,))

    def summarise(self, episodes: List[Memory]) -> List[NewMemory]:
        data = self._ask(SUMMARISE_SYSTEM, render_summarise_user(episodes))
        try:
            return [_new_memory(n) for n in data.get("new", [])]
        except _MALFORMED as e:
            raise ValueError("malformed summary: %r" % (e,))
