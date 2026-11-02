from typing import List

from memory.domain.entities import Memory, MemoryOp, Turn

_KINDS = "fact, decision, feature, preference, episode"

EXTRACT_SYSTEM = """You maintain the long-term memory of a voice coding assistant.
You are given a batch of recent conversation turns and the memories that were recalled during them.
Decide what is worth remembering, and how the recalled memories fared.

Reply with ONE JSON object and nothing else, with exactly this schema:

{
  "new":       [{"kind": "decision", "scope": "project", "content": "...", "importance": 0.7}],
  "reinforce": ["<id>"],
  "irrelevant": ["<id>"],
  "supersede": [{"old_id": "<id>", "content": "...", "kind": "decision", "importance": 0.8}],
  "complete":  ["<feature id>"],
  "forget":    ["<id>"],
  "explicit_ops": [{"turn_id": "<id>", "op": "store"}]
}

Rules:
- kind is one of: """ + _KINDS + """.
- scope is "project" (only true for the current project) or "global" (true for the user in general).
- importance is a number between 0 and 1 (1 = must never be forgotten).
- new: durable facts, decisions, features in progress and preferences. Skip chatter and anything already recalled.
- Every recalled id MUST appear in either "reinforce" (it was useful) or "irrelevant" (it was not).
- supersede: a recalled memory that is now outdated, with its replacement.
- complete: ids of features that are finished. forget: ids the user asked to drop.
- explicit_ops: turns where the user clearly asked for a memory operation. op is one of:
  store, forget, revise, complete, ask, status, undo.
- Use empty lists for anything that does not apply. Never invent ids."""

# resolve_op reply: {"target_ids": ["<id>"], "new": {kind, scope, content, importance} | null}
RESOLVE_SYSTEM = """You carry out one explicit memory operation requested by the user of a voice coding assistant.
You are given the operation, the user's utterance, recent turns and candidate memories.

Reply with ONE JSON object and nothing else, with exactly this schema:

{"target_ids": ["<id>"], "new": {"kind": "decision", "scope": "project", "content": "...", "importance": 0.7}}

Rules:
- target_ids: ids chosen ONLY from the candidates the operation applies to. Use [] if none match.
- new: the memory to write, or null. Required for store (importance 1.0) and revise (the replacement);
  null for forget, complete and ask.
- kind is one of: """ + _KINDS + """.
- scope is "project" or "global". importance is a number between 0 and 1."""

# summarise reply: {"new": [{kind, scope, content, importance}, ...]}
SUMMARISE_SYSTEM = """You condense old episode memories of a voice coding assistant into a few durable memories.

Reply with ONE JSON object and nothing else, with exactly this schema:

{"new": [{"kind": "fact", "scope": "project", "content": "...", "importance": 0.6}]}

Rules:
- Keep only what is still useful later: decisions, facts, preferences. Merge duplicates.
- kind is one of: """ + _KINDS + """.
- scope is "project" or "global". importance is a number between 0 and 1.
- Use {"new": []} if nothing is worth keeping."""


def _render_turns(turns: List[Turn]) -> str:
    return "\n".join(
        "[turn %s] user: %s\nassistant: %s" % (t.id, t.user_text, t.assistant_text) for t in turns
    ) or "(none)"


def _render_memories(memories: List[Memory]) -> str:
    return "\n".join(
        "- id=%s kind=%s scope=%s: %s" % (m.id, m.kind.value, m.scope, m.content) for m in memories
    ) or "(none)"


def render_extract_user(turns: List[Turn], recalled: List[Memory]) -> str:
    return "Turns:\n%s\n\nRecalled memories:\n%s" % (_render_turns(turns), _render_memories(recalled))


def render_resolve_user(
    op: MemoryOp, utterance: str, working_turns: List[Turn], candidates: List[Memory]
) -> str:
    return "Operation: %s\nUtterance: %s\n\nRecent turns:\n%s\n\nCandidate memories:\n%s" % (
        op.value, utterance, _render_turns(working_turns), _render_memories(candidates),
    )


def render_summarise_user(episodes: List[Memory]) -> str:
    return "Episodes to condense:\n%s" % _render_memories(episodes)
