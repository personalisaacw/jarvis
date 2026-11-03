import os
from typing import List

from memory.domain.entities import GLOBAL_SCOPE, Memory, Turn

_PROJECT_PREFIX = "project:"
TURN_TEXT_CHARS = 400  # per user/assistant text, so one long answer cannot flood a prompt


def clip(text: str, limit: int = TURN_TEXT_CHARS) -> str:
    """text, or its first limit-1 characters plus an ellipsis when it is longer than limit."""
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _basename(project_scope: str) -> str:
    path = project_scope[len(_PROJECT_PREFIX):] if project_scope.startswith(_PROJECT_PREFIX) else project_scope
    return os.path.basename(path.rstrip("/\\")) or "this project"


def render_profile(profile, project_scope: str = "") -> str:
    parts = []
    if profile.preferences:
        parts.append("About the user:\n" + "\n".join("- " + m.content for m in profile.preferences))
    if profile.features:
        parts.append(
            "Working on %s: features in progress:\n" % _basename(project_scope)
            + "\n".join("- " + m.content for m in profile.features)
        )
    return "\n\n".join(parts)


def _label(m: Memory, project_scope: str) -> str:
    kind = m.kind.value if hasattr(m.kind, "value") else str(m.kind)
    if m.scope == GLOBAL_SCOPE:
        return kind
    return kind + (", this project" if m.scope == project_scope else ", other project")


def render_memories(memories: List[Memory], project_scope: str) -> str:
    if not memories:
        return ""
    lines = ["- (%s) %s" % (_label(m, project_scope), m.content) for m in memories]
    return "What you remember, most relevant first:\n" + "\n".join(lines)


def render_turns(turns: List[Turn]) -> str:
    if not turns:
        return ""
    lines = []
    for t in turns:
        lines.append("User: " + clip(t.user_text))
        lines.append("JARVIS: " + clip(t.assistant_text))
    return "Recent conversation:\n" + "\n".join(lines)


def render_system_prompt(
    base: str, profile, memories: List[Memory], turns: List[Turn], project_scope: str
) -> str:
    sections = [
        base,
        render_profile(profile, project_scope),
        render_memories(memories, project_scope),
        render_turns(turns),
    ]
    return "\n\n".join(s for s in sections if s)


def render_code_prefix(profile, memories: List[Memory], project_scope: str) -> str:
    body = "\n\n".join(
        s for s in (render_profile(profile, project_scope), render_memories(memories, project_scope)) if s
    )
    return "## Project context\n" + body + "\n\n" if body else ""
