from memory.adapters.prompt_renderer import (
    render_code_prefix,
    render_memories,
    render_profile,
    render_system_prompt,
    render_turns,
)
from memory.domain.entities import (
    GLOBAL_SCOPE,
    Memory,
    MemoryKind,
    Status,
    Tier,
    Turn,
    project_scope,
)
from memory.use_cases.recall import Profile
from tests.memory.conftest import vec

SCOPE = project_scope("/work/jarvis")


def _m(kind, scope, text):
    return Memory(id=text, kind=kind, scope=scope, content=text, embedding=vec(text),
                  importance=1.0, status=Status.CURRENT, tier=Tier.ACTIVE, created_at=0.0)


def _turn():
    return Turn(id="t", session_id="s", at=0.0, project_scope=SCOPE, mode="quick",
                user_text="hi", assistant_text="hello", embedding=vec("t"))


PROFILE = Profile(
    preferences=[_m(MemoryKind.PREFERENCE, GLOBAL_SCOPE, "likes brevity")],
    features=[_m(MemoryKind.FEATURE, SCOPE, "voice diff reviewer")],
    version=1,
)
EMPTY = Profile([], [], 0)


def test_render_profile():
    assert render_profile(PROFILE, SCOPE) == (
        "About the user:\n- likes brevity\n\n"
        "Working on jarvis: features in progress:\n- voice diff reviewer"
    )
    assert render_profile(EMPTY, SCOPE) == ""
    assert render_profile(Profile([], PROFILE.features, 0), SCOPE).startswith("Working on jarvis")


def test_scope_labels():
    out = render_memories([
        _m(MemoryKind.DECISION, SCOPE, "use sqlite"),
        _m(MemoryKind.PREFERENCE, GLOBAL_SCOPE, "short answers"),
        _m(MemoryKind.FACT, project_scope("/o"), "uses flask"),
    ], SCOPE)
    assert out == (
        "What you remember, most relevant first:\n"
        "- (decision, this project) use sqlite\n"
        "- (preference) short answers\n"
        "- (fact, other project) uses flask"
    )
    assert render_memories([], SCOPE) == ""


def test_render_turns():
    assert render_turns([_turn()]) == "Recent conversation:\nUser: hi\nJARVIS: hello"
    assert render_turns([]) == ""


def test_section_order_and_empty_omitted():
    mems = [_m(MemoryKind.FACT, SCOPE, "fact one")]
    out = render_system_prompt("BASE", PROFILE, mems, [_turn()], SCOPE)
    marks = ("BASE", "About the user", "What you remember", "Recent conversation")
    positions = [out.index(s) for s in marks]
    assert positions == sorted(positions)
    assert render_system_prompt("BASE", EMPTY, [], [], SCOPE) == "BASE"
    only_turns = render_system_prompt("BASE", EMPTY, [], [_turn()], SCOPE)
    assert only_turns == "BASE\n\nRecent conversation:\nUser: hi\nJARVIS: hello"


def test_code_prefix():
    mems = [_m(MemoryKind.DECISION, SCOPE, "use sqlite")]
    out = render_code_prefix(PROFILE, mems, SCOPE)
    assert out.startswith("## Project context\n") and out.endswith("\n\n")
    assert "voice diff reviewer" in out and "use sqlite" in out
    assert render_code_prefix(EMPTY, [], SCOPE) == ""


def test_render_turns_caps_long_texts_at_400_chars():
    long_answer = "x" * 2000
    t = Turn(id="t", session_id="s", at=0.0, project_scope=SCOPE, mode="think",
             user_text="q" * 401, assistant_text=long_answer, embedding=vec("t"))
    _, user, jarvis = render_turns([t]).split("\n")
    assert user == "User: " + "q" * 399 + "\u2026"
    assert jarvis == "JARVIS: " + "x" * 399 + "\u2026"
    assert len(user) - len("User: ") == 400 and len(jarvis) - len("JARVIS: ") == 400
    short = render_turns([_turn()])
    assert "\u2026" not in short and short.endswith("JARVIS: hello")
