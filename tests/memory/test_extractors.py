import sys
from types import SimpleNamespace

import pytest

from memory.adapters.extractor_factory import build_extractor
from memory.adapters.extractor_prompts import (
    EXTRACT_SYSTEM,
    render_extract_user,
    render_resolve_user,
    render_summarise_user,
)
from memory.adapters.groq_extractor import GroqMemoryExtractor
from memory.adapters.llm_json import parse_json_object
from memory.adapters.ollama_extractor import OllamaMemoryExtractor
from memory.domain.entities import (
    Extraction,
    Memory,
    MemoryKind,
    MemoryOp,
    NewMemory,
    OpResolution,
    Status,
    Tier,
    Turn,
)
from tests.memory.conftest import FakeExtractor, vec

NEW_JSON = '{"new":[{"kind":"fact","scope":"global","content":"x","importance":0.4}],"reinforce":["a"]}'


class _FakeOpenAI:
    def __init__(self, content):
        self.content = content
        self.last_kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _turn(text, tid="t1"):
    return Turn(
        id=tid, session_id="s", at=1.0, project_scope="project:/p", mode="chat",
        user_text=text, assistant_text="ok", embedding=vec(text),
    )


def _mem_with_id(mid, content="some memory"):
    return Memory(
        id=mid, kind=MemoryKind.FACT, scope="global", content=content, embedding=vec(content),
        importance=0.5, status=Status.CURRENT, tier=Tier.ACTIVE, created_at=1.0,
    )


def _ollama(content):
    return OllamaMemoryExtractor(chat_fn=lambda **kw: {"message": {"content": content}})


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json_object("Sure!\n```json\n{\"new\": []}\n```\nDone.") == {"new": []}


def test_parse_json_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_parse_json_raises_on_invalid_object():
    with pytest.raises(ValueError):
        parse_json_object("{not valid}")


def test_groq_extract_roundtrip_with_fake_client():
    client = _FakeOpenAI(NEW_JSON)
    e = GroqMemoryExtractor(api_key="k", client=client).extract([_turn("hi")], [_mem_with_id("a")])
    assert e.new[0].content == "x" and e.reinforce == ["a"]
    assert client.last_kwargs["response_format"] == {"type": "json_object"}


def test_groq_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert GroqMemoryExtractor().is_available() is False


def test_groq_reads_key_from_env_with_injected_client(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert GroqMemoryExtractor(client=_FakeOpenAI("{}")).is_available() is True


@pytest.mark.parametrize(
    "payload",
    [
        '{"new":[{"kind":"bogus","scope":"global","content":"x","importance":0.4}]}',
        '{"new":[{"kind":"fact"}]}',
        '{"new":[5]}',
        "not json",
    ],
)
def test_groq_malformed_extraction_raises_value_error(payload):
    ex = GroqMemoryExtractor(api_key="k", client=_FakeOpenAI(payload))
    with pytest.raises(ValueError):
        ex.extract([_turn("hi")], [])


def test_ollama_extract_uses_json_format():
    calls = []

    def chat(**kw):
        calls.append(kw)
        return {"message": {"content": '{"new": []}'}}

    OllamaMemoryExtractor(chat_fn=chat).extract([_turn("hi")], [])
    assert calls[0]["format"] == "json"


def test_ollama_malformed_extraction_raises_value_error():
    with pytest.raises(ValueError):
        _ollama('{"supersede":[{}]}').extract([_turn("hi")], [])


def test_ollama_available_when_chat_fn_injected():
    assert OllamaMemoryExtractor(chat_fn=lambda **kw: {}).is_available() is True


def test_ollama_unavailable_when_package_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "ollama":
            raise ImportError("no ollama")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert OllamaMemoryExtractor().is_available() is False


def test_factory_prefers_groq_when_key_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    fake_openai = SimpleNamespace(OpenAI=lambda **kw: _FakeOpenAI("{}"))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    assert isinstance(build_extractor(), GroqMemoryExtractor)


def test_factory_falls_back_to_ollama_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert isinstance(build_extractor(), OllamaMemoryExtractor)


def test_extract_prompt_lists_recalled_ids():
    user = render_extract_user([_turn("hi")], [_mem_with_id("a")])
    assert "a" in user and "hi" in user and "t1" in user


def test_extract_system_states_schema_and_rules():
    for needle in ("reinforce", "irrelevant", "supersede", "explicit_ops", "project", "global", "0 and 1"):
        assert needle in EXTRACT_SYSTEM


def test_resolve_and_summarise_prompts_render():
    r = render_resolve_user(MemoryOp.FORGET, "drop that", [_turn("hi")], [_mem_with_id("m1", "faiss")])
    assert "forget" in r and "drop that" in r and "m1" in r and "faiss" in r
    s = render_summarise_user([_mem_with_id("e1", "episode text")])
    assert "episode text" in s


def test_resolve_op_returns_targets():
    client = _FakeOpenAI('{"target_ids": ["m1"], "new": null}')
    ex = GroqMemoryExtractor(api_key="k", client=client)
    res = ex.resolve_op(MemoryOp.FORGET, "drop that", [_turn("hi")], [_mem_with_id("m1")])
    assert res == OpResolution(target_ids=["m1"], new=None)


def test_resolve_op_parses_new_memory():
    payload = '{"target_ids": ["m1"], "new": {"kind":"decision","scope":"project","content":"use pg","importance":0.9}}'
    ex = GroqMemoryExtractor(api_key="k", client=_FakeOpenAI(payload))
    res = ex.resolve_op(MemoryOp.REVISE, "moved to pg", [], [_mem_with_id("m1")])
    assert res.new == NewMemory(MemoryKind.DECISION, "project", "use pg", 0.9)


def test_resolve_op_malformed_raises_value_error():
    with pytest.raises(ValueError):
        _ollama('{"new": {"kind": "fact"}}').resolve_op(MemoryOp.STORE, "keep", [], [])


def test_summarise_returns_new_memories():
    payload = '{"new":[{"kind":"fact","scope":"global","content":"digest","importance":0.6}]}'
    out = _ollama(payload).summarise([_mem_with_id("e1")])
    assert out == [NewMemory(MemoryKind.FACT, "global", "digest", 0.6)]


def test_fake_extractor_contract():
    first = Extraction.from_dict({"reinforce": ["a"]})
    fx = FakeExtractor([first])
    assert fx.extract([], []) is first and fx.extract([], []) is first
    assert fx.resolve_op(MemoryOp.FORGET, "u", [], [_mem_with_id("m9")]).target_ids == ["m9"]
    assert fx.summarise([]) == []
    fx.raise_next = 1
    with pytest.raises(ValueError):
        fx.extract([], [])
    assert fx.extract([], []) is first
    assert [c[0] for c in fx.calls] == ["extract", "extract", "resolve_op", "summarise", "extract", "extract"]
