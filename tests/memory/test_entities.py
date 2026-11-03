import pytest

from memory.domain.entities import Extraction, MemoryKind, project_scope


def test_project_scope_is_absolute():
    assert project_scope("jarvis").startswith("project:/")


def test_extraction_from_dict_defaults_missing_lists():
    e = Extraction.from_dict({"new": [{"kind": "decision", "scope": "project", "content": "x", "importance": 0.7}]})
    assert e.new[0].kind == MemoryKind.DECISION and e.reinforce == [] and e.is_empty() is False


def test_extraction_from_dict_rejects_bad_kind():
    with pytest.raises(ValueError):
        Extraction.from_dict({"new": [{"kind": "rumour", "scope": "global", "content": "x", "importance": 0.1}]})


def test_extraction_is_empty_ignores_reinforce():
    assert Extraction.from_dict({"reinforce": ["a"], "irrelevant": ["b"]}).is_empty() is True
