import json
import sys
import types
import uuid

import pytest

from memory.adapters.memory_op_seeds import SEEDS
from memory.adapters.sqlite_store import SqliteMemoryStore
from memory.cli import main
from memory.domain.entities import Memory, MemoryKind, Status, Tier
from memory.domain.params import Params
from tests.memory.conftest import FakeEmbedder, vec


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "m.sqlite")


def open_store(path):
    return SqliteMemoryStore(path)


def test_cli_report_runs_on_empty_db(tmp_path, capsys):
    assert main(["report", "--db", str(tmp_path / "m.sqlite")]) == 0
    out = capsys.readouterr().out
    assert "Counts" in out and "last_sleep_at: never" in out


def test_cli_reset_params_writes_history(db, capsys):
    s = open_store(db)
    s.save_params(Params.defaults().with_value("recall_k", 9), [])
    s.close()
    assert main(["reset-params", "--db", db]) == 0
    s = open_store(db)
    try:
        assert s.load_params().as_dict() == Params.defaults().as_dict()
        (row,) = s.params_history(10)
        assert (row["name"], row["old_value"], row["new_value"], row["reason"]) == (
            "recall_k", 9, 6, "manual reset")
    finally:
        s.close()


def test_cli_sleep_dry_run_changes_nothing(db, capsys):
    s = open_store(db)
    m = Memory(id=str(uuid.uuid4()), kind=MemoryKind.FACT, scope="global", content="old fact",
               embedding=vec("old fact"), importance=0.1, status=Status.CURRENT,
               tier=Tier.ACTIVE, created_at=0.0)  # long cold: a real sleep would demote it
    s.add(m)
    s.close()
    assert main(["sleep", "--dry-run", "--db", db]) == 0
    assert "dry run" in capsys.readouterr().out
    s = open_store(db)
    try:
        assert s.get(m.id).tier == Tier.ACTIVE
        assert s.get_meta("last_sleep_at") is None
    finally:
        s.close()


def test_cli_sleep_writes_last_sleep_at(db, capsys):
    assert main(["sleep", "--db", db]) == 0
    s = open_store(db)
    try:
        assert s.get_meta("last_sleep_at") is not None
    finally:
        s.close()


def test_cli_migrate_imports_routes_and_seeds(tmp_path, db, monkeypatch, capsys):
    fake = types.ModuleType("adapters.embeddings")
    fake.HuggingFaceAdapter = FakeEmbedder
    monkeypatch.setitem(sys.modules, "adapters.embeddings", fake)
    (tmp_path / ".vectorstore").mkdir()
    (tmp_path / ".vectorstore" / "metadata.json").write_text(json.dumps(
        [{"text": "play some jazz", "intent": "quick", "timestamp": 5.0}]))
    assert main(["migrate", "--db", db, "--project-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1 route examples" in out
    s = open_store(db)
    try:
        assert len(s.list(None, [MemoryKind.ROUTE_EXAMPLE])) == 1
        assert len(s.list(None, [MemoryKind.MEMORY_OP_EXAMPLE])) == sum(
            len(v) for v in SEEDS.values())
    finally:
        s.close()


def test_cli_error_returns_one_with_message(tmp_path, capsys):
    bad = tmp_path / "file"
    bad.write_text("x")
    assert main(["report", "--db", str(bad / "m.sqlite")]) == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("error:")


def test_cli_default_db_is_under_project_dir(tmp_path, capsys):
    assert main(["report", "--project-dir", str(tmp_path)]) == 0
    assert (tmp_path / ".jarvis" / "memory.sqlite").is_file()
