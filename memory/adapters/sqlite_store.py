import json
import os
import sqlite3
import threading
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from domain.entities import Intent, Utterance
from domain.ports import IVectorStore
from memory.domain.entities import (
    GLOBAL_SCOPE,
    Access,
    AccessKind,
    EventLabel,
    Memory,
    MemoryEvent,
    MemoryKind,
    Status,
    Tier,
    Turn,
)
from memory.domain.params import PARAM_SPECS, Params
from memory.domain.scoring import cosine, strength
from memory.ports.clock import IClock, SystemClock
from memory.ports.store import IMemoryStore

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, scope TEXT NOT NULL, content TEXT NOT NULL,
    label TEXT, importance REAL NOT NULL, status TEXT NOT NULL, tier TEXT NOT NULL,
    completed_at REAL, embedding BLOB NOT NULL, created_at REAL NOT NULL,
    last_accessed_at REAL NOT NULL, supersedes_id TEXT, superseded_by_id TEXT, feature_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status, tier);
CREATE TABLE IF NOT EXISTS accesses (
    id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT NOT NULL, at REAL NOT NULL,
    kind TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_accesses_memory ON accesses(memory_id, at);
CREATE TABLE IF NOT EXISTS memory_events (
    id TEXT PRIMARY KEY, at REAL NOT NULL, label TEXT NOT NULL, memory_id TEXT,
    turn_id TEXT, context TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_at ON memory_events(at);
CREATE TABLE IF NOT EXISTS params (
    name TEXT PRIMARY KEY, value REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS params_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, old_value REAL, new_value REAL,
    old_loss REAL, new_loss REAL, at REAL NOT NULL, reason TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, at REAL NOT NULL, project_scope TEXT NOT NULL,
    mode TEXT NOT NULL, user_text TEXT NOT NULL, assistant_text TEXT NOT NULL,
    embedding BLOB NOT NULL, salience REAL NOT NULL, recalled_ids TEXT NOT NULL,
    extracted_at REAL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, at);
"""

_MEMORY_COLS = (
    "id, kind, scope, content, label, importance, status, tier, completed_at, embedding, "
    "created_at, last_accessed_at, supersedes_id, superseded_by_id, feature_id"
)
_TURN_COLS = (
    "id, session_id, at, project_scope, mode, user_text, assistant_text, embedding, "
    "salience, recalled_ids, extracted_at"
)
_ROUTE_POOL = 50  # candidates re-ranked by cosine x strength in search_similar
_EMPTY = np.zeros((0, 384), dtype=np.float32)


def _blob(arr: np.ndarray) -> bytes:
    return np.asarray(arr).astype(np.float32).tobytes()


def _unblob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _memory_to_row(m: Memory) -> tuple:
    return (
        m.id, MemoryKind(m.kind).value, m.scope, m.content, m.label, m.importance,
        Status(m.status).value, Tier(m.tier).value, m.completed_at, _blob(m.embedding),
        m.created_at, m.last_accessed_at, m.supersedes_id, m.superseded_by_id, m.feature_id,
    )


def _row_to_memory(r: tuple, accesses: Optional[List[Access]] = None) -> Memory:
    return Memory(
        id=r[0], kind=MemoryKind(r[1]), scope=r[2], content=r[3], label=r[4],
        importance=r[5], status=Status(r[6]), tier=Tier(r[7]), completed_at=r[8],
        embedding=_unblob(r[9]), created_at=r[10], supersedes_id=r[12],
        superseded_by_id=r[13], feature_id=r[14], accesses=accesses or [],
    )


def _row_to_turn(r: tuple) -> Turn:
    return Turn(
        id=r[0], session_id=r[1], at=r[2], project_scope=r[3], mode=r[4], user_text=r[5],
        assistant_text=r[6], embedding=_unblob(r[7]), salience=r[8],
        recalled_ids=json.loads(r[9]), extracted_at=r[10],
    )


def _in_clause(values: Iterable[Any]) -> Tuple[str, list]:
    vals = list(values)
    return "(" + ",".join("?" * len(vals)) + ")", vals


class SqliteMemoryStore(IMemoryStore, IVectorStore):
    def __init__(self, path: str = ".jarvis/memory.sqlite", clock: IClock = SystemClock()):
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: Optional[Tuple[List[str], np.ndarray]] = None
        if path != ":memory:":
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        try:
            self._conn = self._open(path)
        except sqlite3.DatabaseError:
            if path == ":memory:":
                raise
            moved = "%s.corrupt-%d" % (path, int(self._clock.now()))
            os.replace(path, moved)
            for suffix in ("-wal", "-shm"):  # stale journals must not attach to the fresh file
                if os.path.exists(path + suffix):
                    os.replace(path + suffix, moved + suffix)
            print("[Memory] WARNING: memory database was unreadable; moved to %s and started "
                  "fresh" % moved)
            self._conn = self._open(path)

    def _open(self, path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(path, check_same_thread=False)
        try:
            with self._lock:
                if path != ":memory:":
                    conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
                conn.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                    (SCHEMA_VERSION,),
                )
                conn.commit()
        except sqlite3.DatabaseError:
            conn.close()
            raise
        return conn

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- memories ----

    def add(self, m: Memory) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (%s) VALUES (%s)" % (_MEMORY_COLS, ",".join("?" * 15)),
                _memory_to_row(m),
            )
            self._conn.execute(
                "INSERT INTO accesses(memory_id, at, kind) VALUES (?, ?, ?)",
                (m.id, m.created_at, AccessKind.CREATED.value),
            )
            self._conn.commit()
            self._cache = None

    def add_many(self, memories: List[Memory]) -> None:
        """Insert all memories in one transaction: either every row lands or none does."""
        with self._lock:
            try:
                with self._conn:
                    for m in memories:
                        self._conn.execute(
                            "INSERT INTO memories (%s) VALUES (%s)"
                            % (_MEMORY_COLS, ",".join("?" * 15)),
                            _memory_to_row(m),
                        )
                        self._conn.execute(
                            "INSERT INTO accesses(memory_id, at, kind) VALUES (?, ?, ?)",
                            (m.id, m.created_at, AccessKind.CREATED.value),
                        )
            finally:
                self._cache = None

    def get(self, id: str) -> Optional[Memory]:
        with self._lock:
            row = self._conn.execute(
                "SELECT %s FROM memories WHERE id = ?" % _MEMORY_COLS, (id,)
            ).fetchone()
            if row is None:
                return None
            return _row_to_memory(row, self._accesses_for([id]).get(id))

    def update(self, m: Memory) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET kind=?, scope=?, content=?, label=?, importance=?, "
                "status=?, tier=?, completed_at=?, embedding=?, supersedes_id=?, "
                "superseded_by_id=?, feature_id=? WHERE id=?",
                (
                    MemoryKind(m.kind).value, m.scope, m.content, m.label, m.importance,
                    Status(m.status).value, Tier(m.tier).value, m.completed_at,
                    _blob(m.embedding), m.supersedes_id, m.superseded_by_id, m.feature_id, m.id,
                ),
            )
            self._conn.commit()
            self._cache = None

    def list(
        self,
        status: Optional[Status] = Status.CURRENT,
        kinds: Optional[List[MemoryKind]] = None,
        scope: Optional[str] = None,
    ) -> List[Memory]:
        with self._lock:
            where, args = self._filters(status, kinds, scope)
            rows = self._conn.execute(
                "SELECT %s FROM memories%s ORDER BY created_at, rowid" % (_MEMORY_COLS, where),
                args,
            ).fetchall()
            accesses = self._accesses_for([r[0] for r in rows])
            return [_row_to_memory(r, accesses.get(r[0])) for r in rows]

    def nearest(
        self,
        vec: np.ndarray,
        k: int,
        kinds: Optional[List[MemoryKind]] = None,
        scope: Optional[str] = None,
        status: Status = Status.CURRENT,
    ) -> List[Tuple[Memory, float]]:
        with self._lock:
            ids, matrix = self._candidates(status, kinds, scope)
            if not ids:
                return []
            sims = self._scores(matrix, vec)
            order = [int(i) for i in np.argsort(-sims, kind="stable")[:k]]
            top = [ids[i] for i in order]
            ph, vals = _in_clause(top)
            rows = {
                r[0]: r
                for r in self._conn.execute(
                    "SELECT %s FROM memories WHERE id IN %s" % (_MEMORY_COLS, ph), vals
                )
            }
            accesses = self._accesses_for(top)
            return [
                (_row_to_memory(rows[ids[i]], accesses.get(ids[i])), float(sims[i]))
                for i in order
            ]

    def max_cosine(
        self,
        vec: np.ndarray,
        kinds: Optional[List[MemoryKind]] = None,
        scope: Optional[str] = None,
    ) -> float:
        with self._lock:
            ids, matrix = self._candidates(Status.CURRENT, kinds, scope)
            if not ids:
                return 0.0
            return float(np.max(self._scores(matrix, vec)))

    # ---- dedup ----

    def dedup(self, kinds: List[MemoryKind], threshold: float, now: float) -> int:
        """Merge near-duplicate current memories within (kind, scope, label); return merges."""
        with self._lock:
            p = self.load_params()
            groups: Dict[Tuple[Any, ...], List[Memory]] = {}
            for m in self.list(Status.CURRENT, kinds):
                groups.setdefault((m.kind, m.scope, m.label), []).append(m)
            merges = 0
            for members in groups.values():
                members.sort(key=lambda m: -strength(m, now, p))  # stable: ties keep age order
                keepers: List[Memory] = []
                for m in members:
                    keeper = next(
                        (k for k in keepers if cosine(k.embedding, m.embedding) >= threshold), None
                    )
                    if keeper is None:
                        keepers.append(m)
                        continue
                    self.move_accesses(m.id, keeper.id)
                    m.status, m.superseded_by_id = Status.SUPERSEDED, keeper.id
                    keeper.supersedes_id = m.id
                    self.update(m)
                    self.update(keeper)
                    merges += 1
            return merges

    # ---- IVectorStore (semantic router examples) ----

    def add_utterance(self, utterance: Utterance, vector: List[float]) -> None:
        with self._lock:
            vec = np.asarray(vector, dtype=np.float32)
            label = Intent(utterance.intent).value
            threshold = self.load_params().get("dedup_threshold")
            for m, sim in self.nearest(vec, _ROUTE_POOL, [MemoryKind.ROUTE_EXAMPLE], GLOBAL_SCOPE):
                if m.label == label and sim >= threshold:
                    self.add_access(m.id, self._clock.now(), AccessKind.REINFORCED)
                    return
            self.add(Memory(
                id=uuid.uuid4().hex, kind=MemoryKind.ROUTE_EXAMPLE, scope=GLOBAL_SCOPE,
                content=utterance.text, embedding=vec, importance=0.5, status=Status.CURRENT,
                tier=Tier.ACTIVE, created_at=utterance.timestamp, label=label,
            ))

    def search_similar(
        self, vector: List[float], limit: int = 3
    ) -> List[Tuple[Utterance, float]]:
        with self._lock:
            p, now = self.load_params(), self._clock.now()
            hits = self.nearest(
                np.asarray(vector, dtype=np.float32), max(limit, _ROUTE_POOL),
                [MemoryKind.ROUTE_EXAMPLE], GLOBAL_SCOPE,
            )
        hits.sort(key=lambda h: -(h[1] * strength(h[0], now, p)))
        return [
            (Utterance(m.content, Intent(m.label), m.created_at), 2.0 - 2.0 * sim)
            for m, sim in hits[:limit]
        ]

    def distill_utterances(self) -> None:
        self.dedup(
            [MemoryKind.ROUTE_EXAMPLE], self.load_params().get("dedup_threshold"), self._clock.now()
        )

    # ---- accesses ----

    def add_access(self, memory_id: str, at: float, kind: AccessKind) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO accesses(memory_id, at, kind) VALUES (?, ?, ?)",
                (memory_id, at, AccessKind(kind).value),
            )
            self._refresh_last_accessed([memory_id])
            self._conn.commit()

    def move_accesses(self, from_id: str, to_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE accesses SET memory_id = ? WHERE memory_id = ?", (to_id, from_id)
            )
            self._refresh_last_accessed([from_id, to_id])
            self._conn.commit()

    # ---- events ----

    def add_event(self, e: MemoryEvent) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO memory_events(id, at, label, memory_id, turn_id, context) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (e.id, e.at, EventLabel(e.label).value, e.memory_id, e.turn_id,
                 json.dumps(e.context)),
            )
            self._conn.commit()

    def delete_events(self, event_ids: List[str]) -> int:
        if not event_ids:
            return 0
        ph, vals = _in_clause(event_ids)
        with self._lock:
            n = self._conn.execute("DELETE FROM memory_events WHERE id IN " + ph, vals).rowcount
            self._conn.commit()
        return n

    def events_since(self, at: float) -> List[MemoryEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, at, label, memory_id, turn_id, context FROM memory_events "
                "WHERE at >= ? ORDER BY at, rowid",
                (at,),
            ).fetchall()
        return [
            MemoryEvent(r[0], r[1], EventLabel(r[2]), r[3], r[4], json.loads(r[5]))
            for r in rows
        ]

    # ---- params ----

    def load_params(self) -> Params:
        with self._lock:
            rows = self._conn.execute("SELECT name, value FROM params").fetchall()
        values = Params.defaults().as_dict()
        values.update({n: v for n, v in rows if n in PARAM_SPECS})
        return Params(values)

    def save_params(
        self, p: Params, history: List[Tuple[str, float, float, float, float, str]]
    ) -> None:
        now = self._clock.now()
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO params(name, value, updated_at) VALUES (?, ?, ?)",
                [(n, float(v), now) for n, v in p.as_dict().items()],
            )
            self._conn.executemany(
                "INSERT INTO params_history(name, old_value, new_value, old_loss, new_loss, "
                "at, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(n, o, nw, ol, nl, now, reason) for n, o, nw, ol, nl, reason in history],
            )
            self._conn.commit()

    def params_history(self, limit: int = 10) -> List[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, name, old_value, new_value, old_loss, new_loss, at, reason "
                "FROM params_history ORDER BY at DESC, id DESC LIMIT ?",
                (limit,),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    # ---- meta ----

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value)
            )
            self._conn.commit()

    # ---- turns ----

    def add_turn(self, t: Turn) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (%s) VALUES (%s)" % (_TURN_COLS, ",".join("?" * 11)),
                (t.id, t.session_id, t.at, t.project_scope, t.mode, t.user_text,
                 t.assistant_text, _blob(t.embedding), t.salience, json.dumps(t.recalled_ids),
                 t.extracted_at),
            )
            self._conn.commit()

    def pending_turns(self) -> List[Turn]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT %s FROM turns WHERE extracted_at IS NULL ORDER BY at, rowid" % _TURN_COLS
            ).fetchall()
        return [_row_to_turn(r) for r in rows]

    def mark_extracted(self, turn_ids: List[str], at: float) -> None:
        if not turn_ids:
            return
        ph, vals = _in_clause(turn_ids)
        with self._lock:
            self._conn.execute(
                "UPDATE turns SET extracted_at = ? WHERE id IN " + ph, [at] + vals
            )
            self._conn.commit()

    def recent_turns(self, session_id: str, n: int) -> List[Turn]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT %s FROM turns WHERE session_id = ? ORDER BY at DESC, rowid DESC LIMIT ?"
                % _TURN_COLS,
                (session_id, n),
            ).fetchall()
        return [_row_to_turn(r) for r in reversed(rows)]

    def delete_turns_before(self, at: float, keep_session: str, keep_n: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM turns WHERE extracted_at IS NOT NULL AND at < ? AND id NOT IN "
                "(SELECT id FROM turns WHERE session_id = ? ORDER BY at DESC, rowid DESC LIMIT ?)",
                (at, keep_session, keep_n),
            )
            self._conn.commit()
            return cur.rowcount

    # ---- purge / report ----

    def purge(self, before: float) -> int:
        with self._lock:
            ids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT m.id FROM memories m WHERE (m.status = ? OR m.tier = ?) AND "
                    "COALESCE((SELECT MAX(a.at) FROM accesses a WHERE a.memory_id = m.id), "
                    "m.created_at) < ?",
                    (Status.FORGOTTEN.value, Tier.ARCHIVED.value, before),
                )
            ]
            if ids:
                ph, vals = _in_clause(ids)
                self._conn.execute("DELETE FROM accesses WHERE memory_id IN " + ph, vals)
                self._conn.execute("DELETE FROM memory_events WHERE memory_id IN " + ph, vals)
                self._conn.execute("DELETE FROM memories WHERE id IN " + ph, vals)
            self._conn.commit()
            self._cache = None
            return len(ids)

    def copy_in_memory(self) -> "SqliteMemoryStore":
        copy = SqliteMemoryStore(":memory:", self._clock)
        with self._lock:
            self._conn.backup(copy._conn)
        return copy

    def count_by(self, kind_tier_scope: bool = True) -> List[Tuple[str, str, str, int]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, tier, scope, COUNT(*) FROM memories WHERE status = ? "
                "GROUP BY kind, tier, scope",
                (Status.CURRENT.value,),
            ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    # ---- private helpers (callers hold the lock) ----

    @staticmethod
    def _filters(
        status: Optional[Status], kinds: Optional[List[MemoryKind]], scope: Optional[str]
    ) -> Tuple[str, List[Any]]:
        clauses: List[str] = []
        args: List[Any] = []
        if status is not None:
            clauses.append("status = ?")
            args.append(Status(status).value)
        if kinds is not None:
            ph, vals = _in_clause(MemoryKind(k).value for k in kinds)
            clauses.append("kind IN " + ph)
            args.extend(vals)
        if scope is not None:
            clauses.append("scope = ?")
            args.append(scope)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", args

    @staticmethod
    def _scores(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
        # Some macOS BLAS builds emit spurious FP warnings from float32 matmul on valid input.
        with np.errstate(all="ignore"):
            return matrix @ np.asarray(vec, dtype=np.float32)

    def _current_cache(self) -> Tuple[List[str], np.ndarray]:
        if self._cache is None:
            rows = self._conn.execute(
                "SELECT id, embedding FROM memories WHERE status = ? ORDER BY rowid",
                (Status.CURRENT.value,),
            ).fetchall()
            matrix = np.stack([_unblob(r[1]) for r in rows]) if rows else _EMPTY
            self._cache = ([r[0] for r in rows], matrix)
        return self._cache

    def _candidates(
        self, status: Status, kinds: Optional[List[MemoryKind]], scope: Optional[str]
    ) -> Tuple[List[str], np.ndarray]:
        if Status(status) is not Status.CURRENT:
            where, args = self._filters(status, kinds, scope)
            rows = self._conn.execute("SELECT id, embedding FROM memories" + where, args).fetchall()
            if not rows:
                return [], _EMPTY
            return [r[0] for r in rows], np.stack([_unblob(r[1]) for r in rows])
        ids, matrix = self._current_cache()
        if kinds is None and scope is None:
            return ids, matrix
        where, args = self._filters(Status.CURRENT, kinds, scope)
        keep = {r[0] for r in self._conn.execute("SELECT id FROM memories" + where, args)}
        mask = [i for i, mid in enumerate(ids) if mid in keep]
        return [ids[i] for i in mask], matrix[mask]

    def _accesses_for(self, ids: List[str]) -> Dict[str, List[Access]]:
        out: Dict[str, List[Access]] = {}
        for start in range(0, len(ids), 500):
            ph, vals = _in_clause(ids[start:start + 500])
            for mid, at, kind in self._conn.execute(
                "SELECT memory_id, at, kind FROM accesses WHERE memory_id IN %s "
                "ORDER BY at, id" % ph,
                vals,
            ):
                out.setdefault(mid, []).append(Access(mid, at, AccessKind(kind)))
        return out

    def _refresh_last_accessed(self, ids: List[str]) -> None:
        for mid in ids:
            self._conn.execute(
                "UPDATE memories SET last_accessed_at = COALESCE("
                "(SELECT MAX(at) FROM accesses WHERE memory_id = ?), created_at) WHERE id = ?",
                (mid, mid),
            )
