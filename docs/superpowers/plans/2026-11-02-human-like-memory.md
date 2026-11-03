# Human-like Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give JARVIS a tiered, decaying, self-tuning memory that remembers recent things vividly, keeps big ideas, revises itself from ordinary conversation, and replaces the FAISS router store.

**Architecture:** A new `memory/` package (domain / ports / use_cases / adapters, mirroring `verifier/`) backed by one SQLite file. Hot path is embedding-only (Net 1 classifier, salience, recall); LLM work (Net 2 batched extraction, op resolution, sleep summaries) runs in daemon threads. A deterministic replay tuner adjusts every numeric knob from labelled events during the sleep pass.

**Tech Stack:** Python 3.9, `sqlite3`, `numpy`, `pytest`; existing `HuggingFaceAdapter` (MiniLM, 384-d) for embeddings; Groq (OpenAI client) with Ollama fallback for extraction.

**Spec:** `docs/superpowers/specs/2026-11-02-human-like-memory-design.md`

## Global Constraints

- Python 3.9 syntax only: `typing.List/Dict/Optional/Tuple`, no `X | Y`, no `match`.
- `memory/` must import with only `numpy` installed; `ollama`, `openai`, `semantic_router`, `faiss` are imported lazily inside the adapter methods that need them.
- Database path `.jarvis/memory.sqlite`; SQLite opened with `check_same_thread=False` and guarded by one `threading.RLock` per store.
- Embeddings are `np.ndarray` float32 of length 384, L2-normalised; stored as raw bytes (`arr.tobytes()`), read back with `np.frombuffer(..., dtype=np.float32)`.
- All background work in daemon threads; nothing on the voice hot path waits on an LLM except Net 1 op resolution.
- Default params, bounds, steps and fixed constants exactly as spec §9.1; decay defaults as §6.
- Tests under `tests/memory/`, no network, no model downloads; fakes from `tests/memory/conftest.py`.
- Commit after every task; message prefix `feat(memory):`, `test(memory):`, or `refactor(router):`.

## Review Focus

1. **Fresh install:** empty DB, no prototypes, no route examples: recall returns nothing, Net 1 falls through, routing returns QUICK with confidence 0, nothing crashes. (Tests in Tasks 8, 9, 4.)
2. **Clock skew:** an access timestamp in the future (negative age) must not raise or produce infinite strength; age is clamped at 0. (Task 2.)
3. **Threaded writes:** extractor thread writes while the main thread recalls; one `RLock` per store, no "database is locked" or cache staleness. (Task 3.)
4. **Sloppy LLM JSON:** Groq/Ollama returning code fences, trailing prose or unknown keys must still parse; missing required keys raise `ValueError` and the batch is retried once. (Tasks 5, 7.)
5. **Confirmation turn:** when Net 1 asked "Should I remember that?", the next utterance ("yes"/"no"/anything else) is consumed as the answer and is never itself classified or routed. (Task 9.)

---

### Task 1: Environment, package skeleton, domain entities and params

**Files:**
- Create: `requirements-dev.txt`, `memory/__init__.py`, `memory/domain/__init__.py`, `memory/domain/entities.py`, `memory/domain/params.py`, `memory/ports/__init__.py`, `memory/ports/clock.py`, `tests/memory/__init__.py`, `tests/memory/conftest.py`
- Test: `tests/memory/test_entities.py`, `tests/memory/test_params.py`

**Interfaces:**
- Produces (`entities.py`): enums `MemoryKind{EPISODE,FACT,DECISION,FEATURE,PREFERENCE,ROUTE_EXAMPLE,MEMORY_OP_EXAMPLE}`, `Status{CURRENT,SUPERSEDED,FORGOTTEN}`, `Tier{ACTIVE,DORMANT,ARCHIVED}`, `AccessKind{CREATED,RECALLED,REINFORCED,EXPLICIT}`, `EventLabel{HIT,FALSE_RECALL,MISS,STALE,EMPTY_EXTRACTION,LATE_EXTRACTION,FALSE_OP,MISSED_OP}`, `MemoryOp{STORE,FORGET,REVISE,COMPLETE,ASK,STATUS,UNDO,NONE}` (all `str, Enum`); constants `GLOBAL_SCOPE = "global"`, `def project_scope(path: str) -> str` returning `"project:" + os.path.abspath(path)`; dataclasses `Access(memory_id: str, at: float, kind: AccessKind)`, `Memory(id, kind, scope, content, embedding: np.ndarray, importance: float, status: Status, tier: Tier, created_at: float, label: Optional[str]=None, completed_at: Optional[float]=None, supersedes_id=None, superseded_by_id=None, feature_id=None, accesses: List[Access]=field(default_factory=list))` with `last_accessed_at` property (max `accesses.at`, else `created_at`), `MemoryEvent(id, at, label: EventLabel, memory_id: Optional[str], turn_id: Optional[str], context: Dict[str, Any])`, `Turn(id, session_id, at, project_scope, mode: str, user_text, assistant_text, embedding: np.ndarray, salience: float=0.0, recalled_ids: List[str]=field(default_factory=list), extracted_at: Optional[float]=None)`, `NewMemory(kind: MemoryKind, scope: str, content: str, importance: float)` (`scope` here is `"project"` or `"global"` as the extractor emits it), `Supersede(old_id, content, kind, importance)`, `ExplicitOp(turn_id: str, op: MemoryOp)`, `Extraction(new, reinforce, irrelevant, supersede, complete, forget, explicit_ops)` with `@classmethod from_dict(cls, d: dict) -> "Extraction"` (missing list keys default to empty; bad enum values raise `ValueError`) and `def is_empty(self) -> bool` (true when `new`, `supersede`, `complete`, `forget` are all empty), `OpResolution(target_ids: List[str], new: Optional[NewMemory])`.
- Produces (`params.py`): `ParamSpec(name, default, min, max, step)`; `PARAM_SPECS: Dict[str, ParamSpec]` with every row of spec §9.1 (names exactly as the table, e.g. `"decay.episode"`, `"mode_prior.quick"`); `FIXED = {"tune_window_days": 30, "min_events": 40, "improvement_margin": 0.05, "max_moves_per_pass": 3, "w_false_recall": 0.5, "w_stale": 1.0, "w_empty_extraction": 0.3, "w_late_extraction": 1.0, "w_false_op": 1.0, "w_missed_op": 0.7}`; class `Params` with `@classmethod defaults()`, `get(name) -> float`, `decay(kind: MemoryKind) -> float`, `mode_prior(mode: str) -> float` (unknown mode → `mode_prior.quick`), `with_value(name, value) -> Params` (clamped to spec bounds, returns a new instance), `is_valid() -> bool` (both invariants of §9.1), `as_dict() -> Dict[str, float]`.
- Produces (`ports/clock.py`): `class IClock(ABC)` with `now() -> float`; `class SystemClock(IClock)`.
- Produces (`conftest.py`): `FakeEmbedder` with `embed_text(text) -> List[float]` (deterministic: seed a `np.random.RandomState` from `hash(text) % 2**32`, 384-d, normalised; identical text → identical vector) and helper `vec(text) -> np.ndarray`; `FrozenClock(IClock)` with `now()`, `advance(seconds)`, `set(t)`; `FakeExtractor` (filled in Task 5; create the class now returning `Extraction.from_dict({})`); fixtures `clock`, `embedder`, `params` (defaults).

- [ ] **Step 1: Create the venv and dev requirements**

Write `requirements-dev.txt` containing `pytest>=7` and `numpy>=1.24`. Run: `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`. Confirm `.venv/` is already in `.gitignore` (it is).

- [ ] **Step 2: Write the failing tests**

`tests/memory/test_entities.py`:
```python
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
```
`tests/memory/test_params.py`:
```python
def test_defaults_match_spec():
    p = Params.defaults()
    assert p.get("decay.episode") == 0.9 and p.get("recall_k") == 6 and p.get("extract_salience_threshold") == 1.5

def test_with_value_clamps_to_bounds():
    assert Params.defaults().with_value("recall_k", 99).get("recall_k") == 15

def test_invariants():
    assert Params.defaults().is_valid()
    assert not Params.defaults().with_value("dormant_threshold", 0.8).is_valid()
    assert not Params.defaults().with_value("memory_op_ask_threshold", 0.7).is_valid()

def test_decay_and_mode_prior_lookup():
    p = Params.defaults()
    assert p.decay(MemoryKind.FEATURE) == 0.2 and p.mode_prior("think") == 1.0 and p.mode_prior("code") == 0.5
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/bin/pytest tests/memory -v`. Expected: ImportError / FAIL.

- [ ] **Step 4: Implement entities, params, clock, conftest as specified in Interfaces**

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/bin/pytest tests/memory -v`. Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements-dev.txt memory tests/memory
git commit -m "feat(memory): domain entities, params and test fakes"
```

---

### Task 2: Scoring (pure functions)

**Files:**
- Create: `memory/domain/scoring.py`
- Test: `tests/memory/test_scoring.py`

**Interfaces:**
- Consumes: Task 1 entities and `Params`.
- Produces: `effective_kind(m: Memory) -> MemoryKind`; `strength(m: Memory, now: float, p: Params) -> float` (spec §6; age in days clamped at ≥ 0); `tier_for(s: float, p: Params) -> Tier`; `scope_weight(memory_scope: str, query_scope: str, p: Params) -> float`; `cosine(a: np.ndarray, b: np.ndarray) -> float`; `recall_score(m: Memory, query_vec: np.ndarray, query_scope: str, now: float, p: Params) -> float`; `salience(max_cosine: float, mode: str, word_count: int, op_fired: bool, p: Params) -> float` (spec §6 formula).

- [ ] **Step 1: Write the failing tests**

```python
def _mem(kind, accesses_days_ago, importance=0.5, completed=None): ...  # builds Memory with Access rows at now - d*86400

def test_episode_fades_before_feature(params):
    now = 1_000_000.0
    ep, ft = _mem(MemoryKind.EPISODE, [10]), _mem(MemoryKind.FEATURE, [10])
    assert strength(ep, now, params) < strength(ft, now, params)

def test_reinforcement_raises_strength(params):
    assert strength(_mem(MemoryKind.FACT, [5, 1]), NOW, params) > strength(_mem(MemoryKind.FACT, [5]), NOW, params)

def test_tier_boundaries(params):
    assert tier_for(0.8, params) == Tier.ACTIVE and tier_for(0.79, params) == Tier.DORMANT and tier_for(0.29, params) == Tier.ARCHIVED

def test_future_access_does_not_explode(params):
    m = _mem(MemoryKind.FACT, [-3])  # access 3 days in the future
    s = strength(m, NOW, params)
    assert math.isfinite(s) and s == pytest.approx(params.get("importance_weight") * 0.5 + 1.0)

def test_completed_feature_decays_like_fact(params):
    done = _mem(MemoryKind.FEATURE, [30], completed=NOW - 86400)
    assert strength(done, NOW, params) == pytest.approx(strength(_mem(MemoryKind.FACT, [30]), NOW, params))

def test_scope_weight_order(params):
    q = project_scope("/a")
    assert scope_weight(q, q, params) > scope_weight(GLOBAL_SCOPE, q, params) > scope_weight(project_scope("/b"), q, params)

def test_salience_novel_beats_duplicate_and_scales_with_mode(params):
    assert salience(0.1, "think", 20, False, params) > salience(0.95, "think", 20, False, params)
    assert salience(0.1, "think", 20, False, params) > salience(0.1, "quick", 20, False, params)
    assert salience(0.5, "quick", 5, True, params) >= 1.0
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/pytest tests/memory/test_scoring.py -v` → ImportError.
- [ ] **Step 3: Implement `memory/domain/scoring.py`** per Interfaces; `strength` uses `max(0.0, (now - a.at) / 86400.0) + 1.0` as the base.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(memory): strength, tier, recall and salience scoring"`

---

### Task 3: SQLite store

**Files:**
- Create: `memory/ports/store.py`, `memory/adapters/__init__.py`, `memory/adapters/sqlite_store.py`
- Test: `tests/memory/test_sqlite_store.py`

**Interfaces:**
- Consumes: Task 1 entities/params, Task 2 `cosine`.
- Produces: `class IMemoryStore(ABC)` and `class SqliteMemoryStore(IMemoryStore)` with `__init__(self, path: str = ".jarvis/memory.sqlite", clock: IClock = SystemClock())` (`path=":memory:"` allowed for tests; creates parent dir; creates schema §5.1–5.6 with `schema_version = 1`). Methods:
  - memories: `add(m: Memory) -> None` (also writes a `created` access), `get(id) -> Optional[Memory]` (with accesses loaded), `update(m: Memory) -> None` (all mutable columns), `list(status: Optional[Status]=Status.CURRENT, kinds: Optional[List[MemoryKind]]=None, scope: Optional[str]=None) -> List[Memory]`, `nearest(vec: np.ndarray, k: int, kinds=None, scope=None, status=Status.CURRENT) -> List[Tuple[Memory, float]]` (cosine, descending; brute force over an in-memory `(ids, matrix)` cache rebuilt lazily after any write), `max_cosine(vec, kinds=None, scope=None) -> float` (0.0 when empty).
  - accesses: `add_access(memory_id, at, kind: AccessKind) -> None`; `move_accesses(from_id, to_id) -> None`.
  - events: `add_event(e: MemoryEvent) -> None`; `events_since(at: float) -> List[MemoryEvent]`.
  - params: `load_params() -> Params` (defaults when table empty), `save_params(p: Params, history: List[Tuple[str, float, float, float, float, str]]) -> None` where each history tuple is `(name, old, new, old_loss, new_loss, reason)`, `params_history(limit: int=10) -> List[dict]`.
  - meta: `get_meta(key) -> Optional[str]`, `set_meta(key, value: str) -> None`.
  - turns: `add_turn(t: Turn) -> None`, `pending_turns() -> List[Turn]` (`extracted_at IS NULL`, oldest first), `mark_extracted(turn_ids: List[str], at: float) -> None`, `recent_turns(session_id: str, n: int) -> List[Turn]` (newest last), `delete_turns_before(at: float, keep_session: str, keep_n: int) -> int`.
  - purge: `purge(before: float) -> int` deletes memories with `status = forgotten` or `tier = archived` whose `last_accessed_at < before`, plus their accesses and events.
  - `count_by(kind_tier_scope: bool=True) -> List[Tuple[str, str, str, int]]` for the report.

- [ ] **Step 1: Write the failing tests** (use `SqliteMemoryStore(":memory:", clock)`):

```python
def test_round_trip_all_fields(store, embedder): ...        # add → get equals, embedding allclose, accesses has one 'created'
def test_list_filters_status_kind_scope(store): ...
def test_nearest_orders_by_cosine_and_respects_scope(store, embedder): ...
def test_cache_invalidates_on_write(store, embedder):
    assert store.max_cosine(vec("hello")) == 0.0
    store.add(_m("hello", embedder)); assert store.max_cosine(vec("hello")) == pytest.approx(1.0)
def test_params_default_then_saved(store):
    assert store.load_params().get("recall_k") == 6
    store.save_params(Params.defaults().with_value("recall_k", 4), [("recall_k", 6, 4, 10.0, 8.0, "replay")])
    assert store.load_params().get("recall_k") == 4 and store.params_history()[0]["reason"] == "replay"
def test_turns_pending_and_mark_extracted(store, clock): ...
def test_recent_turns_newest_last(store): ...
def test_purge_removes_archived_and_forgotten_with_children(store, clock): ...
def test_concurrent_writer_thread(store, embedder):
    # 200 adds from a thread while main thread calls nearest() in a loop; no exception; final count == 200
```

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement `memory/ports/store.py` and `memory/adapters/sqlite_store.py`.** Schema per spec §5; `PRAGMA journal_mode=WAL`; every public method takes `self._lock`; the embedding cache is `Optional[Tuple[List[str], np.ndarray]]` set to `None` on any write.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(memory): sqlite memory store"`

---

### Task 4: Router on the new store, FAISS migration, shared embedding

**Files:**
- Modify: `memory/adapters/sqlite_store.py` (add `IVectorStore` methods), `use_cases/routing.py:14-15`, `seed_db.py:7,13`, `domain/ports.py` (no change expected; confirm)
- Create: `memory/adapters/migrate_faiss.py`
- Delete: `adapters/vector_store.py`
- Test: `tests/memory/test_router_store.py`, `tests/memory/test_migrate_faiss.py`, replace `tests/test_router.py`

**Interfaces:**
- Consumes: `IVectorStore` (`domain/ports.py`), `Utterance`, `Intent` (`domain/entities.py`), Task 3 store.
- Produces: `SqliteMemoryStore` also subclasses `IVectorStore`: `add_utterance(u: Utterance, vector: List[float])` (insert `ROUTE_EXAMPLE`, scope `GLOBAL_SCOPE`, `label = u.intent.value`, importance 0.5; or if nearest route example cosine ≥ `dedup_threshold` → `add_access(REINFORCED)`), `search_similar(vector, limit=3) -> List[Tuple[Utterance, float]]` ordered by `cosine × strength`, distance returned as `2 - 2*cosine` (squared L2 for unit vectors), `distill_utterances()` → calls the store's new `dedup(kinds: List[MemoryKind], threshold: float, now: float) -> int` with `[ROUTE_EXAMPLE]` (pairwise within (kind, scope, label) above `threshold`; keep the higher-strength memory, `move_accesses` from the loser, set the loser `SUPERSEDED` with links; returns merges). Task 10 reuses `dedup` for all kinds. `RouteCommandUseCase.execute(self, text: str, vector: Optional[List[float]] = None) -> RoutingResult` (embeds only when `vector is None`). `migrate_faiss(store: SqliteMemoryStore, metadata_path: str = ".vectorstore/metadata.json", embedder: IEmbeddingEngine) -> int` returns rows imported; no-op returning 0 if the store already has route examples or the file is absent.

- [ ] **Step 1: Write the failing tests**

`tests/memory/test_router_store.py` (uses `FakeEmbedder`, no jarvis import):
```python
def test_empty_store_routes_quick_with_zero_confidence(store, embedder):
    r = RouteCommandUseCase(store, embedder).execute("anything")
    assert r.intent == Intent.QUICK and r.confidence_score == 0.0
def test_learn_then_route_same_text(store, embedder):
    LearnFromFeedbackUseCase(store, embedder).execute("fix the bug in router.py", "code")
    r = RouteCommandUseCase(store, embedder).execute("fix the bug in router.py")
    assert r.intent == Intent.CODE and r.confidence_score == pytest.approx(1.0, abs=1e-5)
def test_execute_accepts_precomputed_vector(store, embedder): ...
def test_duplicate_utterance_reinforces_instead_of_inserting(store, embedder):
    # learn twice → one ROUTE_EXAMPLE row with two accesses beyond 'created'
def test_distill_merges_near_duplicates(store, embedder): ...  # two distinct texts, monkeypatch cosine ≥ 0.9 → one current
```
`tests/memory/test_migrate_faiss.py`: fixture JSON with 3 rows → 3 route examples with the right labels and timestamps as `created_at`; second call returns 0.

Replace `tests/test_router.py` with a 10-line file that imports only `use_cases.routing` and asserts the same three routings as today after seeding the three example sentences from `seed_db.py` through `LearnFromFeedbackUseCase` on an in-memory store. (The current file is already failing: it expects a tuple from `determine_intent` and imports `jarvis.py`.)

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement**: store methods, `migrate_faiss.py`, the `vector=` argument, switch `seed_db.py` to `SqliteMemoryStore()`, delete `adapters/vector_store.py` (`git rm`).
- [ ] **Step 4: Run** `.venv/bin/pytest tests/memory tests/test_router.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "refactor(router): route examples live in the sqlite memory store; retire FAISS"`

---

### Task 5: Extractor port and adapters

**Files:**
- Create: `memory/ports/extractor.py`, `memory/adapters/llm_json.py`, `memory/adapters/extractor_prompts.py`, `memory/adapters/groq_extractor.py`, `memory/adapters/ollama_extractor.py`, `memory/adapters/extractor_factory.py`
- Modify: `tests/memory/conftest.py` (`FakeExtractor`)
- Test: `tests/memory/test_extractors.py`

**Interfaces:**
- Produces: `class IMemoryExtractor(ABC)`: `extract(turns: List[Turn], recalled: List[Memory]) -> Extraction`; `resolve_op(op: MemoryOp, utterance: str, working_turns: List[Turn], candidates: List[Memory]) -> OpResolution`; `summarise(episodes: List[Memory]) -> List[NewMemory]`; `is_available() -> bool`.
  `llm_json.parse_json_object(text: str) -> dict` (strips ``` fences and any prose before the first `{` / after the last `}`; raises `ValueError`).
  `extractor_prompts.EXTRACT_SYSTEM`, `RESOLVE_SYSTEM`, `SUMMARISE_SYSTEM` (strings) and `render_extract_user(turns, recalled) -> str`, `render_resolve_user(op, utterance, working_turns, candidates) -> str`, `render_summarise_user(episodes) -> str` — the prompts state the JSON schema of §7.1 verbatim, instruct `scope` ∈ {`project`,`global`}, importance ∈ [0,1], and that every recalled id must appear in `reinforce` or `irrelevant`.
  `GroqMemoryExtractor(api_key: Optional[str]=None, model: str = GROQ_DEFAULT_MODEL, client=None)` (reads `GROQ_API_KEY` like `GroqTerminalParser`; `client` injectable for tests; uses `response_format={"type": "json_object"}`). `OllamaMemoryExtractor(model: str = "llama3.2:3b", chat_fn=None)` (`chat_fn` defaults to `ollama.chat` imported lazily; `format="json"`). `build_extractor() -> IMemoryExtractor` returns Groq when available else Ollama.
  `FakeExtractor(responses: List[Extraction]=None)` in conftest: pops responses in order (last one repeats), records calls in `.calls`; `resolve_op` returns `self.resolution` (default: first candidate id, no new); `summarise` returns `self.summaries` (default `[]`).

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_json_tolerates_fences_and_prose():
    assert parse_json_object("Sure!\n```json\n{\"new\": []}\n```\nDone.") == {"new": []}
def test_parse_json_raises_on_garbage():
    with pytest.raises(ValueError): parse_json_object("no json here")
def test_groq_extract_roundtrip_with_fake_client():
    client = _fake_openai_client('{"new":[{"kind":"fact","scope":"global","content":"x","importance":0.4}],"reinforce":["a"]}')
    e = GroqMemoryExtractor(api_key="k", client=client).extract([_turn("hi")], [_mem_with_id("a")])
    assert e.new[0].content == "x" and e.reinforce == ["a"]
    assert client.last_kwargs["response_format"] == {"type": "json_object"}
def test_groq_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False); assert GroqMemoryExtractor().is_available() is False
def test_ollama_extract_uses_json_format():
    calls = []; def chat(**kw): calls.append(kw); return {"message": {"content": "{\"new\": []}"}}
    OllamaMemoryExtractor(chat_fn=chat).extract([_turn("hi")], []); assert calls[0]["format"] == "json"
def test_factory_prefers_groq_when_key_set(monkeypatch): ...
def test_extract_prompt_lists_recalled_ids(): assert "a" in render_extract_user([_turn("hi")], [_mem_with_id("a")])
def test_resolve_op_returns_targets(): ...  # fake client → OpResolution(target_ids=["m1"], new=None)
```

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** the port, prompts, parser, both adapters, factory, `FakeExtractor`.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(memory): extractor port with Groq and Ollama adapters"`

---

### Task 6: Extraction buffer (salience accumulation and flush triggers)

**Files:**
- Create: `memory/use_cases/__init__.py`, `memory/use_cases/extraction_buffer.py`
- Test: `tests/memory/test_extraction_buffer.py`

**Interfaces:**
- Consumes: Task 3 store (`add_turn`, `pending_turns`), Task 2 `salience`, `Params`, `IClock`.
- Produces: `class ExtractionBuffer` with `__init__(store: IMemoryStore, params_provider: Callable[[], Params], clock: IClock, on_flush: Callable[[List[Turn], str], None])` (`on_flush(turns, reason)`), `add_turn(turn: Turn, max_cosine: float, op_fired: bool) -> float` (computes and sets `turn.salience` with `len(turn.user_text.split())`, persists via `store.add_turn`, adds to `self.total`, then checks triggers in order `salience`, `max_turns`), `tick() -> None` (called by the caller's timer or before recall; flushes with reason `"idle"` when `clock.now() - last_turn_at ≥ extract_idle_seconds` and the buffer is non-empty), `flush(reason: str) -> None` (no-op if nothing pending; calls `on_flush` with `store.pending_turns()`, resets `total`), `pending_count() -> int`. Flush reasons: `"salience"`, `"max_turns"`, `"idle"`, `"mode_switch"`, `"session_end"`, `"sleep"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_flushes_when_salience_crosses_threshold(buffer, flushed):
    buffer.add_turn(_turn("novel idea one"), max_cosine=0.0, op_fired=False)  # salience 1.0*1.0+bonus<1.5
    assert flushed == []
    buffer.add_turn(_turn("novel idea two"), max_cosine=0.0, op_fired=False)
    assert flushed[0][1] == "salience" and len(flushed[0][0]) == 2
def test_low_salience_turns_wait_for_idle(buffer, flushed, clock):
    for i in range(3): buffer.add_turn(_turn("ok", mode="quick"), max_cosine=0.95, op_fired=False)
    buffer.tick(); assert flushed == []
    clock.advance(91); buffer.tick(); assert flushed[0][1] == "idle"
def test_new_turn_resets_idle_timer(buffer, flushed, clock): ...
def test_max_turns_cap(buffer, flushed): ...   # 10 near-duplicate turns → one flush "max_turns"
def test_flush_empty_is_noop(buffer, flushed): buffer.flush("sleep"); assert flushed == []
def test_total_resets_after_flush(buffer): ...
```

- [ ] **Step 2: Run to verify fail.** — [ ] **Step 3: Implement.** — [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(memory): salience-gated extraction buffer"`

---

### Task 7: Capture batch use case (Net 2 application)

**Files:**
- Create: `memory/use_cases/capture.py`
- Test: `tests/memory/test_capture.py`

**Interfaces:**
- Consumes: store, `IMemoryExtractor`, `IEmbeddingEngine`, `Params`, `IClock`, `tier_for`/`strength`.
- Produces: `class CaptureBatchUseCase` with `__init__(store, extractor, embedder, params_provider, clock, on_profile_change: Optional[Callable[[], None]] = None, id_factory: Callable[[], str] = uuid4-str)` and `execute(turns: List[Turn], project_scope: str, buffer_salience: float) -> Optional[Extraction]`: loads `recalled` = union of `turn.recalled_ids`; calls `extractor.extract`. On `ValueError` or any exception: retries are tracked in `meta` under key `f"retry:{turns[0].id}"`; first failure → leave turns pending and return `None`; second failure → `mark_extracted`, clear the key, return `None`. On success apply items exactly per spec §7.1 bullet list, using a public helper `apply_new(nm: NewMemory, project_scope: str, explicit: bool, turn_id: Optional[str]) -> Memory` shared with Task 9 (`explicit=True` forces importance 1.0 and `AccessKind.EXPLICIT`; resolves `nm.scope == "project"` to `project_scope`). Calls `on_profile_change()` once per batch if any applied `new`/`supersede`/`complete`/`forget` touched a `PREFERENCE` or `FEATURE`. Logs `EventLabel.EMPTY_EXTRACTION` with `context={"buffer_salience": …}` when `extraction.is_empty()`. Marks turns extracted. Returns the `Extraction`.

- [ ] **Step 1: Write the failing tests**

```python
def test_new_inserts_with_project_scope_resolved(capture, store): ...     # scope "project" → "project:/abs/path"
def test_new_dedups_onto_existing_and_reinforces(capture, store, embedder): ...  # same text already stored → no new row, +1 access
def test_new_dedup_onto_dormant_logs_miss(capture, store, clock): ...
def test_reinforce_logs_hit_and_irrelevant_logs_false_recall(capture, store): ...
def test_supersede_links_both_ways_and_logs_stale_when_recalled(capture, store): ...
def test_complete_sets_completed_at(capture, store, clock): ...
def test_forget_marks_forgotten_and_logs_false_recall_if_recalled(capture, store): ...
def test_explicit_ops_add_prototype_and_log_missed_op(capture, store): ...
def test_empty_extraction_logged_with_salience(capture, store): ...
def test_profile_change_callback_fires_only_for_preference_or_feature(store, embedder, clock): ...
def test_failure_retries_once_then_drops(store, embedder, clock):
    ext = FakeExtractor(); ext.raise_next = 2
    uc = CaptureBatchUseCase(store, ext, embedder, lambda: Params.defaults(), clock)
    uc.execute([t], scope, 1.0); assert store.pending_turns()        # still pending
    uc.execute([t], scope, 1.0); assert not store.pending_turns()    # dropped
```

- [ ] **Step 2: Run to verify fail.** — [ ] **Step 3: Implement** (`FakeExtractor` gains `raise_next: int`). — [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(memory): capture batch use case applies extractions and labels events"`

---

### Task 8: Recall, session working set, profile block, prompt renderer

**Files:**
- Create: `memory/use_cases/recall.py`, `memory/adapters/prompt_renderer.py`
- Test: `tests/memory/test_recall.py`, `tests/memory/test_prompt_renderer.py`

**Interfaces:**
- Consumes: store, scoring, `Params`, `IClock`.
- Produces:
  `@dataclass RecallResult(memories: List[Memory], newly_added: List[Memory], working_turns: List[Turn], profile: "Profile")`; `@dataclass Profile(preferences: List[Memory], features: List[Memory], version: int)`.
  `class SessionWorkingSet`: `members() -> List[Memory]` (insertion order), `merge(candidates: List[Memory], patience: int) -> List[Memory]` (returns newly added; increments a miss counter for members not in `candidates`, evicts at `patience`), `reset()`.
  `class RecallUseCase` with `__init__(store, params_provider, clock, session_id: str)` and:
  - `profile(project_scope: str) -> Profile` (cached; `invalidate_profile()` bumps version; built from top `profile_preferences` global `PREFERENCE` by strength + current project `FEATURE` with `completed_at is None` by strength).
  - `execute(query_vec: np.ndarray, project_scope: str, mode: str, include_dormant: bool = False) -> RecallResult` implementing §7.2 steps 1–6 with budgets: `"quick"` → `budget_quick`, `"think"` → `recall_k`, `"code"` → all current project `FEATURE`/`DECISION` plus `recall_k` others; `include_dormant=True` (for `ask`) skips the floor gate and budget and returns the top `recall_k` across active+dormant with their tiers.
  - `last_recalled_ids() -> List[str]`; public attribute `working_set: SessionWorkingSet`.
  `prompt_renderer.render_profile(profile: Profile) -> str`, `render_memories(memories: List[Memory], project_scope: str) -> str` (`"- (decision, this project) …"` / `"- (preference) …"` / `"- (fact, other project) …"`; empty string when none), `render_turns(turns: List[Turn]) -> str`, `render_system_prompt(base: str, profile, memories, turns, project_scope) -> str` (order: base → profile → memories → turns; sections omitted when empty), `render_code_prefix(profile, memories, project_scope) -> str` (`## Project context` block, empty string when nothing to say).

- [ ] **Step 1: Write the failing tests**

```python
def test_empty_store_returns_nothing(recall): r = recall.execute(vec("x"), SCOPE, "think"); assert r.memories == [] and r.profile.features == []
def test_top_k_from_active_only_and_archived_never(recall, store, clock): ...
def test_dormant_only_above_similarity_bar(recall, store, clock): ...
def test_scope_ordering_at_equal_similarity(recall, store, embedder): ...   # same content in three scopes → project, global, other
def test_gate_blocks_below_floor(recall, store, embedder): ...
def test_budget_quick_is_two(recall, store): ...
def test_working_set_keeps_order_appends_and_evicts(params):
    ws = SessionWorkingSet(); a,b,c = _m("a"),_m("b"),_m("c")
    assert ws.merge([a,b], 2) == [a,b]; assert ws.merge([b,c], 2) == [c]; assert [m.id for m in ws.members()] == ["a","b","c"]
    ws.merge([b,c], 2); assert [m.id for m in ws.members()] == ["b","c"]   # a evicted after 2 misses
def test_access_appended_only_on_first_entry(recall, store): ...
def test_profile_excluded_from_recall_and_cached_until_invalidated(recall, store): ...
def test_code_mode_includes_all_project_features_and_decisions(recall, store): ...
def test_include_dormant_returns_tiers_without_gate(recall, store, clock): ...
```
`test_prompt_renderer.py`: section order; empty sections omitted; scope labels; `render_code_prefix` empty when nothing.

- [ ] **Step 2: Run to verify fail.** — [ ] **Step 3: Implement.** — [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(memory): gated, budgeted recall with session working set and profile block"`

---

### Task 9: Memory operations (Net 1) with learning loop

**Files:**
- Create: `memory/adapters/memory_op_seeds.py`, `memory/use_cases/memory_ops.py`
- Test: `tests/memory/test_memory_ops.py`

**Interfaces:**
- Consumes: store, `IMemoryExtractor.resolve_op`, `RecallUseCase`, `CaptureBatchUseCase.apply_new`, `Params`, `IClock`.
- Produces:
  `memory_op_seeds.SEEDS: Dict[MemoryOp, List[str]]` — ≥ 20 phrasings for each of STORE, FORGET, REVISE, COMPLETE, ASK, STATUS, UNDO and ≥ 20 for NONE (ordinary questions/requests that should fall through: "what time is it in Tokyo", "write a python script to list files", …). `seed_memory_ops(store, embedder, clock) -> int` inserts them as `MEMORY_OP_EXAMPLE` (global, importance 0.5, `label = op.value`) if the store has none.
  `@dataclass OpDecision(op: MemoryOp, score: float, action: str)` with `action ∈ {"execute", "ask", "fallthrough"}`.
  `class ClassifyMemoryOpUseCase(store, params_provider, clock)`: `execute(vec: np.ndarray, only: Optional[List[MemoryOp]] = None) -> OpDecision` — nearest `memory_op_k` prototypes (kind `MEMORY_OP_EXAMPLE`), weight `cosine × strength`, vote by label; score = winning share × best cosine of that label; `NONE` or empty store → `fallthrough`.
  `@dataclass OpOutcome(spoken: str, consumed: bool, pending_question: Optional["PendingAsk"])` and `@dataclass PendingAsk(op, utterance, vec, candidates)`.
  `class HandleMemoryOpUseCase(store, extractor, recall, capture, embedder, params_provider, clock, project_scope_provider: Callable[[], str])` with:
  - `handle(utterance: str, vec: np.ndarray, working_turns: List[Turn]) -> OpOutcome` — if a `PendingAsk` exists, interpret `utterance` as the answer (yes-words `{"yes","yeah","yep","sure","do it","please","correct"}` / anything else = no), learn (§7.3 table rows 1–2), execute or drop, clear pending, return `consumed=True`. Else if `self.last_op` was executed on the previous turn, classify with `only=[UNDO]`; on execute → revert (`un-forget`: status back to CURRENT; `un-store`: status FORGOTTEN; `un-supersede`: swap statuses and clear links), negative prototype, `FALSE_OP`. Else classify; `execute` → `_run(op, …)`; `ask` → set pending, return the question text; `fallthrough` → `consumed=False`.
  - `_run(op, utterance, vec, working_turns) -> str` per §7.3 table (candidates for resolution = `recall.last_recalled_ids()` ∪ `recall.working_set.members()` ∪ `store.nearest(vec, 5)`); `ASK` uses `recall.execute(..., include_dormant=True)` and logs `MISS` when the best is dormant/archived or below `recall_similarity_floor`. When logging a `MISS` (here or in `STORE` dedup), if any `store.pending_turns()` has `cosine(turn.embedding, vec) ≥ recall_similarity_floor`, log `LATE_EXTRACTION` as well with `context={"buffer_salience": sum of pending salience}`. `STATUS` returns `self.status_line_provider()` (injected `Callable[[], str]`, Task 12 supplies it).
  - `note_turn_passed()` — called by the main loop for every non-consumed turn so the undo window closes.
  - `learn(utterance, vec, op: MemoryOp, positive: bool, label: Optional[EventLabel])`.

- [ ] **Step 1: Write the failing tests**

```python
def test_seeds_every_op_has_at_least_twenty(): ...
def test_classifier_empty_store_falls_through(classifier): assert classifier.execute(vec("remember this")).action == "fallthrough"
def test_classifier_executes_above_threshold(seeded, classifier, embedder):
    d = classifier.execute(embedder.vec(SEEDS[MemoryOp.FORGET][0])); assert d.op == MemoryOp.FORGET and d.action == "execute"
def test_classifier_none_label_falls_through(seeded, classifier, embedder): ...
def test_classifier_medium_score_asks(seeded, classifier, params_mid): ...   # params with memory_op_threshold raised to 0.95
def test_store_op_writes_importance_one_and_explicit_access(handler, store): ...
def test_forget_resolves_target_and_reads_back(handler, store): ...
def test_revise_supersedes_and_logs_stale_when_recalled(handler, store): ...
def test_complete_sets_completed_at(handler, store, clock): ...
def test_ask_speaks_top_and_logs_miss_on_dormant(handler, store, clock): ...
def test_ask_logs_late_extraction_when_answer_is_still_pending(handler, store): ...
def test_ask_flow_yes_learns_positive_and_executes(handler, store): ...
def test_ask_flow_no_learns_none_and_logs_false_op(handler, store): ...
def test_confirmation_turn_is_consumed_even_if_it_looks_like_a_command(handler): ...  # answer "no, forget it" → treated as "no"
def test_undo_after_forget_restores_and_logs_false_op(handler, store): ...
def test_undo_window_closes_after_one_turn(handler, store): ...
def test_fallthrough_not_consumed(handler): assert handler.handle("what time is it", vec("what time is it"), []).consumed is False
```

- [ ] **Step 2: Run to verify fail.** — [ ] **Step 3: Implement.** — [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(memory): semantic memory operations with confirmation, undo and prototype learning"`

---

### Task 10: Sleep pass

**Files:**
- Create: `memory/use_cases/consolidate.py`
- Modify: `memory/adapters/sqlite_store.py` (if `dedup` from Task 4 needs the `label` grouping)
- Test: `tests/memory/test_sleep.py`

**Interfaces:**
- Consumes: store (`dedup`, `purge`, `list`, `update`, `move_accesses`, meta), `IMemoryExtractor.summarise`, `ExtractionBuffer.flush`, `RecallUseCase.invalidate_profile`, `TuneParamsUseCase` (Task 11; accept any object with `run() -> dict`), `Params`, `IClock`.
- Produces: `@dataclass SleepReport(rescored: int, summarised: int, deduped: int, purged: int, tune: dict, dry_run: bool)`; `class SleepUseCase(store, extractor, buffer: Optional[ExtractionBuffer], tuner, params_provider, clock, on_done: Optional[Callable[[], None]] = None)` with `is_due() -> bool` (meta `last_sleep_at` absent or older than `sleep_interval_hours`) and `run(dry_run: bool = False) -> SleepReport` implementing §8 steps 1–7; dry-run uses `store.copy_in_memory()` (add to store: `copy_in_memory() -> SqliteMemoryStore` via `sqlite3.Connection.backup`). On exception: log, do **not** write `last_sleep_at`, re-raise.

- [ ] **Step 1: Write the failing tests**

```python
def test_is_due_when_never_slept_and_after_interval(sleep, store, clock): ...
def test_pending_buffer_flushed_first(sleep, buffer_spy): ...
def test_rescore_moves_old_episode_to_archived(sleep, store, clock): ...
def test_episodes_fold_into_summary_inheriting_accesses(sleep, store, extractor, clock):
    # 3 episodes 10 days old in scope S; extractor.summaries=[NewMemory(FEATURE,"project","built the reviewer",0.8)]
    # → one FEATURE current/active in S; episodes tier ARCHIVED with feature_id set; summary has ≥ accesses of newest episode
def test_fewer_than_three_episodes_not_summarised(sleep, store): ...
def test_dedup_keeps_stronger_across_kinds(sleep, store): ...
def test_purge_respects_retention(sleep, store, clock): ...
def test_profile_invalidated_and_last_sleep_written(sleep, store, recall_spy, clock): ...
def test_dry_run_writes_nothing(sleep, store): ...
def test_exception_leaves_last_sleep_unset(sleep, store, extractor): ...
```

- [ ] **Step 2: Run to verify fail.** — [ ] **Step 3: Implement.** — [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(memory): sleep pass with summarise, dedup, purge"`

---

### Task 11: Replay tuner

**Files:**
- Create: `memory/use_cases/tune.py`
- Test: `tests/memory/test_tune.py`

**Interfaces:**
- Consumes: store (`events_since`, `get`, `load_params`, `save_params`), scoring, `PARAM_SPECS`, `FIXED`, `IClock`.
- Produces: `class TuneParamsUseCase(store, clock)` with `loss(p: Params, events: List[MemoryEvent], now: float) -> float` (spec §9.3 cost table; per event it reads `context` keys `tier_at_time`, `rank_at_time`, `similarity`, `mode`, `buffer_salience`, `classifier_score`, `budget_rank`; "would still be recalled" = recomputed tier is ACTIVE **and** `similarity ≥ recall_similarity_floor` **and** `rank_at_time < budget(mode)`), and `run() -> dict` returning `{"status": "insufficient_data"|"tuned"|"no_improvement", "events": n, "moves": [{"name","old","new","old_loss","new_loss"}]}` implementing §9.3 algorithm with `FIXED` constants and `Params.is_valid()`.
- Event writers in Tasks 7–9 must have populated `context` with those keys; Task 11 adds a `tests/memory/helpers.py::event(label, memory, **context)` builder used here and in the end-to-end test.

- [ ] **Step 1: Write the failing tests**

```python
def test_insufficient_data_below_min_events(tuner, store): assert tuner.run()["status"] == "insufficient_data"
def test_missed_decisions_lower_decision_decay(tuner, store, clock):
    # 50 MISS events on DECISION memories whose only access is 20 days old (dormant under defaults)
    r = tuner.run(); assert any(m["name"] == "decay.decision" and m["new"] < m["old"] for m in r["moves"])
def test_irrelevant_episodes_raise_episode_decay_or_tighten_recall(tuner, store, clock): ...
def test_empty_extractions_raise_salience_threshold(tuner, store): ...
def test_late_extractions_lower_salience_threshold(tuner, store): ...
def test_false_ops_raise_op_threshold(tuner, store): ...
def test_missed_ops_lower_ask_threshold(tuner, store): ...
def test_bounds_and_invariants_hold_after_many_passes(tuner, store): ...   # run 10 times; every param within spec bounds; is_valid()
def test_margin_prevents_flip_flop(tuner, store): ...                       # second run on same events → no_improvement
def test_at_most_three_moves_per_pass(tuner, store): ...
def test_history_written_with_losses(tuner, store): ...
```

- [ ] **Step 2: Run to verify fail.** — [ ] **Step 3: Implement.** — [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(memory): counterfactual replay tuner"`

---

### Task 12: Memory service facade and CLI

**Files:**
- Create: `memory/service.py`, `memory/cli.py`, `memory/__main__.py`
- Test: `tests/memory/test_service.py`, `tests/memory/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `class MemoryService` — the single object `jarvis.py` talks to:
  - `__init__(embedder: IEmbeddingEngine, project_dir: str, db_path: str = ".jarvis/memory.sqlite", extractor: Optional[IMemoryExtractor] = None, clock: IClock = SystemClock(), store: Optional[SqliteMemoryStore] = None, start_threads: bool = True)` — wires store, params provider (`store.load_params`, cached and refreshed after sleep), buffer (with `on_flush` → daemon thread running `CaptureBatchUseCase` then `store.delete_turns_before(now - 86400, session_id, working_turns)`), recall (`session_id = uuid4`), capture (`on_profile_change=recall.invalidate_profile`), classifier/handler (`status_line_provider=self.status_line`), sleep, tuner; runs `migrate_faiss` and `seed_memory_ops` once; if `start_threads` and sleep `is_due()` → daemon thread `sleep.run()`; starts an idle-tick daemon thread calling `buffer.tick()` every 5 s. When `start_threads=False`, `on_flush` runs capture synchronously (tests).
  - `embed(text) -> np.ndarray`
  - `try_memory_op(transcript: str, vec: np.ndarray) -> Optional[str]` (returns spoken reply when consumed, else `None`)
  - `recall_for_prompt(vec, mode: str) -> RecallResult` and `system_prompt(base: str, result: RecallResult) -> str`, `code_prefix(result) -> str`
  - `record_turn(user_text, assistant_text, mode, vec, recalled: RecallResult, op_fired: bool) -> None` (builds `Turn`, `max_cosine` via store, feeds buffer)
  - `record_coding_turn(prompt: str, accepted: int, rejected: int, files: List[str], cwd: str) -> None` (builds the coding `Turn`, flushes with `"session_end"`)
  - `on_mode_switch_to_code()` → `buffer.flush("mode_switch")`
  - `status_line() -> str` (spec §9.4 sentence) and `report() -> str` (plain text, in this order: counts by kind × tier × scope; per-kind hit/false-recall/miss/stale rates over `tune_window_days`; extraction stats (batches = distinct `turn_id` groups with `extracted_at`, mean turns per batch, empty-extraction rate, late-extraction count); classifier stats (`FALSE_OP`/`MISSED_OP` counts); params vs defaults with a `*` on moved ones; last 10 `params_history` rows; `last_sleep_at` and next due).
  `cli.main(argv: List[str]) -> int` with subcommands `report`, `migrate`, `sleep [--dry-run]`, `reset-params`; constructs `MemoryService(start_threads=False)` with the real `HuggingFaceAdapter` only for `migrate` (lazy import), otherwise a store-only path (report/sleep/reset need no embedder except sleep's dedup, which uses stored vectors). `__main__.py` calls `sys.exit(main(sys.argv[1:]))`.

- [ ] **Step 1: Write the failing tests** (service built with `FakeEmbedder`, `FakeExtractor`, `FrozenClock`, `start_threads=False`, `store=SqliteMemoryStore(":memory:")`):

```python
def test_first_start_seeds_ops_and_migrates(tmp_path, ...): ...
def test_try_memory_op_returns_none_for_ordinary_text(service): ...
def test_record_turn_then_recall_shows_it_after_flush(service): ...   # force flush via service.buffer.flush("idle") and run capture synchronously
def test_record_coding_turn_flushes_session_end(service): ...
def test_status_line_mentions_counts(service): assert "memories" in service.status_line()
def test_cli_report_runs_on_empty_db(tmp_path, capsys): assert main(["report", "--db", str(tmp_path/"m.sqlite")]) == 0
def test_cli_reset_params_writes_history(tmp_path): ...
def test_cli_sleep_dry_run_changes_nothing(tmp_path): ...
```
(`cli` accepts `--db PATH` on every subcommand for testability; default `.jarvis/memory.sqlite`.)

- [ ] **Step 2: Run to verify fail.** — [ ] **Step 3: Implement.** — [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(memory): MemoryService facade and CLI"`

---

### Task 13: Wire into `jarvis.py` and `AgentManager`

**Files:**
- Modify: `jarvis.py:77-86` (construction), `jarvis.py:93-100` (`run_antigravity_coding_task` takes `prefix: str = ""`), `jarvis.py:117-135` (`determine_intent(text, vector=None)`), `jarvis.py:460-540` (main loop order per spec §7), `agent_adapter/manager.py:45-49,201-228` (new `on_coding_turn` callback), `verifier/use_cases/coordinator.py` (`last_completed_session`)
- Test: `tests/test_agent_adapter.py` (add one test), `tests/memory/test_coordinator_summary.py`

**Interfaces:**
- Consumes: `MemoryService` (Task 12).
- Produces: `ReviewSessionCoordinator.last_completed_session: Optional[ReviewSession]` set in every branch that sets `active_session = None` after completion (`trigger_review` no-diff branch, `_advance_and_notify`, `accept_all`, `reject_all`). `AgentManager.__init__(..., on_coding_turn: Optional[Callable[[str, int, int, List[str], str], None]] = None)`; `start_session` stores `self._last_prompt = prompt` (not for continuations); `_finish_review` calls `on_coding_turn(self._last_prompt, accepted, rejected, files, cwd)` using the accepted/rejected count properties on `ReviewSession` (`verifier/domain/entities.py:53,57`) and `[fd.file_path for fd in file_diffs]`, with zeros when `last_completed_session` is `None`.
  `jarvis.py`: `memory_service = MemoryService(embedding_engine, project_dir)` built after `embedding_engine`; `vector_store = memory_service.store`; `agent_manager = AgentManager(on_speech=speak, on_coding_turn=memory_service.record_coding_turn)`. Main loop after the two existing agent checks: `vec = memory_service.embed(transcript)`; `reply = memory_service.try_memory_op(transcript, vec)`; if `reply`: `speak(reply); continue`; `determine_intent(transcript, vector=vec.tolist())` (ruling: the shared vector is of the raw transcript, not `clean_text`; MiniLM is punctuation-insensitive enough and one embedding is the point); route confirmation unchanged; if `mode_key == "code"`: `memory_service.on_mode_switch_to_code()`; `rr = memory_service.recall_for_prompt(vec, "code")`; `run_antigravity_coding_task(transcript, prefix=memory_service.code_prefix(rr))`; `continue`. Else `rr = memory_service.recall_for_prompt(vec, mode_key)`; `messages = [{"role": "system", "content": memory_service.system_prompt(SYSTEM_PROMPTS[mode_key], rr)}, {"role": "user", "content": transcript}]`; accumulate the streamed reply into `full_reply`; after the stream: `memory_service.record_turn(transcript, full_reply, mode_key, vec, rr, op_fired=False)`.
  `run_antigravity_coding_task(prompt, prefix="")` passes `prefix + prompt` to `start_session`.

- [ ] **Step 1: Write the failing tests**

`tests/memory/test_coordinator_summary.py` (fake diff engine + presenter): after `accept_all`, `last_completed_session.accepted_count == n` and `active_session is None`; after `trigger_review` with no diffs, `last_completed_session` is set.
`tests/test_agent_adapter.py::test_finish_review_reports_coding_turn`: `AgentManager(on_speech=None, on_coding_turn=spy)`, set `_last_prompt`, a coordinator stub with `last_completed_session` of 2 accepted / 1 rejected over `["a.py"]` → `_finish_review()` → `spy` called with `("p", 2, 1, ["a.py"], cwd)`.

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** the coordinator field, the manager callback, and the `jarvis.py` edits. Keep every existing print/audit line; add `print(f"[Memory] {len(rr.memories)} recalled, {len(rr.newly_added)} new")` after recall.
- [ ] **Step 4: Run** `.venv/bin/pytest tests/memory tests/test_router.py tests/test_agent_adapter.py -v` → PASS. Then `python -c "import ast,sys; ast.parse(open('jarvis.py').read())"` → no SyntaxError (jarvis.py cannot be imported without audio models).
- [ ] **Step 5: Commit** — `git commit -m "feat(memory): wire MemoryService into the voice loop and coding sessions"`

---

### Task 14: End-to-end test and documentation

**Files:**
- Create: `tests/memory/test_end_to_end.py`
- Modify: `read.me` (new "Memory" section), `.gitignore` (no change expected; confirm `.jarvis/` present)

**Interfaces:** consumes `MemoryService` with fakes.

- [ ] **Step 1: Write the end-to-end test** following spec §11 last row:

```python
def test_peer_memory_lifecycle(service, store, extractor, clock):
    # 1. two brainstorm-like turns (novel text, mode think) → buffer flushed with reason "salience"
    # 2. extractor returns NewMemory(DECISION, "project", "We use FAISS for routing", 0.8) → capture → memory present
    # 3. recall for a related query shows it in working set; newly_added has it; a second recall adds no new access
    # 4. Net 1: "we're not using FAISS anymore" with extractor.resolution → OpResolution([id], NewMemory(DECISION,"project","We use SQLite for routing",0.8))
    #    → old SUPERSEDED, new CURRENT, STALE event logged, spoken reply mentions SQLite
    # 5. recall shows the replacement only
    # 6. three low-salience quick turns → nothing flushed until clock.advance(91); tick → flushed "idle"
    # 7. clock.advance(8 days); extractor.summaries=[...]; service.sleep.run() → episodes archived, summary active, tune status "insufficient_data"
    # 8. 50 synthetic MISS events on decisions → service.sleep.run() → len(moves) ≤ 3 and decay.decision decreased
```

- [ ] **Step 2: Run** → PASS (fix anything it exposes in the owning module, with its own unit test).
- [ ] **Step 3: Document** in `read.me`: what memory does, the two scopes, voice phrasings by op (one example each), `python -m memory report|sleep|migrate|reset-params`, where the DB lives, and that `GROQ_API_KEY` selects the extractor.
- [ ] **Step 4: Run the full suite** `.venv/bin/pytest tests/memory tests/test_router.py tests/test_agent_adapter.py -v` and paste the summary line into the commit body.
- [ ] **Step 5: Commit** — `git commit -m "test(memory): end-to-end lifecycle; docs: memory section"`

---

## Task dependency graph

```
1 → 2 → 3 → 4
          3 → 5 → 7
          3 → 6
          3 → 8
    5,7,8 → 9
    4,5 → 10 ← 11 (10 only needs tuner's run(); 11 needs store+events)
    6..11 → 12 → 13 → 14
```
Independent after Task 3: {4}, {5}, {6}, {8}; after 5: {7}; {11} is independent of 7–10 once Task 3 exists. These may be dispatched in parallel.
