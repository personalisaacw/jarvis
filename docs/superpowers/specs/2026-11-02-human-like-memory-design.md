# Human-like memory for JARVIS

**Date:** 2026-11-02
**Branch:** `feature/human-like-memory` (from `origin/dev`)
**Status:** Draft for review

## 1. Goal

Give JARVIS the memory of a peer programmer you have worked with for a while:

- recent things are vivid, old small things fade and are eventually thrown away;
- big ideas (features in progress, decisions, preferences) persist and get stronger every time they come up;
- ordinary conversation revises memory ("no, we switched to Postgres" replaces the old fact);
- it knows which project it is in, weights that project's memory highest, but can still recall "we solved this in the other repo";
- the numbers that govern fading, recall and consolidation are **measured and tuned automatically** from how the user actually reacts, not hand-edited.

The same engine replaces the router's FAISS training store so route examples also get recency, reinforcement and de-duplication.

## 2. Current state (what this changes)

| Today | Where | After |
|---|---|---|
| Router store: FAISS `IndexFlatL2` + `.vectorstore/metadata.json` with `{text, intent, timestamp}`; `timestamp` unused; nothing decays or merges; `distill_utterances()` is a stub | `adapters/vector_store.py`, `use_cases/routing.py` | Route examples live in the memory DB as `route_example` memories; strength-weighted search; dedup at sleep; FAISS retired with a one-time migration |
| Quick/Think prompts send `[system, transcript]` only; no conversation history | `jarvis.py:503-506` | Prompt gains a "what you remember" block + last *n* turns of this session |
| Coding tasks get the raw transcript as prompt; context lives only inside `agy -c` | `jarvis.py:93-100`, `agent_adapter/manager.py` | Prompt is prefixed with project memory (features in progress, standing decisions) |
| No memory voice commands | — | `MemoryVoiceGrammar` intercepts "remember / forget / what do you remember" before route confirmation |
| Tests: `tests/test_router.py` covers routing only | `tests/` | Those tests keep passing unchanged; new suite under `tests/memory/` |

Ports `IVectorStore` and `IEmbeddingEngine` (`domain/ports.py`) and `RouteCommandUseCase` / `LearnFromFeedbackUseCase` interfaces are **kept**; only their store implementation changes.

## 3. Out of scope

- Any UI (the FastAPI dashboard is untouched; visibility is CLI + voice only).
- Multi-user memory. One user, one database.
- Changing the training-mode route confirmation flow (every utterance still asks "quick, think, or code?"). Memory commands bypass it; nothing else changes.
- Cloud sync of the database.
- Tuning the extractor prompt automatically (the tuner moves numeric params only).

## 4. Architecture

Package layout mirrors `verifier/`:

```
memory/
  __init__.py
  __main__.py              # python -m memory <report|migrate|sleep>
  cli.py
  domain/
    entities.py            # Memory, MemoryKind, Scope, Tier, Status, Link, MemoryEvent, EventLabel
    params.py              # ParamSpec (default/min/max/step), DEFAULT_PARAMS
    scoring.py             # pure: strength(), tier_for(), recall_score()
  ports/
    store.py               # IMemoryStore
    extractor.py           # IMemoryExtractor
    clock.py               # IClock (so sleep/tuner tests can freeze time)
  use_cases/
    capture_turn.py        # CaptureTurnUseCase
    recall.py              # RecallUseCase
    memory_command.py      # HandleMemoryCommandUseCase
    consolidate.py         # SleepUseCase (steps 1–4 of §8)
    tune.py                # TuneParamsUseCase (§9)
  adapters/
    sqlite_store.py        # SqliteMemoryStore(IMemoryStore, IVectorStore)
    groq_extractor.py      # GroqMemoryExtractor
    ollama_extractor.py    # OllamaMemoryExtractor
    extractor_factory.py   # Groq if key present, else Ollama (same pattern as GroqTerminalParser.is_available())
    voice_grammar.py       # MemoryVoiceGrammar
    prompt_renderer.py     # renders recalled memories into prompt text
    migrate_faiss.py       # .vectorstore/metadata.json -> route_example rows
```

Embeddings reuse the existing `HuggingFaceAdapter` (`adapters/embeddings.py`, MiniLM-L6-v2, 384-d, normalised). Database file: `.jarvis/memory.sqlite` (`.jarvis/` is already git-ignored).

Dependencies added: none beyond the standard library `sqlite3` and `numpy` (already present). The `faiss` dependency is removed.

## 5. Data model

### 5.1 `memories`

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | uuid4 |
| `kind` | TEXT | `episode`, `fact`, `decision`, `feature`, `preference`, `route_example` |
| `scope` | TEXT | `global` or `project:<absolute repo path>` |
| `content` | TEXT | 1–2 sentences written by the extractor; for `route_example` the raw utterance |
| `intent` | TEXT NULL | only for `route_example`: `quick`/`think`/`code` |
| `importance` | REAL | 0–1; explicit "remember that" sets 1.0 |
| `status` | TEXT | `current`, `superseded`, `forgotten` |
| `tier` | TEXT | `active`, `dormant`, `archived`; derived, recomputed at sleep and on write |
| `embedding` | BLOB | float32[384] |
| `created_at` | REAL | unix seconds |
| `last_accessed_at` | REAL | denormalised from `accesses` for cheap sorting |
| `supersedes_id` | TEXT NULL | FK to the memory this one replaced |
| `superseded_by_id` | TEXT NULL | FK to the replacement |
| `feature_id` | TEXT NULL | FK: this memory is part of that `feature` |

### 5.2 `accesses`

Every event that counts toward strength: `(memory_id, at, kind)` where `kind ∈ {created, recalled, reinforced, explicit}`. Strength is computed from this table, never stored as the source of truth.

### 5.3 `memory_events` (the tuner's labelled data)

| Column | Notes |
|---|---|
| `id`, `at` | |
| `label` | `hit`, `false_recall`, `miss`, `stale` (see §9.2) |
| `memory_id` | nullable for `miss` when nothing matched |
| `turn_id` | groups events from one exchange |
| `context` | JSON: `{tier_at_time, rank_at_time, similarity, scope, query_scope}` |

### 5.4 `params` and `params_history`

`params(name TEXT PK, value REAL, updated_at REAL)`.
`params_history(id, name, old_value, new_value, old_loss, new_loss, at, reason)`.

### 5.5 `meta`

`last_sleep_at`, `schema_version`.

### 5.6 `turns` (working memory)

`(id, session_id, at, project_scope, user_text, assistant_text, mode)`. Only the last `working_turns` rows of the current session are used; older rows are deleted after capture has run.

## 6. Scoring (pure functions in `memory/domain/scoring.py`)

```
strength(m, now, p) =
    p.importance_weight * m.importance
  + Σ_{a in m.accesses} ((now - a.at) / 86400 + 1) ** (-p.decay[m.kind])

tier_for(strength, p):
    >= p.active_threshold   -> active
    >= p.dormant_threshold  -> dormant
    else                    -> archived

recall_score(m, query, now, p) =
    similarity(m.embedding, query.embedding)   # cosine, normalised vectors
  * strength(m, now, p)
  * scope_weight(m.scope, query.scope, p)      # same project 1.0, global p.scope_global, other project p.scope_other
```

Only `status = current` memories are scored. Reinforcing a memory appends an `accesses` row; nothing else changes.

Default decay rates (all tunable, see §9): `episode` 0.9, `route_example` 0.6, `fact` 0.5, `decision` 0.3, `feature` 0.2, `preference` 0.2. With these, an unused episode drops out of `active` in roughly a week while an unused feature stays active for months; any mention resets the clock for that memory.

## 7. Data flow

### 7.1 Capture (`CaptureTurnUseCase`)

Trigger: after each Quick/Think reply is fully streamed, and when a coding session finishes review (`AgentManager._finish_review`). Runs in a daemon thread; the voice loop never waits on it.

Input: `Turn{user_text, assistant_text | coding_summary, project_scope, recalled_ids}`. For coding sessions `coding_summary` = the original prompt + "accepted N hunks, rejected M hunks in files …" from the review coordinator.

Extractor contract (`IMemoryExtractor.extract(turn, recalled: list[Memory]) -> Extraction`):

```json
{
  "new":       [{"kind": "decision", "scope": "project", "content": "...", "importance": 0.7}],
  "reinforce": ["<id>"],
  "irrelevant": ["<id>"],
  "supersede": [{"old_id": "<id>", "content": "...", "kind": "decision", "importance": 0.8}],
  "forget":    ["<id>"]
}
```

`scope` in the output is `project` or `global`; the use case resolves `project` to the current path. Then, per item:

- `new`: embed; if the nearest `current` memory in the same scope has cosine ≥ `dedup_threshold` → reinforce it instead; else insert (tier computed immediately).
- `reinforce`: append access; log event `hit`.
- `irrelevant`: log event `false_recall` (no access row).
- `supersede`: insert replacement, link both ways, set old `status = superseded`; log `stale` if the old one was recalled this turn.
- `forget`: `status = forgotten`; log `false_recall` if it was recalled this turn.
- Recalled ids not mentioned in `reinforce` or `irrelevant` are logged as neither (unlabelled); the extractor is asked to classify every recalled id so this should be rare.

Malformed or failed extraction: log a warning, store nothing, still record the raw turn in `turns`.

Groq extractor uses `GROQ_DEFAULT_MODEL` with JSON mode. Ollama fallback uses `ACTIVE_MODEL` with `format="json"` and a shorter prompt; if the fallback returns invalid JSON twice, the turn is skipped.

### 7.2 Recall (`RecallUseCase`)

Trigger: before building the Quick/Think prompt, and before building the coding prompt.

1. Embed transcript. Query scope = current project path (JARVIS's own directory today; `AgentManager.start_session(cwd=…)` is the source of truth).
2. Score all `current` memories (brute-force numpy over the embedding matrix; the store keeps it cached in memory and invalidates on write).
3. Result = top `recall_k` from `active` ∪ any `dormant` with similarity ≥ `dormant_recall_similarity`. `archived` is never returned.
4. For each returned memory append an `accesses(kind=recalled)` row and remember `recalled_ids` for capture.
5. Return `RecallResult{memories, working_turns}`.

`prompt_renderer` turns this into:

```
What you remember, most relevant first:
- (decision, this project) We use SQLite for memory, not FAISS.
- (preference) Isaac prefers short spoken answers.
- (feature, this project) In progress: human-like memory with sleep pass and tuner.
Recent conversation:
User: ... / JARVIS: ...
```

For the coding prompt the block is prefixed to the user's request under a `## Project context` heading; the `agy` command line is otherwise unchanged.

### 7.3 Memory voice commands (`MemoryVoiceGrammar`)

Checked in `main_loop` after the active-session / review checks and **before** "Route to quick, think, or code?". Case-insensitive prefix match on the transcript:

| Pattern | Action |
|---|---|
| `remember (that) …` | `CaptureTurn` with `explicit=True`: extractor picks kind/scope, importance forced to 1.0, `accesses(kind=explicit)`. If dedup finds an existing memory that was `dormant`/`archived` → log `miss`. Reply: "Got it." |
| `forget (that\|about X)` | "that" → last recalled memory of this session; "about X" → nearest `current` memory by similarity. Set `forgotten`; log `false_recall` if it was recalled this turn. Reply: "Forgotten." |
| `what do you remember about X` / `did we decide …` / `do you remember …` | Recall with `include_dormant=True`. Speak the top 1–3 contents. If the best match was `dormant`/`archived`, or similarity < `recall_similarity_floor`, log `miss`. |
| `how's your memory` | Speak the one-line summary from the report (§9.4). |

Everything else continues to route confirmation unchanged.

### 7.4 Router integration

`SqliteMemoryStore` implements `IVectorStore`:

- `add_utterance(u, vec)` → insert `route_example` (global scope, importance 0.5) or reinforce a duplicate (cosine ≥ `dedup_threshold`).
- `search_similar(vec, limit)` → top `limit` `current` route examples ordered by `similarity * strength`; returns `(Utterance, squared_l2_distance)` so `RouteCommandUseCase`'s confidence formula is unchanged.
- `distill_utterances()` → the dedup step of sleep restricted to `route_example`.

`jarvis.py` constructs `SqliteMemoryStore` where it built `FaissAdapter`. `seed_db.py` keeps working through the same port. `python -m memory migrate` imports `.vectorstore/metadata.json` once (idempotent: skips if the DB already has route examples) and is also run automatically on first start if the FAISS files exist.

## 8. Sleep pass (`SleepUseCase`)

Trigger: daemon thread at startup if `now - last_sleep_at > sleep_interval_hours`; or `python -m memory sleep [--dry-run]`.

1. **Rescore:** recompute `tier` for every `current` memory.
2. **Summarise episodes:** for each project scope, take `current` episodes older than `episode_summary_age_days` (minimum 3); one extractor call (`IMemoryExtractor.summarise(episodes) -> list[new]`) producing 1–3 `fact`/`feature`/`decision` memories; originals set `feature_id`/link `derived-from` and `tier = archived`. New summaries inherit the newest original's accesses so they do not start cold.
3. **Dedup:** pairwise within (kind, scope) above `dedup_threshold`; keep the one with higher strength, move the other's accesses to it, set the loser `superseded`.
4. **Purge:** delete `archived` or `forgotten` rows (and their accesses/events) older than `retention_days`.
5. **Tune:** `TuneParamsUseCase.run()` (§9).
6. Write `last_sleep_at`.

`--dry-run` executes against a copy of the DB in memory and prints the plan without writing.

## 9. Self-tuning (`TuneParamsUseCase`)

### 9.1 Tunable params

| Name | Default | Min | Max | Step |
|---|---|---|---|---|
| `decay.episode` | 0.9 | 0.3 | 2.0 | 0.1 |
| `decay.route_example` | 0.6 | 0.2 | 1.5 | 0.1 |
| `decay.fact` | 0.5 | 0.1 | 1.5 | 0.05 |
| `decay.decision` | 0.3 | 0.05 | 1.0 | 0.05 |
| `decay.feature` | 0.2 | 0.05 | 1.0 | 0.05 |
| `decay.preference` | 0.2 | 0.05 | 1.0 | 0.05 |
| `importance_weight` | 1.0 | 0.2 | 3.0 | 0.1 |
| `active_threshold` | 0.8 | 0.2 | 3.0 | 0.1 |
| `dormant_threshold` | 0.3 | 0.05 | 1.5 | 0.05 |
| `scope_global` | 0.8 | 0.3 | 1.0 | 0.05 |
| `scope_other` | 0.4 | 0.0 | 1.0 | 0.05 |
| `recall_k` | 6 | 2 | 15 | 1 |
| `dormant_recall_similarity` | 0.75 | 0.5 | 0.95 | 0.02 |
| `recall_similarity_floor` | 0.45 | 0.2 | 0.8 | 0.02 |
| `dedup_threshold` | 0.9 | 0.8 | 0.98 | 0.01 |
| `episode_summary_age_days` | 7 | 2 | 30 | 1 |
| `sleep_interval_hours` | 24 | 6 | 168 | 6 |
| `retention_days` | 60 | 14 | 365 | 7 |
| `working_turns` | 6 | 2 | 20 | 1 |

Invariant enforced after every move: `dormant_threshold < active_threshold`.

Not tuned, fixed in code: `tune_window_days` 30, `min_events` 40, `improvement_margin` 5 %, `max_moves_per_pass` 3, loss weights `w_false_recall` 0.5, `w_stale` 1.0.

### 9.2 Labels

| Label | Source | Interpretation |
|---|---|---|
| `hit` | extractor `reinforce` | recall was right |
| `false_recall` | extractor `irrelevant`; "forget that" right after recall | surfaced noise |
| `miss` | "remember that" for an existing dormant/archived memory; "what do you remember" answered from dormant/archived or below the floor; extractor `new` that dedups onto a dormant/archived memory | faded or ranked too low |
| `stale` | `supersede` of a memory recalled this turn | decayed too slowly |

### 9.3 Method: counterfactual replay

Because strength depends only on `accesses`, `importance`, params and time, the tuner can recompute what each labelled event would have looked like under different params:

```
loss(p) = Σ_events  cost(event, p)
  hit          -> 0 if memory would still be in top recall_k of active at event.at else 1
  false_recall -> w_false_recall if memory would still be recalled else 0
  miss         -> 1 if memory would still be dormant/archived (or absent from top k) else 0
  stale        -> w_stale if memory would still be active else 0
```

Replay for `rank`-dependent decisions uses the `similarity` and candidate set stored in `memory_events.context`, so no re-embedding is needed.

Algorithm per sleep pass:

1. If labelled events in the last `tune_window_days` < `min_events` → record "insufficient data", stop.
2. `base = loss(current)`.
3. For each param in a fixed order, evaluate `loss` at `value ± step` (clamped to bounds, invariants respected). Collect candidates with `loss < base * (1 - improvement_margin)`.
4. Apply the best candidate, set `base` to its loss, repeat from 3 until no candidate improves or `max_moves_per_pass` reached.
5. Each applied move writes `params_history` with old/new value and old/new loss, `reason = "replay"`.

Deterministic, no LLM, finishes in well under a second for thousands of events.

### 9.4 Visibility

`python -m memory report` prints:

- counts by kind × tier × scope;
- per-kind hit / false-recall / miss / stale rates over the window;
- current params vs defaults, flagged when moved;
- last 10 `params_history` rows with loss deltas;
- `last_sleep_at` and next due.

Voice "how's your memory?" → "I hold 212 memories, 48 active. Over the last month I recalled correctly 83 percent of the time and missed 4 times. Last night I slowed how fast decisions fade."

`python -m memory reset-params` restores defaults (writes history rows with `reason = "manual reset"`).

## 10. Error handling

- Extractor unavailable or malformed output → skip capture for that turn, log, continue. Never block speech.
- DB locked / write failure → log and continue; recall falls back to an empty block.
- Migration failure → leave `.vectorstore/` untouched, start with an empty route store, warn loudly.
- Sleep or tuner exception → caught in the thread, logged, `last_sleep_at` not updated so it retries next start.
- All threads are daemon threads, matching the existing pattern in `agent_adapter/manager.py`.

## 11. Testing

All under `tests/memory/`, pytest, no network, no model downloads (a `FakeEmbedder` returns deterministic vectors; a `FakeExtractor` returns canned `Extraction`s; `FrozenClock` controls time).

| Area | Tests |
|---|---|
| `scoring` | episodes fade below active before features given identical access history; a reinforcement raises strength; tier boundaries; superseded/forgotten never score; scope weights ordering |
| `sqlite_store` | round-trip all fields; scope filtering; embedding cache invalidation; `IVectorStore` contract (add, search ordering, distance mapping) equals FAISS behaviour on a fixture; `migrate_faiss` idempotent |
| `capture_turn` | new insert; dedup → reinforce; supersede links both ways and flips status; forget; malformed extraction skips cleanly; events logged with the right labels; explicit remember forces importance 1.0 and logs `miss` when target was dormant |
| `recall` | top-k from active; dormant only above similarity bar; archived never; current project outranks global outranks other project at equal similarity; accesses appended; prompt renderer output |
| `voice_grammar` | each pattern routes to the right use case; non-matching text falls through |
| `sleep` | episodes fold into summaries and are archived; summaries inherit accesses; dedup keeps the stronger; purge respects retention; dry-run writes nothing; `last_sleep_at` untouched on exception |
| `tune` | synthetic logs with known answers: (a) decisions always missed → `decay.decision` decreases; (b) episodes always irrelevant → `decay.episode` increases or `recall_k` decreases; (c) under `min_events` → no move; (d) bounds never exceeded; (e) margin prevents flip-flop over two passes; (f) `dormant_threshold < active_threshold` invariant holds |
| end-to-end | capture → recall shows it → contradict → recall shows the replacement only → advance clock past summary age → sleep → episode archived, summary active → tuner moved at most 3 params |
| regression | `tests/test_router.py` unchanged and passing with the new store |

## 12. Rollout

1. Land everything behind the new store; FAISS files migrated on first start.
2. Params start at defaults; tuner silent until 40 labelled events exist.
3. After a couple of weeks, review `python -m memory report` together and adjust bounds or loss weights if the tuner is drifting somewhere unwanted. Those are code changes, deliberately.

## 13. Later (not in this spec)

- Extractor prompt A/B testing with the same event log.
- Per-project `PROJECT.md`-style export of active features/decisions.
- Showing memory in the dashboard.
- Multi-project detection from the active editor instead of JARVIS's own directory.
