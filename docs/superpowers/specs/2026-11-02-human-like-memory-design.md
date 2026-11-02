# Human-like memory for JARVIS

**Date:** 2026-11-02 (rev 2)
**Branch:** `feature/human-like-memory` (from `origin/dev`)
**Status:** Draft for review

Rev 2 changes, after review: semantic memory operations replace prefix-matched voice commands (§7.3); extraction is salience-gated and debounced instead of per-turn (§7.1); recall is gated, budgeted per mode and kept stable across a session with a profile block and working set (§7.2); the tuner gains cadence and classifier signals (§9).

## 1. Goal

Give JARVIS the memory of a peer programmer you have worked with for a while:

- recent things are vivid, old small things fade and are eventually thrown away;
- big ideas (features in progress, decisions, preferences) persist and get stronger every time they come up;
- ordinary conversation revises memory ("no, we switched to Postgres" replaces the old fact) without magic words;
- it knows which project it is in, weights that project's memory highest, but can still recall "we solved this in the other repo";
- it costs almost nothing on turns that carry no new information, and does its heavy lifting off the hot path;
- the numbers that govern fading, recall, extraction cadence and consolidation are **measured and tuned automatically** from how the user reacts, not hand-edited.

The same engine replaces the router's FAISS training store so route examples also get recency, reinforcement and de-duplication.

## 2. Current state (what this changes)

| Today | Where | After |
|---|---|---|
| Router store: FAISS `IndexFlatL2` + `.vectorstore/metadata.json` with `{text, intent, timestamp}`; `timestamp` unused; nothing decays or merges; `distill_utterances()` is a stub | `adapters/vector_store.py`, `use_cases/routing.py` | Route examples live in the memory DB as `route_example` memories; strength-weighted search; dedup at sleep; FAISS retired with a one-time migration |
| Quick/Think prompts send `[system, transcript]` only; no conversation history | `jarvis.py:503-506` | Prompt gains a cache-stable profile block, a gated and budgeted "what you remember" block, and the last *n* turns of this session |
| Coding tasks get the raw transcript as prompt; context lives only inside `agy -c` | `jarvis.py:93-100`, `agent_adapter/manager.py` | Prompt is prefixed with project memory (features in progress, standing decisions) |
| Transcript is embedded once, inside `RouteCommandUseCase.execute` | `use_cases/routing.py:14` | Embedded once in `main_loop`; the vector is shared by routing, the memory-op classifier, salience and recall |
| No memory voice commands | — | Semantic memory-op classifier (Net 1) intercepts store / forget / revise / complete / ask / status before route confirmation; batched extractor (Net 2) catches implicit updates |
| Tests: `tests/test_router.py` covers routing only | `tests/` | Those tests keep passing unchanged; new suite under `tests/memory/` |

Ports `IVectorStore` and `IEmbeddingEngine` (`domain/ports.py`) and `RouteCommandUseCase` / `LearnFromFeedbackUseCase` interfaces are **kept**; `RouteCommandUseCase.execute` gains an optional `vector=` argument so the shared embedding can be passed in.

## 3. Out of scope

- Any UI (the FastAPI dashboard is untouched; visibility is CLI + voice only).
- Multi-user memory. One user, one database.
- Changing the training-mode route confirmation flow (every non-memory utterance still asks "quick, think, or code?").
- Cloud sync of the database.
- Tuning extractor or classifier *prompts* automatically (the tuner moves numeric params only; the classifier learns prototypes, not prompts).

## 4. Architecture

Package layout mirrors `verifier/`:

```
memory/
  __init__.py
  __main__.py              # python -m memory <report|migrate|sleep|reset-params>
  cli.py
  domain/
    entities.py            # Memory, MemoryKind, Scope, Tier, Status, Link, MemoryEvent, EventLabel,
                           # MemoryOp, Turn, Extraction
    params.py              # ParamSpec (default/min/max/step), DEFAULT_PARAMS, FIXED constants
    scoring.py             # pure: strength(), tier_for(), recall_score(), salience()
  ports/
    store.py               # IMemoryStore
    extractor.py           # IMemoryExtractor
    clock.py               # IClock (so sleep/tuner tests can freeze time)
  use_cases/
    memory_ops.py          # ClassifyMemoryOpUseCase (Net 1) + HandleMemoryOpUseCase
    extraction_buffer.py   # ExtractionBuffer: salience accumulation + flush triggers
    capture.py             # CaptureBatchUseCase (Net 2): applies an Extraction to the store
    recall.py              # RecallUseCase + SessionWorkingSet + ProfileBlock
    consolidate.py         # SleepUseCase (§8)
    tune.py                # TuneParamsUseCase (§9)
  adapters/
    sqlite_store.py        # SqliteMemoryStore(IMemoryStore, IVectorStore)
    groq_extractor.py      # GroqMemoryExtractor
    ollama_extractor.py    # OllamaMemoryExtractor
    extractor_factory.py   # Groq if key present, else Ollama (same pattern as GroqTerminalParser.is_available())
    prompt_renderer.py     # renders profile block, recalled memories and working turns into prompt text
    memory_op_seeds.py     # seed phrasings per MemoryOp
    migrate_faiss.py       # .vectorstore/metadata.json -> route_example rows
```

Embeddings reuse the existing `HuggingFaceAdapter` (`adapters/embeddings.py`, MiniLM-L6-v2, 384-d, normalised). Database file: `.jarvis/memory.sqlite` (`.jarvis/` is already git-ignored).

Dependencies added: none beyond the standard library `sqlite3` and `numpy` (already present). The `faiss` dependency is removed.

### 4.1 Hot path vs background

| On the hot path (synchronous, no LLM) | In the background (daemon threads) |
|---|---|
| embed transcript once | extractor calls (Net 2 batches, Net 1 resolution) |
| Net 1 classification (cosine vs prototypes) | sleep pass |
| salience score + buffer bookkeeping | tuner |
| recall scoring, working-set update, prompt rendering | migration on first start |

The only LLM call that is waited on is the Net 1 *resolution* call when the user has explicitly asked for a memory operation, because it needs a spoken acknowledgement.

## 5. Data model

### 5.1 `memories`

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | uuid4 |
| `kind` | TEXT | `episode`, `fact`, `decision`, `feature`, `preference`, `route_example`, `memory_op_example` |
| `scope` | TEXT | `global` or `project:<absolute repo path>` |
| `content` | TEXT | 1–2 sentences written by the extractor; for the two `*_example` kinds, the raw utterance |
| `label` | TEXT NULL | `route_example`: `quick`/`think`/`code`. `memory_op_example`: a `MemoryOp` or `none` |
| `importance` | REAL | 0–1; explicit store sets 1.0 |
| `status` | TEXT | `current`, `superseded`, `forgotten` |
| `tier` | TEXT | `active`, `dormant`, `archived`; derived, recomputed at sleep and on write |
| `completed_at` | REAL NULL | `feature` only; set by a `complete` op; see §6 |
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
| `label` | see §9.2 |
| `memory_id` | nullable for `miss` when nothing matched, and for cadence/classifier labels |
| `turn_id` | groups events from one exchange |
| `context` | JSON: `{tier_at_time, rank_at_time, similarity, scope, query_scope, mode, buffer_salience, classifier_score}` as applicable |

### 5.4 `params` and `params_history`

`params(name TEXT PK, value REAL, updated_at REAL)`.
`params_history(id, name, old_value, new_value, old_loss, new_loss, at, reason)`.

### 5.5 `meta`

`last_sleep_at`, `schema_version`, `profile_block_version`.

### 5.6 `turns` (working memory and extraction buffer)

`(id, session_id, at, project_scope, mode, user_text, assistant_text, embedding BLOB, salience REAL, recalled_ids JSON, extracted_at REAL NULL)`.

Rows with `extracted_at IS NULL` are the extraction buffer. The last `working_turns` rows of the current session are the verbatim working memory. Rows are deleted once extracted *and* older than the working window.

## 6. Scoring (pure functions in `memory/domain/scoring.py`)

```
effective_kind(m) = fact if m.kind == feature and m.completed_at else m.kind

strength(m, now, p) =
    p.importance_weight * m.importance
  + Σ_{a in m.accesses} ((now - a.at) / 86400 + 1) ** (-p.decay[effective_kind(m)])

tier_for(strength, p):
    >= p.active_threshold   -> active
    >= p.dormant_threshold  -> dormant
    else                    -> archived

recall_score(m, query, now, p) =
    similarity(m.embedding, query.embedding)   # cosine, normalised vectors
  * strength(m, now, p)
  * scope_weight(m.scope, query.scope, p)      # same project 1.0, global p.scope_global, other project p.scope_other

salience(turn, store, p) =
    (1 - max_cosine(turn.embedding, store.current ∪ working_set)) * p.mode_prior[turn.mode]
  + p.salience_length_bonus * min(1, word_count / 40)
  + (1.0 if a Net 1 op fired on this turn else 0)
```

Only `status = current` memories are scored. Reinforcing a memory appends an `accesses` row; nothing else changes. A completed feature stays `current` and recallable ("we built X") but decays at the `fact` rate and is no longer rendered under *in progress*.

Default decay rates (all tunable, §9): `episode` 0.9, `route_example` 0.6, `memory_op_example` 0.6, `fact` 0.5, `decision` 0.3, `feature` 0.2, `preference` 0.2. With these, an unused episode drops out of `active` in roughly a week while an unused feature stays active for months; any mention resets the clock for that memory.

## 7. Data flow

Order of operations in `main_loop` for one utterance, after the existing review-mode and active-session checks:

1. embed transcript once (`vector`);
2. **Net 1** `ClassifyMemoryOpUseCase(vector)` → if it fires, handle (§7.3) and `continue`;
3. route confirmation as today (`determine_intent(transcript, vector=vector)`);
4. **Recall** (§7.2) → prompt; stream the answer;
5. append the turn to `turns` with its salience; **ExtractionBuffer** decides whether to flush (§7.1).

### 7.1 Extraction: salience-gated, debounced batches (Net 2)

Per turn, with no LLM call, `salience()` is computed and added to the buffer total. The buffer is flushed to **one** extractor call (daemon thread) when any trigger fires:

| Trigger | Param | Why |
|---|---|---|
| accumulated salience ≥ threshold | `extract_salience_threshold` | content-driven: a brainstorm crosses it in one or two turns, routine chatter takes many |
| no new turn for N seconds | `extract_idle_seconds` | debounce; a new turn resets the timer |
| buffer length ≥ N turns | `extract_max_turns` | hard cap |
| mode switches to Code; coding session ends | fixed | the goal is about to change / has just been acted on |
| Net 1 op fired | fixed | handled immediately, see §7.3 |

A coding session's turn is `{user_text: original prompt, assistant_text: "accepted N hunks, rejected M in files …"}` produced from the review coordinator in `AgentManager._finish_review`.

Extractor contract (`IMemoryExtractor.extract(turns: list[Turn], recalled: list[Memory]) -> Extraction`):

```json
{
  "new":       [{"kind": "decision", "scope": "project", "content": "...", "importance": 0.7}],
  "reinforce": ["<id>"],
  "irrelevant": ["<id>"],
  "supersede": [{"old_id": "<id>", "content": "...", "kind": "decision", "importance": 0.8}],
  "complete":  ["<feature id>"],
  "forget":    ["<id>"],
  "explicit_ops": [{"turn_id": "<id>", "op": "store"}]
}
```

`scope` is `project` or `global`; the use case resolves `project` to the current path. `recalled` is the union of memories recalled across the batch. `explicit_ops` flags turns where the user clearly asked for an operation; it is the feedback path for Net 1 (§7.3). `CaptureBatchUseCase` then applies each item:

- `new`: embed; if the nearest `current` memory in the same scope has cosine ≥ `dedup_threshold` → reinforce it instead (and log `miss` if it was dormant/archived); else insert (tier computed immediately).
- `reinforce`: append access; log `hit`.
- `irrelevant`: log `false_recall` (no access row).
- `supersede`: insert replacement, link both ways, set old `status = superseded`; log `stale` if the old one was recalled in this batch.
- `complete`: set `completed_at`; log nothing.
- `forget`: `status = forgotten`; log `false_recall` if it was recalled in this batch.
- `explicit_ops`: for each, if Net 1 did not fire on that turn → add the utterance as a positive `memory_op_example` for that op and log `missed_op`.
- If `new`, `supersede`, `complete` and `forget` are all empty → log `empty_extraction` with the batch's `buffer_salience` (cadence was too eager).

Mark the batch's turns `extracted_at`. Malformed or failed extraction: log a warning, leave the turns unextracted so they join the next batch (at most once; on the second failure they are marked extracted and dropped).

Groq extractor uses `GROQ_DEFAULT_MODEL` with JSON mode. Ollama fallback uses `ACTIVE_MODEL` with `format="json"` and a shorter prompt.

### 7.2 Recall: gated, budgeted, stable across the session

Trigger: before building the Quick/Think prompt, and before building the coding prompt.

**Profile block.** Rendered once per session and cached: top `profile_preferences` global preferences by strength, plus this project's `current` features without `completed_at`, by strength. Re-rendered only when sleep runs or a preference/feature is written. It sits directly after the static system prompt so Ollama's prefix cache keeps hitting.

**Relevance recall.**

1. Score all `current` non-example memories with `recall_score` (brute-force numpy over the cached embedding matrix; invalidated on write).
2. Candidates = top `recall_k` from `active` ∪ any `dormant` with similarity ≥ `dormant_recall_similarity`. `archived` is never returned. Memories already in the profile block are excluded.
3. **Gate:** if no candidate has similarity ≥ `recall_similarity_floor`, nothing is added this turn.
4. **Budget per mode:** keep at most `budget[mode]` candidates (`budget_quick` 2, `budget_think` = `recall_k`, Code = all active features/decisions for the project plus `recall_k` others).
5. **Session working set:** candidates are merged into the session's working set. A memory already in the set keeps its position; new ones are appended; a member is evicted after `working_set_patience` consecutive turns in which it was not a candidate. The rendered block is the working set in insertion order, so it changes rarely and the prefix cache stays warm.
6. For each memory *newly* added to the set, append `accesses(kind=recalled)`; members already present are counted as recalled for labelling purposes (`recalled_ids` on the turn) but not re-accessed every turn.

`prompt_renderer` output, in order: static system prompt → profile block → "What you remember" (working set, if non-empty) → "Recent conversation" (last `working_turns` turns). For the coding prompt the profile block and working set go under a `## Project context` heading ahead of the user's request; the `agy` command line is otherwise unchanged.

### 7.3 Semantic memory operations (Net 1) and how they learn

**Classifier.** `memory_op_example` memories form a prototype index (same mechanism as route examples). For each incoming `vector`, take the nearest `memory_op_k` prototypes weighted by `similarity × strength` and vote by `label`. Score = winning label's share × its best similarity.

| Score | Action |
|---|---|
| ≥ `memory_op_threshold` | execute the op |
| ≥ `memory_op_ask_threshold` | confirm by voice first: "Should I remember that?", "Drop the FAISS decision?" |
| below, or winning label `none` | fall through to route confirmation |

Operations and their resolution:

| Op | Example phrasings (seeded, ~20 each) | Resolution |
|---|---|---|
| `store` | "keep in mind…", "for future reference…", "I always want…" | one extractor call with the utterance + working turns → kind/scope/content; importance 1.0; `accesses(kind=explicit)`; if dedup hits a dormant/archived memory → log `miss` |
| `forget` | "drop that", "stop bringing that up", "scrap the diff-reviewer idea" | extractor picks target ids from: last recalled, working set, top-5 by similarity; set `forgotten`; log `false_recall` if it was recalled this turn |
| `revise` | "we're not using FAISS anymore", "that moved to Postgres" | extractor picks target + writes replacement → `supersede`; log `stale` if recalled this turn |
| `complete` | "the reviewer's done", "we shipped that" | extractor picks the feature → `completed_at` |
| `ask` | "what did we decide about…", "where were we on…" | recall with `include_dormant=True`, speak top 1–3; log `miss` if best match was dormant/archived or below `recall_similarity_floor` |
| `status` | "how's your memory" | speak the one-line report (§9.4) |

The resolution extractor call is synchronous (the user is waiting for "Noted" / "Dropped the diff-reviewer feature" / the answer). `forget` and `revise` always read back what they changed. Ollama fallback applies here too.

**Learning loop** (no new user behaviour required):

| Situation | Learned |
|---|---|
| Medium confidence, user says yes | utterance → positive `memory_op_example` for that op |
| Medium confidence, user says no | utterance → `memory_op_example` with label `none`; log `false_op` |
| High confidence, user pushes back in the next utterance ("no, I wasn't asking you to remember", "that was a question") | revert the action (un-forget / un-store / un-supersede); negative example; log `false_op` |
| Net 1 did not fire, Net 2 reports `explicit_ops` for the turn | positive example added automatically; log `missed_op` |

Pushback detection is itself Net 1 with a reserved label `undo`, seeded with ~20 phrasings and only consulted on the turn immediately following an executed op. Prototypes decay and dedup like any memory, so phrasings the user stops using fade.

### 7.4 Router integration

`SqliteMemoryStore` implements `IVectorStore`:

- `add_utterance(u, vec)` → insert `route_example` (global scope, importance 0.5) or reinforce a duplicate (cosine ≥ `dedup_threshold`).
- `search_similar(vec, limit)` → top `limit` `current` route examples ordered by `similarity × strength`; returns `(Utterance, squared_l2_distance)` so `RouteCommandUseCase`'s confidence formula is unchanged.
- `distill_utterances()` → the dedup step of sleep restricted to `route_example`.

`jarvis.py` constructs `SqliteMemoryStore` where it built `FaissAdapter`. `seed_db.py` keeps working through the same port. `python -m memory migrate` imports `.vectorstore/metadata.json` once (idempotent: skips if the DB already has route examples) and is also run automatically on first start if the FAISS files exist.

## 8. Sleep pass (`SleepUseCase`)

Trigger: daemon thread at startup if `now - last_sleep_at > sleep_interval_hours`; or `python -m memory sleep [--dry-run]`.

1. **Flush** any pending extraction buffer first.
2. **Rescore:** recompute `tier` for every `current` memory.
3. **Summarise episodes:** for each project scope, take `current` episodes older than `episode_summary_age_days` (minimum 3); one extractor call (`IMemoryExtractor.summarise(episodes) -> list[new]`) producing 1–3 `fact`/`feature`/`decision` memories; originals set `feature_id`/link `derived-from` and `tier = archived`. New summaries inherit the newest original's accesses so they do not start cold.
4. **Dedup:** pairwise within (kind, scope, label) above `dedup_threshold`; keep the one with higher strength, move the other's accesses to it, set the loser `superseded`. Covers `route_example` and `memory_op_example` too.
5. **Purge:** delete `archived` or `forgotten` rows (and their accesses/events) older than `retention_days`.
6. **Tune:** `TuneParamsUseCase.run()` (§9).
7. **Refresh** the profile block; write `last_sleep_at`.

`--dry-run` executes against a copy of the DB in memory and prints the plan without writing.

## 9. Self-tuning (`TuneParamsUseCase`)

### 9.1 Tunable params

| Group | Name | Default | Min | Max | Step |
|---|---|---|---|---|---|
| decay | `decay.episode` | 0.9 | 0.3 | 2.0 | 0.1 |
| | `decay.route_example` | 0.6 | 0.2 | 1.5 | 0.1 |
| | `decay.memory_op_example` | 0.6 | 0.2 | 1.5 | 0.1 |
| | `decay.fact` | 0.5 | 0.1 | 1.5 | 0.05 |
| | `decay.decision` | 0.3 | 0.05 | 1.0 | 0.05 |
| | `decay.feature` | 0.2 | 0.05 | 1.0 | 0.05 |
| | `decay.preference` | 0.2 | 0.05 | 1.0 | 0.05 |
| strength | `importance_weight` | 1.0 | 0.2 | 3.0 | 0.1 |
| | `active_threshold` | 0.8 | 0.2 | 3.0 | 0.1 |
| | `dormant_threshold` | 0.3 | 0.05 | 1.5 | 0.05 |
| recall | `scope_global` | 0.8 | 0.3 | 1.0 | 0.05 |
| | `scope_other` | 0.4 | 0.0 | 1.0 | 0.05 |
| | `recall_k` | 6 | 2 | 15 | 1 |
| | `budget_quick` | 2 | 0 | 6 | 1 |
| | `dormant_recall_similarity` | 0.75 | 0.5 | 0.95 | 0.02 |
| | `recall_similarity_floor` | 0.45 | 0.2 | 0.8 | 0.02 |
| | `working_set_patience` | 3 | 1 | 10 | 1 |
| | `working_turns` | 6 | 2 | 20 | 1 |
| | `profile_preferences` | 5 | 0 | 15 | 1 |
| extraction | `extract_salience_threshold` | 1.5 | 0.3 | 6.0 | 0.1 |
| | `extract_idle_seconds` | 90 | 20 | 600 | 10 |
| | `extract_max_turns` | 10 | 2 | 30 | 1 |
| | `mode_prior.quick` | 0.5 | 0.1 | 1.5 | 0.1 |
| | `mode_prior.think` | 1.0 | 0.1 | 2.0 | 0.1 |
| | `salience_length_bonus` | 0.3 | 0.0 | 1.0 | 0.05 |
| | `dedup_threshold` | 0.9 | 0.8 | 0.98 | 0.01 |
| memory ops | `memory_op_threshold` | 0.7 | 0.4 | 0.95 | 0.02 |
| | `memory_op_ask_threshold` | 0.5 | 0.3 | 0.9 | 0.02 |
| | `memory_op_k` | 5 | 1 | 15 | 1 |
| sleep | `episode_summary_age_days` | 7 | 2 | 30 | 1 |
| | `sleep_interval_hours` | 24 | 6 | 168 | 6 |
| | `retention_days` | 60 | 14 | 365 | 7 |

Invariants enforced after every move: `dormant_threshold < active_threshold`; `memory_op_ask_threshold < memory_op_threshold`.

Fixed in code (`params.FIXED`): `tune_window_days` 30, `min_events` 40, `improvement_margin` 5 %, `max_moves_per_pass` 3, loss weights `w_false_recall` 0.5, `w_stale` 1.0, `w_empty_extraction` 0.3, `w_late_extraction` 1.0, `w_false_op` 1.0, `w_missed_op` 0.7.

### 9.2 Labels

| Label | Source | Interpretation |
|---|---|---|
| `hit` | extractor `reinforce` | recall was right |
| `false_recall` | extractor `irrelevant`; `forget` of a memory recalled this turn | surfaced noise |
| `miss` | `store`/`new` deduping onto a dormant/archived memory; `ask` answered from dormant/archived or below the floor | faded or ranked too low |
| `stale` | `supersede`/`revise` of a memory recalled this turn | decayed too slowly |
| `empty_extraction` | a batch produced no new/supersede/complete/forget | extraction fired too eagerly |
| `late_extraction` | a `miss` or `ask` whose answer was sitting unextracted in the buffer | extraction fired too late |
| `false_op` | Net 1 confirmed "no", or `undo` after an executed op | classifier too eager |
| `missed_op` | Net 2 `explicit_ops` on a turn where Net 1 did not fire | classifier too shy |

### 9.3 Method: counterfactual replay

Strength depends only on `accesses`, `importance`, params and time, and each event stores the similarity, candidate set, buffer salience or classifier score it was decided with, so the tuner can recompute what each labelled event would have looked like under different params without re-embedding or calling an LLM:

```
loss(p) = Σ_events cost(event, p)
  hit              -> 0 if memory would still be recalled (tier, rank ≤ budget, floor) else 1
  false_recall     -> w_false_recall if memory would still be recalled else 0
  miss             -> 1 if memory would still be dormant/archived or outside the budget else 0
  stale            -> w_stale if memory would still be active else 0
  empty_extraction -> w_empty_extraction if the batch would still have flushed at that salience else 0
  late_extraction  -> w_late_extraction if the turn would still have been unextracted (salience below threshold and within idle/max limits) else 0
  false_op         -> w_false_op if classifier_score would still be ≥ memory_op_threshold else 0
  missed_op        -> w_missed_op if classifier_score would still be < memory_op_ask_threshold else 0
```

Algorithm per sleep pass:

1. If labelled events in the last `tune_window_days` < `min_events` → record "insufficient data", stop.
2. `base = loss(current)`.
3. For each param in a fixed order, evaluate `loss` at `value ± step` (clamped to bounds, invariants respected). Collect candidates with `loss < base × (1 − improvement_margin)`.
4. Apply the best candidate, set `base` to its loss, repeat from 3 until no candidate improves or `max_moves_per_pass` reached.
5. Each applied move writes `params_history` with old/new value and old/new loss, `reason = "replay"`.

Deterministic, no LLM, finishes in well under a second for thousands of events.

### 9.4 Visibility

`python -m memory report` prints:

- counts by kind × tier × scope;
- per-kind hit / false-recall / miss / stale rates over the window;
- extraction stats: batches, mean turns per batch, mean salience at flush, empty-extraction rate, late-extraction count;
- classifier stats: ops executed / asked / declined, `false_op` and `missed_op` counts;
- current params vs defaults, flagged when moved;
- last 10 `params_history` rows with loss deltas;
- `last_sleep_at` and next due.

Voice `status` → "I hold 212 memories, 48 active. Over the last month I recalled correctly 83 percent of the time, missed 4 times and ran 31 extractions. Last night I slowed how fast decisions fade."

`python -m memory reset-params` restores defaults (writes history rows with `reason = "manual reset"`).

## 10. Error handling

- Extractor unavailable or malformed output → batch retried once with the next flush, then dropped; log, continue. Never block speech.
- Net 1 resolution call fails → speak "I couldn't work out what to change, could you say it another way?"; no write; no label.
- DB locked / write failure → log and continue; recall falls back to profile block only, then to nothing.
- Migration failure → leave `.vectorstore/` untouched, start with an empty route store, warn loudly.
- Sleep or tuner exception → caught in the thread, logged, `last_sleep_at` not updated so it retries next start.
- All threads are daemon threads, matching the existing pattern in `agent_adapter/manager.py`.

## 11. Testing

All under `tests/memory/`, pytest, no network, no model downloads (a `FakeEmbedder` returns deterministic vectors; a `FakeExtractor` returns canned `Extraction`s; `FrozenClock` controls time).

| Area | Tests |
|---|---|
| `scoring` | episodes fade below active before features given identical access history; a reinforcement raises strength; tier boundaries; superseded/forgotten never score; scope weight ordering; completed feature decays at fact rate; salience is higher for novel text than for near-duplicates and scales with mode prior |
| `sqlite_store` | round-trip all fields; scope filtering; embedding cache invalidation; `IVectorStore` contract (add, search ordering, distance mapping) equals FAISS behaviour on a fixture; `migrate_faiss` idempotent |
| `extraction_buffer` | flushes on salience threshold, on idle (frozen clock), on max turns, on mode switch; a new turn resets the idle timer; nothing flushes below all triggers |
| `capture` | new insert; dedup → reinforce; supersede links both ways and flips status; complete sets `completed_at`; forget; malformed extraction retried once then dropped; events logged with the right labels; `empty_extraction` logged when a batch yields nothing; `explicit_ops` adds a prototype and logs `missed_op` |
| `recall` | top-k from active; dormant only above similarity bar; archived never; current project outranks global outranks other project at equal similarity; gate adds nothing below the floor; budget per mode; working set keeps order, appends new, evicts after patience; accesses appended only on first entry; profile block excludes its members from recall; renderer output and ordering |
| `memory_ops` | each op resolves to the right use case; thresholds split execute / ask / fall-through; invariants; yes/no on ask adds the right prototype and label; `undo` on the next turn reverts a forget/store/supersede; non-op text falls through unchanged |
| `sleep` | pending buffer flushed first; episodes fold into summaries and are archived; summaries inherit accesses; dedup keeps the stronger across all kinds; purge respects retention; profile block refreshed; dry-run writes nothing; `last_sleep_at` untouched on exception |
| `tune` | synthetic logs with known answers: decisions always missed → `decay.decision` decreases; episodes always irrelevant → `decay.episode` up or `recall_k`/`budget` down; many `empty_extraction` → `extract_salience_threshold` up; many `late_extraction` → down; many `false_op` → `memory_op_threshold` up; many `missed_op` → `memory_op_ask_threshold` down; under `min_events` → no move; bounds never exceeded; margin prevents flip-flop over two passes; both invariants hold |
| end-to-end | brainstorm-like turns flush after two turns while low-salience turns accumulate to the idle flush; capture → recall shows it in the working set → "we're not using FAISS anymore" (Net 1 `revise`) → recall shows the replacement only → advance clock past summary age → sleep → episode archived, summary active → tuner moved at most 3 params |
| regression | `tests/test_router.py` unchanged and passing with the new store and the optional `vector=` argument |

## 12. Rollout

1. Land everything behind the new store; FAISS files migrated on first start.
2. Params start at defaults; Net 1 starts from the seed phrasings only; tuner silent until 40 labelled events exist.
3. After a couple of weeks, review `python -m memory report` together and adjust bounds, loss weights or seed phrasings if the tuner is drifting somewhere unwanted. Those are code changes, deliberately.

## 13. Later (not in this spec)

- Extractor prompt A/B testing with the same event log.
- Per-project `PROJECT.md`-style export of active features/decisions.
- Showing memory in the dashboard.
- Multi-project detection from the active editor instead of JARVIS's own directory.
- A learned (non-prototype) salience gate once there is enough labelled data to train one.

## 14. References consulted

Mem0 (arXiv 2504.19413) for the ADD/UPDATE/DELETE/NOOP update step; LangMem's delayed background processing for debounced extraction; Letta's sleep-time compute for off-hot-path consolidation; Generative Agents (arXiv 2304.03442) for importance-accumulation triggers and recency × relevance × importance retrieval; MemoryBank (arXiv 2305.10250) for Ebbinghaus-style strength reset on recall; SLICE-Mem (arXiv 2610.33015) and related write-time gating work for the cheap pre-extraction gate; Zep (arXiv 2501.13956) for temporal validity of facts, which informed `supersede` links over hard deletes.
