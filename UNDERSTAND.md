# Understand your own repo (30-min protocol, then you own it)

Codex wrote it; this makes it YOURS. Do the three parts in order. Part 3 is the same
scaffold → cold-pass method as your foundations drills.

---

## Part 1 — The 10-minute mental model (read once)

**What this service is:** a FastAPI web server exposing a small research assistant over
financial documents + price data. Three tools, one rule: never answer without evidence.

**Trace ONE request end to end** (this is the single highest-value exercise — do it with the
files open):

`curl "localhost:8010/search?q=apple revenue"` →

1. `app/main.py` — the FastAPI layer. `@app.get("/search")` catches the HTTP request,
   pulls `q` from the query string, calls `tools.search_docs(q)`. main.py does NO logic;
   it only translates HTTP ↔ Python and picks error codes (404 file missing, 422 no text,
   503 model configuration missing). That separation is deliberate: the agent calls the same
   `tools.*` functions directly, without HTTP.
2. `app/tools.py :: search_docs` — thin adapter: calls `rag.search`, converts the result
   to a plain JSON-able dict. Tools return dicts because an LLM agent consumes JSON.
3. `app/rag.py :: search` — the real work:
   - Your query is embedded (Chroma's built-in MiniLM sentence-embedding model, local, free).
   - Chroma compares it by **cosine distance** to every stored chunk vector, returns top-4.
   - Each hit becomes a `Citation` with `doc_id::chunk_id` — that string is why the answer
     is auditable.
   - **The refusal gate:** if the best hit's score (1 − distance) < 0.25, return
     `refused: true` instead of garbage. This is why "black hole entropy" got refused:
     nothing in the 10-Ks is close to it in embedding space.
4. Response flows back up: rag → tools (dict) → main (Pydantic `SearchResponse`) → JSON.

**Where do chunks come from?** `/ingest` → `rag.ingest`: read file (pypdf if PDF) →
`_chunk`: fixed 1000-char windows sliding by 850 (150 overlap so a sentence cut at a
boundary still appears whole in the next chunk) → store in Chroma with ids `doc::0, doc::1…`
Chroma persists to `.chroma/` on disk, so ingested docs survive restarts.

**The price tool** (`get_price_data`): try YOUR spx-news-intraday cached 1-min bars first
(it temporarily adds that repo to `sys.path` and calls its `load_price_bars`); fall back to
yfinance daily; either way `_normalize_bars` forces one shape (UTC DatetimeIndex, lowercase
OHLCV, numeric, close non-null) and `_frame_to_records` caps output at 2000 JSON rows.
Normalization exists because the agent must never care which source the data came from.

**models.py** = Pydantic schemas: typed request/response contracts, free validation + docs.
**agent.py** = bounded ReAct loop with validated tool calls and citation provenance checks.
**eval.py / tests** = metric definitions (hit@k etc.) with 2 real retrieval cases so the
eval harness has something to grow around.

## Part 2 — The 8 questions an interviewer would ask (answer aloud, no peeking)

1. Why chunk at ~1000 chars with overlap, and what breaks with no overlap? (boundary
   sentences get split; retrieval misses facts that straddle chunks)
2. What exactly is stored in Chroma — text, vectors, or both? (both: chunk text + its
   embedding + metadata; query embeds at search time)
3. Why cosine similarity and what does the 0.25 threshold mean physically? (angle between
   embedding vectors; 0.25 ≈ "barely related" — tuned by eye, should be tuned on labeled
   pairs — SAY that it's currently arbitrary, that's honesty points)
4. Why does the tool layer return dicts instead of Pydantic objects? (agent/LLM consumes
   JSON; HTTP layer re-validates into Pydantic at the boundary)
5. Why refuse instead of always returning top-k? (a research tool that guesses is worse
   than useless; refusal is measurable — false-refusal rate is an eval metric)
6. Why normalize price bars to one schema? (two sources, one downstream contract; the
   event study must not branch on data source)
7. Where is the lookahead risk in the event study? (expected-return baseline and bootstrap
   residuals must come from strictly before the event window)
8. What would you change first for production? (real embedder, async ingestion, authn,
   eval-gated threshold; pick one and go one level deep)

## Part 3 — Cold-pass exercises (do tonight or tomorrow's MVP block, ~40 min)

1. **Break it (10m):** set `REFUSE_SCORE_THRESHOLD = 0.9`, run the app, watch every query
   refuse. Set it back. Now you FEEL what the gate does.
2. **Trace it (10m):** put a `print(res["distances"])` in `rag.search`, query twice —
   once on-topic, once nonsense. Look at the actual numbers. Delete the print.
3. **Rebuild from blank (20m, AI OFF):** new file, write `_chunk()` and a minimal
   `search()` against Chroma from memory (client → collection → add → query). Compare with
   the real one. This is the foundations method; after this the repo is yours.

## The 60-second interview version (memorize the shape, not the words)

"It's a research service over financial filings and price data: FastAPI surface, Chroma RAG
with citation-first retrieval and an explicit refusal gate, and typed tools designed to be
called by a ReAct agent. The design center is auditability — every claim carries a
doc::chunk citation, weak evidence refuses rather than guesses, and the event-study tool
enforces a pre-event-only baseline so there's no lookahead. The eval harness measures
hit@k, citation grounding, and false-refusal rate, because an agent you can't score is an
agent you can't trust."

---

## Day 4 — Own the ReAct agent (10 minutes)

### Trace one `/research` request

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"question":"What did Apple report about services revenue?"}'
```

1. `app/main.py` validates the body as `ResearchRequest`, then calls
   `agent.run_agent(question)`.
2. `run_agent` loads the configured model and sends the system prompt plus the question.
3. The model chooses an action. For example, it can request `search_docs`; it is not routed
   by a keyword `if` statement.
4. Python parses that action as JSON, validates the arguments with Pydantic, and dispatches
   only a name in `TOOLS`. The model cannot create a fourth tool.
5. The JSON tool observation goes back into the message history. The model can choose another
   tool or produce a final action. This repeats for at most six steps.
6. Every passage returned by `search_docs` enters a citation registry keyed by
   `doc_id::chunk_id`.
7. Before accepting a non-refused answer, `_finalize` checks that at least one citation was
   declared, each id exists in that registry, and each id appears in the answer text.
8. `main.py` converts the result into `ResearchResponse`, so FastAPI validates the final API
   shape before returning JSON.

### What “ReAct” means here

ReAct means the model alternates between deciding what evidence it needs and observing the
result of a deterministic tool call:

```text
model action → validated tool execution → observation → next model action
```

The model has freedom over tool choice and order. It does not have freedom to bypass argument
validation, call arbitrary Python, exceed the step budget, or invent accepted citations.

### Five-question self-quiz

Answer aloud without opening `app/agent.py`:

1. What part is genuinely model-directed, and what part remains deterministic?
2. Why use a maximum step count instead of letting the model continue until it is satisfied?
3. What happens when the model requests an unknown tool or sends `k=0` to `search_docs`?
4. What does the citation guard prove, and what does it explicitly **not** prove?
5. Why do the tests inject `ScriptedModel` instead of calling the configured API?

Expected points:

1. The model chooses tool and order; Python validates, dispatches, and accepts or refuses.
2. Boundedness controls cost, latency, and loops that never converge.
3. The loop returns a structured error observation so the model can recover within its budget.
4. It proves citation provenance, not sentence-level semantic grounding.
5. Tests remain deterministic, fast, free, and independent of credentials or provider uptime.

### 60-second interview version

"I implemented a bounded ReAct loop over exactly three research tools. The LLM decides which
tool to call and in what order, but every action uses a strict JSON contract and Pydantic
argument validation. Tool results return as observations for at most six steps. Retrieved
passages enter a citation registry, and code refuses any final answer that omits citations,
fabricates an id, or fails to place the id in the answer text. That enforces provenance; I am
careful not to overclaim semantic grounding, which is measured separately in the eval harness.
The model interface is injected in tests, so all agent behavior is deterministic and no test
spends API credits."
