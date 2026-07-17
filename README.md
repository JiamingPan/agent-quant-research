# Agent Quant Research

Citation-grounded RAG and eval infrastructure for financial research documents.

A RAG + agent service over financial documents and price data that produces reproducible,
leakage-checked research memos and event studies. **Not** live trading — no buy/sell claims.
The point is rigorous, reproducible research *infrastructure*: answers must use retrieved
citations, weak-evidence questions get refused, and the event-study tool is leakage-checked.

This is a standalone public MVP repo. It intentionally excludes private trading strategy,
backtests, runbooks, live execution, broker automation, and proprietary data.

## Eval (offline reproducible baseline)

| Metric | What it measures | Result |
|---|---|---|
| hit@1 | expected document is the first result | **1.00** (3 cases) |
| MRR | rank of the expected document | **1.00** (3 cases) |
| citation-grounding rate | accepted answers carry retrieved citations | **1.00** (3 accepted answers) |
| tool-call success rate | expected tool dispatched successfully | **1.00** (4 calls) |
| refusal accuracy | answerable vs weak-evidence decision is correct | **1.00** (4 cases) |
| leakage guard rate | accepts clean baseline and rejects cutoff leak | **1.00** (2 checks) |
| trajectory contract rate | scripted tool order matches the public trace | **1.00** (5 cases) |
| trace completeness rate | every tool attempt has all public trace fields | **1.00** (7 attempts) |
| recovery contract rate | injected failure reaches the expected recovery action | **1.00** (1 case) |
| mean tool steps | attempted tool actions per orchestration case | **1.40** (5 cases) |

These are deliberately small **offline smoke-baseline** results. Retrieval and refusal use
real Chroma embeddings over three sanitized fixtures. Agent grounding, tool-call, trace, and
trajectory values use deterministic observation-driven models to exercise the real ReAct loop;
they measure system contracts, not live-LLM planning quality. See
[eval/results.json](eval/results.json).

Reproduce the checked-in result without credentials:

```bash
.venv/bin/python -m app.eval_harness --output eval/results.json
```

> "I built a RAG agent" is weak. "I built a RAG agent and characterized its citation-grounding
> and retrieval quality on N queries, with a leakage-checked event-study tool" is the claim.

## Architecture

```
Client / API
     │
FastAPI + Pydantic        /ingest  /research  /event-study  /documents
     │
Python orchestration runtime
     ├──▶ LLM routing policy: choose next action
     ├──▶ Pydantic validation + bounded synchronous execution
     └──▶ 3 tools, including RAG (Chroma): retrieve → cite → refuse-if-weak
     │
Eval harness: retrieval · grounding · trajectories · recovery · refusal · leakage
```

## The 3 tools (exactly three)
1. `search_docs` — RAG retrieval over ingested docs; returns passages **with citations**;
   refuses when evidence is weak.
2. `get_price_data` — price/return series for a ticker over a window.
3. `run_event_study` — log abnormal returns and CAR around an event date + a
   **pre-event bootstrap CI**, leakage-checked.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
export AGENT_MODEL="your-model-name"
# Optional for an OpenAI-compatible provider:
# export OPENAI_BASE_URL="https://provider.example/v1"
uvicorn app.main:app --reload
# then: http://127.0.0.1:8000/docs
```

## Local Smoke Test

In Swagger, run `POST /ingest` with:

```json
{
  "path": "/absolute/path/to/agent-quant-research/sample.txt",
  "doc_id": "sample_apple"
}
```

`path` is a server-side file path, so use the absolute path on the machine running Uvicorn.
From the repo root, run `pwd` and append `/sample.txt`.

Then run `GET /search` with:

```text
q = what did Apple say about services revenue?
k = 4
```

The response should include a retrieved passage from `sample_apple::0`.

## Retrieval Contract

`/search` returns retrieved passages, not a generated answer. Each passage includes:

- `citation`: source pointer in the form `doc_id::chunk_id`, such as `sample_apple::0`.
- `distance`: raw Chroma distance; lower is closer.
- `score`: `1.0 - distance`; higher is closer.
- `score_kind`: currently `cosine_similarity`, because this collection is explicitly configured with Chroma's cosine HNSW space.

The refusal rule is intentionally simple for the MVP: if the top passage score is below
`REFUSE_SCORE_THRESHOLD = 0.25`, the API refuses rather than pretending it found evidence.

## Research Agent

`POST /research` runs a bounded ReAct-style loop over exactly the three tools above. The
model chooses which tool to call and in what order; application code retains control:

1. The model emits one strict JSON tool or final action.
2. Pydantic rejects unknown fields, missing arguments, and invalid values.
3. Only names in the three-tool registry can execute.
4. Tool observations return to the model, for at most six model steps.
5. A successful final answer must name citation identifiers actually returned by
   `search_docs`, and those identifiers must appear in the answer text.
6. The response exposes a bounded tool trace: step, selected tool, normalized arguments,
   success/failure, and a sanitized error category. It never exposes model chain-of-thought or
   full tool result payloads.

This guard proves **citation provenance**: the model cannot return a fabricated source id.
It does not prove that every sentence is semantically supported by the cited passage. That
claim-level grounding measurement belongs to the Day 5 corpus evaluation.

Example:

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"question":"What did Apple report about services revenue?"}'
```

If the model configuration is missing, the endpoint returns HTTP 503. Weak retrieval,
fabricated citations, malformed model actions, and step exhaustion return an explicit
research refusal instead of an unsupported answer.

### Orchestration ownership

The orchestration is hybrid. The LLM chooses the semantic next action: which tool to call,
with which arguments, and whether to call another tool or finalize. Python retains operational
control: it validates the action, executes one tool synchronously, returns the observation,
records the public trace, enforces the step budget, and accepts or refuses the final result.

There is no separate queue or background scheduler in this MVP. Scheduling is the direct
one-tool-at-a-time dispatch inside `run_agent`. A separate scheduler would become justified for
parallel, long-running, resumable, or resource-constrained jobs.

### Optional live model-routing smoke test

The checked-in metrics are deterministic. To make a small provider-backed check of actual model
tool selection, configure an OpenAI-compatible endpoint and run:

```bash
export OPENAI_API_KEY="your-key"
export AGENT_MODEL="your-model"
# Optional:
# export OPENAI_BASE_URL="https://provider.example/v1"
.venv/bin/python -m app.live_eval
```

The command uses two short cases, deterministic price/event fixtures, and an ephemeral Chroma
collection. Each case is capped at three model steps, so the default run can make at most six
model API calls. It reports only tool sequences, refusal state, step count, and latency; it does
not persist prompts, answers, citations, raw observations, or credentials. This is a smoke test,
not a statistically meaningful model benchmark, and it is deliberately excluded from CI.

## Price Data Tool

`get_price_data(ticker, start, end)` is a thin tool wrapper. It first tries cached 1-minute
bars from a local `spx-news-intraday` checkout, using `SPX_NEWS_INTRADAY_ROOT` if set or
`~/spx-news-intraday` if present. If that loader is unavailable, it falls back to optional
`yfinance` daily data. The returned payload is JSON-safe for the future agent loop:
`ticker`, `start`, `end`, `source`, `n_rows`, `columns`, and `rows`.
Public tool calls retain at most 2,000 rows; the event-study implementation requests the
full internal range so minute-bar truncation cannot discard its baseline or event window.

## Event Study Tool

`run_event_study(ticker, event_date, window)` computes close-to-close log abnormal returns
around an event. `event_date` accepts either `YYYY-MM-DD` or a timezone-aware ISO timestamp.
The MVP baseline is intentionally simple and auditable:

1. Load prices around the event. A timestamp before 16:00 New York time aligns to that
   observed trading day; a timestamp at/after 16:00, on a weekend, or on a holiday aligns
   to the next observed trading day.
2. Convert prices to daily close-to-close log returns in basis points.
3. Locate `[-window, +window]` by trading observations, then fit expected log return as
   the mean of returns strictly before the first observation in that window.
4. Report abnormal return (`AR = actual - expected`) and cumulative abnormal return
   (`CAR = cumsum(AR)`) for those trading observations.
5. Build a 95% percentile interval around CAR from 1,000 resamples of centered
   pre-event returns.

The leakage check is the main point: an explicit assertion rejects any baseline date on or
after the event-window start. The response returns the baseline and bootstrap source dates,
`n_pre_obs`, and a `passed`/`failed` leakage status. If prices are unavailable and alignment
cannot be established, `event_date` is `null` and leakage status is `not_run` rather than a
guessed result.

The response also returns `event_input`, the aligned `event_date`, and an `alignment` object
containing the New York-local timestamp and rule used. Timestamp inputs must include a
timezone offset. This MVP assumes the regular 16:00 close; an exchange calendar is still
needed for scheduled early-close sessions.

Example after-close request:

```json
{
  "ticker": "AAPL",
  "event_date": "2026-01-09T16:30:00-05:00",
  "window": 1
}
```

For the interview explanation and 10-minute self-quiz, see
[EVENT_STUDY_INTERVIEW.md](EVENT_STUDY_INTERVIEW.md).

## Test

```bash
.venv/bin/python -m pytest -q
```

## Build status
- [x] Day 1–2: FastAPI skeleton + `/ingest` + Chroma + `search_docs` (citations + refusal)
- [x] Day 3 foundation: `get_price_data` cache-first wrapper
- [x] Day 3 event study: `run_event_study` (pre-event bootstrap CAR CI + leakage check)
- [x] Day 4: bounded ReAct agent loop over the 3 tools + `/research`
- [x] Day 5 foundation: eval metric helpers + RAG refusal/citation regression tests
- [x] Day 5 corpus eval: isolated fixtures + reproducible offline metrics above
- [x] Day 6: orchestration traces + all-three-tool trajectory/recovery eval + capped live smoke
- [ ] Day 6–7: Dockerize, polish, make public

## Explicitly out of scope (known production path, deliberately not built)
Live trading · buy/sell claims · private strategy logic · backtesting · risk metrics · Qdrant
hybrid retrieval · reranking · LangGraph · MCP · AWS/Redis.
