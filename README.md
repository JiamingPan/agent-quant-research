# Agent Quant Research

A RAG + agent service over financial documents and price data that produces reproducible,
leakage-checked research memos and event studies. **Not** live trading — no buy/sell claims.
The point is rigorous, reproducible research *infrastructure*: every claim cites a retrieved
passage, weak-evidence questions get refused, and the event-study tool is leakage-checked.

This is a standalone public MVP repo. It intentionally excludes private trading strategy,
backtests, runbooks, live execution, broker automation, and proprietary data.

## Eval (the headline — fill in as the harness lands)

| Metric | What it measures | Result |
|---|---|---|
| hit@k | retrieval: is the right passage in the top-k? | metric helper tested |
| MRR | retrieval: how high is the right passage ranked? | metric helper tested |
| citation-grounding rate | does every claim trace to a retrieved passage? | metric helper tested |
| tool-call success rate | agent picks + calls the right tool | metric helper tested |
| refusal-when-weak | refuses when evidence is insufficient | RAG behavior tested |
| leakage check | event study uses no look-ahead data | _[TODO]_ |

> "I built a RAG agent" is weak. "I built a RAG agent and characterized its citation-grounding
> and retrieval quality on N queries, with a leakage-checked event-study tool" is the claim.

## Architecture

```
Client / API
     │
FastAPI + Pydantic        /ingest  /research  /event-study  /documents
     │
Agent (ReAct, 3 tools) ──▶ RAG core (Chroma): retrieve → cite → refuse-if-weak
     │
Eval harness: hit@k · MRR · grounding · tool success · refusal · leakage
```

## The 3 tools (exactly three)
1. `search_docs` — RAG retrieval over ingested docs; returns passages **with citations**;
   refuses when evidence is weak.
2. `get_price_data` — price/return series for a ticker over a window.
3. `run_event_study` — abnormal returns around an event date + **bootstrap CI**, leakage-checked.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# then: http://127.0.0.1:8000/docs
```

## Test

```bash
python -m pytest -q
```

## Build status
- [x] Day 1–2: FastAPI skeleton + `/ingest` + Chroma + `search_docs` (citations + refusal)
- [ ] Day 3: `get_price_data` + `run_event_study` (bootstrap CI + leakage check)
- [ ] Day 4: ReAct agent loop over the 3 tools + `/research`
- [x] Day 5 foundation: eval metric helpers + RAG refusal/citation regression tests
- [ ] Day 5 corpus eval: labeled query set + numbers above
- [ ] Day 6–7: Dockerize, polish, make public

## Explicitly out of scope (known production path, deliberately not built)
Live trading · buy/sell claims · private strategy logic · backtesting · risk metrics · Qdrant
hybrid retrieval · reranking · LangGraph · MCP · AWS/Redis.
