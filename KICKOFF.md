# Agent Quant Research - Public MVP Complete

The MVP is a runnable financial-research service with citation-grounded retrieval, three
validated tools, bounded LLM orchestration, explicit refusal behavior, and a deterministic
offline evaluation. It is intentionally research infrastructure, not a trading strategy.

## What is implemented

- FastAPI endpoints: `/health`, `/ingest`, `/search`, `/documents`, `/research`, and
  `/event-study`.
- Chroma RAG: overlapping chunks, cosine retrieval, `doc_id::chunk_id` citations, and a
  weak-evidence refusal gate.
- Exactly three tools: document search, price retrieval, and a leakage-checked event study.
- Public daily-price fallback through installed `yfinance`; an external private cache is
  optional and is not part of this repository.
- Bounded ReAct loop: the LLM selects actions while Python validates, dispatches, traces, and
  enforces citations and termination.
- Deterministic offline eval for retrieval, refusal, citation provenance, tool trajectories,
  failure recovery, and leakage guards.
- Non-root Docker image and credential-free GitHub Actions checks.

## Final verification

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m app.eval_harness --output /tmp/eval-results.json
diff -u eval/results.json /tmp/eval-results.json
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`. Ingest `sample.txt`, search for Apple services revenue,
then search for an unrelated scientific topic and confirm that weak evidence is refused.

## Interview points

- **Grounding:** retrieval returns source passages and stable `doc_id::chunk_id` identifiers.
  The agent may cite only identifiers that were actually retrieved.
- **Refusal:** a top result is not automatically sufficient evidence. The explicit threshold
  makes unsupported-answer behavior testable.
- **Orchestration:** the LLM is the routing policy; Python remains the controlled runtime with
  schemas, an allowlist, a six-step cap, safe traces, and deterministic termination.
- **Leakage:** event-study baseline returns and bootstrap residuals must be strictly earlier
  than the event window. Code asserts this invariant and tests both pass and failure paths.
- **Evaluation honesty:** perfect scores on a tiny sanitized corpus prove MVP contracts, not
  production accuracy or live-model intelligence.
- **Release boundary:** Docker proves reproducible packaging; GitHub Actions proves the clean
  build and offline checks. Neither is a production cloud deployment.

## Deliberately out of scope

No private strategy logic, broker execution, buy/sell output, proprietary datasets, vLLM,
LangGraph, MCP, queueing system, or cloud deployment. Add those only when a measured product
requirement justifies them.
