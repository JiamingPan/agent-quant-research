# Kickoff — MVP Increment (Day 1-2)

Scaffold is in place. The first increment is to make the RAG path real and prove it works.
Keep the implementation small enough that every design choice can be explained from first
principles.

## What's already here (runnable skeleton)
- FastAPI app with `/health`, `/ingest`, `/search`, `/documents` live; `/research` + `/event-study` return 501.
- `app/rag.py` — working chunk → Chroma → retrieve-with-citations → refuse-if-weak.
- `app/tools.py` — `search_docs` wired; `get_price_data` / `run_event_study` stubbed (Day 3).
- `app/agent.py` — ReAct loop stub (Day 4).
- `tests/test_eval.py` — metric definitions + passing unit tests (Day 5 fills the real harness).

## Checklist
1. `pip install -r requirements.txt`, then `uvicorn app.main:app --reload` → open `/docs`.
2. Ingest a couple of real financial PDFs (a 10-K, an earnings call) via `POST /ingest`.
3. Hit `GET /search?q=...` — confirm you get cited passages, and that an off-topic query **refuses**.
4. Tune `REFUSE_SIMILARITY` in `rag.py` on a few real queries so refusal fires sensibly.
5. Commit. `git init` if needed; push to the standalone public repo.

## Talking points to lock in while building
- **Why refusal matters:** a research tool that confidently answers with no evidence is worse than
  one that says "insufficient evidence." Refusal-when-weak is a rigor feature, not a limitation.
- **Chunking tradeoff:** bigger chunks = more context but noisier retrieval; overlap avoids
  splitting a fact across a boundary. You picked ~1000 chars / 150 overlap — be ready to defend it.
- **Citations = grounding:** every passage carries (doc_id, chunk_id) so a claim traces back to
  source. This is what makes the later citation-grounding eval possible.
- **Scope discipline:** exactly 3 tools, and a README TODO list of what you deliberately didn't
  build (reranking, hybrid retrieval, LangGraph). Knowing the production path without spending the
  time is the signal.

## Don't
- Add a 4th tool. Add LangGraph/MCP. Gold-plate retrieval. The MVP's job is threshold + hedge,
  not to beat agent specialists. Ship the eval table; that's the headline.
