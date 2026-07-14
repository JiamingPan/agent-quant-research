# Offline Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce deterministic, corpus-backed Day 5 metrics using real Chroma retrieval and the real bounded agent loop without external model calls.

**Architecture:** Add narrow dependency-injection points for the Chroma collection and agent tool registry. Build an `app.eval_harness` orchestrator around an ephemeral collection, checked-in fixtures, and a deterministic `AgentModel`, then serialize stable metrics for the README.

**Tech Stack:** Python 3.9+, Chroma, Pydantic 2, pytest, existing FastAPI application modules.

## Global Constraints

- Keep exactly the existing three tools.
- Do not call an external model during the default evaluation or tests.
- Do not mutate the persistent Chroma collection.
- Do not tune `REFUSE_SCORE_THRESHOLD` from this four-case fixture.
- Label deterministic agent results as contract metrics, not live-model quality.
- Do not add private SPX data, strategy logic, provider credentials, or machine paths.

---

### Task 1: Explicit Evaluation Isolation

**Files:**
- Modify: `app/rag.py`
- Modify: `app/agent.py`
- Create: `tests/test_eval_harness.py`

**Interfaces:**
- `rag.ingest(path, doc_id=None, *, collection=None) -> tuple[str, int]`.
- `rag.search(query, k=4, *, collection=None) -> tuple[list[Citation], bool, str | None]`.
- `agent.run_agent(question, max_steps=6, model=None, *, tool_registry=None) -> dict`.

- [ ] **Step 1: Write failing isolation tests**

Create fake collection objects that record `add` and `query` calls. Assert an explicitly supplied
collection receives ingest/search calls while a monkeypatched `rag._collection` receives none.
Create a one-step fake model and a supplied three-tool registry; assert `run_agent` calls its
`search_docs` function while a monkeypatched global `agent.TOOLS["search_docs"]` raises if used.

- [ ] **Step 2: Verify the tests fail on the current signatures**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py -k "collection or tool_registry" -q`

Expected: failures report unexpected `collection` and `tool_registry` keyword arguments.

- [ ] **Step 3: Add collection injection to RAG**

Inside each function select `active_collection = collection or _collection`, then call
`active_collection.add` or `active_collection.query`. Preserve all existing return shapes and
default behavior.

- [ ] **Step 4: Add tool-registry injection to the agent**

Pass the active registry into `_dispatch_tool`; look up both the argument model and callable by
the selected tool name. Default to `TOOLS` when no registry is supplied. Do not permit names
outside `TOOL_ARGUMENT_MODELS`, even if a supplied registry contains them.

- [ ] **Step 5: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py -q`

Expected: isolation tests pass.

Run: `.venv/bin/python -m pytest -q`

Expected: all pre-existing and new tests pass.

---

### Task 2: Corpus-Backed Harness and Stable Result

**Files:**
- Create: `app/eval_harness.py`
- Create: `eval/corpus/manifest.json`
- Create: `eval/corpus/apple_services.txt`
- Create: `eval/corpus/nvidia_data_center.txt`
- Create: `eval/corpus/fed_policy.txt`
- Create: `eval/cases.json`
- Create: `eval/results.json`
- Modify: `tests/test_eval_harness.py`

**Interfaces:**
- `OfflineEvalModel(question: str)` implements `complete(messages) -> str`.
- `run_offline_eval(corpus_dir: Path, cases_path: Path) -> dict`.
- CLI: `python -m app.eval_harness --output eval/results.json`.

- [ ] **Step 1: Write failing model and aggregation tests**

Test that `OfflineEvalModel` first emits a `search_docs` action containing the question, then
converts a successful observation to a final action with the top citation. Test that a refused
observation becomes a refused final action. With fake search and agent dependencies, assert the
suite emits all six metric names, counts answerable/all cases correctly, and passes both leakage
guard checks.

- [ ] **Step 2: Verify failures identify missing harness types**

Run: `.venv/bin/python -m pytest tests/test_eval_harness.py -k "offline or aggregate" -q`

Expected: import or attribute failures for `OfflineEvalModel` and `run_offline_eval`.

- [ ] **Step 3: Add sanitized fixtures and Pydantic case models**

Manifest entries contain only relative filename and stable doc id. Cases contain `question`,
nullable `expected_doc_id`, and `should_refuse`. Validate that answerable cases have an expected
document id and refusal cases omit it.

- [ ] **Step 4: Implement the harness**

Create an ephemeral cosine collection, ingest every manifest document through `rag.ingest`, and
define an eval `search_docs` adapter around
`rag.search(query, k=k, collection=collection)`. Compute
retrieval metrics on answerable cases. Run every case through `agent.run_agent` with a fresh
`OfflineEvalModel` and the injected three-tool registry. Aggregate refusal, grounding, and tool
records using the existing metric helpers. Exercise `_assert_no_baseline_leakage` once with safe
dates and once with a cutoff leak that must raise.

- [ ] **Step 5: Run the real corpus evaluation**

Run: `.venv/bin/python -m app.eval_harness --output eval/results.json`

Expected: JSON output contains three corpus documents, three retrieval cases, four agent/refusal
cases, and numeric values from zero to one for all metrics.

- [ ] **Step 6: Verify determinism**

Run the same command a second time and then run `git diff --exit-code eval/results.json` after
staging the first generated result.

Expected: the second run does not change the result.

---

### Task 3: Publish Metrics and Ownership Quiz

**Files:**
- Modify: `README.md`
- Modify: `UNDERSTAND.md`

**Interfaces:**
- Consumes: exact values and counts from `eval/results.json`.
- Produces: honest public metric table, rerun command, Day 5 trace, and self-quiz.

- [ ] **Step 1: Replace README placeholders with measured values**

Report hit@3, MRR, citation-grounding rate, tool-call success rate, refusal accuracy, and leakage
guard rate with case counts. Add the exact rerun command. State that retrieval/refusal use real
Chroma behavior while agent/tool/grounding values are deterministic contract metrics.

- [ ] **Step 2: Append the Day 5 ownership section**

Explain the fixture-to-report flow, why the collection is ephemeral, why the deterministic model
is useful, why 100% on a contract metric is not proof of live-model quality, and why four cases
are a smoke baseline rather than a statistically persuasive benchmark. Add five questions with
an answer key and a sixty-second interview explanation.

- [ ] **Step 3: Run final verification**

Run: `.venv/bin/python -m app.eval_harness --output eval/results.json`

Run: `.venv/bin/python -m pytest -q`

Run: `git diff --check`

Expected: deterministic result, complete green suite, and no whitespace errors.

- [ ] **Step 4: Review privacy and public claims**

Search the new diff for private source names, strategy terms, API-key values, absolute user paths,
and claims that offline contract metrics measure live-model intelligence. Remove any such content.

- [ ] **Step 5: Commit and push main**

```bash
git add app eval tests README.md UNDERSTAND.md docs/superpowers
git commit -m "feat: add reproducible offline eval harness"
git push origin main
```
