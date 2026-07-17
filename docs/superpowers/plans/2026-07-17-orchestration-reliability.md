# Orchestration Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add inspectable agent execution traces, deterministic all-three-tool trajectory evaluation, and a capped optional live-model routing smoke test.

**Architecture:** Keep the existing synchronous bounded `run_agent` loop. The LLM continues to propose semantic actions, while Python validates and dispatches them; additive trace fields expose only tool metadata. Offline scripted trajectories verify runtime contracts, while a separate credential-gated command measures live model routing against ephemeral RAG and deterministic price/event fixtures.

**Tech Stack:** Python 3.9+, FastAPI, Pydantic v2, Chroma, OpenAI-compatible chat completions, pytest.

## Global Constraints

- Keep exactly three tools: `search_docs`, `get_price_data`, and `run_event_study`.
- Do not add vLLM, LangGraph, MCP, a multi-agent system, a queue, or a fourth tool.
- Never persist raw model messages, chain-of-thought, full tool payloads, credentials, private market data, or private strategy logic.
- Keep normal tests and deterministic evaluation network-free and credential-free.
- Cap the default live smoke run at two cases and three model steps per case: at most six paid model calls.
- Keep provider-dependent live results separate from `eval/results.json`.

---

## File Map

- `app/agent.py`: trace collection, normalized dispatch metadata, step accounting, and public environment-backed model factory.
- `app/models.py`: additive `AgentTraceStep` and research response fields.
- `app/main.py`: serialize trace and step count from the agent result.
- `app/eval.py`: pure orchestration contract metric functions.
- `app/eval_harness.py`: scripted all-three-tool trajectories and aggregate offline metrics.
- `app/live_eval.py`: optional two-case live-model routing command.
- `eval/orchestration_cases.json`: deterministic trajectory definitions.
- `eval/results.json`: reproducible offline result after the new metrics are added.
- `tests/test_agent.py`: runtime and API trace behavior.
- `tests/test_eval.py`: pure metric behavior.
- `tests/test_eval_harness.py`: offline trajectory aggregation.
- `tests/test_live_eval.py`: live-runner behavior with a fake model; no network calls.
- `README.md`: architecture, commands, metrics, and limitation language.
- `UNDERSTAND.md`: orchestration ownership and evaluation self-quiz.

---

### Task 1: Add Safe Runtime Traces

**Files:**
- Modify: `app/agent.py:145-278`
- Modify: `app/models.py:35-45`
- Modify: `app/main.py:45-57`
- Test: `tests/test_agent.py`

**Interfaces:**
- Produces: `run_agent(question: str, max_steps: int = 6, model: AgentModel | None = None,
  *, tool_registry: Mapping[str, Any] | None = None) -> dict` with additive
  `trace: list[dict]` and `steps_used: int`.
- Produces: `model_from_env() -> AgentModel` as the public replacement for `_default_model()`.
- Produces: API model `AgentTraceStep(step, tool, arguments, ok, error)`.
- Preserves: existing answer, citation, confidence, refusal, tool validation, and step-limit behavior.

- [ ] **Step 1: Write failing trace tests**

Add assertions to the existing successful, recovery, exhaustion, and endpoint tests:

```python
assert result["trace"] == [
    {
        "step": 1,
        "tool": "search_docs",
        "arguments": {"query": "Apple services", "k": 4},
        "ok": True,
        "error": None,
    }
]
assert result["steps_used"] == 2
```

For the unknown-tool recovery test, assert a sanitized failure followed by success:

```python
assert result["trace"] == [
    {
        "step": 1,
        "tool": "trade_stock",
        "arguments": {"ticker": "AAPL"},
        "ok": False,
        "error": "unknown tool: trade_stock",
    },
    {
        "step": 2,
        "tool": "search_docs",
        "arguments": {"query": "Apple", "k": 4},
        "ok": True,
        "error": None,
    },
]
assert result["steps_used"] == 3
```

For exhaustion, assert two trace entries and `steps_used == 2`. Update the mocked endpoint
result and expected JSON to include the additive fields.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent.py -q
```

Expected: failures for missing `trace` and `steps_used` fields.

- [ ] **Step 3: Add dispatch metadata and sanitized trace helpers**

Make `_dispatch_tool` include normalized arguments in every observation. Successful validation
uses `arguments.model_dump()`; unknown or invalid actions use `dict(action.arguments)`.

```python
def _sanitized_trace_error(observation: dict) -> str | None:
    error = str(observation.get("error") or "")
    if not error:
        return None
    if error.startswith("invalid arguments:"):
        return "invalid arguments"
    if error.startswith("tool failed:"):
        parts = error.split(":", 2)
        return ":".join(parts[:2])
    return error


def _trace_entry(step: int, action: ToolAction, observation: dict) -> dict:
    return {
        "step": step,
        "tool": action.name,
        "arguments": dict(observation.get("arguments") or action.arguments),
        "ok": bool(observation.get("ok")),
        "error": _sanitized_trace_error(observation),
    }
```

Do not remove the detailed observation error sent back to the model; only the public trace is
sanitized.

- [ ] **Step 4: Thread trace and step count through every exit path**

Change `_refusal` and `_finalize` to accept trace metadata:

```python
def _refusal(
    reason: str,
    *,
    trace: list[dict] | None = None,
    steps_used: int = 0,
) -> dict:
    return {
        "answer": reason,
        "citations": [],
        "confidence": 0.0,
        "refused": True,
        "trace": list(trace or []),
        "steps_used": steps_used,
    }
```

Enumerate model calls from one, append a trace entry after each tool attempt, and preserve the
trace for malformed actions, final answers, and exhaustion:

```python
trace: list[dict] = []
for step in range(1, max_steps + 1):
    raw_action = model.complete(messages)
    try:
        action = _parse_action(raw_action)
    except AgentActionError as exc:
        return _refusal(
            f"Refused: malformed model action: {exc}",
            trace=trace,
            steps_used=step,
        )
    if isinstance(action, FinalAction):
        return _finalize(action, citation_registry, trace, step)
    observation = _dispatch_tool(action, tool_registry=tool_registry)
    trace.append(_trace_entry(step, action, observation))
```

Rename `_default_model` to `model_from_env` and make `run_agent` call the public name.

- [ ] **Step 5: Add the API response models and serialization**

Add to `app/models.py`:

```python
class AgentTraceStep(BaseModel):
    step: int = Field(ge=1)
    tool: str
    arguments: dict[str, object]
    ok: bool
    error: Optional[str] = None


class ResearchResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    refused: bool = False
    trace: list[AgentTraceStep] = Field(default_factory=list)
    steps_used: int = Field(0, ge=0)
```

In `app/main.py`, construct `AgentTraceStep` objects and pass `steps_used` from `out`.

- [ ] **Step 6: Run focused and full tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_agent.py -q
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit runtime tracing**

```bash
git add app/agent.py app/models.py app/main.py tests/test_agent.py
git commit -m "feat: expose bounded agent tool traces"
```

---

### Task 2: Evaluate Offline Orchestration Contracts

**Files:**
- Create: `eval/orchestration_cases.json`
- Modify: `app/eval.py`
- Modify: `app/eval_harness.py`
- Modify: `tests/test_eval.py`
- Modify: `tests/test_eval_harness.py`

**Interfaces:**
- Consumes: `run_agent` trace entries from Task 1.
- Produces: pure metrics `trajectory_contract_rate`, `trace_completeness_rate`,
  `recovery_contract_rate`, and `mean_tool_steps`.
- Produces: `OrchestrationCase` and `OrchestrationEvalModel` in `app.eval_harness`.
- Produces: four additive orchestration metrics in `run_offline_eval()`.

- [ ] **Step 1: Write failing pure metric tests**

Add to `tests/test_eval.py`:

```python
def test_orchestration_contract_metrics():
    trajectories = [
        {"expected": ["search_docs"], "actual": ["search_docs"]},
        {
            "expected": ["search_docs", "run_event_study"],
            "actual": ["run_event_study", "search_docs"],
        },
    ]
    traces = [
        [{"step": 1, "tool": "search_docs", "arguments": {}, "ok": True, "error": None}],
        [{"step": 1, "tool": "get_price_data", "arguments": {}, "ok": False, "error": "tool failed: RuntimeError"}],
    ]
    recoveries = [{"required": True, "recovered": True}, {"required": False, "recovered": False}]

    assert trajectory_contract_rate(trajectories) == 0.5
    assert trace_completeness_rate(traces) == 1.0
    assert recovery_contract_rate(recoveries) == 1.0
    assert mean_tool_steps(traces) == 1.0
```

- [ ] **Step 2: Run the metric test and confirm import failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_eval.py::test_orchestration_contract_metrics -q
```

Expected: import failure because the metric functions do not exist.

- [ ] **Step 3: Implement the pure metrics**

Add to `app/eval.py`:

```python
def trajectory_contract_rate(records: Sequence[Mapping[str, object]]) -> float:
    if not records:
        return 0.0
    return sum(record.get("expected") == record.get("actual") for record in records) / len(records)


def trace_completeness_rate(traces: Sequence[Sequence[Mapping[str, object]]]) -> float:
    entries = [entry for trace in traces for entry in trace]
    if not entries:
        return 0.0
    required = {"step", "tool", "arguments", "ok", "error"}
    return sum(required.issubset(entry.keys()) for entry in entries) / len(entries)


def recovery_contract_rate(records: Sequence[Mapping[str, object]]) -> float:
    required = [record for record in records if bool(record.get("required"))]
    if not required:
        return 0.0
    return sum(bool(record.get("recovered")) for record in required) / len(required)


def mean_tool_steps(traces: Sequence[Sequence[Mapping[str, object]]]) -> float:
    if not traces:
        return 0.0
    return sum(len(trace) for trace in traces) / len(traces)
```

Export/import the new functions in the tests and run `tests/test_eval.py -q`.

- [ ] **Step 4: Add five explicit trajectory fixtures**

Create `eval/orchestration_cases.json` with strict cases shaped as follows:

```json
[
  {
    "case_id": "search_only",
    "question": "What did Apple report about Services revenue?",
    "steps": [
      {"tool": "search_docs", "arguments": {"query": "Apple Services revenue", "k": 3}, "should_succeed": true}
    ],
    "recovery_required": false
  },
  {
    "case_id": "price_only",
    "question": "Retrieve SPY prices for the first week of January 2026.",
    "steps": [
      {"tool": "get_price_data", "arguments": {"ticker": "SPY", "start": "2026-01-02", "end": "2026-01-09"}, "should_succeed": true}
    ],
    "recovery_required": false
  },
  {
    "case_id": "event_only",
    "question": "Run a one-day SPY event study around January 7, 2026.",
    "steps": [
      {"tool": "run_event_study", "arguments": {"ticker": "SPY", "event_date": "2026-01-07", "window": 1}, "should_succeed": true}
    ],
    "recovery_required": false
  },
  {
    "case_id": "search_then_event",
    "question": "Summarize the Fed policy excerpt and evaluate SPY around January 7, 2026.",
    "steps": [
      {"tool": "search_docs", "arguments": {"query": "Federal Reserve inflation interest rates", "k": 3}, "should_succeed": true},
      {"tool": "run_event_study", "arguments": {"ticker": "SPY", "event_date": "2026-01-07", "window": 1}, "should_succeed": true}
    ],
    "recovery_required": false
  },
  {
    "case_id": "failure_then_search",
    "question": "Recover from unavailable price data and retrieve Apple Services evidence.",
    "steps": [
      {"tool": "get_price_data", "arguments": {"ticker": "FAIL", "start": "2026-01-02", "end": "2026-01-09"}, "should_succeed": false},
      {"tool": "search_docs", "arguments": {"query": "Apple Services revenue", "k": 3}, "should_succeed": true}
    ],
    "recovery_required": true
  }
]
```

- [ ] **Step 5: Write the failing harness aggregation test**

Extend `test_run_offline_eval_aggregates_stable_contract_metrics` to assert:

```python
assert result["cases"]["orchestration"] == 5
assert result["metrics"]["trajectory_contract_rate"] == 1.0
assert result["metrics"]["trace_completeness_rate"] == 1.0
assert result["metrics"]["recovery_contract_rate"] == 1.0
assert result["metrics"]["mean_tool_steps"] == 1.4
```

Run the test and expect failure because `run_offline_eval` does not load orchestration cases.

- [ ] **Step 6: Implement scripted trajectory execution**

Add strict Pydantic models:

```python
class OrchestrationStep(_StrictModel):
    tool: str = Field(min_length=1)
    arguments: dict[str, Any]
    should_succeed: bool


class OrchestrationCase(_StrictModel):
    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    steps: list[OrchestrationStep] = Field(min_length=1)
    recovery_required: bool
```

Implement `OrchestrationEvalModel` so each call emits the next scripted tool action. After all
tools, it returns a grounded final answer when a successful `search_docs` observation supplied
a citation; otherwise it returns an explicit refusal. Record no separate call history: use the
real `run_agent` trace as the evaluation source.

Use deterministic tools:

```python
def fixed_prices(ticker: str, start: str, end: str) -> dict:
    if ticker == "FAIL":
        raise RuntimeError("injected unavailable price fixture")
    return {"ticker": ticker, "start": start, "end": end, "source": "eval_fixture", "n_rows": 2, "rows": []}


def fixed_event(ticker: str, event_date: str, window: int = 5) -> dict:
    return {"ticker": ticker, "event_date": event_date, "window": window, "source": "eval_fixture", "car_bps": 12.0}
```

For each case, call `run_agent` with `max_steps=len(case.steps) + 1`, compare trace tool names and
success flags with the fixture, and aggregate the four pure metrics. A recovery counts only when
the failed expected step is followed by the expected successful step.

- [ ] **Step 7: Run focused and full tests**

```bash
.venv/bin/python -m pytest tests/test_eval.py tests/test_eval_harness.py -q
.venv/bin/python -m pytest -q
```

Expected: all tests pass and the original metrics remain unchanged.

- [ ] **Step 8: Commit offline orchestration evaluation**

```bash
git add app/eval.py app/eval_harness.py eval/orchestration_cases.json tests/test_eval.py tests/test_eval_harness.py
git commit -m "feat: evaluate multi-tool orchestration contracts"
```

---

### Task 3: Add the Capped Live-Model Smoke Command

**Files:**
- Create: `app/live_eval.py`
- Create: `tests/test_live_eval.py`
- Modify: `app/eval_harness.py`

**Interfaces:**
- Consumes: `agent.model_from_env`, `agent.run_agent`, `OrchestrationCase`, and the ephemeral eval corpus.
- Produces: `run_live_eval(model=None, collection=None) -> dict`.
- Produces CLI: `.venv/bin/python -m app.live_eval [--output PATH]`.
- Guarantees: default `max_cases == 2`, `max_steps == 3`, and `max_model_calls == 6`.

- [ ] **Step 1: Expose a shared eval collection loader**

Refactor the existing corpus-ingest loop in `app/eval_harness.py` into:

```python
def load_eval_collection(corpus_dir: Path = DEFAULT_CORPUS_DIR, collection: Any = None) -> tuple[Any, int]:
    corpus_dir = Path(corpus_dir)
    manifest = _load_models(corpus_dir / "manifest.json", CorpusEntry)
    active_collection = collection if collection is not None else _ephemeral_collection()
    for item in manifest:
        document_path = corpus_dir / item.filename
        if not document_path.is_file():
            raise FileNotFoundError(f"missing eval corpus file: {item.filename}")
        rag.ingest(str(document_path), item.doc_id, collection=active_collection)
    return active_collection, len(manifest)
```

Make `run_offline_eval` call this helper. Run `tests/test_eval_harness.py -q` and expect it to pass.

- [ ] **Step 2: Write network-free tests for the live runner**

Create `tests/test_live_eval.py` with a scripted model producing exactly five model decisions:

```python
outputs = [
    '{"type":"tool","name":"search_docs","arguments":{"query":"Apple Services revenue","k":3}}',
    '{"type":"final","answer":"Supported [apple::0].","citation_ids":["apple::0"],"confidence":0.8,"refused":false}',
    '{"type":"tool","name":"search_docs","arguments":{"query":"Federal Reserve inflation interest rates","k":3}}',
    '{"type":"tool","name":"run_event_study","arguments":{"ticker":"SPY","event_date":"2026-01-07","window":1}}',
    '{"type":"final","answer":"Supported [apple::0].","citation_ids":["apple::0"],"confidence":0.8,"refused":false}'
]
```

Inject `HarnessCollection` or an equivalent local collection and assert:

```python
assert result["mode"] == "live_model_smoke"
assert result["n_cases"] == 2
assert result["max_model_calls"] == 6
assert result["model_calls_used"] == 5
assert result["ordered_trajectory_accuracy"] == 1.0
assert result["tool_coverage"] == 1.0
assert "answer" not in json.dumps(result)
```

Also test that no model is constructed at import time and missing environment configuration
raises `AgentConfigurationError` only when `run_live_eval()` is called without an injected model.

- [ ] **Step 3: Run the live-runner tests and confirm failure**

```bash
.venv/bin/python -m pytest tests/test_live_eval.py -q
```

Expected: import failure because `app.live_eval` does not exist.

- [ ] **Step 4: Implement the live runner with fixed scope**

In `app/live_eval.py`, define:

```python
LIVE_CASE_IDS = ("search_only", "search_then_event")
MAX_STEPS_PER_CASE = 3
MAX_MODEL_CALLS = len(LIVE_CASE_IDS) * MAX_STEPS_PER_CASE
```

Load only those cases from `eval/orchestration_cases.json`. Build a registry containing real
ephemeral `search_docs` plus deterministic `get_price_data` and `run_event_study` fixtures. For
each case, measure elapsed time with `time.perf_counter()`, call `run_agent` with
`max_steps=MAX_STEPS_PER_CASE`, and retain only:

```python
{
    "case_id": case.case_id,
    "expected_tools": expected_tools,
    "actual_tools": actual_tools,
    "ordered_match": actual_tools == expected_tools,
    "refused": bool(agent_result["refused"]),
    "steps_used": int(agent_result["steps_used"]),
    "elapsed_seconds": round(elapsed, 3),
}
```

Aggregate `model_calls_used`, exact ordered trajectory accuracy, expected-tool coverage (observed
expected tools divided by all expected tools), and the hard maximum. Do not include final answer
text, raw messages, citations, or tool results. Wrap provider failures without their message so
credentials or provider response bodies cannot reach output:

```python
try:
    agent_result = agent.run_agent(
        case.question,
        max_steps=MAX_STEPS_PER_CASE,
        model=active_model,
        tool_registry=registry,
    )
except Exception as exc:
    raise RuntimeError(
        f"live model case {case.case_id} failed: {type(exc).__name__}"
    ) from exc
```

- [ ] **Step 5: Implement the CLI**

Use `argparse` with only an optional `--output Path`. Print sorted indented JSON. When output is
provided, create its parent directory and write the same JSON with a trailing newline. Catch
`AgentConfigurationError` or the sanitized `RuntimeError`, print its message to stderr, and exit
nonzero.

- [ ] **Step 6: Run focused and full tests**

```bash
.venv/bin/python -m pytest tests/test_live_eval.py tests/test_eval_harness.py -q
.venv/bin/python -m pytest -q
```

Expected: all tests pass with no network calls.

- [ ] **Step 7: Commit the optional live smoke command**

```bash
git add app/live_eval.py app/eval_harness.py tests/test_live_eval.py
git commit -m "feat: add capped live agent routing smoke test"
```

---

### Task 4: Document, Verify, Run Once, and Publish the Increment

**Files:**
- Modify: `README.md`
- Modify: `UNDERSTAND.md`
- Modify: `eval/results.json`

**Interfaces:**
- Consumes: offline and live commands from Tasks 2 and 3.
- Produces: public documentation that accurately distinguishes LLM routing, Python orchestration,
  synchronous scheduling, offline contracts, and live routing quality.

- [ ] **Step 1: Reproduce and save deterministic metrics**

```bash
.venv/bin/python -m app.eval_harness --output eval/results.json
```

Expected: existing six metrics remain `1.0`; the new contract rates are `1.0`; mean tool steps is
`1.4`; orchestration case count is `5`.

- [ ] **Step 2: Update the README**

Add four orchestration rows to the eval table and explain:

```text
The LLM chooses the semantic next action. Python validates and executes that action,
returns the observation, and owns the step budget and termination. The current scheduler
is synchronous: one tool executes at a time in the same process.
```

Document the live command:

```bash
export OPENAI_API_KEY="your-key"
export AGENT_MODEL="your-model"
export OPENAI_BASE_URL="https://provider.example/v1"  # optional
.venv/bin/python -m app.live_eval
```

State explicitly that it uses two cases and no more than six model calls, and that live results
are smoke evidence rather than a benchmark.

- [ ] **Step 3: Add the orchestration self-quiz**

Append questions and concise answers to `UNDERSTAND.md` covering:

1. Which decisions belong to the LLM?
2. Which controls remain deterministic?
3. Why is the synchronous loop both orchestrator and scheduler in this MVP?
4. Why does the trace omit model reasoning and full observations?
5. What does offline trajectory accuracy prove, and what does it not prove?
6. Why should objective tool routing be code-scored instead of LLM-judged?

- [ ] **Step 4: Run the full deterministic verification**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m app.eval_harness --output /tmp/agent-quant-eval-check.json
git diff --check
```

Expected: all tests pass, the eval command exits zero, and `git diff --check` prints nothing.

- [ ] **Step 5: Run one tightly capped private-credential smoke test**

Load the existing credential into the shell without printing it, map the existing model variable
to `AGENT_MODEL`, and run:

```bash
.venv/bin/python -m app.live_eval --output /tmp/agent-quant-live-smoke.json
```

Expected: at most six model calls. Inspect only the aggregate routing result. Do not copy the
credential file, output file, private data, or provider-specific response into the repository.

- [ ] **Step 6: Scan the staged public diff for secret-shaped content**

```bash
git grep -nE 'sk-[A-Za-z0-9_-]{20,}' -- . ':!docs/superpowers/plans/2026-07-17-orchestration-reliability.md'
git status --short
```

Expected: no credential value appears; only intended public files are modified.

- [ ] **Step 7: Commit documentation and measured offline results**

```bash
git add README.md UNDERSTAND.md eval/results.json
git commit -m "docs: explain orchestration reliability metrics"
```

- [ ] **Step 8: Verify commit history and push**

```bash
git status --short --branch
git log --oneline -6
git push origin main
```

Expected: clean `main`, the four implementation commits are present, and `origin/main` advances
to the documentation commit.
