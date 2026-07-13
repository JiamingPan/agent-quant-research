# ReAct Agent Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bounded, citation-enforced ReAct loop over the existing three research tools and expose it through `POST /research`.

**Architecture:** `app/agent.py` owns a provider-independent loop whose model dependency returns strict JSON actions. The loop validates tool arguments, dispatches only the existing registry, records citations returned by retrieval, and accepts a final answer only when its citation identifiers are proven to come from those observations. `app/main.py` remains a thin HTTP adapter, while tests inject a scripted model and monkeypatched tools.

**Tech Stack:** Python 3.9+, Pydantic 2, FastAPI, OpenAI Python SDK, pytest.

## Global Constraints

- Keep exactly three tools: `search_docs`, `get_price_data`, and `run_event_study`.
- Keep the loop bounded at six steps by default.
- No external API calls in tests.
- A non-refused answer must cite passages actually returned by `search_docs`.
- Invalid or insufficient evidence must produce an explicit refusal.
- Do not add LangGraph, MCP, streaming, memory, trading logic, or private strategy data.
- Preserve unrelated working-tree changes.

## File Map

- Create `tests/test_agent.py`: scripted-model unit tests for loop behavior and citation enforcement.
- Modify `app/agent.py`: action schemas, model protocol and adapter, tool dispatch, citation guard, and bounded loop.
- Modify `app/main.py`: connect `POST /research` and map missing model configuration to HTTP 503.
- Modify `requirements.txt`: make the OpenAI SDK an installed runtime dependency.
- Modify `README.md`: document configuration, request example, and completed Day 4 status.
- Modify `UNDERSTAND.md`: add the ReAct request trace and ten-minute ownership quiz without deleting existing material.

---

### Task 1: Bounded Core Loop and Citation Guard

**Files:**
- Create: `tests/test_agent.py`
- Modify: `app/agent.py`

**Interfaces:**
- Consumes: existing `TOOLS` mapping from tool names to JSON-returning callables.
- Produces: `AgentModel.complete(messages: list[dict[str, str]]) -> str` and `run_agent(question: str, max_steps: int = 6, model: AgentModel | None = None) -> dict`.
- Produces response keys: `answer`, `citations`, `confidence`, `refused`.

- [ ] **Step 1: Write a scripted model and failing grounded-answer test**

```python
class ScriptedModel:
    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages.copy())
        return next(self.outputs)


def test_run_agent_retrieves_then_returns_grounded_answer(monkeypatch):
    monkeypatch.setitem(agent.TOOLS, "search_docs", lambda query, k=4: {
        "passages": [{
            "doc_id": "apple",
            "chunk_id": 2,
            "citation": "apple::2",
            "text": "Services revenue reached a record level.",
            "distance": 0.2,
            "score": 0.8,
            "score_kind": "cosine_similarity",
        }],
        "refused": False,
        "reason": None,
    })
    model = ScriptedModel([
        '{"type":"tool","name":"search_docs","arguments":{"query":"Apple services","k":4}}',
        '{"type":"final","answer":"Services reached a record [apple::2].","citation_ids":["apple::2"],"confidence":0.9,"refused":false}',
    ])

    result = agent.run_agent("What happened to services?", model=model)

    assert result["refused"] is False
    assert result["citations"][0]["citation"] == "apple::2"
```

- [ ] **Step 2: Run the test and verify the stub fails**

Run: `.venv/bin/python -m pytest tests/test_agent.py::test_run_agent_retrieves_then_returns_grounded_answer -q`

Expected: FAIL because `run_agent` raises `NotImplementedError` and does not accept `model`.

- [ ] **Step 3: Implement the minimal model protocol and strict action schemas**

Add to `app/agent.py`:

```python
class AgentModel(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


class ToolAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool"]
    name: str
    arguments: dict[str, Any]


class FinalAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["final"]
    answer: str
    citation_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    refused: bool
```

Define Pydantic argument models with `extra="forbid"`:

```python
class SearchDocsArgs(BaseModel):
    query: str
    k: int = Field(default=4, ge=1, le=20)

class GetPriceDataArgs(BaseModel):
    ticker: str
    start: str
    end: str

class RunEventStudyArgs(BaseModel):
    ticker: str
    event_date: str
    window: int = Field(default=5, ge=0)
```

- [ ] **Step 4: Implement dispatch, observation recording, and final validation**

Implement `_parse_action`, `_dispatch_tool`, `_register_citations`, `_finalize`, and
`_refusal` as separate helpers. `_parse_action` uses `json.loads`, selects `ToolAction` or
`FinalAction` from the `type` discriminator, and converts JSON or Pydantic failures into an
`AgentActionError`. `_dispatch_tool` looks up the argument model below, validates with
`model_validate`, invokes the matching `TOOLS` callable with `model_dump()`, and returns either
`{"ok": true, "tool": name, "result": result}` or a JSON-safe error observation.

```python
TOOL_ARGUMENT_MODELS = {
    "search_docs": SearchDocsArgs,
    "get_price_data": GetPriceDataArgs,
    "run_event_study": RunEventStudyArgs,
}
```

`_finalize` must reject a non-refused answer when `citation_ids` is empty, an identifier is
absent from the registry, or an identifier is absent from `answer`. It returns citation objects
in declared order with duplicates removed.

`run_agent` must append the raw assistant action and a JSON `TOOL_OBSERVATION` message after
every tool call. Unknown tools, bad arguments, and tool exceptions become observations with
`{"ok": false, "error": "specific validation or dispatch message"}` so the model may recover.
Malformed JSON or an unsupported action type returns a refusal beginning with
`Malformed model action:`
immediately. Exhausting `max_steps` returns a bounded-loop refusal.

- [ ] **Step 5: Run the grounded-answer test**

Run: `.venv/bin/python -m pytest tests/test_agent.py::test_run_agent_retrieves_then_returns_grounded_answer -q`

Expected: `1 passed`.

- [ ] **Step 6: Add failing coverage for multiple tools and failure paths**

Add tests named `test_model_can_call_price_then_search_before_answer`,
`test_unknown_tool_is_observed_and_model_can_recover`,
`test_invalid_arguments_are_observed_and_model_can_recover`,
`test_retrieval_refusal_cannot_become_uncited_answer`,
`test_fabricated_citation_is_refused`,
`test_citation_missing_from_answer_text_is_refused`, `test_malformed_action_is_refused`, and
`test_max_steps_exhaustion_refuses`. Each test uses complete JSON strings in `ScriptedModel`;
the final test supplies two tool actions, calls `run_agent(question, max_steps=2, model=model)`, and asserts the
result is refused with `confidence == 0.0` and an answer containing `maximum 2 steps`.

The recovery tests inspect `ScriptedModel.messages` and assert the next model call contains an
`ok: false` observation. The max-step test uses repeated valid tool actions with `max_steps=2`.

- [ ] **Step 7: Run the agent tests and fix only core-loop defects**

Run: `.venv/bin/python -m pytest tests/test_agent.py -q`

Expected: all agent tests pass with no external API calls.

- [ ] **Step 8: Commit the core loop**

```bash
git add app/agent.py tests/test_agent.py
git commit -m "feat: add bounded citation-grounded agent loop"
```

---

### Task 2: Default Model Adapter and Research Endpoint

**Files:**
- Modify: `app/agent.py`
- Modify: `app/main.py`
- Modify: `requirements.txt`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `run_agent` from Task 1 with its optional `model` parameter left unset.
- Produces: `AgentConfigurationError`, the `OpenAICompatibleModel.complete` method, and live `POST /research`.

- [ ] **Step 1: Add failing endpoint tests**

```python
def test_research_endpoint_returns_agent_result(monkeypatch):
    monkeypatch.setattr(agent, "run_agent", lambda question: {
        "answer": "Grounded [apple::2].",
        "citations": [citation_dict],
        "confidence": 0.9,
        "refused": False,
    })
    response = TestClient(app).post("/research", json={"question": "What happened?"})
    assert response.status_code == 200
    assert response.json()["question"] == "What happened?"


def test_research_endpoint_maps_missing_model_config_to_503(monkeypatch):
    def fail(question):
        raise agent.AgentConfigurationError("Set OPENAI_API_KEY and AGENT_MODEL.")
    monkeypatch.setattr(agent, "run_agent", fail)
    response = TestClient(app).post("/research", json={"question": "What happened?"})
    assert response.status_code == 503
```

- [ ] **Step 2: Run endpoint tests and verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agent.py -k research_endpoint -q`

Expected: FAIL because `/research` still returns 501 and `AgentConfigurationError` is absent.

- [ ] **Step 3: Add the lazy OpenAI-compatible adapter**

Implement:

```python
class AgentConfigurationError(RuntimeError):
    pass


class OpenAICompatibleModel:
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("model returned empty content")
        return content


def _default_model() -> AgentModel:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("AGENT_MODEL", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    if not api_key or not model:
        raise AgentConfigurationError("Set OPENAI_API_KEY and AGENT_MODEL.")
    return OpenAICompatibleModel(api_key=api_key, model=model, base_url=base_url)
```

The adapter lazily imports `OpenAI`, calls `client.chat.completions.create` with temperature
zero, and returns the first message content. Empty content raises a runtime error. Add
`openai>=1.30` to `requirements.txt`.

- [ ] **Step 4: Wire `POST /research`**

Import `agent` in `app/main.py`, call `agent.run_agent(req.question)`, and construct the response:

```python
return ResearchResponse(
    question=req.question,
    answer=out["answer"],
    citations=[Citation(**item) for item in out["citations"]],
    confidence=out["confidence"],
    refused=out["refused"],
)
```

Map only `AgentConfigurationError` to `HTTPException(503, str(exc))`. Do not convert internal
programming errors to 4xx responses.

- [ ] **Step 5: Run endpoint and agent tests**

Run: `.venv/bin/python -m pytest tests/test_agent.py -q`

Expected: all tests pass.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`

Expected: all existing RAG, tool, event-study, eval, agent, and API tests pass.

- [ ] **Step 7: Commit the provider and endpoint integration**

```bash
git add app/agent.py app/main.py requirements.txt tests/test_agent.py
git commit -m "feat: expose ReAct research endpoint"
```

---

### Task 3: Documentation and Ownership Quiz

**Files:**
- Modify: `README.md`
- Modify: `UNDERSTAND.md`

**Interfaces:**
- Consumes: completed `/research` behavior and environment variables from Task 2.
- Produces: runnable setup instructions, request trace, limitations, and self-quiz.

- [ ] **Step 1: Update README configuration and smoke test**

Document `OPENAI_API_KEY`, `AGENT_MODEL`, and optional `OPENAI_BASE_URL`. Add a `/research`
request example and explain that the LLM selects tools while code validates arguments,
provenance, and loop bounds. Mark Day 4 complete. State that citation provenance is enforced,
while claim-level semantic grounding is measured separately by the Day 5 corpus eval.

- [ ] **Step 2: Extend `UNDERSTAND.md` without replacing existing material**

Append:

- A trace from `POST /research` through FastAPI, model action, tool dispatch, observation,
  citation registry, and response validation.
- A concise explanation of ReAct: alternate model decisions with deterministic tool execution.
- A ten-minute five-question self-quiz covering model freedom, boundedness, argument validation,
  citation provenance versus semantic grounding, and refusal.
- A sixty-second interview explanation of the agent increment.

- [ ] **Step 3: Run documentation consistency checks**

Run: `rg -n "Day 4|OPENAI_API_KEY|AGENT_MODEL|/research|citation provenance" README.md UNDERSTAND.md`

Expected: both documents describe the live endpoint and no text still claims the loop returns 501.

- [ ] **Step 4: Run final verification**

Run: `.venv/bin/python -m pytest -q`

Expected: the complete suite passes.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md UNDERSTAND.md
git commit -m "docs: explain the ReAct research workflow"
```

## Final Review

- Confirm only the three registered tools are callable.
- Confirm all model-driven actions are bounded and validated.
- Confirm citation objects come only from retrieved passages.
- Confirm no private SPX strategy logic or data entered the public repository.
- Report exact test counts and any verification that could not be run.
