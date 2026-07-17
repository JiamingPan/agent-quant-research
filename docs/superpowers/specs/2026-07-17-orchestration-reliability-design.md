# Orchestration Reliability Design

## Context

The MVP already has a bounded model-tool loop over `search_docs`, `get_price_data`, and
`run_event_study`. The model proposes the next semantic action, while Python validates the
action, executes the selected tool synchronously, returns the observation, and enforces the
step and citation rules.

The current offline harness exercises that loop, but every agent case calls only
`search_docs`. It therefore proves the basic loop and citation contract, not observable
multi-tool orchestration or recovery after a tool failure. The next increment closes that
gap before release packaging.

## Goals

1. Make each tool execution inspectable without exposing model chain-of-thought or full tool
   payloads.
2. Prove offline that the runtime records and executes single-tool, multi-tool, and recovery
   trajectories correctly.
3. Add a deliberately small optional live-model smoke test that measures actual model tool
   selection against deterministic tools.
4. Keep the existing three-tool boundary and synchronous bounded runtime.
5. Use existing private credentials only for a local verification run; never copy credentials,
   private data, or private strategy code into this public repository.

## Non-Goals

- vLLM installation or deployment
- LangGraph, MCP, a multi-agent system, or a fourth tool
- A queue, background worker, parallel scheduler, persistence, or human approval workflow
- Persisting raw model messages, chain-of-thought, or full price/event-study observations
- Treating a few live smoke cases as a statistically meaningful benchmark

## Runtime Trace

`run_agent` will accumulate one trace entry for every attempted tool action. Each entry will
contain:

- `step`: one-based model step number
- `tool`: requested tool name
- `arguments`: normalized validated arguments on success; the parsed proposal on validation
  failure
- `ok`: whether validation and execution succeeded
- `error`: a concise sanitized failure description when `ok` is false, otherwise `null`

The final agent result will also include `steps_used`, counting model calls. The FastAPI
`ResearchResponse` will expose these fields. The trace will not include raw model output,
hidden reasoning, tool result bodies, credentials, or environment variables.

This is a hybrid control model: the LLM chooses the next tool, but deterministic code owns
validation, dispatch, observation delivery, the step budget, and termination.

## Offline Orchestration Evaluation

Add deterministic trajectories covering:

1. `search_docs`
2. `get_price_data`
3. `run_event_study`
4. `search_docs` followed by `run_event_study`
5. A failed tool call followed by a valid recovery action

The price and event-study calls will use deterministic test doubles, so CI never requires
network access, market data, or credentials. The existing Chroma-backed retrieval fixtures
remain isolated in an ephemeral collection.

Offline metrics will be named as contract metrics rather than model-quality metrics:

- `trajectory_contract_rate`: recorded tool sequence exactly matches the scripted sequence
- `trace_completeness_rate`: every attempted tool call has a complete trace record
- `recovery_contract_rate`: a recoverable failure is followed by the expected valid action
- `mean_tool_steps`: average number of attempted tool actions per orchestration case

Existing retrieval, citation, refusal, tool-success, and leakage metrics remain unchanged.

## Optional Live-Model Smoke Test

Add a separate command that uses `OpenAICompatibleModel` with a tiny fixed corpus and
deterministic tool fixtures. It will not run under normal `pytest` or GitHub Actions.

The default run will use two short cases:

1. A document-only question expected to select `search_docs`.
2. A combined document-and-event question expected to select both `search_docs` and
   `run_event_study` before finalizing.

The command will cap each case at three model steps, so the default run makes at most six
model API calls. It will report tool coverage, ordered trajectory match, refusal status,
steps used, and elapsed time. Results will be printed or written to a user-selected path;
provider-dependent results will not replace the deterministic checked-in baseline.

For the one local verification run, credentials may be loaded from an existing private
environment file and mapped to the public app's variable names in the invoking shell. The
public code will know nothing about that private file or its location.

## Error Handling

- Invalid tool names and arguments remain non-fatal tool observations so the model can recover
  within its remaining step budget.
- Provider/configuration errors fail the optional live command clearly without changing the
  offline test result.
- Step exhaustion returns the existing explicit refusal and preserves the trace that led to it.
- Trace generation must not change the existing citation provenance rules.

## Testing

1. Unit tests for successful, failed, and exhausted traces.
2. API tests proving `/research` serializes trace entries and `steps_used`.
3. Offline harness tests for all orchestration contract metrics.
4. Full `pytest` run and deterministic eval reproduction.
5. One capped live-model smoke run using an existing private credential, with no secret values
   printed or persisted.

## Documentation

Update the README architecture and evaluation sections to distinguish:

- LLM routing policy
- Python orchestration runtime
- Synchronous scheduling/execution
- Deterministic offline contract evaluation
- Optional live model-routing evaluation

Add a short orchestration section to `UNDERSTAND.md` covering who controls each stage and why
objective routing checks should remain code-scored rather than LLM-judged.

## Acceptance Criteria

- Existing behavior remains backward compatible except for additive research-response fields.
- Offline tests cover all three tools, one multi-tool sequence, and one recovery sequence.
- No raw chain-of-thought or tool payloads appear in traces.
- The live smoke command has a hard maximum of six model calls by default.
- The repository contains no copied credentials, private market data, or private strategy logic.
- Full tests and deterministic evaluation pass before commit and push.
