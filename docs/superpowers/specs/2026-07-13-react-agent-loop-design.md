# ReAct Agent Loop Design

## Objective

Implement the Day 4 increment: a bounded ReAct-style agent over the existing
`search_docs`, `get_price_data`, and `run_event_study` tools, then expose it through
`POST /research`.

The model decides which tools to call and in what order. Application code retains control
over tool access, argument validation, citation provenance, refusal, and the step limit.

## Scope

This increment includes:

- A provider adapter for an environment-configured OpenAI-compatible model.
- An injectable model interface so tests do not call an external API.
- A strict JSON action contract for tool calls and final answers.
- A maximum-six-step agent loop over exactly three tools.
- Citation provenance enforcement using passages returned by `search_docs`.
- Explicit refusal for weak evidence, malformed actions, invalid tools or arguments,
  fabricated citations, and step exhaustion.
- A live `POST /research` endpoint.
- Focused agent and endpoint tests.
- README and `UNDERSTAND.md` documentation for the completed workflow.

This increment excludes additional tools, LangGraph, MCP, conversation memory, streaming,
live trading, strategy logic, and a full corpus-level evaluation run.

## Components

### Model Adapter

The agent depends on a small interface that accepts the current message history and returns
one JSON action. The default adapter lazily constructs an OpenAI-compatible client using
environment configuration. Tests inject a deterministic fake adapter.

Keeping the adapter separate makes the loop testable and prevents provider-specific response
objects from leaking into tool dispatch or API code.

### Action Contract

Every model turn must produce exactly one JSON object in one of these forms:

```json
{
  "type": "tool",
  "name": "search_docs",
  "arguments": {"query": "services revenue", "k": 4}
}
```

```json
{
  "type": "final",
  "answer": "Services revenue reached a record level [apple_10k::3].",
  "citation_ids": ["apple_10k::3"],
  "confidence": 0.82,
  "refused": false
}
```

Refusals use the final form with `refused: true`, an empty citation list, and an explanation
of what evidence or input is missing.

### Tool Dispatch

The loop dispatches only names in the existing three-tool registry. Each tool has an explicit
argument schema. Unknown tools, unknown arguments, missing required arguments, invalid types,
or tool exceptions are returned to the model as structured error observations. Repeated bad
actions remain bounded by `max_steps`.

Tool observations are JSON-serializable dictionaries appended to the conversation. The model
may use multiple tools before answering.

### Citation Guard

Every successful `search_docs` observation contributes its returned passages to an in-memory
citation registry keyed by the stable `doc_id::chunk_id` citation string.

A non-refused final answer is accepted only when:

1. It contains at least one citation identifier.
2. Every declared identifier exists in the registry.
3. Every declared identifier also appears in the answer text.

The API returns the full citation objects from that registry. This prevents the model from
inventing source identifiers. The guard establishes citation provenance; claim-level semantic
grounding remains a separate evaluation problem for Day 5.

If retrieval itself refuses or returns no passages, the model may continue with another tool,
but it cannot produce a non-refused research answer without valid document evidence.

### Loop Termination

The loop ends when the model returns a valid final answer or when `max_steps` is exhausted.
Malformed model output, unsupported action types, or a final answer that fails the citation
guard produces a deterministic refusal. This keeps failures visible and prevents an extra
unbounded repair loop.

### API

`POST /research` passes `ResearchRequest.question` to `run_agent` and validates the returned
payload with `ResearchResponse`. Missing model configuration is reported as an HTTP 503 because
the service is correctly formed but its external model dependency is unavailable. Invalid
model output or exhausted reasoning returns a normal refused research response rather than a
server error.

## Data Flow

1. FastAPI validates the incoming question.
2. `run_agent` creates the system and user messages.
3. The model returns a JSON tool action.
4. The loop validates and dispatches the selected tool.
5. The structured observation is appended to message history.
6. Steps 3-5 repeat until the model returns a final action.
7. The citation guard validates identifiers against retrieved passages.
8. FastAPI validates and returns the research response.

## Error Handling

- Missing API key or model dependency: HTTP 503 from `/research`.
- Unknown tool or invalid arguments: structured observation, allowing the model to recover.
- Tool exception: structured observation without exposing a traceback to the model or API.
- Malformed model action: refused response with zero confidence.
- Fabricated or omitted citation: refused response with zero confidence.
- Step exhaustion: refused response explaining that the bounded loop did not finish.

## Testing

Tests use a scripted fake model and monkeypatched tools. Required cases are:

- One-tool retrieval followed by a grounded final answer.
- Multiple tool calls selected by the model before the final answer.
- Retrieval refusal leading to agent refusal.
- Unknown tool and invalid arguments represented as observations.
- Fabricated, omitted, or unmentioned citation rejected by the guard.
- Maximum-step exhaustion.
- `/research` maps a successful agent result to the response schema.
- `/research` maps missing model configuration to HTTP 503.

The complete existing test suite must remain green.

## Acceptance Criteria

- The model, not keyword rules, selects among exactly three tools.
- No external API call occurs in tests.
- The loop cannot exceed `max_steps`.
- Returned citations originate from actual `search_docs` observations.
- Weak or invalid evidence produces an explicit refusal.
- `/research` no longer returns 501.
- README build status and `UNDERSTAND.md` accurately describe the implemented path.
