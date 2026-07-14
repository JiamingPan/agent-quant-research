# Offline Evaluation Harness Design

## Objective

Implement the Day 5 evaluation increment so the repository reports reproducible measured
results rather than only tested metric helpers.

The default evaluation is deterministic and offline. It evaluates the real Chroma retrieval
path and real bounded agent loop without calling a paid model. The report must label agent
results as contract metrics, not as evidence of live-model planning quality.

## Scope

This increment includes:

- Three small sanitized financial-document fixtures.
- Labeled answerable and refusal cases stored as JSON.
- An ephemeral Chroma collection that cannot alter the app's persistent collection.
- Real retrieval measurements: hit@3 and mean reciprocal rank.
- Real RAG refusal accuracy on answerable and off-topic cases.
- Real agent-loop contract measurements using an injected deterministic eval model.
- Citation-grounding and tool-call-success metrics from those offline agent runs.
- Two leakage-guard checks: accept a strictly pre-window baseline and reject a baseline date
  at the cutoff.
- A deterministic checked-in `eval/results.json` report.
- README metrics and a Day 5 `UNDERSTAND.md` self-quiz.

This increment excludes paid live-model evaluation, model comparison, threshold tuning,
human grading, private SPX data, trading-strategy evaluation, and production observability.

## Corpus and Cases

The checked-in corpus contains short sanitized excerpts representing three distinct topics:

- Apple services revenue.
- NVIDIA data-center and accelerated-computing demand.
- Federal Reserve inflation and target-rate commentary.

The excerpts are evaluation fixtures, not authoritative reproductions of complete filings or
transcripts. Each fixture has a stable public `doc_id` in a manifest.

The case file contains three answerable queries with expected document ids and one clearly
off-topic query that should refuse. The harness evaluates retrieval only on answerable cases
and refusal behavior on all cases.

## Isolation

`rag.ingest` and `rag.search` gain an optional collection parameter. Existing API callers omit
it and continue using the persistent application collection. The harness passes an ephemeral
collection created specifically for the run.

`run_agent` gains an optional tool-registry parameter. Existing API callers omit it and use the
three production tools. The harness supplies a registry whose `search_docs` entry targets the
ephemeral collection. The registry still contains exactly the same three tool names.

These dependency-injection points avoid global monkeypatching and make isolation explicit and
testable.

## Offline Agent Model

The deterministic eval model implements the same `AgentModel.complete` interface as the real
provider adapter:

1. On its first turn, request `search_docs` for the case question.
2. Parse the resulting structured tool observation.
3. If retrieval refused, return a refused final action.
4. Otherwise return a short answer containing the top retrieved citation id.

This exercises the production JSON parser, argument validation, dispatcher, observation loop,
citation registry, final citation guard, and refusal path. It does not measure whether an
external LLM independently chooses the correct tool. The report and README state this limit.

## Metrics

The result contains:

- `hit_at_3`: expected document present in the first three retrieved passages.
- `mrr`: mean reciprocal rank of the expected document.
- `refusal_accuracy`: correct refusal/non-refusal decisions across all cases.
- `citation_grounding_rate`: fraction of accepted offline-agent answers carrying returned
  citations. This is a contract metric, not semantic entailment.
- `tool_call_success_rate`: fraction of deterministic expected `search_docs` calls that were
  dispatched as `search_docs` and completed successfully.
- `leakage_guard_rate`: fraction of the two explicit guard behaviors that worked.

The JSON report also records case counts and a note distinguishing retrieval quality, system
contract behavior, and live-model quality.

## Error Handling

- Invalid manifest or case shape fails the CLI with a clear validation error.
- Missing corpus files fail before evaluation.
- An empty answerable case set returns zero retrieval metrics rather than dividing by zero.
- Agent refusal remains a normal measured result.
- The harness restores no global state because dependencies are injected explicitly.
- The checked-in output omits timestamps and machine paths so repeated runs are diff-stable.

## Testing

Focused tests must prove:

- Explicit Chroma collections isolate ingest and search from the default collection.
- A supplied agent tool registry is used instead of the global registry.
- The deterministic eval model performs one tool action and then a grounded or refused final
  action from the observation.
- The suite aggregates retrieval, refusal, grounding, tool-call, and leakage metrics correctly.
- Result serialization is deterministic and contains no absolute paths.

The complete existing test suite must remain green.

## Acceptance Criteria

- `python -m app.eval_harness --output eval/results.json` runs without model credentials.
- The harness uses real Chroma embeddings and the real `run_agent` implementation.
- Evaluation does not modify `.chroma` or previously ingested documents.
- `eval/results.json` contains measured values and stable case counts.
- README labels offline contract metrics honestly.
- `UNDERSTAND.md` explains what each metric proves and does not prove.
- No private SPX data, identifiers, strategy logic, or credentials are added.
