from __future__ import annotations

from app.eval import (
    citation_grounding_rate,
    evaluate_retrieval_cases,
    hit_at_k,
    mean_reciprocal_rank,
    reciprocal_rank,
    refusal_accuracy,
    tool_call_success_rate,
)

RETRIEVAL_EVAL_CASES = [
    {
        "query": "services revenue reached a record level",
        "expected_doc_id": "apple_10k_excerpt",
    },
    {
        "query": "data center revenue increased sharply",
        "expected_doc_id": "nvidia_earnings_excerpt",
    },
]


def test_hit_at_k_basic():
    assert hit_at_k(["a", "b", "c"], "b", k=2) == 1
    assert hit_at_k(["a", "b", "c"], "c", k=2) == 0


def test_mrr_basic():
    assert reciprocal_rank(["a", "b", "c"], "a") == 1.0
    assert reciprocal_rank(["a", "b", "c"], "c") == 1 / 3
    assert reciprocal_rank(["a", "b"], "z") == 0.0
    assert mean_reciprocal_rank([
        (["a", "b", "c"], "a"),
        (["a", "b", "c"], "c"),
        (["a", "b"], "z"),
    ]) == (1.0 + 1 / 3 + 0.0) / 3


def test_grounding_rate_requires_claim_citations():
    claims = [
        {"text": "Revenue rose.", "citations": [{"doc_id": "10k", "chunk_id": 1}]},
        {"text": "Margins fell.", "citations": []},
        {"text": "Guidance was raised.", "citations": [{"doc_id": "call", "chunk_id": 2}]},
    ]

    assert citation_grounding_rate(claims) == 2 / 3
    assert citation_grounding_rate([]) == 0.0


def test_tool_call_success_rate():
    calls = [
        {"expected": "search_docs", "actual": "search_docs", "succeeded": True},
        {"expected": "get_price_data", "actual": "search_docs", "succeeded": True},
        {"expected": "run_event_study", "actual": "run_event_study", "succeeded": False},
    ]

    assert tool_call_success_rate(calls) == 1 / 3
    assert tool_call_success_rate([]) == 0.0


def test_refusal_accuracy():
    cases = [
        {"should_refuse": True, "refused": True},
        {"should_refuse": False, "refused": False},
        {"should_refuse": True, "refused": False},
    ]

    assert refusal_accuracy(cases) == 2 / 3
    assert refusal_accuracy([]) == 0.0


def test_retrieval_eval_cases_score_expected_source_docs():
    ranked = {
        "services revenue reached a record level": [
            {"doc_id": "apple_10k_excerpt"},
            {"doc_id": "nvidia_earnings_excerpt"},
        ],
        "data center revenue increased sharply": [
            {"doc_id": "apple_10k_excerpt"},
            {"doc_id": "nvidia_earnings_excerpt"},
        ],
    }

    def fake_search(query: str, k: int) -> list[dict[str, str]]:
        return ranked[query][:k]

    summary = evaluate_retrieval_cases(RETRIEVAL_EVAL_CASES, fake_search, k=2)

    assert summary == {
        "n_cases": 2,
        "hit_at_k": 1.0,
        "mrr": 0.75,
    }
