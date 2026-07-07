"""Evaluation metrics for retrieval, grounding, and tool behavior."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def hit_at_k(ranked_doc_ids: Sequence[str], relevant_doc_id: str, k: int) -> int:
    """Return 1 when the relevant doc is present in the first k results."""
    return int(relevant_doc_id in ranked_doc_ids[:k])


def reciprocal_rank(ranked_doc_ids: Sequence[str], relevant_doc_id: str) -> float:
    """Return 1/rank for the first relevant hit, or 0 when absent."""
    for rank, doc_id in enumerate(ranked_doc_ids, 1):
        if doc_id == relevant_doc_id:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    cases: Iterable[tuple[Sequence[str], str]],
) -> float:
    """Mean reciprocal rank over `(ranked_doc_ids, relevant_doc_id)` cases."""
    values = [reciprocal_rank(ranked, relevant) for ranked, relevant in cases]
    return sum(values) / len(values) if values else 0.0


def citation_grounding_rate(claims: Sequence[Mapping[str, object]]) -> float:
    """Fraction of claims with at least one citation attached."""
    if not claims:
        return 0.0
    grounded = 0
    for claim in claims:
        citations = claim.get("citations", [])
        grounded += int(bool(citations))
    return grounded / len(claims)


def tool_call_success_rate(calls: Sequence[Mapping[str, object]]) -> float:
    """Fraction of tool calls that selected the expected tool and succeeded."""
    if not calls:
        return 0.0
    successes = 0
    for call in calls:
        successes += int(
            call.get("expected") == call.get("actual")
            and bool(call.get("succeeded"))
        )
    return successes / len(calls)


def refusal_accuracy(cases: Sequence[Mapping[str, bool]]) -> float:
    """Fraction of weak/answerable cases where refusal behavior is correct."""
    if not cases:
        return 0.0
    correct = 0
    for case in cases:
        correct += int(case.get("should_refuse") == case.get("refused"))
    return correct / len(cases)
