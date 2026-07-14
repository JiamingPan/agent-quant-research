"""Deterministic offline evaluation over the real RAG and agent contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import chromadb
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import agent, rag, tools
from .eval import (
    citation_grounding_rate,
    evaluate_retrieval_cases,
    refusal_accuracy,
    tool_call_success_rate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = REPO_ROOT / "eval" / "corpus"
DEFAULT_CASES_PATH = REPO_ROOT / "eval" / "cases.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "eval" / "results.json"
RETRIEVAL_K = 3


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusEntry(_StrictModel):
    doc_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)


class EvalCase(_StrictModel):
    question: str = Field(min_length=1)
    expected_doc_id: Optional[str] = None
    should_refuse: bool

    @model_validator(mode="after")
    def validate_expected_document(self) -> "EvalCase":
        if self.should_refuse and self.expected_doc_id is not None:
            raise ValueError("refusal cases must not declare expected_doc_id")
        if not self.should_refuse and not self.expected_doc_id:
            raise ValueError("answerable cases require expected_doc_id")
        return self


class OfflineEvalModel:
    """Deterministic model that reacts to one real `search_docs` observation."""

    def __init__(self, question: str):
        self.question = question
        self._requested_search = False
        self.call_records: list[dict[str, object]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self._requested_search:
            self._requested_search = True
            return json.dumps(
                {
                    "type": "tool",
                    "name": "search_docs",
                    "arguments": {"query": self.question, "k": RETRIEVAL_K},
                },
                separators=(",", ":"),
            )

        observation = _last_tool_observation(messages)
        actual_tool = str(observation.get("tool", ""))
        succeeded = bool(observation.get("ok"))
        self.call_records.append(
            {
                "expected": "search_docs",
                "actual": actual_tool,
                "succeeded": succeeded,
            }
        )
        if not succeeded:
            return _final_refusal("Tool execution failed during offline evaluation.")

        result = observation.get("result")
        result = result if isinstance(result, dict) else {}
        passages = result.get("passages") or []
        if bool(result.get("refused")) or not passages:
            reason = str(result.get("reason") or "No sufficiently relevant passage found.")
            return _final_refusal(reason)

        top_passage = passages[0]
        citation_id = str(top_passage.get("citation", "")).strip()
        if not citation_id:
            return _final_refusal("Retrieved passage has no citation identifier.")
        return json.dumps(
            {
                "type": "final",
                "answer": f"Retrieved supporting evidence [{citation_id}].",
                "citation_ids": [citation_id],
                "confidence": 1.0,
                "refused": False,
            },
            separators=(",", ":"),
        )


def _last_tool_observation(messages: list[dict[str, str]]) -> dict:
    if not messages:
        raise ValueError("offline eval model expected a tool observation")
    content = messages[-1].get("content", "")
    prefix = "TOOL_OBSERVATION\n"
    if not content.startswith(prefix):
        raise ValueError("offline eval model received an invalid tool observation")
    payload = json.loads(content[len(prefix) :])
    if not isinstance(payload, dict):
        raise ValueError("tool observation must be a JSON object")
    return payload


def _final_refusal(reason: str) -> str:
    return json.dumps(
        {
            "type": "final",
            "answer": reason,
            "citation_ids": [],
            "confidence": 0.0,
            "refused": True,
        },
        separators=(",", ":"),
    )


def _load_models(path: Path, model_type: type[BaseModel]) -> list[BaseModel]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path.name} must contain a JSON list")
    return [model_type.model_validate(item) for item in payload]


def _ephemeral_collection() -> Any:
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        "eval_docs",
        metadata={"hnsw:space": rag.DISTANCE_METRIC},
    )


def _leakage_guard_result() -> dict[str, float | int]:
    cutoff = pd.Timestamp("2026-01-08", tz="UTC")
    safe_dates = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-06", tz="UTC"), pd.Timestamp("2026-01-07", tz="UTC")]
    )
    tools._assert_no_baseline_leakage(safe_dates, cutoff)
    passed = 1

    leaking_dates = safe_dates.append(
        pd.DatetimeIndex([pd.Timestamp("2026-01-08", tz="UTC")])
    )
    try:
        tools._assert_no_baseline_leakage(leaking_dates, cutoff)
    except AssertionError:
        passed += 1
    return {"n_checks": 2, "n_passed": passed, "rate": passed / 2}


def run_offline_eval(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    cases_path: Path = DEFAULT_CASES_PATH,
    *,
    collection: Any = None,
) -> dict:
    """Run the reproducible offline suite and return a JSON-safe result."""
    corpus_dir = Path(corpus_dir)
    cases_path = Path(cases_path)
    manifest = _load_models(corpus_dir / "manifest.json", CorpusEntry)
    cases = _load_models(cases_path, EvalCase)
    active_collection = collection if collection is not None else _ephemeral_collection()

    for item in manifest:
        document_path = corpus_dir / item.filename
        if not document_path.is_file():
            raise FileNotFoundError(f"missing eval corpus file: {item.filename}")
        rag.ingest(str(document_path), item.doc_id, collection=active_collection)

    def eval_search(query: str, k: int = RETRIEVAL_K) -> dict:
        passages, refused, reason = rag.search(
            query,
            k=k,
            collection=active_collection,
        )
        return {
            "passages": [passage.model_dump() for passage in passages],
            "refused": refused,
            "reason": reason,
        }

    answerable_cases = [case for case in cases if not case.should_refuse]
    retrieval_cases = [
        {"query": case.question, "expected_doc_id": str(case.expected_doc_id)}
        for case in answerable_cases
    ]
    retrieval_at_1 = evaluate_retrieval_cases(
        retrieval_cases,
        lambda query, k: eval_search(query, k)["passages"],
        k=1,
    )
    retrieval_ranking = evaluate_retrieval_cases(
        retrieval_cases,
        lambda query, k: eval_search(query, k)["passages"],
        k=RETRIEVAL_K,
    )

    registry = {
        "search_docs": eval_search,
        "get_price_data": tools.get_price_data,
        "run_event_study": tools.run_event_study,
    }
    claims: list[dict[str, object]] = []
    refusal_records: list[dict[str, bool]] = []
    call_records: list[dict[str, object]] = []
    for case in cases:
        model = OfflineEvalModel(case.question)
        result = agent.run_agent(
            case.question,
            model=model,
            tool_registry=registry,
        )
        refusal_records.append(
            {"should_refuse": case.should_refuse, "refused": bool(result["refused"])}
        )
        if not result["refused"]:
            claims.append({"text": result["answer"], "citations": result["citations"]})
        call_records.extend(model.call_records)

    leakage = _leakage_guard_result()
    return {
        "mode": "offline_contract",
        "corpus_documents": len(manifest),
        "cases": {
            "retrieval": len(answerable_cases),
            "agent": len(cases),
            "refusal": len(cases),
            "leakage_guard": int(leakage["n_checks"]),
        },
        "metrics": {
            "hit_at_1": float(retrieval_at_1["hit_at_k"]),
            "mrr": float(retrieval_ranking["mrr"]),
            "citation_grounding_rate": citation_grounding_rate(claims),
            "tool_call_success_rate": tool_call_success_rate(call_records),
            "refusal_accuracy": refusal_accuracy(refusal_records),
            "leakage_guard_rate": float(leakage["rate"]),
        },
        "notes": [
            "Retrieval and refusal metrics use real Chroma search over sanitized fixtures.",
            "Agent grounding and tool metrics are deterministic contract checks, not live-model quality.",
            "This small corpus is a reproducible smoke baseline, not a statistical benchmark.",
        ],
    }


def _write_result(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    result = run_offline_eval(args.corpus_dir, args.cases)
    _write_result(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
