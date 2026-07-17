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
    mean_tool_steps,
    recovery_contract_rate,
    refusal_accuracy,
    tool_call_success_rate,
    trace_completeness_rate,
    trajectory_contract_rate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = REPO_ROOT / "eval" / "corpus"
DEFAULT_CASES_PATH = REPO_ROOT / "eval" / "cases.json"
DEFAULT_ORCHESTRATION_CASES_PATH = REPO_ROOT / "eval" / "orchestration_cases.json"
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


class OrchestrationStep(_StrictModel):
    tool: str = Field(min_length=1)
    arguments: dict[str, Any]
    should_succeed: bool


class OrchestrationCase(_StrictModel):
    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    steps: list[OrchestrationStep] = Field(min_length=1)
    recovery_required: bool


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


class OrchestrationEvalModel:
    """Scripted model used to verify the runtime's trajectory contracts."""

    def __init__(self, case: OrchestrationCase):
        self.case = case
        self._next_step = 0
        self._citation_id: str | None = None

    def complete(self, messages: list[dict[str, str]]) -> str:
        if messages and messages[-1].get("content", "").startswith("TOOL_OBSERVATION\n"):
            observation = _last_tool_observation(messages)
            if observation.get("ok") and observation.get("tool") == "search_docs":
                result = observation.get("result")
                result = result if isinstance(result, dict) else {}
                passages = result.get("passages") or []
                if passages:
                    citation_id = str(passages[0].get("citation", "")).strip()
                    self._citation_id = citation_id or None

        if self._next_step < len(self.case.steps):
            step = self.case.steps[self._next_step]
            self._next_step += 1
            return json.dumps(
                {"type": "tool", "name": step.tool, "arguments": step.arguments},
                separators=(",", ":"),
            )

        if self._citation_id:
            return json.dumps(
                {
                    "type": "final",
                    "answer": f"Retrieved supporting evidence [{self._citation_id}].",
                    "citation_ids": [self._citation_id],
                    "confidence": 1.0,
                    "refused": False,
                },
                separators=(",", ":"),
            )
        return _final_refusal("Completed tool contract case without document evidence.")


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


def load_eval_collection(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    collection: Any = None,
) -> tuple[Any, int]:
    """Ingest the sanitized eval corpus into an isolated collection."""
    corpus_dir = Path(corpus_dir)
    manifest = _load_models(corpus_dir / "manifest.json", CorpusEntry)
    active_collection = collection if collection is not None else _ephemeral_collection()
    for item in manifest:
        document_path = corpus_dir / item.filename
        if not document_path.is_file():
            raise FileNotFoundError(f"missing eval corpus file: {item.filename}")
        rag.ingest(str(document_path), item.doc_id, collection=active_collection)
    return active_collection, len(manifest)


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


def fixed_price_data(ticker: str, start: str, end: str) -> dict:
    """Deterministic price fixture for orchestration evaluation."""
    if ticker == "FAIL":
        raise RuntimeError("injected unavailable price fixture")
    return {
        "ticker": ticker,
        "start": start,
        "end": end,
        "source": "eval_fixture",
        "n_rows": 2,
        "rows": [],
    }


def fixed_event_study(ticker: str, event_date: str, window: int = 5) -> dict:
    """Deterministic event-study fixture for orchestration evaluation."""
    return {
        "ticker": ticker,
        "event_date": event_date,
        "window": window,
        "source": "eval_fixture",
        "car_bps": 12.0,
    }


def _run_orchestration_contracts(
    cases: list[OrchestrationCase],
    search_fn: Any,
) -> dict[str, float]:
    registry = {
        "search_docs": search_fn,
        "get_price_data": fixed_price_data,
        "run_event_study": fixed_event_study,
    }
    trajectories: list[dict[str, object]] = []
    traces: list[list[dict[str, object]]] = []
    recoveries: list[dict[str, object]] = []

    for case in cases:
        result = agent.run_agent(
            case.question,
            max_steps=len(case.steps) + 1,
            model=OrchestrationEvalModel(case),
            tool_registry=registry,
        )
        trace = result["trace"]
        expected_tools = [step.tool for step in case.steps]
        expected_success = [step.should_succeed for step in case.steps]
        actual_tools = [str(entry["tool"]) for entry in trace]
        actual_success = [bool(entry["ok"]) for entry in trace]
        trajectories.append({"expected": expected_tools, "actual": actual_tools})
        traces.append(trace)
        recoveries.append(
            {
                "required": case.recovery_required,
                "recovered": (
                    actual_tools == expected_tools
                    and actual_success == expected_success
                    and False in actual_success
                    and any(actual_success[actual_success.index(False) + 1 :])
                ),
            }
        )

    return {
        "trajectory_contract_rate": trajectory_contract_rate(trajectories),
        "trace_completeness_rate": trace_completeness_rate(traces),
        "recovery_contract_rate": recovery_contract_rate(recoveries),
        "mean_tool_steps": mean_tool_steps(traces),
    }


def run_offline_eval(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    cases_path: Path = DEFAULT_CASES_PATH,
    orchestration_cases_path: Path = DEFAULT_ORCHESTRATION_CASES_PATH,
    *,
    collection: Any = None,
) -> dict:
    """Run the reproducible offline suite and return a JSON-safe result."""
    corpus_dir = Path(corpus_dir)
    cases_path = Path(cases_path)
    cases = _load_models(cases_path, EvalCase)
    orchestration_cases = _load_models(
        Path(orchestration_cases_path),
        OrchestrationCase,
    )
    active_collection, corpus_count = load_eval_collection(corpus_dir, collection)

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
    orchestration = _run_orchestration_contracts(orchestration_cases, eval_search)
    return {
        "mode": "offline_contract",
        "corpus_documents": corpus_count,
        "cases": {
            "retrieval": len(answerable_cases),
            "agent": len(cases),
            "refusal": len(cases),
            "leakage_guard": int(leakage["n_checks"]),
            "orchestration": len(orchestration_cases),
        },
        "metrics": {
            "hit_at_1": float(retrieval_at_1["hit_at_k"]),
            "mrr": float(retrieval_ranking["mrr"]),
            "citation_grounding_rate": citation_grounding_rate(claims),
            "tool_call_success_rate": tool_call_success_rate(call_records),
            "refusal_accuracy": refusal_accuracy(refusal_records),
            "leakage_guard_rate": float(leakage["rate"]),
            **orchestration,
        },
        "notes": [
            "Retrieval and refusal metrics use real Chroma search over sanitized fixtures.",
            "Agent grounding and tool metrics are deterministic contract checks, not live-model quality.",
            "Orchestration metrics cover scripted trajectories and recovery, not model routing intelligence.",
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
