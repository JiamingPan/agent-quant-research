"""Capped live-model routing smoke test over deterministic evaluation tools."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import agent, rag
from .eval_harness import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_ORCHESTRATION_CASES_PATH,
    OrchestrationCase,
    fixed_event_study,
    fixed_price_data,
    load_eval_collection,
)

LIVE_CASE_IDS = ("search_only", "search_then_event")
MAX_STEPS_PER_CASE = 3
MAX_MODEL_CALLS = len(LIVE_CASE_IDS) * MAX_STEPS_PER_CASE


def _load_live_cases(path: Path) -> list[OrchestrationCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{Path(path).name} must contain a JSON list")
    cases = [OrchestrationCase.model_validate(item) for item in payload]
    selected = [case for case in cases if case.case_id in LIVE_CASE_IDS]
    selected_by_id = {case.case_id: case for case in selected}
    missing = [case_id for case_id in LIVE_CASE_IDS if case_id not in selected_by_id]
    if missing:
        raise ValueError(f"missing live orchestration cases: {', '.join(missing)}")
    return [selected_by_id[case_id] for case_id in LIVE_CASE_IDS]


def run_live_eval(
    *,
    model: agent.AgentModel | None = None,
    collection: Any = None,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    cases_path: Path = DEFAULT_ORCHESTRATION_CASES_PATH,
) -> dict:
    """Run two live model-routing cases with at most six model calls."""
    active_model = model if model is not None else agent.model_from_env()
    cases = _load_live_cases(Path(cases_path))
    active_collection, _ = load_eval_collection(Path(corpus_dir), collection)

    def eval_search(query: str, k: int = 3) -> dict:
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

    registry = {
        "search_docs": eval_search,
        "get_price_data": fixed_price_data,
        "run_event_study": fixed_event_study,
    }
    case_results: list[dict[str, object]] = []
    expected_tool_count = 0
    covered_tool_count = 0

    for case in cases:
        expected_tools = [step.tool for step in case.steps]
        started = time.perf_counter()
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
        elapsed = time.perf_counter() - started
        actual_tools = [str(entry["tool"]) for entry in agent_result["trace"]]
        expected_tool_count += len(expected_tools)
        covered_tool_count += sum(tool in actual_tools for tool in expected_tools)
        case_results.append(
            {
                "case_id": case.case_id,
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "ordered_match": actual_tools == expected_tools,
                "refused": bool(agent_result["refused"]),
                "steps_used": int(agent_result["steps_used"]),
                "elapsed_seconds": round(elapsed, 3),
            }
        )

    return {
        "mode": "live_model_smoke",
        "n_cases": len(case_results),
        "max_model_calls": MAX_MODEL_CALLS,
        "model_calls_used": sum(int(case["steps_used"]) for case in case_results),
        "ordered_trajectory_accuracy": sum(
            bool(case["ordered_match"]) for case in case_results
        )
        / len(case_results),
        "tool_coverage": covered_tool_count / expected_tool_count,
        "cases": case_results,
    }


def _write_result(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run_live_eval()
    except (agent.AgentConfigurationError, RuntimeError) as exc:
        print(f"live eval failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if args.output is not None:
        _write_result(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
