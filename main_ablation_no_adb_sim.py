import argparse
import json
import random
import threading
import time
from typing import Dict, List

from sim_no_adb_utils import COMMON_ELEMENT_TEXTS, avg, parse_action_from_llm_text, simulate_llm_output, text_similarity


def _explore_similarity(task: str, rounds: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    records: List[Dict] = []
    for i in range(1, rounds + 1):
        element = rng.choice(COMMON_ELEMENT_TEXTS)
        score, jac, seq = text_similarity(task, element)
        time.sleep(1.0)
        records.append(
            {
                "explore_step": i,
                "element_text": element,
                "similarity": score,
                "jaccard": jac,
                "seq_ratio": seq,
            }
        )
    return records


def _run_variant(args, variant: str) -> Dict:
    rng = random.Random(args.seed + (0 if variant == "reasoning_only" else 111))
    planning_latencies: List[float] = []
    step_latencies: List[float] = []
    details: List[Dict] = []

    for itr in range(1, args.max_itr + 1):
        step_start = time.time()
        explore_records: List[Dict] = []
        rounds = rng.randint(3, 5)

        explorer_state: Dict = {}
        if variant == "reasoning_plus_explorer":
            def _worker():
                explorer_state["records"] = _explore_similarity(args.task, rounds, args.seed + itr * 97)

            th = threading.Thread(target=_worker, daemon=True)
            th.start()

        reasoning_start = time.time()
        time.sleep(max(0.0, args.reasoning_sleep_sec))
        llm_text = simulate_llm_output(args.task, history_len=itr - 1, role="single")
        action_obj = parse_action_from_llm_text(llm_text)
        reasoning_end = time.time()

        if variant == "reasoning_plus_explorer":
            th.join()
            explore_records = explorer_state.get("records", [])

        planning_end = time.time()
        planning_ms = (planning_end - reasoning_start) * 1000
        step_ms = (planning_end - step_start) * 1000
        planning_latencies.append(planning_ms)
        step_latencies.append(step_ms)

        details.append(
            {
                "step": itr,
                "variant": variant,
                "action": action_obj,
                "reasoning_latency_ms": (reasoning_end - reasoning_start) * 1000,
                "planning_latency_ms": planning_ms,
                "step_latency_ms": step_ms,
                "explore_rounds": rounds if variant == "reasoning_plus_explorer" else 0,
                "explore_records": explore_records,
            }
        )

    return {
        "variant": variant,
        "avg_planning_latency_ms": avg(planning_latencies),
        "avg_step_latency_ms": avg(step_latencies),
        "details": details,
    }


def run_ablation_no_adb(args):
    print("### Running Ablation Simulation (No ADB) ###")
    print(f"[Config] task={args.task}")
    print(f"[Config] max_itr={args.max_itr}, reasoning_sleep_sec={args.reasoning_sleep_sec}")

    reasoning_only = _run_variant(args, "reasoning_only")
    reasoning_plus_explorer = _run_variant(args, "reasoning_plus_explorer")

    print("\n=== Ablation Summary ===")
    print(
        f"reasoning_only: planning={reasoning_only['avg_planning_latency_ms']:.3f} ms, "
        f"step={reasoning_only['avg_step_latency_ms']:.3f} ms"
    )
    print(
        f"reasoning_plus_explorer: planning={reasoning_plus_explorer['avg_planning_latency_ms']:.3f} ms, "
        f"step={reasoning_plus_explorer['avg_step_latency_ms']:.3f} ms"
    )

    results = {
        "reasoning_only": reasoning_only,
        "reasoning_plus_explorer": reasoning_plus_explorer,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[Saved] ablation traces -> {args.output_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="Search papers on Mobile GUI Agent on Google Scholar.",
        help="User instruction for ablation simulation.",
    )
    parser.add_argument("--max_itr", type=int, default=5, help="Iterations for each ablation variant.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--reasoning_sleep_sec",
        type=float,
        default=4.0,
        help="Reasoning wait used by both ablation variants.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional output JSON file for ablation records.",
    )
    args = parser.parse_args()

    run_ablation_no_adb(args)

