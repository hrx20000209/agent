import argparse
import json
import random
import threading
import time
from typing import Dict, List

from sim_no_adb_utils import (
    COMMON_ELEMENT_TEXTS,
    avg,
    call_llama_cpp_with_image,
    ensure_screenshot_path,
    parse_action_from_llm_text,
    text_similarity,
)

SYSTEM_PROMPT = """You are a mobile GUI agent.
Decide ONE next GUI action from the given task, history and screenshot.
Use:
<thinking>...</thinking>
<tool_call>{"name":"mobile_use","arguments":{"action":"...", "...":"..."}}</tool_call>
""".strip()


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


def _run_variant(args, variant: str, screenshot_path: str) -> Dict:
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
        if args.reasoning_sleep_sec > 0:
            time.sleep(args.reasoning_sleep_sec)

        clue_line = ""
        if variant == "reasoning_plus_explorer":
            clue_line = f"Exploration is running in parallel (rounds={rounds}, each sleep=1s)."

        user_prompt = (
            f"Task:\n{args.task}\n\n"
            f"Step Index: {itr}\n"
            f"Variant: {variant}\n"
            f"{clue_line}\n\n"
            "Output one next action now."
        )
        llm_text = call_llama_cpp_with_image(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            screenshot_path=screenshot_path,
            api_url=args.llama_api_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
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
    print("### Running Ablation (No ADB, llama.cpp) ###")
    print(f"[Config] task={args.task}")
    print(f"[Config] max_itr={args.max_itr}, reasoning_sleep_sec={args.reasoning_sleep_sec}")
    screenshot_path = ensure_screenshot_path(args.screenshot_path)
    print(f"[Config] llama_api_url={args.llama_api_url}")
    print(f"[Config] screenshot_path={screenshot_path}")

    reasoning_only = _run_variant(args, "reasoning_only", screenshot_path=screenshot_path)
    reasoning_plus_explorer = _run_variant(args, "reasoning_plus_explorer", screenshot_path=screenshot_path)

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
        default=0.0,
        help="Optional extra wait before llama.cpp request.",
    )
    parser.add_argument(
        "--screenshot_path",
        type=str,
        default="./screenshot.png",
        help="Input screenshot path (default: repo-root screenshot.png).",
    )
    parser.add_argument(
        "--llama_api_url",
        type=str,
        default="http://localhost:8100/v1/chat/completions",
        help="llama.cpp OpenAI-compatible endpoint.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max new tokens per request.")
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional output JSON file for ablation records.",
    )
    args = parser.parse_args()

    run_ablation_no_adb(args)
