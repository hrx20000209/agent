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
    safe_json,
    text_similarity,
)


SYSTEM_PROMPT = """You are a mobile GUI agent.
Decide ONE next GUI action from the given task, history, and screenshot.

Output format:
<thinking>
one-sentence rationale
</thinking>
<tool_call>
{"name":"mobile_use","arguments":{"action":"...", "...":"..."}}
</tool_call>
""".strip()


def _explore_loop_with_similarity(task: str, rounds: int, seed: int, output: Dict):
    rng = random.Random(seed)
    records: List[Dict] = []
    start_time = time.time()

    for i in range(1, rounds + 1):
        element = rng.choice(COMMON_ELEMENT_TEXTS)
        step_start = time.time()
        score, jac, seq = text_similarity(task, element)
        # Keep the same exploration timing as main: each exploration sleeps 1 second.
        time.sleep(1.0)
        records.append(
            {
                "explore_step": i,
                "element_text": element,
                "similarity": score,
                "jaccard": jac,
                "seq_ratio": seq,
                "explore_step_latency_ms": (time.time() - step_start) * 1000,
            }
        )

    output["records"] = records
    output["exploration_latency_ms"] = (time.time() - start_time) * 1000


def _explore_loop_no_similarity(rounds: int, seed: int, output: Dict):
    rng = random.Random(seed)
    records: List[Dict] = []
    start_time = time.time()

    for i in range(1, rounds + 1):
        element = rng.choice(COMMON_ELEMENT_TEXTS)
        step_start = time.time()
        # Ablation point: remove similarity computation, keep explore thread and sleep.
        time.sleep(1.0)
        records.append(
            {
                "explore_step": i,
                "element_text": element,
                "similarity": None,
                "explore_step_latency_ms": (time.time() - step_start) * 1000,
            }
        )

    output["records"] = records
    output["exploration_latency_ms"] = (time.time() - start_time) * 1000


def _run_variant(args, variant_name: str, enable_similarity: bool, screenshot_path: str) -> Dict:
    rng = random.Random(args.seed + (0 if enable_similarity else 10007))
    history: List[str] = []
    steps: List[Dict] = []

    perception_latency_list: List[float] = []
    reasoning_latency_list: List[float] = []
    exploration_latency_list: List[float] = []
    planning_latency_list: List[float] = []
    operation_latency_list: List[float] = []
    end_to_end_latency_list: List[float] = []

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()

        time.sleep(max(0.0, args.perception_sleep_ms / 1000.0))
        perception_end_time = time.time()
        perception_latency = (perception_end_time - start_time) * 1000
        perception_latency_list.append(perception_latency)

        rounds = rng.randint(3, 5)
        explorer_output: Dict = {}

        if enable_similarity:
            thread_target = _explore_loop_with_similarity
            thread_args = (args.task, rounds, args.seed + itr * 97, explorer_output)
        else:
            thread_target = _explore_loop_no_similarity
            thread_args = (rounds, args.seed + itr * 97, explorer_output)

        explorer_thread = threading.Thread(target=thread_target, args=thread_args, daemon=True)
        explorer_thread.start()

        user_prompt = (
            f"Task:\n{args.task}\n\n"
            f"Action History:\n{history[-6:] if history else 'None'}\n\n"
            f"Ablation Variant: {variant_name}\n"
            "Output the next action now."
        )

        reasoning_start = time.time()
        if args.reasoning_sleep_sec > 0:
            time.sleep(args.reasoning_sleep_sec)
        llm_output = call_llama_cpp_with_image(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            screenshot_path=screenshot_path,
            api_url=args.llama_api_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        action_obj = parse_action_from_llm_text(llm_output)
        reasoning_end = time.time()
        reasoning_latency = (reasoning_end - reasoning_start) * 1000
        reasoning_latency_list.append(reasoning_latency)

        explorer_thread.join()
        planning_end_time = time.time()
        planning_latency = (planning_end_time - perception_end_time) * 1000
        planning_latency_list.append(planning_latency)

        explore_records = explorer_output.get("records", [])
        exploration_latency = float(explorer_output.get("exploration_latency_ms", 0.0))
        exploration_latency_list.append(exploration_latency)

        operation_start = time.time()
        time.sleep(max(0.0, args.operation_sleep_ms / 1000.0))
        executed_action = f"simulate_execute::{action_obj.get('action_type', 'wait')}"
        operation_end = time.time()
        operation_latency = (operation_end - operation_start) * 1000
        operation_latency_list.append(operation_latency)

        history.append(safe_json(action_obj))
        step_latency = (operation_end - start_time) * 1000
        end_to_end_latency_list.append(step_latency)

        steps.append(
            {
                "step": itr,
                "variant": variant_name,
                "action": action_obj,
                "llm_output": llm_output,
                "explore_rounds": rounds,
                "explore_records": explore_records,
                "executed_action": executed_action,
                "perception_latency_ms": perception_latency,
                "reasoning_latency_ms": reasoning_latency,
                "exploration_latency_ms": exploration_latency,
                "planning_latency_ms": planning_latency,
                "operation_latency_ms": operation_latency,
                "step_latency_ms": step_latency,
            }
        )

    return {
        "variant": variant_name,
        "avg_perception_latency_ms": avg(perception_latency_list),
        "avg_reasoning_latency_ms": avg(reasoning_latency_list),
        "avg_exploration_latency_ms": avg(exploration_latency_list),
        "avg_planning_latency_ms": avg(planning_latency_list),
        "avg_operation_latency_ms": avg(operation_latency_list),
        "avg_step_latency_ms": avg(end_to_end_latency_list),
        "details": steps,
    }


def run_ablation_no_adb(args):
    print("### Running Ablation (No ADB, llama.cpp) ###")
    print(f"[Config] task={args.task}")
    print(f"[Config] max_itr={args.max_itr}")
    screenshot_path = ensure_screenshot_path(args.screenshot_path)
    print(f"[Config] llama_api_url={args.llama_api_url}")
    print(f"[Config] screenshot_path={screenshot_path}")

    with_similarity = _run_variant(
        args,
        variant_name="with_similarity",
        enable_similarity=True,
        screenshot_path=screenshot_path,
    )
    no_similarity_keep_thread = _run_variant(
        args,
        variant_name="no_similarity_keep_thread",
        enable_similarity=False,
        screenshot_path=screenshot_path,
    )

    print("\n=== Ablation Summary ===")
    print(
        f"with_similarity: "
        f"exploration={with_similarity['avg_exploration_latency_ms']:.3f} ms, "
        f"planning={with_similarity['avg_planning_latency_ms']:.3f} ms, "
        f"step={with_similarity['avg_step_latency_ms']:.3f} ms"
    )
    print(
        f"no_similarity_keep_thread: "
        f"exploration={no_similarity_keep_thread['avg_exploration_latency_ms']:.3f} ms, "
        f"planning={no_similarity_keep_thread['avg_planning_latency_ms']:.3f} ms, "
        f"step={no_similarity_keep_thread['avg_step_latency_ms']:.3f} ms"
    )

    results = {
        "with_similarity": with_similarity,
        "no_similarity_keep_thread": no_similarity_keep_thread,
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
        "--perception_sleep_ms",
        type=float,
        default=20.0,
        help="Perception simulation sleep per iteration.",
    )
    parser.add_argument(
        "--reasoning_sleep_sec",
        type=float,
        default=0.0,
        help="Optional extra wait before llama.cpp request.",
    )
    parser.add_argument(
        "--operation_sleep_ms",
        type=float,
        default=40.0,
        help="Operation simulation sleep per iteration.",
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
