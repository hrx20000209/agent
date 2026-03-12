import argparse
import json
import time
from typing import Dict, List

from sim_no_adb_utils import (
    avg,
    call_llama_cpp_with_image,
    ensure_screenshot_path,
    parse_action_from_llm_text,
    safe_json,
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


def run_baseline_no_explore(args):
    print("### Running Main Baseline (No Explore Thread, llama.cpp) ###")
    print(f"[Config] task={args.task}")
    screenshot_path = ensure_screenshot_path(args.screenshot_path)
    print(f"[Config] max_itr={args.max_itr}")
    print(f"[Config] llama_api_url={args.llama_api_url}")
    print(f"[Config] screenshot_path={screenshot_path}")

    history: List[str] = []
    steps: List[Dict] = []

    perception_latency_list: List[float] = []
    reasoning_latency_list: List[float] = []
    planning_latency_list: List[float] = []
    operation_latency_list: List[float] = []
    end_to_end_latency_list: List[float] = []

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        # Perception simulation.
        time.sleep(max(0.0, args.perception_sleep_ms / 1000.0))
        perception_end_time = time.time()
        perception_latency = (perception_end_time - start_time) * 1000
        perception_latency_list.append(perception_latency)

        # Reasoning (single llama.cpp call, no exploration thread).
        user_prompt = (
            f"Task:\n{args.task}\n\n"
            f"Action History:\n{history[-6:] if history else 'None'}\n\n"
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

        planning_end_time = reasoning_end
        planning_latency = (planning_end_time - perception_end_time) * 1000
        planning_latency_list.append(planning_latency)

        # Operation simulation.
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
                "action": action_obj,
                "llm_output": llm_output,
                "executed_action": executed_action,
                "perception_latency_ms": perception_latency,
                "reasoning_latency_ms": reasoning_latency,
                "planning_latency_ms": planning_latency,
                "operation_latency_ms": operation_latency,
                "step_latency_ms": step_latency,
            }
        )

        print("[Reasoning] Parsed action:", action_obj)
        print("[Execution] Action done:", executed_action)
        print(
            f"Perception latency: {perception_latency:.3f} ms, "
            f"Reasoning latency: {reasoning_latency:.3f} ms, "
            f"Planning latency: {planning_latency:.3f} ms, "
            f"Operation latency: {operation_latency:.3f} ms"
        )
        print(f"Step latency: {step_latency:.3f} ms")

        action_type = str(action_obj.get("action_type", "")).lower()
        if action_type in {"finish", "done", "exit", "stop", "terminate"}:
            print("[Stop] finish-like action detected, exiting loop.")
            break

    print("\n=== Finished all iterations (main baseline: no explore thread) ===")
    print(
        f"Perception latency: {avg(perception_latency_list):.3f} ms, "
        f"Reasoning latency: {avg(reasoning_latency_list):.3f} ms, "
        f"Planning latency: {avg(planning_latency_list):.3f} ms, "
        f"Operation latency: {avg(operation_latency_list):.3f} ms, "
        f"End-to-end latency: {avg(end_to_end_latency_list):.3f} ms"
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(steps, f, ensure_ascii=False, indent=2)
        print(f"[Saved] step traces -> {args.output_json}")

    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="Search papers on Mobile GUI Agent on Google Scholar.",
        help="User instruction for baseline run.",
    )
    parser.add_argument("--max_itr", type=int, default=10, help="Maximum benchmark iterations.")
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
        help="Optional extra wait before each llama.cpp request.",
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
        default="./resized_screenshot.png",
        help="Input screenshot path (default: repo-root screenshot.png).",
    )
    parser.add_argument( 
        "--llama_api_url",
        type=str,
        default="http://localhost:8081/v1/chat/completions",
        help="llama.cpp OpenAI-compatible endpoint.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max new tokens per request.")
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional output JSON file for step-level records.",
    )
    args = parser.parse_args()

    run_baseline_no_explore(args)
