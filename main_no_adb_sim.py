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
    compute_phash,
    ensure_screenshot_path,
    parse_action_from_llm_text,
    safe_json,
    text_similarity,
    verify_with_anchor_phash,
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


def _explorer_similarity_loop(
    task: str,
    explore_rounds: int,
    seed: int,
    output: Dict,
    screenshot_path: str,
    anchor_hash,
    enable_thread_verification: bool,
    phash_threshold: int,
):
    rng = random.Random(seed)
    records: List[Dict] = []
    start_time = time.time()

    for step in range(1, explore_rounds + 1):
        element_text = rng.choice(COMMON_ELEMENT_TEXTS)
        step_start = time.time()
        score, jaccard, seq_ratio = text_similarity(task, element_text)
        # Requirement: each exploration sleeps 1 second.
        time.sleep(1.0)
        step_latency_ms = (time.time() - step_start) * 1000

        records.append(
            {
                "explore_step": step,
                "element_text": element_text,
                "similarity": score,
                "jaccard": jaccard,
                "seq_ratio": seq_ratio,
                "explore_step_latency_ms": step_latency_ms,
            }
        )

    output["records"] = records
    output["exploration_latency_ms"] = (time.time() - start_time) * 1000
    if enable_thread_verification:
        output["thread_verification"] = verify_with_anchor_phash(
            screenshot_path=screenshot_path,
            anchor_hash=anchor_hash,
            threshold=phash_threshold,
        )
    else:
        output["thread_verification"] = None


def run_single_step_agent_no_adb(args):
    print("### Running Single-Step Agent (No ADB, llama.cpp) ###")
    print(f"[Config] task={args.task}")
    screenshot_path = ensure_screenshot_path(args.screenshot_path)
    print(f"[Config] max_itr={args.max_itr}, explore_rounds=3~5")
    print(f"[Config] llama_api_url={args.llama_api_url}")
    print(f"[Config] screenshot_path={screenshot_path}")

    rng = random.Random(args.seed)
    history: List[str] = []
    steps: List[Dict] = []

    perception_latency_list: List[float] = []
    planning_latency_list: List[float] = []
    reasoning_latency_list: List[float] = []
    exploration_latency_list: List[float] = []
    thread_verification_latency_list: List[float] = []
    main_verification_latency_list: List[float] = []
    operation_latency_list: List[float] = []
    end_to_end_latency_list: List[float] = []
    thread_verify_fail_count = 0
    main_verify_fail_count = 0

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        # Perception simulation (replace screenshot/XML collection).
        time.sleep(max(0.0, args.perception_sleep_ms / 1000.0))
        perception_end_time = time.time()
        perception_latency = (perception_end_time - start_time) * 1000
        perception_latency_list.append(perception_latency)

        # Anchor hash for this iteration; thread/main verification both compare to this.
        anchor_hash = compute_phash(screenshot_path)

        # Explorer simulation: similarity-only, 3-5 rounds, each round sleep=1s.
        explore_rounds = rng.randint(3, 5)
        explorer_output: Dict = {}
        explorer_thread = threading.Thread(
            target=_explorer_similarity_loop,
            args=(
                args.task,
                explore_rounds,
                args.seed + itr * 9973,
                explorer_output,
                screenshot_path,
                anchor_hash,
                bool(args.enable_thread_verification),
                int(args.phash_threshold),
            ),
            daemon=True,
        )
        explorer_thread.start()

        # Main loop waits for reasoning (real llama.cpp call).
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

        # Wait explorer to finish this simulation round.
        explorer_thread.join()
        planning_end_time = time.time()
        planning_latency = (planning_end_time - perception_end_time) * 1000
        planning_latency_list.append(planning_latency)

        explore_records = explorer_output.get("records", [])
        exploration_latency = float(explorer_output.get("exploration_latency_ms", 0.0))
        exploration_latency_list.append(exploration_latency)
        thread_verification = explorer_output.get("thread_verification")
        thread_verify_latency_step = 0.0
        if thread_verification:
            thread_verify_latency_step = float(thread_verification.get("verification_latency_ms", 0.0))
            thread_verification_latency_list.append(thread_verify_latency_step)
            if not bool(thread_verification.get("ok", False)):
                thread_verify_fail_count += 1
        best_record = max(explore_records, key=lambda x: x["similarity"]) if explore_records else None

        main_verification = None
        main_verify_latency_step = 0.0
        if args.enable_main_verification:
            main_verification = verify_with_anchor_phash(
                screenshot_path=screenshot_path,
                anchor_hash=anchor_hash,
                threshold=args.phash_threshold,
            )
            main_verify_latency_step = float(main_verification.get("verification_latency_ms", 0.0))
            main_verification_latency_list.append(main_verify_latency_step)
            if not bool(main_verification.get("ok", False)):
                main_verify_fail_count += 1

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

        step_info = {
            "step": itr,
            "action": action_obj,
            "llm_output": llm_output,
            "explore_rounds": explore_rounds,
            "explore_records": explore_records,
            "best_explore_record": best_record,
            "thread_verification": thread_verification,
            "main_verification": main_verification,
            "executed_action": executed_action,
            "perception_latency_ms": perception_latency,
            "reasoning_latency_ms": reasoning_latency,
            "exploration_latency_ms": exploration_latency,
            "planning_latency_ms": planning_latency,
            "operation_latency_ms": operation_latency,
            "step_latency_ms": step_latency,
        }
        steps.append(step_info)

        if best_record:
            print(
                f"[Explorer] rounds={explore_rounds}, "
                f"best_similarity={best_record['similarity']:.3f}, "
                f"best_element=\"{best_record['element_text']}\""
            )
        if thread_verification is not None:
            print(
                f"[Thread Verify] ok={thread_verification['ok']} "
                f"diff={thread_verification['diff']} threshold={thread_verification['threshold']}"
            )
        if main_verification is not None:
            print(
                f"[Main Verify] ok={main_verification['ok']} "
                f"diff={main_verification['diff']} threshold={main_verification['threshold']}"
            )
        print("[Reasoning] Parsed action:", action_obj)
        print("[Execution] Action done:", executed_action)
        print(
            f"Perception latency: {perception_latency:.3f} ms, "
            f"Reasoning latency: {reasoning_latency:.3f} ms, "
            f"Exploration latency: {exploration_latency:.3f} ms, "
            f"ThreadVerify latency: {thread_verify_latency_step:.3f} ms, "
            f"MainVerify latency: {main_verify_latency_step:.3f} ms, "
            f"Planning latency: {planning_latency:.3f} ms, "
            f"Operation latency: {operation_latency:.3f} ms"
        )
        print(f"Step latency: {step_latency:.3f} ms")

    print("\n=== Finished all iterations (no-adb simulation) ===")
    print(
        f"Perception latency: {avg(perception_latency_list):.3f} ms, "
        f"Reasoning latency: {avg(reasoning_latency_list):.3f} ms, "
        f"Exploration latency: {avg(exploration_latency_list):.3f} ms, "
        f"ThreadVerify latency: {avg(thread_verification_latency_list):.3f} ms, "
        f"MainVerify latency: {avg(main_verification_latency_list):.3f} ms, "
        f"Planning latency: {avg(planning_latency_list):.3f} ms, "
        f"Operation latency: {avg(operation_latency_list):.3f} ms, "
        f"End-to-end latency: {avg(end_to_end_latency_list):.3f} ms"
    )
    print(f"Thread verification failures: {thread_verify_fail_count}, Main verification failures: {main_verify_fail_count}")

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
        help="User instruction for the simulation.",
    )
    parser.add_argument("--max_itr", type=int, default=10, help="Maximum benchmark iterations.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--reasoning_sleep_sec",
        type=float,
        default=0.0,
        help="Optional extra wait before each llama.cpp request.",
    )
    parser.add_argument(
        "--perception_sleep_ms",
        type=float,
        default=20.0,
        help="Perception simulation sleep per iteration.",
    )
    parser.add_argument(
        "--operation_sleep_ms",
        type=float,
        default=40.0,
        help="Operation simulation sleep per iteration.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional output JSON file for step-level records.",
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
        default="http://localhost:8081/v1/chat/completions",
        help="llama.cpp OpenAI-compatible endpoint.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max new tokens per request.")
    parser.add_argument("--phash_threshold", type=int, default=8, help="pHash verification threshold.")
    parser.add_argument(
        "--disable_thread_verification",
        dest="enable_thread_verification",
        action="store_false",
        help="Disable pHash verification in explore thread.",
    )
    parser.add_argument(
        "--disable_main_verification",
        dest="enable_main_verification",
        action="store_false",
        help="Disable pHash verification in main process after reasoning + exploration.",
    )
    parser.set_defaults(enable_thread_verification=True, enable_main_verification=True)
    args = parser.parse_args()

    run_single_step_agent_no_adb(args)
