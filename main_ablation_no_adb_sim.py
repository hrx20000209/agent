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


def _explore_loop(
    task: str,
    rounds: int,
    seed: int,
    output: Dict,
    screenshot_path: str,
    anchor_hash,
    enable_similarity: bool,
    enable_thread_verification: bool,
    phash_threshold: int,
):
    rng = random.Random(seed)
    records: List[Dict] = []
    start_time = time.time()

    for i in range(1, rounds + 1):
        element = rng.choice(COMMON_ELEMENT_TEXTS)
        step_start = time.time()

        if enable_similarity:
            score, jac, seq = text_similarity(task, element)
        else:
            score, jac, seq = None, None, None

        # Keep same exploration budget: each explore step sleeps 1 second.
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

    if enable_thread_verification:
        output["thread_verification"] = verify_with_anchor_phash(
            screenshot_path=screenshot_path,
            anchor_hash=anchor_hash,
            threshold=phash_threshold,
        )
    else:
        output["thread_verification"] = None


def _run_variant(
    args,
    variant_name: str,
    enable_similarity: bool,
    enable_thread_verification: bool,
    enable_main_verification: bool,
    screenshot_path: str,
) -> Dict:
    variant_seed_offset = {
        "full_system": 11,
        "ablate_module1_random_explore": 101,
        "ablate_module2_no_thread_phash": 211,
        "ablate_module3_no_main_verification": 307,
    }.get(variant_name, 0)
    rng = random.Random(args.seed + variant_seed_offset)
    history: List[str] = []
    steps: List[Dict] = []

    perception_latency_list: List[float] = []
    reasoning_latency_list: List[float] = []
    exploration_latency_list: List[float] = []
    planning_latency_list: List[float] = []
    operation_latency_list: List[float] = []
    thread_verify_latency_list: List[float] = []
    main_verify_latency_list: List[float] = []
    end_to_end_latency_list: List[float] = []

    thread_verify_fail_count = 0
    main_verify_fail_count = 0

    print(f"\n--- Variant: {variant_name} ---")
    print(
        f"[Variant Config] similarity={enable_similarity}, "
        f"thread_verify={enable_thread_verification}, "
        f"main_verify={enable_main_verification}"
    )

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n[{variant_name}] Iteration {itr}")

        time.sleep(max(0.0, args.perception_sleep_ms / 1000.0))
        perception_end_time = time.time()
        perception_latency = (perception_end_time - start_time) * 1000
        perception_latency_list.append(perception_latency)

        # Anchor hash for both thread/main verification.
        anchor_hash = compute_phash(screenshot_path)

        rounds = rng.randint(3, 5)
        explorer_output: Dict = {}
        explorer_thread = threading.Thread(
            target=_explore_loop,
            args=(
                args.task,
                rounds,
                args.seed + itr * 97,
                explorer_output,
                screenshot_path,
                anchor_hash,
                enable_similarity,
                enable_thread_verification,
                int(args.phash_threshold),
            ),
            daemon=True,
        )
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

        thread_verification = explorer_output.get("thread_verification")
        if thread_verification is not None:
            thread_verify_latency = float(thread_verification.get("verification_latency_ms", 0.0))
            thread_verify_latency_list.append(thread_verify_latency)
            if not bool(thread_verification.get("ok", False)):
                thread_verify_fail_count += 1

        main_verification = None
        if enable_main_verification:
            main_verification = verify_with_anchor_phash(
                screenshot_path=screenshot_path,
                anchor_hash=anchor_hash,
                threshold=args.phash_threshold,
            )
            main_verify_latency = float(main_verification.get("verification_latency_ms", 0.0))
            main_verify_latency_list.append(main_verify_latency)
            if not bool(main_verification.get("ok", False)):
                main_verify_fail_count += 1

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
            "variant": variant_name,
            "action": action_obj,
            "llm_output": llm_output,
            "explore_rounds": rounds,
            "explore_records": explore_records,
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

        print(
            f"[{variant_name}] rounds={rounds}, "
            f"reasoning={reasoning_latency:.3f} ms, "
            f"explore={exploration_latency:.3f} ms, "
            f"step={step_latency:.3f} ms"
        )

    result = {
        "variant": variant_name,
        "config": {
            "enable_similarity": enable_similarity,
            "enable_thread_verification": enable_thread_verification,
            "enable_main_verification": enable_main_verification,
            "phash_threshold": int(args.phash_threshold),
        },
        "avg_perception_latency_ms": avg(perception_latency_list),
        "avg_reasoning_latency_ms": avg(reasoning_latency_list),
        "avg_exploration_latency_ms": avg(exploration_latency_list),
        "avg_thread_verify_latency_ms": avg(thread_verify_latency_list),
        "avg_main_verify_latency_ms": avg(main_verify_latency_list),
        "avg_planning_latency_ms": avg(planning_latency_list),
        "avg_operation_latency_ms": avg(operation_latency_list),
        "avg_step_latency_ms": avg(end_to_end_latency_list),
        "thread_verify_fail_count": thread_verify_fail_count,
        "main_verify_fail_count": main_verify_fail_count,
        "details": steps,
    }

    print(
        f"[Variant Summary] {variant_name}: "
        f"explore={result['avg_exploration_latency_ms']:.3f} ms, "
        f"thread_verify={result['avg_thread_verify_latency_ms']:.3f} ms, "
        f"main_verify={result['avg_main_verify_latency_ms']:.3f} ms, "
        f"step={result['avg_step_latency_ms']:.3f} ms"
    )
    return result


def run_ablation_no_adb(args):
    print("### Running Ablation (No ADB, llama.cpp) ###")
    print(f"[Config] task={args.task}")
    print(f"[Config] max_itr={args.max_itr}")
    screenshot_path = ensure_screenshot_path(args.screenshot_path)
    print(f"[Config] llama_api_url={args.llama_api_url}")
    print(f"[Config] screenshot_path={screenshot_path}")

    full_system = _run_variant(
        args,
        variant_name="full_system",
        enable_similarity=True,
        enable_thread_verification=True,
        enable_main_verification=True,
        screenshot_path=screenshot_path,
    )

    # Module-1 ablation: random explore (remove embedding/similarity scoring in thread).
    ablate_module1_random_explore = _run_variant(
        args,
        variant_name="ablate_module1_random_explore",
        enable_similarity=False,
        enable_thread_verification=True,
        enable_main_verification=True,
        screenshot_path=screenshot_path,
    )

    # Module-2 ablation: remove thread pHash verification.
    ablate_module2_no_thread_phash = _run_variant(
        args,
        variant_name="ablate_module2_no_thread_phash",
        enable_similarity=True,
        enable_thread_verification=False,
        enable_main_verification=True,
        screenshot_path=screenshot_path,
    )

    # Module-3 ablation: remove main-process verification.
    ablate_module3_no_main_verification = _run_variant(
        args,
        variant_name="ablate_module3_no_main_verification",
        enable_similarity=True,
        enable_thread_verification=True,
        enable_main_verification=False,
        screenshot_path=screenshot_path,
    )

    print("\n=== Ablation Summary ===")
    for key, res in [
        ("full_system", full_system),
        ("ablate_module1_random_explore", ablate_module1_random_explore),
        ("ablate_module2_no_thread_phash", ablate_module2_no_thread_phash),
        ("ablate_module3_no_main_verification", ablate_module3_no_main_verification),
    ]:
        print(
            f"{key}: "
            f"explore={res['avg_exploration_latency_ms']:.3f} ms, "
            f"thread_verify={res['avg_thread_verify_latency_ms']:.3f} ms, "
            f"main_verify={res['avg_main_verify_latency_ms']:.3f} ms, "
            f"step={res['avg_step_latency_ms']:.3f} ms"
        )

    results = {
        "full_system": full_system,
        "ablate_module1_random_explore": ablate_module1_random_explore,
        "ablate_module2_no_thread_phash": ablate_module2_no_thread_phash,
        "ablate_module3_no_main_verification": ablate_module3_no_main_verification,
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
        "--phash_threshold",
        type=int,
        default=8,
        help="pHash verification threshold.",
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
