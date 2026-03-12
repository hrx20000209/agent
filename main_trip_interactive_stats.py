import argparse
import json
import os
import re
import subprocess
import time
from typing import Any, Dict, List

from PIL import Image

from MobileAgentE.controller import get_a11y_tree, get_screenshot
from MobileAgentE.tree import parse_a11y_tree
from agents.mai_prompt import MAI_MOBILE_SYS_PROMPT, build_user_prompt as build_mai_user_prompt
from agents.utils import execute_action
from sim_no_adb_utils import avg, call_llama_cpp_with_image, compute_phash, parse_action_from_llm_text, safe_json


SYSTEM_PROMPT = MAI_MOBILE_SYS_PROMPT


def _history_to_prompt_text(history: List[str], max_items: int = 8) -> str:
    if not history:
        return "None"
    return "\n".join(history[-max_items:])


def _build_user_prompt(task: str, history: List[str]) -> str:
    return build_mai_user_prompt(task, _history_to_prompt_text(history))


def _apply_task_intent_guard(task: str, action_obj: Dict[str, Any]) -> Dict[str, Any]:
    txt = (task or "").lower()
    if re.search(r"(press|tap|click)\s+home|home button|go home", txt):
        return {"action_type": "press_home", "action_inputs": {"button": "home"}}
    if re.search(r"(press|tap|click)\s+back|back button|go back", txt):
        return {"action_type": "press_back", "action_inputs": {"button": "back"}}
    if re.search(r"(press|tap|click)\s+enter|enter key", txt):
        return {"action_type": "press_enter", "action_inputs": {"button": "enter"}}
    return action_obj


def _ensure_adb_device(adb_path: str):
    proc = subprocess.run([adb_path, "get-state"], capture_output=True, text=True)
    state = (proc.stdout or "").strip().lower()
    if proc.returncode != 0 or state != "device":
        raise RuntimeError(
            f"No adb device in 'device' state. adb_path={adb_path}, "
            f"stdout={proc.stdout.strip()}, stderr={proc.stderr.strip()}"
        )


def _count_interactive_elements(xml_path: str) -> int:
    root = parse_a11y_tree(xml_path)
    count = 0
    for node in getattr(root, "children", []) or []:
        if (
            bool(getattr(node, "clickable", False))
            or bool(getattr(node, "focusable", False))
            or bool(getattr(node, "scrollable", False))
            or bool(getattr(node, "long_clickable", False))
            or bool(getattr(node, "checkable", False))
        ):
            count += 1
    return int(count)


def run_trip_interactive_stats(args):
    print("### Running Trip Interactive Element Statistics ###")
    print(f"[Config] task={args.task}")
    print(f"[Config] max_itr={args.max_itr}, scale={args.scale}")
    print(f"[Config] llama_api_url={args.llama_api_url}")
    _ensure_adb_device(args.adb_path)

    os.makedirs(os.path.dirname(args.screenshot_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.xml_path) or ".", exist_ok=True)

    history: List[str] = []
    steps: List[Dict[str, Any]] = []
    page_counts: List[int] = []
    unique_page_counts: Dict[str, int] = {}

    for itr in range(1, args.max_itr + 1):
        print(f"\n================ Iteration {itr} ================\n")

        get_screenshot(args, args.screenshot_path, scale=args.scale)
        width, height = Image.open(args.screenshot_path).size

        get_a11y_tree(args, args.xml_path)
        interactive_count = _count_interactive_elements(args.xml_path)
        page_hash = str(compute_phash(args.screenshot_path))
        page_counts.append(interactive_count)
        if page_hash not in unique_page_counts:
            unique_page_counts[page_hash] = interactive_count

        reasoning_start = time.time()
        llm_output = call_llama_cpp_with_image(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(args.task, history),
            screenshot_path=args.screenshot_path,
            api_url=args.llama_api_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        action_obj = parse_action_from_llm_text(llm_output)
        action_obj = _apply_task_intent_guard(args.task, action_obj)
        reasoning_latency_ms = (time.time() - reasoning_start) * 1000

        operation_start = time.time()
        executed_action = execute_action(
            action_obj=action_obj,
            width=width,
            height=height,
            adb=args.adb_path,
            coord_scale=args.scale,
        )
        if args.post_action_wait_sec > 0:
            time.sleep(args.post_action_wait_sec)
        operation_latency_ms = (time.time() - operation_start) * 1000

        if executed_action is not None:
            history.append(str(executed_action))
        else:
            history.append(safe_json(action_obj))

        step_record = {
            "step": itr,
            "interactive_elements": interactive_count,
            "page_hash": page_hash,
            "action": action_obj,
            "executed_action": executed_action,
            "llm_output": llm_output,
            "reasoning_latency_ms": reasoning_latency_ms,
            "operation_latency_ms": operation_latency_ms,
        }
        steps.append(step_record)

        print(
            f"[Page] interactive_elements={interactive_count}, "
            f"unique_pages={len(unique_page_counts)}"
        )
        print("[Reasoning] Parsed action:", action_obj)
        print("[Execution] Action done:", executed_action)

        action_type = str(action_obj.get("action_type", "")).lower()
        if action_type in {"terminate", "finished", "done", "finish", "stop", "exit"}:
            print("[Stop] finish-like action detected, exiting loop.")
            break

    summary = {
        "task": args.task,
        "total_page_visits": len(page_counts),
        "unique_pages": len(unique_page_counts),
        "avg_interactive_per_page_visit": avg(page_counts),
        "avg_interactive_per_unique_page": avg(unique_page_counts.values()),
        "min_interactive_per_page_visit": min(page_counts) if page_counts else 0,
        "max_interactive_per_page_visit": max(page_counts) if page_counts else 0,
        "sum_interactive_per_page_visit": int(sum(page_counts)),
    }

    print("\n=== Interactive Element Statistics ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    output = {"summary": summary, "steps": steps, "unique_page_counts": unique_page_counts}
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"[Saved] statistics -> {args.output_json}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="Search attractions in Los Angeles in Trip app and open the first attraction.",
        help="Task instruction to complete while collecting interactive-element statistics.",
    )
    parser.add_argument("--max_itr", type=int, default=15, help="Maximum agent iterations.")
    parser.add_argument("--adb_path", type=str, default="adb", help="ADB path.")
    parser.add_argument(
        "--screenshot_path",
        type=str,
        default="./screenshot/screenshot.png",
        help="Local path to store captured screenshot.",
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default="./screenshot/a11y.xml",
        help="Local path to store accessibility tree xml.",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Screenshot downscale factor.")
    parser.add_argument(
        "--llama_api_url",
        type=str,
        default="http://localhost:8081/v1/chat/completions",
        help="llama.cpp OpenAI-compatible endpoint.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max new tokens per request.")
    parser.add_argument(
        "--post_action_wait_sec",
        type=float,
        default=0.35,
        help="Extra wait after action execution for UI settle.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="./trip_interactive_stats.json",
        help="Output json path.",
    )
    parser.add_argument("--on_device", action="store_true", help="Unused, kept for compatibility.")
    args = parser.parse_args()

    run_trip_interactive_stats(args)
