import argparse
import json
import os
import re
import subprocess
import time
from typing import Dict, List

from PIL import Image

from MobileAgentE.controller import get_screenshot
from agents.mai_prompt import MAI_MOBILE_SYS_PROMPT, build_user_prompt as build_mai_user_prompt
from agents.utils import execute_action
from sim_no_adb_utils import (
    avg,
    call_llama_cpp_with_image,
    parse_action_from_llm_text,
    safe_json,
)


SYSTEM_PROMPT = MAI_MOBILE_SYS_PROMPT


def _history_to_prompt_text(history: List[str], max_items: int = 8) -> str:
    if not history:
        return "None"
    return "\n".join(history[-max_items:])


def _build_user_prompt(task: str, history: List[str]) -> str:
    return build_mai_user_prompt(task, _history_to_prompt_text(history))


def _apply_task_intent_guard(task: str, action_obj: Dict) -> Dict:
    """
    Guard for smaller models: if task explicitly asks a system button,
    normalize output to system action even when model emits a nav-bar click.
    """
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


def run_baseline_adb(args):
    print("### Running Baseline with Real ADB Execution ###")
    print(f"[Config] task={args.task}")
    print(
        f"[Config] max_itr={args.max_itr}, scale={args.scale}, "
        f"reasoning_sleep_sec={args.reasoning_sleep_sec}"
    )
    print(f"[Config] llama_api_url={args.llama_api_url}")
    _ensure_adb_device(args.adb_path)

    os.makedirs(os.path.dirname(args.screenshot_path) or ".", exist_ok=True)

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

        # Perception: capture current phone screenshot.
        get_screenshot(args, args.screenshot_path, scale=args.scale)
        width, height = Image.open(args.screenshot_path).size
        perception_end_time = time.time()
        perception_latency = (perception_end_time - start_time) * 1000
        perception_latency_list.append(perception_latency)

        # Reasoning: same API style as previous simulation scripts.
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
        if args.reasoning_sleep_sec > 0:
            time.sleep(args.reasoning_sleep_sec)
        reasoning_end = time.time()
        reasoning_latency = (reasoning_end - reasoning_start) * 1000
        reasoning_latency_list.append(reasoning_latency)

        planning_end_time = reasoning_end
        planning_latency = (planning_end_time - perception_end_time) * 1000
        planning_latency_list.append(planning_latency)

        # Real execution via adb.
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
        operation_end = time.time()
        operation_latency = (operation_end - operation_start) * 1000
        operation_latency_list.append(operation_latency)

        if executed_action is not None:
            history.append(str(executed_action))
        else:
            history.append(safe_json(action_obj))

        step_latency = (operation_end - start_time) * 1000
        end_to_end_latency_list.append(step_latency)

        steps.append(
            {
                "step": itr,
                "action": action_obj,
                "llm_output": llm_output,
                "executed_action": executed_action,
                "screenshot_path": args.screenshot_path,
                "width": width,
                "height": height,
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
        if action_type in {"terminate", "finished", "done", "finish", "stop", "exit"}:
            print("[Stop] finish-like action detected, exiting loop.")
            break

    print("\n=== Finished all iterations (baseline adb) ===")
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
        default="Open the setting and turn off Bluetooth.",
        help="User instruction for the baseline run.",
    )
    parser.add_argument("--max_itr", type=int, default=10, help="Maximum benchmark iterations.")
    parser.add_argument("--adb_path", type=str, default="adb", help="ADB path.")
    parser.add_argument(
        "--screenshot_path",
        type=str,
        default="./screenshot/screenshot.png",
        help="Local path to store each captured screenshot.",
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
        "--reasoning_sleep_sec",
        type=float,
            default=19,
        help="Extra sleep added to each step reasoning to simulate real inference latency.",
    )
    parser.add_argument(
        "--post_action_wait_sec",
        type=float,
        default=0.35,
        help="Extra wait after action execution for UI settle.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional output JSON file for step-level records.",
    )
    parser.add_argument("--on_device", action="store_true", help="Unused, kept for compatibility.")
    args = parser.parse_args()

    run_baseline_adb(args)
