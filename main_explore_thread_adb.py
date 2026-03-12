import argparse
import copy
import json
import os
import random
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from MobileAgentE.controller import back, get_a11y_tree, get_screenshot, home, tap
from MobileAgentE.tree import parse_a11y_tree
from Explorer.utils import click_node_center, collect_clickable_nodes, node_to_text
from agents.mai_prompt import MAI_MOBILE_SYS_PROMPT, build_user_prompt as build_mai_user_prompt
from agents.utils import execute_action
from sim_no_adb_utils import (
    avg,
    call_llama_cpp_with_image,
    compute_phash,
    parse_action_from_llm_text,
    safe_json,
    text_similarity,
    verify_with_anchor_phash,
)


SYSTEM_PROMPT = MAI_MOBILE_SYS_PROMPT


def _explorer_loop(
    args,
    task: str,
    depth: int,
    seed: int,
    output: Dict,
    xml_path: str,
    probe_screenshot_path: str,
    width: int,
    height: int,
    anchor_hash,
    enable_thread_verification: bool,
    phash_threshold: int,
    click_sleep_sec: float,
    rollback_sleep_sec: float,
):
    rng = random.Random(seed)
    depth = max(1, int(depth))
    width = max(2, int(width))
    height = max(2, int(height))
    click_records: List[Dict] = []
    rollback_records: List[Dict] = []
    used_bounds = set()
    start_time = time.time()

    for step in range(1, depth + 1):
        step_start = time.time()

        best_node = None
        best_score = -1.0
        best_jaccard = 0.0
        best_seq_ratio = 0.0
        best_text = ""
        n_candidates = 0

        try:
            get_a11y_tree(args, xml_path)
            root = parse_a11y_tree(xml_path)
            candidates = collect_clickable_nodes(root)
            n_candidates = len(candidates)
            for node in candidates:
                bounds = str(getattr(node, "bounds", "") or "").strip()
                if bounds and bounds in used_bounds:
                    continue
                element_text = node_to_text(node)
                score, jaccard, seq_ratio = text_similarity(task, element_text)
                if score > best_score:
                    best_node = node
                    best_score = score
                    best_jaccard = jaccard
                    best_seq_ratio = seq_ratio
                    best_text = element_text
        except Exception as exc:
            n_candidates = 0
            best_node = None
            best_score = -1.0
            best_jaccard = 0.0
            best_seq_ratio = 0.0
            best_text = f"[a11y_error] {exc}"

        if best_node is not None:
            coordinate, bounds = click_node_center(args.adb_path, best_node)
            if bounds:
                used_bounds.add(str(getattr(best_node, "bounds", "") or "").strip())
            if coordinate is None:
                # Keep fixed depth behavior even if bounds parse fails.
                fx = int(width * 0.5)
                fy = int(height * (0.35 + 0.2 * (step - 1)))
                tap(args.adb_path, fx, fy)
                coordinate = (fx, fy)
                bounds = None
                fallback = True
            else:
                fallback = False
        else:
            # Fallback click to preserve depth-bounded exploration behavior.
            jitter = rng.randint(-40, 40)
            fx = int(width * 0.5 + jitter)
            fy = int(height * (0.35 + 0.2 * (step - 1)))
            fx = max(1, min(width - 2, fx))
            fy = max(1, min(height - 2, fy))
            tap(args.adb_path, fx, fy)
            coordinate = (fx, fy)
            bounds = None
            fallback = True
            if best_score < 0:
                best_score = 0.0

        if click_sleep_sec > 0:
            time.sleep(click_sleep_sec)

        click_records.append(
            {
                "explore_step": step,
                "phase": "click",
                "candidates": n_candidates,
                "element_text": best_text,
                "similarity": best_score,
                "jaccard": best_jaccard,
                "seq_ratio": best_seq_ratio,
                "coordinate": [int(coordinate[0]), int(coordinate[1])],
                "bounds": list(bounds) if isinstance(bounds, tuple) else bounds,
                "fallback_click": bool(fallback),
                "explore_step_latency_ms": (time.time() - step_start) * 1000,
            }
        )

    # Level-1 rollback: issue exactly `depth` backs for depth-bounded exploration.
    for step in range(1, depth + 1):
        rollback_start = time.time()
        back(args.adb_path)
        if rollback_sleep_sec > 0:
            time.sleep(rollback_sleep_sec)
        try:
            get_screenshot(args, probe_screenshot_path, scale=args.scale)
            verify = verify_with_anchor_phash(
                screenshot_path=probe_screenshot_path,
                anchor_hash=anchor_hash,
                threshold=phash_threshold,
            )
        except Exception as exc:
            verify = {
                "ok": False,
                "diff": -1,
                "threshold": int(phash_threshold),
                "error": str(exc),
                "verification_latency_ms": 0.0,
            }
        verify["rollback_step"] = step
        verify["phase"] = "rollback"
        verify["rollback_step_latency_ms"] = (time.time() - rollback_start) * 1000
        rollback_records.append(verify)

    output["records"] = click_records
    output["rollback_records"] = rollback_records
    output["explore_click_count"] = len(click_records)
    output["rollback_count"] = len(rollback_records)
    output["exploration_latency_ms"] = (time.time() - start_time) * 1000

    if enable_thread_verification:
        # Ensure latest post-rollback frame exists for the final verification.
        if not os.path.exists(probe_screenshot_path):
            get_screenshot(args, probe_screenshot_path, scale=args.scale)
        output["thread_verification"] = verify_with_anchor_phash(
            screenshot_path=probe_screenshot_path,
            anchor_hash=anchor_hash,
            threshold=phash_threshold,
        )
    else:
        output["thread_verification"] = None


def _history_to_prompt_text(history: List[str], max_items: int = 8) -> str:
    if not history:
        return "None"
    return "\n".join(history[-max_items:])


def _build_user_prompt(task: str, history: List[str], clue: Optional[str]) -> str:
    base = build_mai_user_prompt(task, _history_to_prompt_text(history))
    if clue:
        base += f"\n\n[Exploration Clue]\n{clue}"
    return base


def _should_start_explore(
    itr: int,
    last_explore_itr: int,
    no_effect_streak: int,
    rng: random.Random,
    explore_probability: float,
    max_explore_gap: int,
    stall_trigger: int,
) -> Tuple[bool, str]:
    if itr == 1:
        return True, "bootstrap"
    if no_effect_streak >= stall_trigger:
        return True, f"stall_streak={no_effect_streak}"
    if last_explore_itr <= 0 or (itr - last_explore_itr) >= max_explore_gap:
        return True, f"max_gap={max_explore_gap}"
    if rng.random() < explore_probability:
        return True, f"prob={explore_probability:.2f}"
    return False, "skip_by_policy"


def _apply_task_intent_guard(task: str, action_obj: Dict) -> Dict:
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


def _recover_to_anchor_with_home_replay(
    args,
    replay_actions: List[Dict[str, Any]],
    width: int,
    height: int,
    anchor_hash,
    phash_threshold: int,
    screenshot_path: str,
) -> Dict[str, Any]:
    start_ts = time.time()
    replay_log: List[Dict[str, Any]] = []

    home(args.adb_path)
    if args.recover_home_wait_sec > 0:
        time.sleep(args.recover_home_wait_sec)

    for idx, past_action in enumerate(replay_actions, 1):
        action_copy = copy.deepcopy(past_action)
        try:
            executed = execute_action(
                action_obj=action_copy,
                width=width,
                height=height,
                adb=args.adb_path,
                coord_scale=args.scale,
            )
            replay_log.append({"idx": idx, "ok": True, "executed_action": executed, "action": action_copy})
        except Exception as exc:
            replay_log.append({"idx": idx, "ok": False, "error": str(exc), "action": action_copy})
        if args.recover_replay_step_wait_sec > 0:
            time.sleep(args.recover_replay_step_wait_sec)

    get_screenshot(args, screenshot_path, scale=args.scale)
    verify = verify_with_anchor_phash(
        screenshot_path=screenshot_path,
        anchor_hash=anchor_hash,
        threshold=phash_threshold,
    )
    return {
        "ok": bool(verify.get("ok", False)),
        "verify": verify,
        "replay_len": len(replay_actions),
        "replay_log": replay_log,
        "recovery_latency_ms": (time.time() - start_ts) * 1000,
    }


def run_explore_thread_adb(args):
    print("### Running ADB Agent with Explore Thread ###")
    print(f"[Config] task={args.task}")
    print(
        f"[Config] max_itr={args.max_itr}, explore_depth={args.explore_depth}, rollback_steps={args.explore_depth}, scale={args.scale}, "
        f"explore_probability={args.explore_probability}, max_explore_gap={args.max_explore_gap}, "
        f"reasoning_sleep_sec={args.reasoning_sleep_sec}, explore_step_sleep_sec={args.explore_step_sleep_sec}, "
        f"rollback_step_sleep_sec={args.rollback_step_sleep_sec}, "
        f"level2_recovery={args.enable_level2_recovery}, require_verified_state={args.require_verified_state}"
    )
    print(f"[Config] llama_api_url={args.llama_api_url}")
    _ensure_adb_device(args.adb_path)
    os.makedirs(os.path.dirname(args.screenshot_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.explore_xml_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.explore_probe_screenshot_path) or ".", exist_ok=True)

    random_gen = random.Random(args.seed)
    history: List[str] = []
    steps: List[Dict] = []
    executed_action_trace: List[Dict[str, Any]] = []
    pending_clue: Optional[str] = None
    last_explore_itr = 0
    no_effect_streak = 0

    perception_latency_list: List[float] = []
    reasoning_latency_list: List[float] = []
    exploration_latency_list: List[float] = []
    planning_latency_list: List[float] = []
    thread_verify_latency_list: List[float] = []
    main_verify_latency_list: List[float] = []
    recovery_latency_list: List[float] = []
    operation_latency_list: List[float] = []
    end_to_end_latency_list: List[float] = []
    thread_verify_fail_count = 0
    main_verify_fail_count = 0
    recovery_count = 0

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        get_screenshot(args, args.screenshot_path, scale=args.scale)
        width, height = Image.open(args.screenshot_path).size
        perception_end_time = time.time()
        perception_latency = (perception_end_time - start_time) * 1000
        perception_latency_list.append(perception_latency)

        anchor_hash = compute_phash(args.screenshot_path)
        should_explore, explore_reason = _should_start_explore(
            itr=itr,
            last_explore_itr=last_explore_itr,
            no_effect_streak=no_effect_streak,
            rng=random_gen,
            explore_probability=float(args.explore_probability),
            max_explore_gap=int(args.max_explore_gap),
            stall_trigger=int(args.stall_trigger),
        )
        explore_depth = int(args.explore_depth) if should_explore else 0
        explorer_output: Dict = {}
        explorer_thread = None

        if should_explore:
            explorer_thread = threading.Thread(
                target=_explorer_loop,
                args=(
                    args,
                    args.task,
                    explore_depth,
                    args.seed + itr * 1009,
                    explorer_output,
                    args.explore_xml_path,
                    args.explore_probe_screenshot_path,
                    width,
                    height,
                    anchor_hash,
                    bool(args.enable_thread_verification),
                    int(args.phash_threshold),
                    float(args.explore_step_sleep_sec),
                    float(args.rollback_step_sleep_sec),
                ),
                daemon=True,
            )
            explorer_thread.start()
            last_explore_itr = itr

        reasoning_start = time.time()
        llm_output = call_llama_cpp_with_image(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(args.task, history, pending_clue),
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

        if explorer_thread is not None:
            explorer_thread.join()
            sync_end_time = time.time()
        else:
            sync_end_time = reasoning_end

        explore_records = explorer_output.get("records", []) if should_explore else []
        rollback_records = explorer_output.get("rollback_records", []) if should_explore else []
        explore_click_count = int(explorer_output.get("explore_click_count", 0)) if should_explore else 0
        rollback_count = int(explorer_output.get("rollback_count", 0)) if should_explore else 0
        exploration_latency = float(explorer_output.get("exploration_latency_ms", 0.0)) if should_explore else 0.0
        exploration_latency_list.append(exploration_latency)
        best_record = max(explore_records, key=lambda x: x["similarity"]) if explore_records else None

        thread_verification = explorer_output.get("thread_verification") if should_explore else None
        if thread_verification is not None:
            thread_v_ms = float(thread_verification.get("verification_latency_ms", 0.0))
            thread_verify_latency_list.append(thread_v_ms)
            if not bool(thread_verification.get("ok", False)):
                thread_verify_fail_count += 1

        # State verification and recovery before executing reasoning action.
        state_verified = True
        state_recovery = None
        main_verification = None
        if should_explore and args.enable_main_verification:
            get_screenshot(args, args.screenshot_path, scale=args.scale)
            main_verification = verify_with_anchor_phash(
                screenshot_path=args.screenshot_path,
                anchor_hash=anchor_hash,
                threshold=args.phash_threshold,
            )
            main_v_ms = float(main_verification.get("verification_latency_ms", 0.0))
            main_verify_latency_list.append(main_v_ms)
            if not bool(main_verification.get("ok", False)):
                main_verify_fail_count += 1
                state_verified = False
            else:
                state_verified = True
        elif should_explore and thread_verification is not None:
            state_verified = bool(thread_verification.get("ok", False))

        if should_explore and not state_verified and args.enable_level2_recovery:
            recovery_count += 1
            state_recovery = _recover_to_anchor_with_home_replay(
                args=args,
                replay_actions=executed_action_trace,
                width=width,
                height=height,
                anchor_hash=anchor_hash,
                phash_threshold=int(args.phash_threshold),
                screenshot_path=args.screenshot_path,
            )
            recovery_latency_list.append(float(state_recovery.get("recovery_latency_ms", 0.0)))
            state_verified = bool(state_recovery.get("ok", False))

            if args.rerun_reasoning_on_recovery:
                rerun_start = time.time()
                llm_output = call_llama_cpp_with_image(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=_build_user_prompt(args.task, history, pending_clue),
                    screenshot_path=args.screenshot_path,
                    api_url=args.llama_api_url,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                action_obj = parse_action_from_llm_text(llm_output)
                action_obj = _apply_task_intent_guard(args.task, action_obj)
                if args.reasoning_sleep_sec > 0:
                    time.sleep(args.reasoning_sleep_sec)
                rerun_ms = (time.time() - rerun_start) * 1000
                reasoning_latency += rerun_ms

        reasoning_latency_list.append(reasoning_latency)
        planning_end_time = time.time() if should_explore else sync_end_time
        planning_latency = (planning_end_time - perception_end_time) * 1000
        planning_latency_list.append(planning_latency)

        skipped_due_to_unverified_state = bool(
            should_explore and args.require_verified_state and not state_verified
        )

        operation_start = time.time()
        executed_action = None
        action_screen_diff = 0
        action_effective = False
        if not skipped_due_to_unverified_state:
            before_action_hash = compute_phash(args.screenshot_path)
            executed_action = execute_action(
                action_obj=action_obj,
                width=width,
                height=height,
                adb=args.adb_path,
                coord_scale=args.scale,
            )
            if args.post_action_wait_sec > 0:
                time.sleep(args.post_action_wait_sec)
            get_screenshot(args, args.screenshot_path, scale=args.scale)
            after_action_hash = compute_phash(args.screenshot_path)
            action_screen_diff = int(after_action_hash - before_action_hash)
            action_effective = action_screen_diff > int(args.action_effect_threshold)
            if action_effective:
                no_effect_streak = 0
            else:
                no_effect_streak += 1
        else:
            no_effect_streak += 1

        operation_end = time.time()
        operation_latency = (operation_end - operation_start) * 1000
        operation_latency_list.append(operation_latency)

        if skipped_due_to_unverified_state:
            history.append("skip_execution_unverified_state")
        elif executed_action is not None:
            history.append(str(executed_action))
        else:
            history.append(safe_json(action_obj))

        if (not skipped_due_to_unverified_state) and executed_action is not None:
            executed_action_trace.append(copy.deepcopy(action_obj))

        if best_record:
            pending_clue = (
                f"best_element={best_record['element_text']}, "
                f"similarity={best_record['similarity']:.3f}, depth={explore_depth}"
            )
        elif should_explore:
            pending_clue = None

        step_latency = (operation_end - start_time) * 1000
        end_to_end_latency_list.append(step_latency)

        steps.append(
            {
                "step": itr,
                "action": action_obj,
                "llm_output": llm_output,
                "explore_enabled": should_explore,
                "explore_reason": explore_reason,
                "explore_depth": explore_depth,
                "explore_click_count": explore_click_count,
                "rollback_count": rollback_count,
                "explore_records": explore_records,
                "rollback_records": rollback_records,
                "best_explore_record": best_record,
                "thread_verification": thread_verification,
                "main_verification": main_verification,
                "state_verified_before_execution": state_verified,
                "state_recovery": state_recovery,
                "skipped_due_to_unverified_state": skipped_due_to_unverified_state,
                "action_effective": action_effective,
                "action_screen_diff": action_screen_diff,
                "no_effect_streak": no_effect_streak,
                "executed_action": executed_action,
                "screenshot_path": args.screenshot_path,
                "width": width,
                "height": height,
                "perception_latency_ms": perception_latency,
                "reasoning_latency_ms": reasoning_latency,
                "exploration_latency_ms": exploration_latency,
                "planning_latency_ms": planning_latency,
                "operation_latency_ms": operation_latency,
                "step_latency_ms": step_latency,
            }
        )

        print(f"[ExplorePolicy] enabled={should_explore}, reason={explore_reason}")
        if should_explore:
            print(f"[Explorer] click_count={explore_click_count}, rollback_count={rollback_count}")
        if best_record:
            print(
                f"[Explorer] depth={explore_depth}, best_similarity={best_record['similarity']:.3f}, "
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
        print(f"[State Verify] restored={state_verified}")
        if state_recovery is not None:
            rec_verify = state_recovery.get("verify", {}) or {}
            print(
                f"[Level2 Recovery] ok={state_recovery.get('ok')} replay_len={state_recovery.get('replay_len')} "
                f"diff={rec_verify.get('diff')} threshold={rec_verify.get('threshold')}"
            )
        if skipped_due_to_unverified_state:
            print("[Execution] skipped due to unverified state after rollback/recovery.")
        print("[Reasoning] Parsed action:", action_obj)
        print("[Execution] Action done:", executed_action)
        print(
            f"[Action Effect] diff={action_screen_diff}, threshold={args.action_effect_threshold}, "
            f"effective={action_effective}, no_effect_streak={no_effect_streak}"
        )
        print(
            f"Perception latency: {perception_latency:.3f} ms, "
            f"Reasoning latency: {reasoning_latency:.3f} ms, "
            f"Exploration latency: {exploration_latency:.3f} ms, "
            f"Planning latency: {planning_latency:.3f} ms, "
            f"Operation latency: {operation_latency:.3f} ms"
        )
        print(f"Step latency: {step_latency:.3f} ms")

        action_type = str(action_obj.get("action_type", "")).lower()
        if action_type in {"terminate", "finished", "done", "finish", "stop", "exit"}:
            print("[Stop] finish-like action detected, exiting loop.")
            break

    print("\n=== Finished all iterations (explore thread + adb) ===")
    print(
        f"Perception latency: {avg(perception_latency_list):.3f} ms, "
        f"Reasoning latency: {avg(reasoning_latency_list):.3f} ms, "
        f"Exploration latency: {avg(exploration_latency_list):.3f} ms, "
        f"ThreadVerify latency: {avg(thread_verify_latency_list):.3f} ms, "
        f"MainVerify latency: {avg(main_verify_latency_list):.3f} ms, "
        f"Recovery latency: {avg(recovery_latency_list):.3f} ms, "
        f"Planning latency: {avg(planning_latency_list):.3f} ms, "
        f"Operation latency: {avg(operation_latency_list):.3f} ms, "
        f"End-to-end latency: {avg(end_to_end_latency_list):.3f} ms"
    )
    print(
        f"Thread verification failures: {thread_verify_fail_count}, "
        f"Main verification failures: {main_verify_fail_count}, "
        f"Level2 recoveries: {recovery_count}"
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
        help="User instruction for the explore-thread run.",
    )
    parser.add_argument("--max_itr", type=int, default=10, help="Maximum benchmark iterations.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
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
        default=8,
        help="Extra sleep added after each LLM response to simulate reasoning latency.",
    )
    parser.add_argument(
        "--explore_step_sleep_sec",
        type=float,
        default=0.35,
        help="Per-step sleep in explore thread (usually smaller than baseline reasoning sleep).",
    )
    parser.add_argument(
        "--rollback_step_sleep_sec",
        type=float,
        default=0.20,
        help="Per-step sleep after each rollback(back) action in explore thread.",
    )
    parser.add_argument(
        "--explore_depth",
        type=int,
        default=2,
        help="Depth-bounded exploration steps. d=2 means click twice then rollback twice.",
    )
    parser.add_argument(
        "--explore_xml_path",
        type=str,
        default="./screenshot/a11y.xml",
        help="Temporary path for exploration thread accessibility tree dump.",
    )
    parser.add_argument(
        "--explore_probe_screenshot_path",
        type=str,
        default="./screenshot/explore_probe.png",
        help="Temporary screenshot path used by exploration rollback verification.",
    )
    parser.add_argument(
        "--explore_probability",
        type=float,
        default=0.35,
        help="Probability of optional exploration when not forced by bootstrap/gap/stall.",
    )
    parser.add_argument(
        "--max_explore_gap",
        type=int,
        default=3,
        help="Force one exploration if this many iterations passed since last exploration.",
    )
    parser.add_argument(
        "--stall_trigger",
        type=int,
        default=1,
        help="Force exploration when consecutive no-effect actions reach this threshold.",
    )
    parser.add_argument("--phash_threshold", type=int, default=8, help="pHash verification threshold.")
    parser.add_argument(
        "--action_effect_threshold",
        type=int,
        default=3,
        help="Post-action pHash diff threshold to judge whether action changed the page.",
    )
    parser.add_argument(
        "--recover_home_wait_sec",
        type=float,
        default=0.35,
        help="Wait time after pressing home in Level-2 recovery.",
    )
    parser.add_argument(
        "--recover_replay_step_wait_sec",
        type=float,
        default=0.15,
        help="Wait time between replayed actions in Level-2 recovery.",
    )
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
    parser.add_argument(
        "--disable_level2_recovery",
        dest="enable_level2_recovery",
        action="store_false",
        help="Disable Level-2 home+replay recovery after failed state verification.",
    )
    parser.add_argument(
        "--disable_rerun_reasoning_on_recovery",
        dest="rerun_reasoning_on_recovery",
        action="store_false",
        help="Disable re-running reasoning after Level-2 recovery.",
    )
    parser.add_argument(
        "--allow_unverified_execution",
        dest="require_verified_state",
        action="store_false",
        help="Allow executing reasoning action even when state verification/recovery fails.",
    )
    parser.set_defaults(
        enable_thread_verification=True,
        enable_main_verification=True,
        enable_level2_recovery=True,
        rerun_reasoning_on_recovery=True,
        require_verified_state=True,
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

    run_explore_thread_adb(args)
