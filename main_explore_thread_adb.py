import argparse
import copy
import json
import os
import random
import re
import subprocess
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


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_resource_id(resource_id: str) -> str:
    rid = _clean_text(resource_id)
    if not rid:
        return ""
    rid = rid.split("/")[-1]
    rid = rid.split(":")[-1]
    rid = rid.replace("_", " ")
    rid = re.sub(r"[^a-zA-Z0-9 ]+", " ", rid)
    return _clean_text(rid)


def _goal_app_keywords(goal: str) -> List[str]:
    text = _clean_text(goal).lower()
    out: List[str] = []
    seen = set()
    for token in re.findall(r"[a-z0-9]{4,}", text):
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= 24:
            break
    return out


def _goal_tokens(goal: str) -> List[str]:
    text = _clean_text(goal).lower()
    out: List[str] = []
    seen = set()
    for token in re.findall(r"[a-z0-9]{3,}", text):
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= 48:
            break
    return out


def _text_similarity_score(merged: str, goal_tokens: List[str], app_keywords: List[str]) -> float:
    merged_low = _clean_text(merged).lower()
    if not merged_low:
        return 0.0

    score = 0.0
    goal_token_set = set(goal_tokens)

    for kw in app_keywords:
        if kw and kw in merged_low:
            score += 1.5 + min(len(kw), 12) * 0.06

    merged_tokens = set(re.findall(r"[a-z0-9]{3,}", merged_low))
    if merged_tokens and goal_token_set:
        overlap = merged_tokens.intersection(goal_token_set)
        if overlap:
            score += float(len(overlap)) * 1.2
            score += float(len(overlap)) / max(1.0, float(len(merged_tokens)))

    return float(score)


def _element_text(node: Any) -> str:
    text = _clean_text(getattr(node, "text", ""))
    desc = _clean_text(getattr(node, "content_desc", ""))
    rid = _normalize_resource_id(str(getattr(node, "resource_id", "")))
    merged = " ".join(x for x in [text, desc, rid] if x)
    return _clean_text(merged).lower()


def _is_bad_element_text(text: str) -> bool:
    low = (text or "").lower()
    if not low:
        return True
    bad_patterns = [
        "inputmethod",
        "systemui",
        "navigation bar",
        "key_pos_",
        "keyboard",
    ]
    return any(p in low for p in bad_patterns)


def _history_to_prompt_text(history: List[str], max_items: int = 8) -> str:
    if not history:
        return "None"
    return "\n".join(history[-max_items:])


def _build_user_prompt(task: str, history: List[str], clue: Optional[str]) -> str:
    base = build_mai_user_prompt(task, _history_to_prompt_text(history))
    if clue:
        base += f"\n\n[Exploration Clue]\n{clue}"
    return base


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


def _collect_ranked_candidates(args, task: str, xml_path: str, used_bounds: set[str]) -> List[Dict[str, Any]]:
    get_a11y_tree(args, xml_path)
    root = parse_a11y_tree(xml_path)
    nodes = collect_clickable_nodes(root)
    app_keywords = _goal_app_keywords(task)
    goal_tokens = _goal_tokens(task)

    candidates = []
    for idx, node in enumerate(nodes):
        bounds_key = _clean_text(getattr(node, "bounds", ""))
        if bounds_key and bounds_key in used_bounds:
            continue

        merged = _element_text(node)
        if _is_bad_element_text(merged):
            continue

        score_rule = _text_similarity_score(merged, goal_tokens=goal_tokens, app_keywords=app_keywords)
        score_soft, jaccard, seq_ratio = text_similarity(task, merged)
        final_score = 0.72 * score_rule + 0.28 * score_soft

        candidates.append(
            {
                "idx": idx,
                "node": node,
                "bounds_key": bounds_key,
                "element_text": merged,
                "score": float(final_score),
                "score_rule": float(score_rule),
                "score_soft": float(score_soft),
                "jaccard": float(jaccard),
                "seq_ratio": float(seq_ratio),
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def _run_serial_exploration(
    args,
    task: str,
    rounds: int,
    width: int,
    height: int,
    anchor_hash,
) -> Dict[str, Any]:
    rounds = max(1, int(rounds))
    width = max(2, int(width))
    height = max(2, int(height))
    used_bounds: set[str] = set()

    click_records: List[Dict[str, Any]] = []
    rollback_records: List[Dict[str, Any]] = []
    top_candidates_debug: List[List[Dict[str, Any]]] = []

    start_ts = time.time()
    clicked_count = 0

    for step in range(1, rounds + 1):
        step_start = time.time()
        selected = None
        shortlist = []

        try:
            ranked = _collect_ranked_candidates(args=args, task=task, xml_path=args.explore_xml_path, used_bounds=used_bounds)
            shortlist = [
                {
                    "text": c["element_text"],
                    "score": c["score"],
                    "score_rule": c["score_rule"],
                    "score_soft": c["score_soft"],
                }
                for c in ranked[: max(1, int(args.explore_shortlist_k))]
            ]
            top_candidates_debug.append(shortlist)
            for cand in ranked:
                if cand["score"] >= float(args.explore_score_threshold):
                    selected = cand
                    break
        except Exception as exc:
            top_candidates_debug.append([{"text": f"[collect_error] {exc}", "score": -1.0}])
            selected = None

        if selected is None:
            if not bool(args.explore_allow_fallback_click):
                break
            fx = int(width * 0.5)
            fy = int(height * (0.35 + 0.2 * (step - 1)))
            fx = max(1, min(width - 2, fx))
            fy = max(1, min(height - 2, fy))
            tap(args.adb_path, fx, fy)
            coord = [fx, fy]
            bounds = None
            best_text = "fallback_click"
            best_score = 0.0
            jaccard = 0.0
            seq_ratio = 0.0
            fallback_click = True
        else:
            node = selected["node"]
            coord_raw, bounds = click_node_center(args.adb_path, node)
            if coord_raw is None:
                if not bool(args.explore_allow_fallback_click):
                    break
                fx = int(width * 0.5)
                fy = int(height * (0.35 + 0.2 * (step - 1)))
                tap(args.adb_path, fx, fy)
                coord = [fx, fy]
                bounds = None
                fallback_click = True
            else:
                coord = [int(coord_raw[0]), int(coord_raw[1])]
                fallback_click = False

            best_text = selected["element_text"]
            best_score = float(selected["score"])
            jaccard = float(selected["jaccard"])
            seq_ratio = float(selected["seq_ratio"])
            if selected["bounds_key"]:
                used_bounds.add(selected["bounds_key"])

        clicked_count += 1
        if args.explore_step_sleep_sec > 0:
            time.sleep(args.explore_step_sleep_sec)

        click_records.append(
            {
                "explore_step": step,
                "phase": "click",
                "element_text": best_text,
                "similarity": best_score,
                "jaccard": jaccard,
                "seq_ratio": seq_ratio,
                "coordinate": coord,
                "bounds": list(bounds) if isinstance(bounds, tuple) else bounds,
                "fallback_click": bool(fallback_click),
                "shortlist": shortlist,
                "explore_step_latency_ms": (time.time() - step_start) * 1000,
            }
        )

    # rollback exactly clicked_count times
    for r in range(1, clicked_count + 1):
        rb_start = time.time()
        back(args.adb_path)
        if args.rollback_step_sleep_sec > 0:
            time.sleep(args.rollback_step_sleep_sec)

        get_screenshot(args, args.explore_probe_screenshot_path, scale=args.scale)
        verify = verify_with_anchor_phash(
            screenshot_path=args.explore_probe_screenshot_path,
            anchor_hash=anchor_hash,
            threshold=args.phash_threshold,
        )
        verify["phase"] = "rollback"
        verify["rollback_step"] = r
        verify["rollback_step_latency_ms"] = (time.time() - rb_start) * 1000
        rollback_records.append(verify)

    get_screenshot(args, args.explore_probe_screenshot_path, scale=args.scale)
    final_verify = verify_with_anchor_phash(
        screenshot_path=args.explore_probe_screenshot_path,
        anchor_hash=anchor_hash,
        threshold=args.phash_threshold,
    )

    return {
        "records": click_records,
        "rollback_records": rollback_records,
        "top_candidates_debug": top_candidates_debug,
        "explore_click_count": clicked_count,
        "rollback_count": len(rollback_records),
        "thread_verification": final_verify if args.enable_thread_verification else None,
        "exploration_latency_ms": (time.time() - start_ts) * 1000,
    }


def run_explore_thread_adb(args):
    print("### Running ADB Agent with Serial Exploration ###")
    print(f"[Config] task={args.task}")
    print(
        f"[Config] max_itr={args.max_itr}, explore_rounds={args.explore_rounds}, scale={args.scale}, "
        f"explore_probability={args.explore_probability}, max_explore_gap={args.max_explore_gap}, "
        f"reasoning_sleep_sec={args.reasoning_sleep_sec}, explore_step_sleep_sec={args.explore_step_sleep_sec}, "
        f"rollback_step_sleep_sec={args.rollback_step_sleep_sec}, score_threshold={args.explore_score_threshold}, "
        f"level2_recovery={args.enable_level2_recovery}, require_verified_state={args.require_verified_state}"
    )
    print(f"[Config] llama_api_url={args.llama_api_url}")

    _ensure_adb_device(args.adb_path)
    os.makedirs(os.path.dirname(args.screenshot_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.explore_xml_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.explore_probe_screenshot_path) or ".", exist_ok=True)

    random_gen = random.Random(args.seed)
    history: List[str] = []
    steps: List[Dict[str, Any]] = []
    executed_action_trace: List[Dict[str, Any]] = []
    pending_clue: Optional[str] = None
    last_explore_itr = 0
    no_effect_streak = 0

    perception_latency_list: List[float] = []
    exploration_latency_list: List[float] = []
    reasoning_latency_list: List[float] = []
    planning_latency_list: List[float] = []
    main_verify_latency_list: List[float] = []
    recovery_latency_list: List[float] = []
    operation_latency_list: List[float] = []
    end_to_end_latency_list: List[float] = []

    main_verify_fail_count = 0
    recovery_count = 0

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        # Perception at start of iteration
        get_screenshot(args, args.screenshot_path, scale=args.scale)
        width, height = Image.open(args.screenshot_path).size
        perception_end = time.time()
        perception_latency = (perception_end - start_time) * 1000
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

        # Serial exploration first (before VLM)
        explore_output: Dict[str, Any] = {}
        explore_rounds = int(args.explore_rounds) if should_explore else 0
        if should_explore and explore_rounds > 0:
            explore_output = _run_serial_exploration(
                args=args,
                task=args.task,
                rounds=explore_rounds,
                width=width,
                height=height,
                anchor_hash=anchor_hash,
            )
            last_explore_itr = itr

        explore_records = explore_output.get("records", []) if should_explore else []
        rollback_records = explore_output.get("rollback_records", []) if should_explore else []
        top_candidates_debug = explore_output.get("top_candidates_debug", []) if should_explore else []
        explore_click_count = int(explore_output.get("explore_click_count", 0)) if should_explore else 0
        rollback_count = int(explore_output.get("rollback_count", 0)) if should_explore else 0
        exploration_latency = float(explore_output.get("exploration_latency_ms", 0.0)) if should_explore else 0.0
        exploration_latency_list.append(exploration_latency)
        best_record = max(explore_records, key=lambda x: x.get("similarity", -1.0)) if explore_records else None
        thread_verification = explore_output.get("thread_verification") if should_explore else None

        # verification and level2 recovery BEFORE reasoning
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
            main_verify_latency_list.append(float(main_verification.get("verification_latency_ms", 0.0)))
            state_verified = bool(main_verification.get("ok", False))
            if not state_verified:
                main_verify_fail_count += 1
        elif should_explore and thread_verification is not None:
            state_verified = bool(thread_verification.get("ok", False))

        if should_explore and (not state_verified) and args.enable_level2_recovery:
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

        # Reasoning after serial exploration + verification
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
        reasoning_latency = (time.time() - reasoning_start) * 1000
        reasoning_latency_list.append(reasoning_latency)

        planning_end = time.time()
        planning_latency = (planning_end - perception_end) * 1000
        planning_latency_list.append(planning_latency)

        skipped_due_to_unverified_state = bool(
            should_explore and args.require_verified_state and not state_verified
        )

        # Execution
        operation_start = time.time()
        executed_action = None
        action_screen_diff = 0
        action_effective = False

        if not skipped_due_to_unverified_state:
            before_hash = compute_phash(args.screenshot_path)
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
            after_hash = compute_phash(args.screenshot_path)
            action_screen_diff = int(after_hash - before_hash)
            action_effective = action_screen_diff > int(args.action_effect_threshold)
            if action_effective:
                no_effect_streak = 0
            else:
                no_effect_streak += 1
        else:
            no_effect_streak += 1

        operation_latency = (time.time() - operation_start) * 1000
        operation_latency_list.append(operation_latency)

        if skipped_due_to_unverified_state:
            history.append("skip_execution_unverified_state")
        elif executed_action is not None:
            history.append(str(executed_action))
            executed_action_trace.append(copy.deepcopy(action_obj))
        else:
            history.append(safe_json(action_obj))

        # this-step exploration knowledge -> next-step reasoning
        if best_record:
            pending_clue = (
                f"best_element={best_record.get('element_text')}, "
                f"similarity={float(best_record.get('similarity', 0.0)):.3f}, "
                f"explore_rounds={explore_rounds}"
            )
        elif should_explore:
            pending_clue = None

        step_latency = (time.time() - start_time) * 1000
        end_to_end_latency_list.append(step_latency)

        steps.append(
            {
                "step": itr,
                "llm_output": llm_output,
                "action": action_obj,
                "executed_action": executed_action,
                "explore_enabled": should_explore,
                "explore_reason": explore_reason,
                "explore_rounds": explore_rounds,
                "explore_click_count": explore_click_count,
                "rollback_count": rollback_count,
                "explore_records": explore_records,
                "rollback_records": rollback_records,
                "top_candidates_debug": top_candidates_debug,
                "best_explore_record": best_record,
                "thread_verification": thread_verification,
                "main_verification": main_verification,
                "state_verified_before_execution": state_verified,
                "state_recovery": state_recovery,
                "skipped_due_to_unverified_state": skipped_due_to_unverified_state,
                "action_effective": action_effective,
                "action_screen_diff": action_screen_diff,
                "no_effect_streak": no_effect_streak,
                "perception_latency_ms": perception_latency,
                "exploration_latency_ms": exploration_latency,
                "reasoning_latency_ms": reasoning_latency,
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
                f"[Explorer] best_similarity={float(best_record.get('similarity', 0.0)):.3f}, "
                f"best_element=\"{best_record.get('element_text', '')}\""
            )
        if thread_verification is not None:
            print(
                f"[Thread Verify] ok={thread_verification.get('ok')} "
                f"diff={thread_verification.get('diff')} threshold={thread_verification.get('threshold')}"
            )
        if main_verification is not None:
            print(
                f"[Main Verify] ok={main_verification.get('ok')} "
                f"diff={main_verification.get('diff')} threshold={main_verification.get('threshold')}"
            )
        print(f"[State Verify] restored={state_verified}")
        if state_recovery is not None:
            rv = state_recovery.get("verify", {}) or {}
            print(
                f"[Level2 Recovery] ok={state_recovery.get('ok')} replay_len={state_recovery.get('replay_len')} "
                f"diff={rv.get('diff')} threshold={rv.get('threshold')}"
            )
        if skipped_due_to_unverified_state:
            print("[Execution] skipped due to unverified state after rollback/recovery.")

        print("[Reasoning] Parsed action:", action_obj)
        print("[Execution] Action done:", executed_action)
        print(
            f"Perception latency: {perception_latency:.3f} ms, "
            f"Exploration latency: {exploration_latency:.3f} ms, "
            f"Reasoning latency: {reasoning_latency:.3f} ms, "
            f"Planning latency: {planning_latency:.3f} ms, "
            f"Operation latency: {operation_latency:.3f} ms"
        )
        print(f"Step latency: {step_latency:.3f} ms")

        action_type = str(action_obj.get("action_type", "")).lower()
        if action_type in {"terminate", "finished", "done", "finish", "stop", "exit"}:
            print("[Stop] finish-like action detected, exiting loop.")
            break

    print("\n=== Finished all iterations (serial explore + adb) ===")
    print(
        f"Perception latency: {avg(perception_latency_list):.3f} ms, "
        f"Exploration latency: {avg(exploration_latency_list):.3f} ms, "
        f"Reasoning latency: {avg(reasoning_latency_list):.3f} ms, "
        f"MainVerify latency: {avg(main_verify_latency_list):.3f} ms, "
        f"Recovery latency: {avg(recovery_latency_list):.3f} ms, "
        f"Planning latency: {avg(planning_latency_list):.3f} ms, "
        f"Operation latency: {avg(operation_latency_list):.3f} ms, "
        f"End-to-end latency: {avg(end_to_end_latency_list):.3f} ms"
    )
    print(f"Main verification failures: {main_verify_fail_count}, Level2 recoveries: {recovery_count}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(steps, f, ensure_ascii=False, indent=2)
        print(f"[Saved] step traces -> {args.output_json}")

    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Open the stopwatch.")
    parser.add_argument("--max_itr", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--adb_path", type=str, default="adb")
    parser.add_argument("--screenshot_path", type=str, default="./screenshot/screenshot.png")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--llama_api_url", type=str, default="http://localhost:8081/v1/chat/completions")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)

    # reasoning is short now because exploration is already done serially
    parser.add_argument("--reasoning_sleep_sec", type=float, default=0.2)

    parser.add_argument("--explore_rounds", type=int, default=2)
    parser.add_argument("--explore_step_sleep_sec", type=float, default=0.12)
    parser.add_argument("--rollback_step_sleep_sec", type=float, default=0.10)
    parser.add_argument("--explore_score_threshold", type=float, default=0.25)
    parser.add_argument("--explore_shortlist_k", type=int, default=5)
    parser.add_argument(
        "--explore_allow_fallback_click",
        action="store_true",
        help="Allow fallback center click when no candidate passes score threshold.",
    )

    parser.add_argument("--explore_xml_path", type=str, default="./screenshot/a11y.xml")
    parser.add_argument("--explore_probe_screenshot_path", type=str, default="./screenshot/explore_probe.png")

    parser.add_argument("--explore_probability", type=float, default=0.35)
    parser.add_argument("--max_explore_gap", type=int, default=3)
    parser.add_argument("--stall_trigger", type=int, default=1)

    parser.add_argument("--phash_threshold", type=int, default=8)
    parser.add_argument("--action_effect_threshold", type=int, default=3)

    parser.add_argument("--recover_home_wait_sec", type=float, default=0.35)
    parser.add_argument("--recover_replay_step_wait_sec", type=float, default=0.15)

    parser.add_argument(
        "--disable_main_verification",
        dest="enable_main_verification",
        action="store_false",
    )
    parser.add_argument(
        "--disable_thread_verification",
        dest="enable_thread_verification",
        action="store_false",
    )
    parser.add_argument(
        "--disable_level2_recovery",
        dest="enable_level2_recovery",
        action="store_false",
    )
    parser.add_argument(
        "--allow_unverified_execution",
        dest="require_verified_state",
        action="store_false",
    )
    parser.set_defaults(
        enable_main_verification=True,
        enable_thread_verification=True,
        enable_level2_recovery=True,
        require_verified_state=True,
    )

    parser.add_argument("--post_action_wait_sec", type=float, default=0.35)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--on_device", action="store_true")
    args = parser.parse_args()

    run_explore_thread_adb(args)
