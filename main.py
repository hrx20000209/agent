import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import json
import threading
import argparse
import shutil
from PIL import Image

from MobileAgentE.controller import get_screenshot, get_a11y_tree
from MobileAgentE.api import (
    inference_chat,
    inference_chat_ollama,
    inference_chat_llama_cpp,
)
from MobileAgentE.tree import parse_a11y_tree, print_tree
from MobileAgentE.agents import OneStepAgent  # ✅ 换成新的 Agent 和 InfoPool
from agents.mai_ui_agent import MAIOneStepAgent
from agents.utils import execute_action
# from Explorer.online_explorer import A11yTreeOnlineExplorer
from Explorer.GoalExplorer import A11yTreeOnlineExplorer
from Explorer.utils import ensure_dir, mark_and_save_explore_click, phash

########################################
#              CONFIG
########################################
REASONING_MODEL = "qwen-vl-plus"
LOG_DIR = "./logs/single_step_agent"

### LLM ###
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = "sk-1aa21ba323d044a092e3579753ec1548"
USAGE_TRACKING_JSONL = None

########################################
#        LLM CALL FUNCTION
########################################
def get_reasoning_response(chat, model=REASONING_MODEL):
    """唯一的 LLM 调用"""
    temperature = 0.0
    return inference_chat_llama_cpp(chat, temperature=temperature)
    # 如果你改回 qwen2.5vl / dashscope，就把上面这一行替换成下方分支即可
    # if model == "qwen2.5vl:3b":
    #     return inference_chat_ollama(chat, model=model, temperature=0.0)
    # else:
    # return inference_chat(chat, model, API_URL, API_KEY,
    #                       usage_tracking_jsonl=USAGE_TRACKING_JSONL,
    #                       temperature=temperature)


def _coord_to_px(coord, width, height, coord_space="auto"):
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return None
    x, y = float(coord[0]), float(coord[1])
    mode = str(coord_space or "auto").strip().lower()
    if mode in {"norm1", "normalized", "0_1"}:
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            px, py = int(x * width), int(y * height)
        else:
            px, py = int(x), int(y)
    elif mode in {"norm1000", "1000", "0_1000"}:
        if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
            px, py = int((x / 1000.0) * width), int((y / 1000.0) * height)
        else:
            px, py = int(x), int(y)
    elif mode in {"pixel", "px"}:
        px, py = int(x), int(y)
    else:
        # auto: keep old behavior first to avoid breaking existing pixel outputs.
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            px, py = int(x * width), int(y * height)
        else:
            px, py = int(x), int(y)
    px = max(0, min(width - 1, px))
    py = max(0, min(height - 1, py))
    return px, py


def _action_signature(action_obj, width, height):
    if not isinstance(action_obj, dict):
        return "none"
    at = str(action_obj.get("action_type", "")).lower()
    ai = action_obj.get("action_inputs", {}) or {}
    coord_space = str(action_obj.get("coord_space") or ai.get("coord_space") or "auto").lower()
    if at in {"click", "long_press"}:
        pt = _coord_to_px(ai.get("coordinate"), width, height, coord_space=coord_space)
        return f"{at}:{pt}"
    if at in {"swipe", "drag"}:
        s = _coord_to_px(ai.get("start_coordinate") or ai.get("coordinate"), width, height, coord_space=coord_space)
        e = _coord_to_px(ai.get("end_coordinate"), width, height, coord_space=coord_space)
        d = ai.get("direction")
        return f"{at}:s={s}:e={e}:d={d}"
    if at == "open_app":
        return f"open_app:{ai.get('content') or ai.get('value') or ai.get('text')}"
    return at


def _settle_profile(action_type: str):
    action_type = (action_type or "").lower()
    if action_type == "open_app":
        return 1.0, 2.4
    if action_type in {"click", "long_press", "swipe", "drag", "press_back", "press_home"}:
        return 0.45, 1.6
    if action_type in {"type", "press_enter"}:
        return 0.30, 1.0
    return 0.20, 0.8


def wait_ui_settle(args, screenshot_path, scale, action_type):
    """
    Wait for post-action UI to settle by frame similarity, not fixed sleep only.
    """
    base_wait, settle_timeout = _settle_profile(action_type)
    time.sleep(base_wait)

    tmp_path = screenshot_path + ".settle_tmp"
    prev_h = None
    start = time.time()
    while time.time() - start < settle_timeout:
        get_screenshot(args, tmp_path, scale=scale)
        cur_h = phash(tmp_path)
        if prev_h is not None:
            diff = prev_h - cur_h
            if diff <= 3:
                break
        prev_h = cur_h
        time.sleep(0.2)

    if os.path.exists(tmp_path):
        shutil.move(tmp_path, screenshot_path)
    else:
        get_screenshot(args, screenshot_path, scale=scale)


########################################
#         SINGLE-STEP MAIN LOOP
########################################
def run_single_step_agent(args):
    """
    单步 Agent 框架：
    每一轮都只做一次 LLM 调用 -> 输出动作 -> 执行 -> 再截图。
    """

    input_dir = "/sdcard" if args.on_device else "./screenshot"
    screenshot_path = os.path.join(input_dir, "screenshot.png")
    xml_path = os.path.join(input_dir, "a11y.xml")

    os.makedirs(input_dir, exist_ok=True)

    print("### Running Single-Step Agent ###")

    # Initialize unified agent
    agent = MAIOneStepAgent(args.adb_path, coord_space=args.coord_space)

    ui_lock = threading.Lock()
    stop_event = threading.Event()
    rollback_done_event = threading.Event()

    scale = float(getattr(args, "scale", 1.0) or 1.0)
    get_screenshot(args, screenshot_path, scale=scale)
    width, height = Image.open(screenshot_path).size

    explorer = A11yTreeOnlineExplorer(
        adb_path=args.adb_path,
        args=args,
        xml_path="./screenshot/a11y.xml",
        explore_vis_dir="explore_results",
        ui_lock=ui_lock,
        stop_event=stop_event,
        rollback_done_event=rollback_done_event,
        width=width,
        height=height,
        explorer_mode=args.explorer_mode,
    )

    clues = None
    pending_explore_payload = None

    steps = []
    history = []
    reasoning_vis_dir = "reasoning_results"
    ensure_dir(reasoning_vis_dir)
    last_action_sig = None
    last_no_effect_repeat = 0
    execution_feedback = ""

    perception_latency_list = []
    screenshot_latency_list = []
    a11y_tree_latency_list = []
    planning_latency_list = []
    operation_latency_list = []
    end_to_end_latency_list = []

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        # --- Perception ---
        get_screenshot(args, screenshot_path, scale=scale)
        width, height = Image.open(screenshot_path).size
        screenshot_time = time.time()
        screenshot_latency = (screenshot_time - start_time) * 1000
        screenshot_latency_list.append(screenshot_latency)

        get_a11y_tree(args, xml_path)
        a11y_tree_time = time.time()
        a11y_tree_latency = (a11y_tree_time - screenshot_time) * 1000
        a11y_tree_latency_list.append(a11y_tree_latency)

        # tree = parse_a11y_tree(xml_path=xml_path)
        # print_tree(tree)

        perception_end_time = time.time()
        perception_latency = (perception_end_time - start_time) * 1000
        perception_latency_list.append(perception_latency)
        print("[Perception] Captured screenshot:", screenshot_path, f"size=({width},{height})")

        # k-step exploration memory is injected at step k+1 after page matching.
        if pending_explore_payload:
            source_itr = pending_explore_payload.get("source_itr")
            pending_explore_candidates = pending_explore_payload.get("candidates") or []
            clues = explorer.build_prompt_clues_from_candidates(
                candidates=pending_explore_candidates,
                current_screenshot_path=screenshot_path,
                max_items=4,
                last_reasoning_action=(history[-1] if history else ""),
            )
            if clues:
                clues = (
                    f"[Clue Source] exploration_iteration={source_itr} -> reasoning_iteration={itr}\n"
                    + clues
                )
        else:
            clues = None

        if execution_feedback:
            fb_block = f"[Execution Feedback from previous step]\n{execution_feedback}\n"
            clues = (clues + "\n" + fb_block) if clues else fb_block

        explorer.set_runtime_focus(history_tail=history[-3:], clues_text=clues)

        # --- Single-step reasoning ---
        explorer.start(
            max_steps=10,
            max_depth=2,
            leaf_width=3,
            time_budget_sec=(args.explore_time_budget_sec if args.explore_time_budget_sec > 0 else None),
        )     # parallel exploration

        action_obj = agent.run_step(
            args.task,
            screenshot_path,
            width, height,
            history=history,
            llm_api_func=get_reasoning_response,
            clues=clues,
            scale=scale
        )

        rollback_done_event.clear()

        explorer.stop()  # 先停 exploration
        # Defensive: only run final rollback after exploration thread fully exits.
        if explorer.thread is None or not explorer.thread.is_alive():
            explorer.fast_rollback(step=itr)

        rollback_done_event.wait()  # 等 rollback 结束

        pending_explore_payload = {
            "source_itr": itr,
            "candidates": explorer.consume_iteration_candidates(),
        }

        planning_end_time = time.time()
        planning_latency = (planning_end_time - perception_end_time) * 1000
        planning_latency_list.append(planning_latency)
        print("[Reasoning] Parsed action:", action_obj)

        # --- Finish condition ---
        if action_obj:
            action_type = action_obj.get("action_type", "")
            if isinstance(action_type, str) and action_type.lower() in ["finish", "done", "exit", "stop"]:
                print("✅ Task finished by model (by action_type).")
                break

        # --- Execution ---
        action_type = (action_obj or {}).get("action_type", "unknown")
        action_sig = _action_signature(action_obj, width, height)
        with ui_lock:
            print(f"[INFO] Action executing...")
            get_screenshot(args, screenshot_path, scale=scale)
            before_hash = phash(screenshot_path)
            executed_action = execute_action(
                action_obj,
                width,
                height,
                args.adb_path,
                coord_scale=scale,
            )
        wait_ui_settle(args, screenshot_path, scale, action_type)
        after_hash = phash(screenshot_path)
        screen_diff = before_hash - after_hash
        no_effect = screen_diff <= 3
        if no_effect:
            if action_sig == last_action_sig:
                last_no_effect_repeat += 1
            else:
                last_no_effect_repeat = 1
            execution_feedback = (
                f"last_action={executed_action} had tiny screen change(diff={screen_diff}); "
                f"avoid repeating same action/coordinate on unchanged page"
            )
            if last_no_effect_repeat >= 2:
                execution_feedback += f"; repeated_no_effect_count={last_no_effect_repeat}"
        else:
            last_no_effect_repeat = 0
            execution_feedback = ""
        last_action_sig = action_sig

        # Save a debug frame for reasoning action (same visual style as exploration).
        width, height = Image.open(screenshot_path).size
        action_inputs = (action_obj or {}).get("action_inputs", {}) or {}
        coord_space = str((action_obj or {}).get("coord_space") or action_inputs.get("coord_space") or "auto").lower()

        marker_xy = None
        marker_bounds = None
        coord = action_inputs.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            marker_xy = _coord_to_px(coord, width, height, coord_space=coord_space)

        if action_type in ["swipe", "drag"]:
            sc = action_inputs.get("start_coordinate") or action_inputs.get("coordinate")
            ec = action_inputs.get("end_coordinate")
            if isinstance(sc, (list, tuple)) and len(sc) == 2:
                marker_xy = _coord_to_px(sc, width, height, coord_space=coord_space)
            if isinstance(ec, (list, tuple)) and len(ec) == 2 and marker_xy is not None:
                ex, ey = _coord_to_px(ec, width, height, coord_space=coord_space)
                marker_bounds = (
                    min(marker_xy[0], ex),
                    min(marker_xy[1], ey),
                    max(marker_xy[0], ex),
                    max(marker_xy[1], ey),
                )

        mark_and_save_explore_click(
            screenshot_path=screenshot_path,
            save_dir=reasoning_vis_dir,
            step_idx=itr,
            xy=marker_xy,
            bounds=marker_bounds,
            text=f"reasoning_action={action_type} | inputs={action_inputs} | executed={executed_action}",
            extra_lines=[
                f"iteration={itr}",
                f"task={args.task}",
                f"screen_diff={screen_diff}",
                f"no_effect={no_effect}",
                f"history_tail={history[-4:] if history else []}",
            ],
            bottom_lines=(
                ["[Injected Clues for This Reasoning Step]"]
                + explorer.get_last_clue_debug_lines()
                + [ln[:140] for ln in (clues or "None").splitlines()[:14]]
            ),
        )

        if executed_action is not None:
            if no_effect:
                history.append(f"{executed_action} [NO_EFFECT diff={screen_diff}]")
            else:
                history.append(str(executed_action))
            explorer.action_history.append(action_obj)

        steps.append({
            "step": itr,
            "operation": "execution",
            "executed_action": executed_action
        })

        operation_end_time = time.time()
        operation_latency = (operation_end_time - planning_end_time) * 1000
        operation_latency_list.append(operation_latency)
        print("[Execution] Action done:", executed_action)

        end_time = time.time()
        step_latency = (end_time - start_time) * 1000
        end_to_end_latency_list.append(step_latency)
        print(f"Perception latency: {perception_latency:.3f} ms, "
              f"Screenshot latency: {screenshot_latency:.3f} ms, A11Y Tree latency: {a11y_tree_latency:.3f} ms, "
              f"Planning latency: {planning_latency:.3f} ms, "
              f"Operation latency: {operation_latency:.3f} ms",)
        print(f"Step latency: {step_latency:.3f} ms",)


    avg_perception_latency = sum(perception_latency_list) / len(perception_latency_list)
    avg_screenshot_latency = sum(screenshot_latency_list) / len(screenshot_latency_list)
    avg_a11y_tree_latency = sum(a11y_tree_latency_list) / len(a11y_tree_latency_list)
    avg_planning_latency = sum(planning_latency_list) / len(planning_latency_list)
    avg_operation_latency = sum(operation_latency_list) / len(operation_latency_list)
    avg_end_to_end_latency = sum(end_to_end_latency_list) / len(end_to_end_latency_list)

    print("\n=== Finished all iterations ===")
    print(f"Perception latency: {avg_perception_latency:.3f} ms, "
          f"Screenshot latency: {avg_screenshot_latency:.3f} ms, A11Y Tree latency: {avg_a11y_tree_latency:.3f} ms, "
          f"Planning Latency: {avg_planning_latency:.3f} ms, "
          f"Operation Latency: {avg_operation_latency:.3f} ms, "
          f"End-to-end latency: {avg_end_to_end_latency:.3f} ms")

    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    task = "Create a new folder in Markor named folder_20250917_002630"
    # task = "Record an audio clip using Audio Recorder app and save it."
    # task = "Run the stopwatch"
    parser.add_argument("--task", type=str, default=task,
                        help="User instruction for the single-step agent")
    parser.add_argument("--max_itr", type=int, default=10,
                        help="Maximum iterations for the agent")
    parser.add_argument("--adb_path", type=str, default="adb", help="ADB path.")
    parser.add_argument("--screenshot_path", type=str, default="./screenshot/screenshot.png", help="Screenshot path.")
    parser.add_argument("--on_device", action="store_true", help="Run on-device or on server.")
    parser.add_argument("--scale", type=float, default=1.0, help="Screenshot downscale factor (>1 means smaller image).")
    parser.add_argument(
        "--explorer_mode",
        type=str,
        default="collect_demo",
        choices=["collect_demo", "task"],
        help="collect_demo prioritizes coverage/traces; task prioritizes strict task relevance.",
    )
    parser.add_argument(
        "--explore_time_budget_sec",
        type=float,
        default=0.0,
        help="Optional exploration time cap per reasoning step; 0 means stop when reasoning returns.",
    )
    parser.add_argument(
        "--coord_space",
        type=str,
        default="auto",
        choices=["auto", "pixel", "norm1", "norm1000"],
        help="How to interpret model coordinates for execution and debug markers.",
    )
    args = parser.parse_args()

    run_single_step_agent(args)
