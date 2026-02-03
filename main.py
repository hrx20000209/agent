import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import json
import argparse
from PIL import Image

from MobileAgentE.controller import get_screenshot, get_a11y_tree
from MobileAgentE.api import (
    inference_chat,
    inference_chat_ollama,
    inference_chat_llama_cpp,
)
from MobileAgentE.tree import parse_a11y_tree, print_tree
from MobileAgentE.agents import OneStepAgent, InfoPool  # ✅ 换成新的 Agent 和 InfoPool
from agents.mai_ui_agent import MAIOneStepAgent
# from Explorer.online_explorer import A11yTreeOnlineExplorer
from Explorer.GoalExplorer import A11yTreeOnlineExplorer

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
    agent = MAIOneStepAgent(args.adb_path)
    # explorer = A11yTreeOnlineExplorer(
    #     adb_path=args.adb_path,
    #     args=args,
    #     xml_path=xml_path,
    # )

    explorer = A11yTreeOnlineExplorer(
        adb_path=args.adb_path,
        args=args,
        xml_path="./screenshot/a11y.xml",
        task_text=args.task,
        explore_vis_dir="explore_results",
        screenshot_path="explore_screenshot.png",
    )

    explore_logs = []
    clues = None

    steps = []
    history = []

    perception_latency_list = []
    screenshot_latency_list = []
    a11y_tree_latency_list = []
    planning_latency_list = []
    operation_latency_list = []
    end_to_end_latency_list = []

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        scale = 1.0

        # --- Perception ---
        get_screenshot(args, screenshot_path, scale=scale)
        screenshot_time = time.time()
        screenshot_latency = (screenshot_time - start_time) * 1000
        screenshot_latency_list.append(screenshot_latency)

        get_a11y_tree(args, xml_path)
        a11y_tree_time = time.time()
        a11y_tree_latency = (a11y_tree_time - screenshot_time) * 1000
        a11y_tree_latency_list.append(a11y_tree_latency)
        w, h = Image.open(screenshot_path).size

        tree = parse_a11y_tree(xml_path=xml_path)
        print_tree(tree)

        info_pool = InfoPool(
            instruction=args.task,
            width=w,
            height=h,
            tree=tree
        )

        perception_end_time = time.time()
        perception_latency = (perception_end_time - start_time) * 1000
        perception_latency_list.append(perception_latency)
        print("[Perception] Captured screenshot:", screenshot_path, f"size=({w},{h})")


        # --- Single-step reasoning ---
        explorer.start(max_steps=5)     # parallel exploration

        action_obj = agent.run_step(
            args.task,
            screenshot_path,
            w, h,
            history=history,
            llm_api_func=get_reasoning_response,
            clues=clues,
            scale=scale
        )

        time.sleep(20)

        explorer.stop()
        clues = explorer.build_prompt_clues()  # ⭐ 拿到压缩结果
        explorer.fast_rollback()

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
        executed_action = agent.execute_action(action_obj, info_pool)

        history.append(str(executed_action))

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
    parser.add_argument("--task", type=str, default="Record an audio clip using Audio Recorder app and save it.",
                        help="User instruction for the single-step agent")
    parser.add_argument("--max_itr", type=int, default=10,
                        help="Maximum iterations for the agent")
    parser.add_argument("--adb_path", type=str, default="adb", help="ADB path.")
    parser.add_argument("--on_device", action="store_true", help="Run on-device or on server.")
    args = parser.parse_args()

    run_single_step_agent(args)
