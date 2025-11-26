import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import json
from PIL import Image

from MobileAgentE.controller import get_screenshot, get_a11y_tree
from MobileAgentE.api import (
    inference_chat,
    inference_chat_ollama,
    inference_chat_llama_cpp,
)
from MobileAgentE.tree import parse_a11y_tree, print_tree
from MobileAgentE.agents import OneStepAgent, InfoPool  # ✅ 换成新的 Agent 和 InfoPool

########################################
#              CONFIG
########################################
ADB_PATH = os.environ.get("ADB_PATH", "adb")
REASONING_MODEL = "qwen-vl-plus"
SLEEP_BETWEEN_STEPS = 3
SCREENSHOT_DIR = "screenshot"
xml_path = os.path.join(SCREENSHOT_DIR, "a11y.xml")
LOG_DIR = "./logs/single_step_agent"

### LLM ###
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = "sk-1aa21ba323d044a092e3579753ec1548"
USAGE_TRACKING_JSONL = None

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


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
def run_single_step_agent(
        instruction: str,
        max_itr: int = 10,
        run_name: str = "single_step",
):
    """
    单步 Agent 框架：
    每一轮都只做一次 LLM 调用 -> 输出动作 -> 执行 -> 再截图。
    """
    print("### Running Single-Step Agent ###")

    # Init dirs
    run_dir = f"{LOG_DIR}/{run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    log_json_path = os.path.join(run_dir, "steps.json")

    # Initialize unified agent
    agent = OneStepAgent(adb_path=ADB_PATH)

    steps = []
    history = []

    perception_latency_list = []
    screenshot_latency_list = []
    a11y_tree_latency_list = []
    planning_latency_list = []
    operation_latency_list = []
    end_to_end_latency_list = []

    for itr in range(1, max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        # --- Perception ---
        screenshot_path = os.path.join(SCREENSHOT_DIR, "screenshot.jpg")
        get_screenshot(ADB_PATH)
        screenshot_time = time.time()
        screenshot_latency = (screenshot_time - start_time) * 1000
        screenshot_latency_list.append(screenshot_latency)

        get_a11y_tree(ADB_PATH)
        a11y_tree_time = time.time()
        a11y_tree_latency = (a11y_tree_time - screenshot_time) * 1000
        a11y_tree_latency_list.append(a11y_tree_latency)
        w, h = Image.open(screenshot_path).size

        tree = parse_a11y_tree(xml_path=xml_path)
        # print_tree(tree)

        info_pool = InfoPool(
            instruction=instruction,
            width=w,
            height=h,
            tree=tree
        )

        perception_end_time = time.time()
        perception_latency = (perception_end_time - start_time) * 1000
        perception_latency_list.append(perception_latency)
        print("[Perception] Captured screenshot:", screenshot_path, f"size=({w},{h})")

        # --- Single-step reasoning ---
        action_obj = agent.run_step(
            instruction,
            screenshot_path,
            w, h,
            history=history,
            llm_api_func=get_reasoning_response
        )

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

        # time.sleep(SLEEP_BETWEEN_STEPS)

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

    with open(log_json_path, "w") as f:
        json.dump(steps, f, indent=4)
    return steps


if __name__ == "__main__":
    run_single_step_agent("Open Chrome and search for newest paper about GUI agent.")
