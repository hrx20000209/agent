import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import json
import argparse

from MobileAgentE.controller import get_screenshot, get_a11y_tree
from MobileAgentE.api import inference_chat_llama_cpp_xml_only
from MobileAgentE.tree import parse_a11y_tree, print_tree
from MobileAgentE.agents import OneStepAgent_XML, InfoPool


########################################
#        LLM CALL FUNCTION
########################################
def get_reasoning_response(chat):
    """
    chat = {
        instruction: str
        xml: str(xml content)
        history: list(str)
    }
    """
    return inference_chat_llama_cpp_xml_only(
        chat=chat,
        temperature=0.0
    )


########################################
#         SINGLE-STEP MAIN LOOP
########################################
def run_single_step_agent(args):
    """
    完整版本：
    - ❗不使用 screenshot
    - ✔ 只使用 XML tree
    - ✔ 保留你全部 latency 统计
    """

    input_dir = "/sdcard" if args.on_device else "./screenshot"
    xml_path = os.path.join(input_dir, "a11y.xml")
    os.makedirs(input_dir, exist_ok=True)

    print("### Running Single-Step Agent (XML-only mode) ###")

    agent = OneStepAgent_XML(args.adb_path)

    steps = []
    history = []

    perception_latency_list = []
    screenshot_latency_list = []       # 虽然不使用 screenshot，但保留字段
    a11y_tree_latency_list = []
    planning_latency_list = []
    operation_latency_list = []
    end_to_end_latency_list = []

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        # --- Screenshot placeholder ---
        screenshot_time = time.time()
        screenshot_latency = (screenshot_time - start_time) * 1000
        screenshot_latency_list.append(screenshot_latency)

        print("[Perception] Screenshot skipped (XML-only mode)")

        # --- A11Y tree ---
        get_a11y_tree(args, xml_path)
        a11y_tree_time = time.time()
        a11y_tree_latency = (a11y_tree_time - screenshot_time) * 1000
        a11y_tree_latency_list.append(a11y_tree_latency)

        xml_str = open(xml_path, "r", encoding="utf-8").read()
        tree = parse_a11y_tree(xml_path=xml_path)
        # print_tree(tree)
        # print(xml_str)

        # fake width/height
        info_pool = InfoPool(
            instruction=args.task,
            width=1080,
            height=2400,
            tree=tree
        )

        perception_end_time = time.time()
        perception_latency = (perception_end_time - start_time) * 1000
        perception_latency_list.append(perception_latency)


        # --- Reasoning ---
        llm_start = time.time()
        action_obj = agent.run_step(
            args.task,
            width=0, height=0,
            history=history,
            llm_api_func=get_reasoning_response,
            xml_str=xml_str
        )
        llm_end = time.time()

        planning_latency = (llm_end - llm_start) * 1000
        planning_latency_list.append(planning_latency)

        print("[Reasoning] Parsed action:", action_obj)

        # --- Finish condition ---
        if isinstance(action_obj, dict):
            action_type = action_obj.get("action_type", "")
            if isinstance(action_type, str) and action_type.lower() in ["finish", "done", "exit", "stop"]:
                print("✓ Model designated task completion")
                break

        # --- Execute ---
        op_start = time.time()
        executed_action = agent.execute_action(action_obj, info_pool)
        op_end = time.time()

        operation_latency = (op_end - op_start) * 1000
        operation_latency_list.append(operation_latency)

        print("[Execution] Done:", executed_action)
        history.append(str(executed_action))

        steps.append({
            "step": itr,
            "operation": "execution",
            "executed_action": executed_action
        })

        end_time = time.time()
        step_latency = (end_time - start_time) * 1000
        end_to_end_latency_list.append(step_latency)

        print(
            f"Perception latency: {perception_latency:.3f} ms, "
            f"A11Y Tree latency: {a11y_tree_latency:.3f} ms, "
            f"Planning latency: {planning_latency:.3f} ms, "
            f"Operation latency: {operation_latency:.3f} ms"
        )
        print(f"Step latency: {step_latency:.3f} ms")

    # Print averages
    print("\n=== Finished all iterations (XML-only) ===")
    print(f"Perception latency: {sum(perception_latency_list)/len(perception_latency_list):.3f} ms")
    print(f"A11Y Tree latency: {sum(a11y_tree_latency_list)/len(a11y_tree_latency_list):.3f} ms")
    print(f"Planning Latency: {sum(planning_latency_list)/len(planning_latency_list):.3f} ms")
    print(f"Operation Latency: {sum(operation_latency_list)/len(operation_latency_list):.3f} ms")
    print(f"End-to-end latency: {sum(end_to_end_latency_list)/len(end_to_end_latency_list):.3f} ms")

    return steps


########################################
#                  MAIN
########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str,
        default="Open Chrome and search for newest paper about GUI agent.",
        help="User instruction")

    parser.add_argument("--max_itr", type=int, default=10)
    parser.add_argument("--adb_path", type=str, default="adb")
    parser.add_argument("--on_device", action="store_true")

    args = parser.parse_args()

    run_single_step_agent(args)
