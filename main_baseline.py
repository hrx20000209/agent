import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import json
import argparse
from typing import List, Dict, Tuple

from PIL import Image

from MobileAgentE.controller import get_screenshot
from MobileAgentE.api import inference_chat_llama_cpp
from agents.mai_ui_agent import MAIOneStepAgent, add_chat


MANAGER_SYSTEM_PROMPT = """You are the Manager in a mobile multi-agent system.
Your role:
1) Track progress of the user task.
2) Maintain/update a concise high-level plan.
3) Output stable next subgoals for an Executor agent.

Output strictly with sections:
### Thought ###
...
### Historical Operations ###
...
### Plan ###
...
""".strip()


EXECUTOR_SYSTEM_PROMPT = """You are the Executor in a mobile multi-agent system.
You get user request, current plan, and latest action history. Decide ONE next GUI action.

Output format:
<thinking>
one-sentence rationale
</thinking>
<tool_call>
{"name":"mobile_use","arguments":{"action":"...", "...":"..."}}
</tool_call>

Action space:
{"action":"click","coordinate":[x,y]}
{"action":"long_press","coordinate":[x,y]}
{"action":"type","text":"..."}
{"action":"swipe","direction":"up/down/left/right","coordinate":[x,y]}
{"action":"drag","start_coordinate":[x1,y1],"end_coordinate":[x2,y2]}
{"action":"open","text":"app_name"}
{"action":"system_button","button":"back/home/enter/menu"}
{"action":"wait"}
{"action":"terminate","status":"success/fail"}
{"action":"answer","text":"..."}
""".strip()


def _llm_api(chat, args) -> str:
    return inference_chat_llama_cpp(
        chat,
        api_url=args.llama_api_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def _history_to_text(history: List[str], max_items: int = 8) -> str:
    if not history:
        return "None"
    return "\n".join(history[-max_items:])


def _safe_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class MultiAgentBaseline:
    def __init__(self, coord_space: str = "auto"):
        self._parser = MAIOneStepAgent(adb_path="adb", coord_space=coord_space)
        self.plan = ""
        self.completed = "No completed subgoal."

    @staticmethod
    def _init_chat(system_prompt: str):
        return [["system", [{"type": "text", "text": system_prompt}]]]

    def _build_manager_prompt(self, task: str, history: List[str]) -> str:
        if not self.plan:
            return (
                f"### User Request ###\n{task}\n\n"
                f"### Latest Action History ###\n{_history_to_text(history)}\n\n"
                "Create an initial concise plan with ordered subgoals.\n"
                "Keep it practical and app-oriented."
            )

        return (
            f"### User Request ###\n{task}\n\n"
            f"### Historical Operations ###\n{self.completed}\n\n"
            f"### Existing Plan ###\n{self.plan}\n\n"
            f"### Latest Action History ###\n{_history_to_text(history)}\n\n"
            "Update the plan based on current screenshot and recent history.\n"
            "If already complete, set Plan to: Finished."
        )

    def _build_executor_prompt(self, task: str, history: List[str]) -> str:
        active_plan = self.plan if self.plan else "No plan yet."
        return (
            f"### User Request ###\n{task}\n\n"
            f"### Overall Plan ###\n{active_plan}\n\n"
            f"### Latest Action History ###\n{_history_to_text(history)}\n\n"
            "Pick exactly one next atomic action that best advances the first unfinished subgoal."
        )

    def _update_plan_from_output(self, manager_output: str):
        if not manager_output:
            return

        if "### Historical Operations ###" in manager_output:
            tmp = manager_output.split("### Historical Operations ###", 1)[1]
            if "### Plan ###" in tmp:
                self.completed = tmp.split("### Plan ###", 1)[0].strip() or self.completed

        if "### Plan ###" in manager_output:
            plan_text = manager_output.split("### Plan ###", 1)[1].strip()
            if plan_text:
                self.plan = plan_text
        elif not self.plan:
            self.plan = manager_output.strip()

    def run_step(
        self,
        task: str,
        screenshot_path: str,
        history: List[str],
        llm_api_func,
    ) -> Tuple[Dict, float, float, str, str]:
        manager_chat = self._init_chat(MANAGER_SYSTEM_PROMPT)
        manager_chat = add_chat(
            "user",
            self._build_manager_prompt(task, history),
            manager_chat,
            image=screenshot_path,
        )
        manager_start = time.time()
        manager_output = llm_api_func(manager_chat)
        manager_latency_ms = (time.time() - manager_start) * 1000

        self._update_plan_from_output(manager_output or "")

        executor_chat = self._init_chat(EXECUTOR_SYSTEM_PROMPT)
        executor_chat = add_chat(
            "user",
            self._build_executor_prompt(task, history),
            executor_chat,
            image=screenshot_path,
        )
        executor_start = time.time()
        executor_output = llm_api_func(executor_chat)
        executor_latency_ms = (time.time() - executor_start) * 1000

        action_obj = self._parser.parse_action(executor_output or "")
        return action_obj, manager_latency_ms, executor_latency_ms, manager_output, executor_output


def run_multi_agent_baseline(args):
    input_dir = "/sdcard" if args.on_device else "./screenshot"
    screenshot_path = os.path.join(input_dir, "screenshot.png")
    if not args.on_device:
        os.makedirs(input_dir, exist_ok=True)

    print("### Running multi-agent baseline ###")
    print(f"[Config] task={args.task}")
    print(f"[Config] max_itr={args.max_itr}, scale={args.scale}")

    agent = MultiAgentBaseline(coord_space=args.coord_space)
    history: List[str] = []
    steps: List[Dict] = []

    screenshot_latency_list: List[float] = []
    manager_latency_list: List[float] = []
    executor_latency_list: List[float] = []
    planning_latency_list: List[float] = []
    end_to_end_latency_list: List[float] = []

    scale = float(getattr(args, "scale", 1.0) or 1.0)

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        get_screenshot(args, screenshot_path, scale=scale)
        width, height = Image.open(screenshot_path).size
        screenshot_time = time.time()
        screenshot_latency = (screenshot_time - start_time) * 1000
        screenshot_latency_list.append(screenshot_latency)

        action_obj, manager_ms, executor_ms, manager_out, executor_out = agent.run_step(
            task=args.task,
            screenshot_path=screenshot_path,
            history=history,
            llm_api_func=lambda chat: _llm_api(chat, args),
        )
        planning_end_time = time.time()
        planning_latency = (planning_end_time - screenshot_time) * 1000
        planning_latency_list.append(planning_latency)

        manager_latency_list.append(manager_ms)
        executor_latency_list.append(executor_ms)

        print("[Perception] Captured screenshot:", screenshot_path, f"size=({width},{height})")
        print("[Manager] output:", (manager_out or "")[:200].replace("\n", " "))
        print("[Executor] parsed action:", action_obj)

        history.append(_safe_json(action_obj) if isinstance(action_obj, dict) else str(executor_out))

        step_latency = (planning_end_time - start_time) * 1000
        end_to_end_latency_list.append(step_latency)

        steps.append(
            {
                "step": itr,
                "action": action_obj,
                "plan": agent.plan,
                "screenshot_latency_ms": screenshot_latency,
                "manager_latency_ms": manager_ms,
                "executor_latency_ms": executor_ms,
                "planning_latency_ms": planning_latency,
                "step_latency_ms": step_latency,
            }
        )

        print(
            f"Screenshot latency: {screenshot_latency:.3f} ms, "
            f"Manager latency: {manager_ms:.3f} ms, Executor latency: {executor_ms:.3f} ms, "
            f"Planning latency: {planning_latency:.3f} ms"
        )
        print(f"Step latency: {step_latency:.3f} ms")

    print("\n=== Finished all iterations (multi-agent baseline) ===")
    print(
        f"Screenshot latency: {_avg(screenshot_latency_list):.3f} ms, "
        f"Manager latency: {_avg(manager_latency_list):.3f} ms, "
        f"Executor latency: {_avg(executor_latency_list):.3f} ms, "
        f"Planning latency: {_avg(planning_latency_list):.3f} ms, "
        f"End-to-end latency: {_avg(end_to_end_latency_list):.3f} ms"
    )
    return steps


def run_input_pruning_baseline(args):
    input_dir = "/sdcard" if args.on_device else "./screenshot"
    screenshot_path = os.path.join(input_dir, "screenshot.png")
    if not args.on_device:
        os.makedirs(input_dir, exist_ok=True)

    print("### Running input-pruning baseline (m3a screenshot agent) ###")
    print(f"[Config] task={args.task}")
    print(f"[Config] max_itr={args.max_itr}, pruning_scale={args.pruning_scale}")

    agent = MAIOneStepAgent(args.adb_path, coord_space=args.coord_space)
    history: List[str] = []
    steps: List[Dict] = []

    screenshot_latency_list: List[float] = []
    planning_latency_list: List[float] = []
    end_to_end_latency_list: List[float] = []

    scale = float(getattr(args, "pruning_scale", 2.0) or 2.0)

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        get_screenshot(args, screenshot_path, scale=scale)
        width, height = Image.open(screenshot_path).size
        screenshot_time = time.time()
        screenshot_latency = (screenshot_time - start_time) * 1000
        screenshot_latency_list.append(screenshot_latency)

        action_obj = agent.run_step(
            args.task,
            screenshot_path,
            width,
            height,
            history=history,
            llm_api_func=lambda chat: _llm_api(chat, args),
            clues=None,
            scale=scale,
        )
        planning_end_time = time.time()
        planning_latency = (planning_end_time - screenshot_time) * 1000
        planning_latency_list.append(planning_latency)

        print("[Perception] Captured screenshot:", screenshot_path, f"size=({width},{height})")
        print("[Reasoning] Parsed action:", action_obj)

        history.append(_safe_json(action_obj) if isinstance(action_obj, dict) else str(action_obj))

        step_latency = (planning_end_time - start_time) * 1000
        end_to_end_latency_list.append(step_latency)

        steps.append(
            {
                "step": itr,
                "action": action_obj,
                "screenshot_latency_ms": screenshot_latency,
                "planning_latency_ms": planning_latency,
                "step_latency_ms": step_latency,
                "effective_scale": scale,
            }
        )

        print(
            f"Screenshot latency: {screenshot_latency:.3f} ms, "
            f"Planning latency: {planning_latency:.3f} ms"
        )
        print(f"Step latency: {step_latency:.3f} ms")

    print("\n=== Finished all iterations (input-pruning baseline) ===")
    print(
        f"Screenshot latency: {_avg(screenshot_latency_list):.3f} ms, "
        f"Planning latency: {_avg(planning_latency_list):.3f} ms, "
        f"End-to-end latency: {_avg(end_to_end_latency_list):.3f} ms"
    )
    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=str,
        default="multi_agent",
        choices=["multi_agent", "input_pruning"],
        help="Which baseline to run.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Search papers on Mobile GUI Agent on Google Scholar.",
        help="User instruction for baseline run.",
    )
    parser.add_argument("--max_itr", type=int, default=10, help="Maximum benchmark iterations.")
    parser.add_argument("--adb_path", type=str, default="adb", help="ADB path.")
    parser.add_argument("--on_device", action="store_true", help="Run on-device or on server.")
    parser.add_argument(
        "--coord_space",
        type=str,
        default="auto",
        choices=["auto", "pixel", "norm1", "norm1000"],
        help="How to interpret model coordinates for parser canonicalization.",
    )
    parser.add_argument(
        "--llama_api_url",
        type=str,
        default="http://localhost:8081/v1/chat/completions",
        help="llama.cpp OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max new tokens per call.")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Screenshot scale for multi_agent baseline (>1 means smaller image).",
    )
    parser.add_argument(
        "--pruning_scale",
        type=float,
        default=2.0,
        help="Screenshot scale for input_pruning baseline (>1 means smaller image).",
    )
    args = parser.parse_args()

    if args.baseline == "multi_agent":
        run_multi_agent_baseline(args)
    else:
        run_input_pruning_baseline(args)
