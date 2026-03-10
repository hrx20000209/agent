import argparse
import json
import time
from typing import Dict, List, Tuple

from sim_no_adb_utils import avg, parse_action_from_llm_text, safe_json, simulate_llm_output


def _history_to_text(history: List[str], max_items: int = 6) -> str:
    if not history:
        return "None"
    return "\n".join(history[-max_items:])


class MultiAgentBaselineSim:
    def __init__(self):
        self.plan = ""
        self.completed = "No completed subgoal."

    def _build_manager_output(self, task: str, history: List[str]) -> str:
        if not self.plan:
            plan = (
                "1. Locate relevant app entry.\n"
                "2. Perform one atomic action toward task.\n"
                "3. Verify completion and finish."
            )
        elif len(history) >= 3:
            plan = "Finished."
        else:
            plan = self.plan

        return (
            "### Thought ###\n"
            f"Keep progress stable for task: {task}\n"
            "### Historical Operations ###\n"
            f"{_history_to_text(history)}\n"
            "### Plan ###\n"
            f"{plan}\n"
        )

    def _update_plan_from_output(self, manager_output: str):
        if "### Plan ###" in manager_output:
            plan_text = manager_output.split("### Plan ###", 1)[1].strip()
            if plan_text:
                self.plan = plan_text

        if "### Historical Operations ###" in manager_output:
            tmp = manager_output.split("### Historical Operations ###", 1)[1]
            if "### Plan ###" in tmp:
                self.completed = tmp.split("### Plan ###", 1)[0].strip() or self.completed

    def run_step(
        self,
        task: str,
        history: List[str],
        manager_sleep_sec: float,
        executor_sleep_sec: float,
    ) -> Tuple[Dict, float, float, str, str]:
        manager_start = time.time()
        time.sleep(max(0.0, manager_sleep_sec))
        manager_output = self._build_manager_output(task, history)
        manager_latency_ms = (time.time() - manager_start) * 1000
        self._update_plan_from_output(manager_output)

        executor_start = time.time()
        time.sleep(max(0.0, executor_sleep_sec))
        executor_output = simulate_llm_output(task, history_len=len(history), role="executor")
        action_obj = parse_action_from_llm_text(executor_output)
        executor_latency_ms = (time.time() - executor_start) * 1000

        return action_obj, manager_latency_ms, executor_latency_ms, manager_output, executor_output


def run_multi_agent_baseline_sim(args):
    print("### Running multi-agent baseline simulation (No ADB) ###")
    print(f"[Config] task={args.task}")
    print(f"[Config] max_itr={args.max_itr}")

    agent = MultiAgentBaselineSim()
    history: List[str] = []
    steps: List[Dict] = []

    perception_latency_list: List[float] = []
    manager_latency_list: List[float] = []
    executor_latency_list: List[float] = []
    planning_latency_list: List[float] = []
    end_to_end_latency_list: List[float] = []

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        time.sleep(max(0.0, args.perception_sleep_ms / 1000.0))
        perception_time = time.time()
        perception_latency = (perception_time - start_time) * 1000
        perception_latency_list.append(perception_latency)

        action_obj, manager_ms, executor_ms, manager_out, executor_out = agent.run_step(
            task=args.task,
            history=history,
            manager_sleep_sec=args.manager_sleep_sec,
            executor_sleep_sec=args.executor_sleep_sec,
        )

        planning_end_time = time.time()
        planning_latency = (planning_end_time - perception_time) * 1000
        planning_latency_list.append(planning_latency)
        manager_latency_list.append(manager_ms)
        executor_latency_list.append(executor_ms)

        history.append(safe_json(action_obj))
        step_latency = (planning_end_time - start_time) * 1000
        end_to_end_latency_list.append(step_latency)

        steps.append(
            {
                "step": itr,
                "action": action_obj,
                "manager_output": manager_out,
                "executor_output": executor_out,
                "plan": agent.plan,
                "perception_latency_ms": perception_latency,
                "manager_latency_ms": manager_ms,
                "executor_latency_ms": executor_ms,
                "planning_latency_ms": planning_latency,
                "step_latency_ms": step_latency,
            }
        )

        print("[Manager] output:", (manager_out or "").splitlines()[0:4])
        print("[Executor] parsed action:", action_obj)
        print(
            f"Perception latency: {perception_latency:.3f} ms, "
            f"Manager latency: {manager_ms:.3f} ms, "
            f"Executor latency: {executor_ms:.3f} ms, "
            f"Planning latency: {planning_latency:.3f} ms"
        )
        print(f"Step latency: {step_latency:.3f} ms")

    print("\n=== Finished all iterations (multi-agent baseline simulation) ===")
    print(
        f"Perception latency: {avg(perception_latency_list):.3f} ms, "
        f"Manager latency: {avg(manager_latency_list):.3f} ms, "
        f"Executor latency: {avg(executor_latency_list):.3f} ms, "
        f"Planning latency: {avg(planning_latency_list):.3f} ms, "
        f"End-to-end latency: {avg(end_to_end_latency_list):.3f} ms"
    )
    return steps


def run_input_pruning_baseline_sim(args):
    print("### Running input-pruning baseline simulation (No ADB) ###")
    print(f"[Config] task={args.task}")
    print(f"[Config] max_itr={args.max_itr}")

    history: List[str] = []
    steps: List[Dict] = []

    perception_latency_list: List[float] = []
    planning_latency_list: List[float] = []
    end_to_end_latency_list: List[float] = []

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        time.sleep(max(0.0, args.perception_sleep_ms / 1000.0))
        perception_time = time.time()
        perception_latency = (perception_time - start_time) * 1000
        perception_latency_list.append(perception_latency)

        planning_start = time.time()
        time.sleep(max(0.0, args.pruning_sleep_sec))
        llm_output = simulate_llm_output(args.task, history_len=len(history), role="single")
        action_obj = parse_action_from_llm_text(llm_output)
        planning_end = time.time()

        planning_latency = (planning_end - planning_start) * 1000
        planning_latency_list.append(planning_latency)
        history.append(safe_json(action_obj))

        step_latency = (planning_end - start_time) * 1000
        end_to_end_latency_list.append(step_latency)

        steps.append(
            {
                "step": itr,
                "action": action_obj,
                "llm_output": llm_output,
                "perception_latency_ms": perception_latency,
                "planning_latency_ms": planning_latency,
                "step_latency_ms": step_latency,
            }
        )

        print("[Reasoning] Parsed action:", action_obj)
        print(
            f"Perception latency: {perception_latency:.3f} ms, "
            f"Planning latency: {planning_latency:.3f} ms"
        )
        print(f"Step latency: {step_latency:.3f} ms")

    print("\n=== Finished all iterations (input-pruning baseline simulation) ===")
    print(
        f"Perception latency: {avg(perception_latency_list):.3f} ms, "
        f"Planning latency: {avg(planning_latency_list):.3f} ms, "
        f"End-to-end latency: {avg(end_to_end_latency_list):.3f} ms"
    )
    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=str,
        default="multi_agent",
        choices=["multi_agent", "input_pruning"],
        help="Which baseline simulation to run.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Search papers on Mobile GUI Agent on Google Scholar.",
        help="User instruction for simulation.",
    )
    parser.add_argument("--max_itr", type=int, default=10, help="Maximum benchmark iterations.")
    parser.add_argument(
        "--perception_sleep_ms",
        type=float,
        default=15.0,
        help="Perception simulation sleep per iteration.",
    )
    parser.add_argument(
        "--manager_sleep_sec",
        type=float,
        default=1.2,
        help="Manager-stage simulated latency.",
    )
    parser.add_argument(
        "--executor_sleep_sec",
        type=float,
        default=1.4,
        help="Executor-stage simulated latency.",
    )
    parser.add_argument(
        "--pruning_sleep_sec",
        type=float,
        default=1.8,
        help="Input-pruning single-stage simulated latency.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional output JSON file for step-level records.",
    )
    args = parser.parse_args()

    if args.baseline == "multi_agent":
        run_steps = run_multi_agent_baseline_sim(args)
    else:
        run_steps = run_input_pruning_baseline_sim(args)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(run_steps, f, ensure_ascii=False, indent=2)
        print(f"[Saved] step traces -> {args.output_json}")

