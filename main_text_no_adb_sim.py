import argparse
import json
import time
from typing import Dict, List

from sim_no_adb_utils import (
    avg,
    call_llama_cpp_text_only,
    ensure_screenshot_path,
    parse_action_from_llm_text,
)

SYSTEM_PROMPT = """You are an Android UI agent. Your goal is to complete the given task step by step.

You will be given:
1. the task instruction
2. the previous action history
3. the current page information

Decide the next single action to perform on the current page.

Do not output any explanation, rationale, or thinking.
Do not use <thinking> tags.
Output only one <tool_call> block and nothing else.

Output format:
<tool_call>
{"name":"mobile_use","arguments":{"action":"...", "...":"..."}}
</tool_call>
""".strip()


def build_androidworld_like_prompt(task: str, action_history: List[str], ui_elements: List[Dict]) -> str:
    history_text = "\n".join(
        [f"{i + 1}. {h}" for i, h in enumerate(action_history)]
    ) if action_history else "None yet."

    ui_lines = []
    for i, elem in enumerate(ui_elements, start=1):
        line = (
            f"UI Element {i}: "
            f'text="{elem.get("text", "")}", '
            f'content_desc="{elem.get("content_desc", "")}", '
            f'clickable={elem.get("clickable", False)}, '
            f'editable={elem.get("editable", False)}, '
            f'bounds={elem.get("bounds", [])}'
        )
        ui_lines.append(line)

    ui_text = "\n".join(ui_lines) if ui_lines else "No UI elements detected."

    prompt = f"""Task:
{task}

Action History:
{history_text}

Current Page Information:
{ui_text}

Please decide the next single action to perform on the current page.
Return your answer in the required structured format.
"""
    return prompt


def get_mock_ui_elements_for_androidworld() -> List[Dict]:
    return [
        {
            "text": "Google Scholar",
            "content_desc": "Page title",
            "clickable": False,
            "editable": False,
            "bounds": [24, 40, 420, 88],
        },
        {
            "text": "Search papers",
            "content_desc": "Search box",
            "clickable": True,
            "editable": True,
            "bounds": [36, 120, 980, 210],
        },
        {
            "text": "Search",
            "content_desc": "Search button",
            "clickable": True,
            "editable": False,
            "bounds": [860, 120, 1010, 210],
        },
        {
            "text": "Menu",
            "content_desc": "Navigation menu",
            "clickable": True,
            "editable": False,
            "bounds": [0, 96, 96, 192],
        },
        {
            "text": "Profile",
            "content_desc": "User profile",
            "clickable": True,
            "editable": False,
            "bounds": [930, 40, 1080, 110],
        },
    ]


def get_mock_action_history(itr: int) -> List[str]:
    base_history = [
        'open Chrome',
        'click the search box',
        'type "Mobile GUI Agent"',
    ]
    return base_history[: max(0, min(itr - 1, len(base_history)))]


def run_text_only_sim(args):
    print("### Running AndroidWorld-like llama.cpp Benchmark ###")
    print(f"[Config] task={args.task}")

    screenshot_path = ensure_screenshot_path(args.screenshot_path)
    print(f"[Config] screenshot_path={screenshot_path}")
    print(f"[Config] llama_api_url={args.llama_api_url}")
    print(f"[Config] max_itr={args.max_itr}")

    steps: List[Dict] = []
    prompt_build_latency_list: List[float] = []
    reasoning_latency_list: List[float] = []
    parse_latency_list: List[float] = []
    end_to_end_latency_list: List[float] = []

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        # Build AndroidWorld-like prompt
        prompt_start = time.time()
        mock_history = get_mock_action_history(itr)
        mock_ui_elements = get_mock_ui_elements_for_androidworld()
        prompt_text = build_androidworld_like_prompt(
            task=args.task,
            action_history=mock_history,
            ui_elements=mock_ui_elements,
        )
        prompt_end = time.time()
        prompt_build_latency_ms = (prompt_end - prompt_start) * 1000
        prompt_build_latency_list.append(prompt_build_latency_ms)

        print("[Prompt]")
        print(prompt_text)

        # Optional extra sleep before llama call
        if args.reasoning_sleep_sec > 0:
            time.sleep(args.reasoning_sleep_sec)

        # Real llama.cpp call
        reasoning_start = time.time()
        llm_text = call_llama_cpp_text_only(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt_text,
            api_url=args.llama_api_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        reasoning_end = time.time()
        reasoning_latency_ms = (reasoning_end - reasoning_start) * 1000
        reasoning_latency_list.append(reasoning_latency_ms)

        # Parse model output
        parse_start = time.time()
        action_obj = parse_action_from_llm_text(llm_text)
        parse_end = time.time()
        parse_latency_ms = (parse_end - parse_start) * 1000
        parse_latency_list.append(parse_latency_ms)

        step_latency_ms = (parse_end - start_time) * 1000
        end_to_end_latency_list.append(step_latency_ms)

        steps.append(
            {
                "step": itr,
                "prompt_text": prompt_text,
                "llm_text": llm_text,
                "action": action_obj,
                "prompt_build_latency_ms": prompt_build_latency_ms,
                "reasoning_latency_ms": reasoning_latency_ms,
                "parse_latency_ms": parse_latency_ms,
                "step_latency_ms": step_latency_ms,
            }
        )

        preview = llm_text.replace("\n", " ")[:200]
        print(f"[LLM Output] {preview}")
        print(f"[Parse] Parsed action: {action_obj}")
        print(
            f"Prompt build latency: {prompt_build_latency_ms:.3f} ms, "
            f"Reasoning latency: {reasoning_latency_ms:.3f} ms, "
            f"Parse latency: {parse_latency_ms:.3f} ms"
        )
        print(f"Step latency: {step_latency_ms:.3f} ms")

    print("\n=== Finished all iterations ===")
    print(
        f"Prompt build latency: {avg(prompt_build_latency_list):.3f} ms, "
        f"Reasoning latency: {avg(reasoning_latency_list):.3f} ms, "
        f"Parse latency: {avg(parse_latency_list):.3f} ms, "
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
        default="Search papers on Mobile GUI Agent on Google Scholar.",
        help="Task instruction for AndroidWorld-like prompt simulation.",
    )
    parser.add_argument("--max_itr", type=int, default=10, help="Maximum benchmark iterations.")
    parser.add_argument(
        "--reasoning_sleep_sec",
        type=float,
        default=0.0,
        help="Optional extra wait before each llama.cpp request.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional output JSON file for step-level records.",
    )
    parser.add_argument(
        "--screenshot_path",
        type=str,
        default="./screenshot.png",
        help="Input screenshot path.",
    )
    parser.add_argument(
        "--llama_api_url",
        type=str,
        default="http://localhost:8100/v1/chat/completions",
        help="llama.cpp OpenAI-compatible endpoint.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max new tokens per request.")
    args = parser.parse_args()

    run_text_only_sim(args)