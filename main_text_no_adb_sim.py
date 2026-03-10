import argparse
import json
import time
from typing import Dict, List

from sim_no_adb_utils import avg, parse_action_from_llm_text, simulate_llm_output


def _load_llm_texts(args) -> List[str]:
    if args.llm_text_file:
        texts: List[str] = []
        with open(args.llm_text_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                # Support plain text lines or jsonl with {"llm_text": "..."}
                if s.startswith("{") and s.endswith("}"):
                    try:
                        obj = json.loads(s)
                        if isinstance(obj, dict) and isinstance(obj.get("llm_text"), str):
                            texts.append(obj["llm_text"])
                            continue
                    except Exception:
                        pass
                texts.append(s)
        return texts

    if args.llm_text:
        return [args.llm_text]

    # Fallback to built-in simulated outputs.
    return [
        simulate_llm_output(args.task, history_len=0, role="single"),
        simulate_llm_output(args.task, history_len=1, role="single"),
        simulate_llm_output(args.task, history_len=2, role="single"),
    ]


def run_text_only_sim(args):
    print("### Running Text-Only Simulation (No ADB) ###")
    print(f"[Config] max_itr={args.max_itr}")
    if args.llm_text:
        print("[Config] input_source=--llm_text")
    elif args.llm_text_file:
        print(f"[Config] input_source=--llm_text_file ({args.llm_text_file})")
    else:
        print("[Config] input_source=built-in simulated LLM outputs")

    llm_texts = _load_llm_texts(args)
    if not llm_texts:
        raise RuntimeError("No LLM text found. Provide --llm_text or --llm_text_file.")

    steps: List[Dict] = []
    input_wait_latency_list: List[float] = []
    parse_latency_list: List[float] = []
    end_to_end_latency_list: List[float] = []

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        # Simulate waiting for LLM output to arrive.
        time.sleep(max(0.0, args.input_wait_ms / 1000.0))
        input_ready_time = time.time()
        input_wait_ms = (input_ready_time - start_time) * 1000
        input_wait_latency_list.append(input_wait_ms)

        llm_text = llm_texts[(itr - 1) % len(llm_texts)]

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
                "llm_text": llm_text,
                "action": action_obj,
                "input_wait_latency_ms": input_wait_ms,
                "parse_latency_ms": parse_latency_ms,
                "step_latency_ms": step_latency_ms,
            }
        )

        preview = llm_text.replace("\n", " ")[:120]
        print(f"[Input] LLM text preview: {preview}")
        print(f"[Parse] Parsed action: {action_obj}")
        print(
            f"Input wait latency: {input_wait_ms:.3f} ms, "
            f"Parse latency: {parse_latency_ms:.3f} ms"
        )
        print(f"Step latency: {step_latency_ms:.3f} ms")

    print("\n=== Finished all iterations (text-only simulation) ===")
    print(
        f"Input wait latency: {avg(input_wait_latency_list):.3f} ms, "
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
        help="Only used when built-in simulated LLM outputs are selected.",
    )
    parser.add_argument("--max_itr", type=int, default=10, help="Maximum benchmark iterations.")
    parser.add_argument(
        "--llm_text",
        type=str,
        default="",
        help="Direct LLM output text as input for parsing.",
    )
    parser.add_argument(
        "--llm_text_file",
        type=str,
        default="",
        help="Optional file path containing one LLM output per line (or JSONL with llm_text field).",
    )
    parser.add_argument(
        "--input_wait_ms",
        type=float,
        default=0.0,
        help="Simulated waiting time before receiving each LLM text input.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional output JSON file for step-level records.",
    )
    args = parser.parse_args()

    run_text_only_sim(args)

