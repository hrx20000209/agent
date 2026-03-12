import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import json
import argparse
import subprocess
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

from MobileAgentE.controller import get_a11y_tree
from MobileAgentE.api import inference_chat_llama_cpp_xml_only
from agents.mai_ui_agent import MAIOneStepAgent


XML_ONLY_SYSTEM_PROMPT = """You are a mobile GUI agent.
You are given:
- a user task
- action history
- the current UI accessibility XML tree

Decide ONE next action.

Output format:
<thinking>
one-sentence reasoning
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


def _history_to_text(history: List[str], max_items: int = 10) -> str:
    if not history:
        return "None"
    return "\n".join(history[-max_items:])


def _safe_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _read_xml_text(args, xml_path: str) -> str:
    if not args.on_device:
        with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    cmd = f"{args.adb_path} shell cat /sdcard/a11y.xml"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to read on-device xml: {proc.stderr}")
    return proc.stdout


def _sanitize_text(v: str, max_chars: int) -> str:
    s = (v or "").replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
    if max_chars > 0 and len(s) > max_chars:
        return s[:max_chars] + "..."
    return s


def _parse_bool(s: str) -> bool:
    return str(s).lower() == "true"


def _build_pruned_tree_text_lightweight(xml_path: str, max_nodes: int = 0, max_field_chars: int = 72) -> Tuple[str, int]:
    tree = ET.parse(xml_path)
    hierarchy = tree.getroot()
    first = hierarchy.find("node")

    if first is None:
        return "<a11y_leaf_nodes>\n</a11y_leaf_nodes>", 0

    leaves = []

    def _dfs(element):
        children = element.findall("node")
        if children:
            for c in children:
                _dfs(c)
            return

        attrib = element.attrib
        clickable = _parse_bool(attrib.get("clickable", "false"))
        long_clickable = _parse_bool(attrib.get("long-clickable", "false"))
        focusable = _parse_bool(attrib.get("focusable", "false"))
        scrollable = _parse_bool(attrib.get("scrollable", "false"))
        checkable = _parse_bool(attrib.get("checkable", "false"))
        selected = _parse_bool(attrib.get("selected", "false"))
        text = attrib.get("text", "")
        content_desc = attrib.get("content-desc", "")

        useful = (
            clickable
            or long_clickable
            or focusable
            or scrollable
            or checkable
            or bool(str(text).strip())
            or bool(str(content_desc).strip())
        )
        if not useful:
            return

        leaves.append(
            {
                "uid": f"E{len(leaves) + 1:04d}",
                "class_name": attrib.get("class", ""),
                "text": text,
                "content_desc": content_desc,
                "resource_id": attrib.get("resource-id", ""),
                "bounds": attrib.get("bounds", ""),
                "clickable": clickable,
                "focusable": focusable,
                "scrollable": scrollable,
                "selected": selected,
            }
        )

    _dfs(first)

    keep_n = len(leaves) if int(max_nodes) <= 0 else min(len(leaves), int(max_nodes))

    lines = ["<a11y_leaf_nodes>"]
    for idx, node in enumerate(leaves[:keep_n], start=1):
        uid = _sanitize_text(str(node.get("uid", "") or ""), max_field_chars)
        cls = _sanitize_text(str(node.get("class_name", "") or ""), max_field_chars)
        txt = _sanitize_text(str(node.get("text", "") or ""), max_field_chars)
        desc = _sanitize_text(str(node.get("content_desc", "") or ""), max_field_chars)
        rid = _sanitize_text(str(node.get("resource_id", "") or ""), max_field_chars)
        bounds = _sanitize_text(str(node.get("bounds", "") or ""), max_field_chars)

        clickable = int(bool(node.get("clickable", False)))
        focusable = int(bool(node.get("focusable", False)))
        scrollable = int(bool(node.get("scrollable", False)))
        selected = int(bool(node.get("selected", False)))

        lines.append(
            f"{idx}. uid={uid} class={cls} text=\"{txt}\" desc=\"{desc}\" "
            f"rid=\"{rid}\" clickable={clickable} focusable={focusable} "
            f"scrollable={scrollable} selected={selected} bounds={bounds}"
        )

    if keep_n < len(leaves):
        lines.append(f"... ({len(leaves) - keep_n} more nodes truncated)")
    lines.append("</a11y_leaf_nodes>")
    return "\n".join(lines), len(leaves)


def _build_pruned_tree_text(
    xml_path: str,
    max_nodes: int = 0,
    max_field_chars: int = 72,
    prefer_main_parser: bool = False,
) -> Tuple[str, int]:
    if prefer_main_parser:
        try:
            old_offline = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            from MobileAgentE.tree import parse_a11y_tree  # lazy import to avoid heavy init unless requested
            root = parse_a11y_tree(xml_path=xml_path)
            leaves = list(getattr(root, "children", []) or [])

            keep_n = len(leaves) if int(max_nodes) <= 0 else min(len(leaves), int(max_nodes))

            lines = ["<a11y_leaf_nodes>"]
            for idx, node in enumerate(leaves[:keep_n], start=1):
                uid = _sanitize_text(str(getattr(node, "uid", "") or ""), max_field_chars)
                cls = _sanitize_text(str(getattr(node, "class_name", "") or ""), max_field_chars)
                txt = _sanitize_text(str(getattr(node, "text", "") or ""), max_field_chars)
                desc = _sanitize_text(str(getattr(node, "content_desc", "") or ""), max_field_chars)
                rid = _sanitize_text(str(getattr(node, "resource_id", "") or ""), max_field_chars)
                bounds = _sanitize_text(str(getattr(node, "bounds", "") or ""), max_field_chars)

                clickable = int(bool(getattr(node, "clickable", False)))
                focusable = int(bool(getattr(node, "focusable", False)))
                scrollable = int(bool(getattr(node, "scrollable", False)))
                selected = int(bool(getattr(node, "selected", False)))

                lines.append(
                    f"{idx}. uid={uid} class={cls} text=\"{txt}\" desc=\"{desc}\" "
                    f"rid=\"{rid}\" clickable={clickable} focusable={focusable} "
                    f"scrollable={scrollable} selected={selected} bounds={bounds}"
                )
            if keep_n < len(leaves):
                lines.append(f"... ({len(leaves) - keep_n} more nodes truncated)")
            lines.append("</a11y_leaf_nodes>")
            return "\n".join(lines), len(leaves)
        except Exception:
            pass
        finally:
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline

    return _build_pruned_tree_text_lightweight(
        xml_path=xml_path,
        max_nodes=max_nodes,
        max_field_chars=max_field_chars,
    )


def _llm_api(chat, args) -> str:
    return inference_chat_llama_cpp_xml_only(
        chat,
        api_url=args.llama_api_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def _is_context_overflow_error(exc: Exception) -> bool:
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)

    body = ""
    if resp is not None:
        try:
            body = (resp.text or "").lower()
        except Exception:
            body = ""

    msg = str(exc).lower()
    if status != 400:
        return False
    return any(
        k in (body + " " + msg)
        for k in [
            "exceed_context_size_error",
            "exceeds the available context size",
            "context size",
            "n_ctx",
            "n_prompt_tokens",
        ]
    )


class XMLOnlyAgent:
    def __init__(self, coord_space: str = "auto"):
        self._parser = MAIOneStepAgent(adb_path="adb", coord_space=coord_space)

    def init_chat(self):
        return [["system", [{"type": "text", "text": XML_ONLY_SYSTEM_PROMPT}]]]

    def build_user_prompt(self, task: str, history: List[str], xml_text: str) -> str:
        return (
            f"Task:\n{task}\n\n"
            f"Action History:\n{_history_to_text(history)}\n\n"
            f"Accessibility Tree (text):\n{xml_text}\n\n"
            "Output the next action now."
        )

    def run_step(self, task: str, history: List[str], xml_text: str, llm_api_func):
        chat = self.init_chat()
        user_prompt = self.build_user_prompt(task, history, xml_text)
        chat.append(["user", [{"type": "text", "text": user_prompt}]])
        llm_output = llm_api_func(chat)
        action_obj = self._parser.parse_action(llm_output or "")
        return action_obj, llm_output


def _run_step_with_adaptive_xml(agent: XMLOnlyAgent, args, task: str, history: List[str], raw_xml_text: str):
    if args.max_xml_chars > 0:
        xml_limit = min(len(raw_xml_text), int(args.max_xml_chars))
    else:
        xml_limit = len(raw_xml_text)

    min_chars = max(256, int(args.min_xml_chars))
    retries_left = int(args.max_context_retries)
    tried = 0

    while True:
        tried += 1
        xml_text = raw_xml_text[:xml_limit]
        if xml_limit < len(raw_xml_text):
            xml_text += "\n<!-- truncated -->"

        try:
            action_obj, llm_output = agent.run_step(
                task=task,
                history=history,
                xml_text=xml_text,
                llm_api_func=lambda chat: _llm_api(chat, args),
            )
            return action_obj, llm_output, len(xml_text), tried
        except Exception as e:
            if retries_left <= 0 or not _is_context_overflow_error(e) or xml_limit <= min_chars:
                raise

            next_limit = max(min_chars, int(xml_limit * float(args.xml_shrink_ratio)))
            if next_limit >= xml_limit:
                next_limit = max(min_chars, xml_limit - 512)

            if next_limit >= xml_limit:
                raise

            print(
                f"[Warn] Context overflow at input_chars={xml_limit}. "
                f"Retry with input_chars={next_limit}."
            )
            xml_limit = next_limit
            retries_left -= 1


def run_xml_only_agent(args):
    input_dir = "/sdcard" if args.on_device else "./screenshot"
    xml_path = os.path.join(input_dir, "a11y.xml")
    if not args.on_device:
        os.makedirs(input_dir, exist_ok=True)

    print("### Running XML-only LLM benchmark ###")
    print(f"[Config] task={args.task}")
    print(f"[Config] max_itr={args.max_itr}, llama_api_url={args.llama_api_url}")

    agent = XMLOnlyAgent(coord_space=args.coord_space)
    history: List[str] = []
    steps: List[Dict] = []

    a11y_tree_latency_list: List[float] = []
    planning_latency_list: List[float] = []
    end_to_end_latency_list: List[float] = []

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        get_a11y_tree(args, xml_path)
        if (not args.on_device) and args.use_pruned_tree:
            try:
                raw_xml_text, total_nodes = _build_pruned_tree_text(
                    xml_path=xml_path,
                    max_nodes=args.max_tree_nodes,
                    max_field_chars=args.max_field_chars,
                    prefer_main_parser=args.prefer_main_tree_parser,
                )
                print(
                    f"[Perception] UI text source=pruned_tree "
                    f"(nodes_total={total_nodes}, max_nodes={args.max_tree_nodes})"
                )
                if args.tree_preview_lines > 0:
                    preview_lines = raw_xml_text.splitlines()[: int(args.tree_preview_lines)]
                    print(f"[Tree Preview] showing first {len(preview_lines)} lines:")
                    for ln in preview_lines:
                        print(ln[:240])
            except Exception as e:
                print(f"[Warn] pruned_tree build failed, fallback to raw xml: {e}")
                raw_xml_text = _read_xml_text(args, xml_path)
                print("[Perception] UI text source=raw_xml")
        else:
            raw_xml_text = _read_xml_text(args, xml_path)
            source = "raw_xml_on_device" if args.on_device else "raw_xml"
            print(f"[Perception] UI text source={source}")
            if args.tree_preview_lines > 0:
                preview_lines = raw_xml_text.splitlines()[: int(args.tree_preview_lines)]
                print(f"[Tree Preview] showing first {len(preview_lines)} lines:")
                for ln in preview_lines:
                    print(ln[:240])

        a11y_tree_time = time.time()
        a11y_tree_latency = (a11y_tree_time - start_time) * 1000
        a11y_tree_latency_list.append(a11y_tree_latency)

        action_obj, llm_output, used_xml_chars, retry_count = _run_step_with_adaptive_xml(
            agent=agent,
            args=args,
            task=args.task,
            history=history,
            raw_xml_text=raw_xml_text,
        )

        planning_end_time = time.time()
        planning_latency = (planning_end_time - a11y_tree_time) * 1000
        planning_latency_list.append(planning_latency)

        print("[Reasoning] Parsed action:", action_obj)
        print(f"[Reasoning] used_input_chars={used_xml_chars}, retry_count={retry_count}")

        history_item = _safe_json(action_obj) if isinstance(action_obj, dict) else str(llm_output)
        history.append(history_item)

        step_latency = (planning_end_time - start_time) * 1000
        end_to_end_latency_list.append(step_latency)

        steps.append(
            {
                "step": itr,
                "action": action_obj,
                "raw_output": llm_output,
                "used_input_chars": used_xml_chars,
                "retry_count": retry_count,
                "a11y_tree_latency_ms": a11y_tree_latency,
                "planning_latency_ms": planning_latency,
                "step_latency_ms": step_latency,
            }
        )

        print(
            f"A11Y Tree latency: {a11y_tree_latency:.3f} ms, "
            f"Planning latency: {planning_latency:.3f} ms"
        )
        print(f"Step latency: {step_latency:.3f} ms")

    if not end_to_end_latency_list:
        print("No iterations executed.")
        return steps

    avg_a11y_tree_latency = sum(a11y_tree_latency_list) / len(a11y_tree_latency_list)
    avg_planning_latency = sum(planning_latency_list) / len(planning_latency_list)
    avg_end_to_end_latency = sum(end_to_end_latency_list) / len(end_to_end_latency_list)

    print("\n=== Finished all iterations ===")
    print(
        f"A11Y Tree latency: {avg_a11y_tree_latency:.3f} ms, "
        f"Planning latency: {avg_planning_latency:.3f} ms, "
        f"End-to-end latency: {avg_end_to_end_latency:.3f} ms"
    )
    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="Search papers on Mobile GUI Agent on Google Scholar.",
        help="User instruction for the XML-only agent",
    )
    parser.add_argument("--max_itr", type=int, default=10, help="Maximum benchmark iterations")
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
        "--max_xml_chars",
        type=int,
        default=0,
        help="Truncate UI text to this many chars before sending. <=0 means no truncation.",
    )
    parser.add_argument(
        "--min_xml_chars",
        type=int,
        default=1200,
        help="Minimum XML chars when adaptive shrinking retries.",
    )
    parser.add_argument(
        "--xml_shrink_ratio",
        type=float,
        default=0.65,
        help="Adaptive XML shrink ratio when context overflows.",
    )
    parser.add_argument(
        "--max_context_retries",
        type=int,
        default=6,
        help="Max retries for context-overflow backoff.",
    )
    parser.add_argument(
        "--use_pruned_tree",
        dest="use_pruned_tree",
        action="store_true",
        default=True,
        help="Use parse_a11y_tree pruned leaf text instead of raw XML (recommended).",
    )
    parser.add_argument(
        "--disable_pruned_tree",
        dest="use_pruned_tree",
        action="store_false",
        help="Disable pruned tree and send raw XML text.",
    )
    parser.add_argument(
        "--max_tree_nodes",
        type=int,
        default=0,
        help="Max number of pruned leaf nodes kept in tree text. <=0 means keep all parsed nodes.",
    )
    parser.add_argument(
        "--max_field_chars",
        type=int,
        default=72,
        help="Max chars per text/desc/id field in pruned tree text.",
    )
    parser.add_argument(
        "--prefer_main_tree_parser",
        action="store_true",
        help="Try using MobileAgentE.tree.parse_a11y_tree first, then fallback to lightweight parser.",
    )
    parser.add_argument(
        "--tree_preview_lines",
        type=int,
        default=8,
        help="Print first N lines of current tree text for debugging; 0 disables preview.",
    )
    args = parser.parse_args()

    run_xml_only_agent(args)
