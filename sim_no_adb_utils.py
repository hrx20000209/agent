import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Tuple

import imagehash
from PIL import Image
from MobileAgentE.api import inference_chat_llama_cpp
from agents.mai_parser import parse_tagged_text
from agents.mai_ui_agent import add_chat


COMMON_ELEMENT_TEXTS: List[str] = [
    "Search input box",
    "Search button",
    "Settings menu",
    "Send message button",
    "Back button",
    "Home tab",
    "Profile icon",
    "Notification center",
    "Share button",
    "Save draft",
    "Play or pause media",
    "Start recording",
    "Stopwatch start button",
    "Confirm dialog",
]


def avg(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def text_similarity(task: str, element_text: str) -> Tuple[float, float, float]:
    """
    Returns:
        score, jaccard, seq_ratio
    """
    task_tokens = _tokenize(task)
    elem_tokens = _tokenize(element_text)

    task_set = set(task_tokens)
    elem_set = set(elem_tokens)
    if not task_set and not elem_set:
        jaccard = 1.0
    elif not task_set or not elem_set:
        jaccard = 0.0
    else:
        jaccard = len(task_set & elem_set) / max(1, len(task_set | elem_set))

    task_norm = " ".join(task_tokens)
    elem_norm = " ".join(elem_tokens)
    seq_ratio = SequenceMatcher(None, task_norm, elem_norm).ratio()

    score = 0.65 * jaccard + 0.35 * seq_ratio
    return score, jaccard, seq_ratio


def _extract_search_query(task: str) -> str:
    """
    Very lightweight heuristic for common "search xxx on yyy" tasks.
    """
    text = (task or "").strip()
    lower = text.lower()
    if "search" in lower:
        m = re.search(r"search\s+(.*?)\s+(?:on|in)\s+", text, flags=re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return text[:48] if text else "query"


def infer_action_from_task(task: str, history_len: int) -> Dict[str, Any]:
    lower = (task or "").lower()

    if "search" in lower:
        query = _extract_search_query(task)
        if history_len == 0:
            return {"action": "click", "coordinate": [520, 200]}
        if history_len == 1:
            return {"action": "type", "text": query}
        if history_len == 2:
            return {"action": "system_button", "button": "enter"}
        return {"action": "wait"}

    if "record" in lower or "audio" in lower:
        if history_len == 0:
            return {"action": "open", "text": "Recorder"}
        if history_len == 1:
            return {"action": "click", "coordinate": [540, 1850]}
        return {"action": "terminate", "status": "success"}

    if "stopwatch" in lower:
        if history_len == 0:
            return {"action": "open", "text": "Clock"}
        if history_len == 1:
            return {"action": "click", "coordinate": [540, 1700]}
        return {"action": "terminate", "status": "success"}

    if history_len >= 2:
        return {"action": "terminate", "status": "success"}
    return {"action": "wait"}


def simulate_llm_output(task: str, history_len: int, role: str = "single") -> str:
    action = infer_action_from_task(task, history_len=history_len)

    if role == "manager":
        return (
            "### Thought ###\n"
            "Track progress and keep plan concise.\n"
            "### Historical Operations ###\n"
            f"history_len={history_len}\n"
            "### Plan ###\n"
            "1. Identify target UI entry.\n"
            "2. Perform one atomic action.\n"
            "3. Verify and finish if done.\n"
        )

    if role == "executor":
        rationale = "Pick one safe action that advances the first unfinished subgoal."
    else:
        rationale = "Choose one next action from current progress."

    payload = {"name": "mobile_use", "arguments": action}
    return (
        "<thinking>\n"
        f"{rationale}\n"
        "</thinking>\n"
        "<tool_call>\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        "</tool_call>"
    )


def _extract_first_json_object(text: str) -> str:
    start = -1
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start:idx + 1]
    return ""


def _extract_action_args(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}

    # Standard MAI format: {"name":"mobile_use","arguments":{...}}
    if str(obj.get("name", "")).strip().lower() == "mobile_use":
        args = obj.get("arguments")
        return args if isinstance(args, dict) else {}

    # Some models output {"name":"click", ...} directly.
    if "name" in obj and "arguments" not in obj:
        name = str(obj.get("name", "")).strip().lower()
        if name:
            args = dict(obj)
            args.pop("name", None)
            args["action"] = name
            return args

    # Some outputs are already flat action JSON.
    if "action" in obj:
        return obj

    # Fallback for malformed nesting.
    args = obj.get("arguments")
    if isinstance(args, dict):
        if "action" in args:
            return args
        name = str(obj.get("name", "")).strip().lower()
        if name:
            merged = dict(args)
            merged["action"] = merged.get("action", name)
            return merged

    return {}


def _infer_coord_space(inputs: Dict[str, Any]) -> str:
    vals: List[float] = []
    for key in ("coordinate", "start_coordinate", "end_coordinate"):
        coord = inputs.get(key)
        if not isinstance(coord, (list, tuple)) or len(coord) < 2:
            continue
        try:
            vals.extend([abs(float(coord[0])), abs(float(coord[1]))])
        except (TypeError, ValueError):
            continue

    if not vals:
        return "1000"

    vmax = max(vals)
    if vmax <= 1.05:
        return "norm1"
    if vmax <= 1000.0:
        return "1000"
    return "pixel"


def parse_action_from_llm_text(llm_text: str) -> Dict[str, Any]:
    raw = (llm_text or "").strip()
    if not raw:
        return {"action_type": "wait", "action_inputs": {}}

    # MAI official parser path first.
    try:
        parsed = parse_tagged_text(raw)
        tool_call = parsed.get("tool_call")
        if isinstance(tool_call, dict):
            args = _extract_action_args(tool_call)
            if args:
                action = str(args.get("action") or args.get("action_type") or "wait").strip().lower()
                action_inputs = {k: v for k, v in args.items() if k not in {"action", "action_type"}}
                return _canonicalize_action(action, action_inputs)
    except Exception:
        pass

    # Generic JSON fallback.
    segment = raw
    m = re.search(r"<tool_call>(.*?)</tool_call>", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        segment = m.group(1).strip()

    json_block = _extract_first_json_object(segment) or _extract_first_json_object(raw)
    if json_block:
        try:
            obj = json.loads(json_block)
        except Exception:
            obj = None

        if isinstance(obj, dict):
            args = _extract_action_args(obj)
            if args:
                action = str(args.get("action") or args.get("action_type") or "wait").strip().lower()
                action_inputs = {k: v for k, v in args.items() if k not in {"action", "action_type"}}
                return _canonicalize_action(action, action_inputs)

    lower = raw.lower()
    if "terminate" in lower or '"status":"success"' in lower:
        return {"action_type": "terminate", "action_inputs": {"status": "success"}}
    if '"action":"type"' in lower or " type " in lower:
        return {"action_type": "type", "action_inputs": {"content": ""}}
    if '"action":"click"' in lower or "click" in lower:
        return {
            "action_type": "click",
            "action_inputs": {"coordinate": [500, 500]},
            "coord_space": "1000",
        }
    if "wait" in lower:
        return {"action_type": "wait", "action_inputs": {}}
    return {"action_type": "wait", "action_inputs": {}}


def _canonicalize_action(action: str, action_inputs: Dict[str, Any]) -> Dict[str, Any]:
    act = (action or "wait").strip().lower()
    inp = dict(action_inputs or {})

    if act == "open":
        act = "open_app"
    elif act in {"done", "finish", "finished", "answer", "stop", "exit"}:
        act = "terminate"

    if act == "system_button":
        button = str(inp.get("button", "")).strip().lower()
        if button == "back":
            act = "press_back"
        elif button == "home":
            act = "press_home"
        elif button in {"enter", "ok"}:
            act = "press_enter"
        else:
            act = "wait"
            inp = {}

    if act == "type":
        # execute_action expects "content"
        if "content" not in inp:
            text_val = inp.get("text")
            if text_val is None:
                text_val = inp.get("value", "")
            inp["content"] = text_val or ""

    if act == "open_app":
        # execute_action supports content/value/text/app_name; keep all if present
        if "app_name" in inp and "content" not in inp:
            inp["content"] = inp.get("app_name")
        elif "text" in inp and "content" not in inp:
            inp["content"] = inp.get("text")
        elif "value" in inp and "content" not in inp:
            inp["content"] = inp.get("value")

    if act == "swipe" and "coordinate" not in inp and "start_coordinate" not in inp:
        # Keep MAI semantics while remaining executable by current controller.
        inp["coordinate"] = [500, 500]

    out = {"action_type": act, "action_inputs": inp}

    if act in {"click", "long_press", "swipe", "drag"}:
        out["coord_space"] = _infer_coord_space(inp)

    return out


def ensure_screenshot_path(screenshot_path: str) -> str:
    abs_path = os.path.abspath(screenshot_path or "./screenshot.png")
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"screenshot not found: {abs_path}. "
            "Please place screenshot.png at repo root or pass --screenshot_path."
        )
    return abs_path


def call_llama_cpp_with_image(
    system_prompt: str,
    user_prompt: str,
    screenshot_path: str,
    api_url: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    chat = [["system", [{"type": "text", "text": system_prompt}]]]
    chat = add_chat("user", user_prompt, chat, image=screenshot_path)
    try:
        return inference_chat_llama_cpp(
            chat,
            api_url=api_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(f"llama.cpp request failed at {api_url}: {exc}") from exc


def compute_phash(image_path: str):
    img = Image.open(image_path).convert("L")
    return imagehash.phash(img)


def _to_hash_obj(hash_value):
    if isinstance(hash_value, str):
        return imagehash.hex_to_hash(hash_value.strip())
    return hash_value


def verify_with_anchor_phash(
    screenshot_path: str,
    anchor_hash,
    threshold: int = 8,
) -> Dict[str, Any]:
    verify_start = time.time()
    curr_hash = compute_phash(screenshot_path)
    anchor_obj = _to_hash_obj(anchor_hash)
    diff = int(curr_hash - anchor_obj)
    ok = diff <= int(threshold)
    verify_ms = (time.time() - verify_start) * 1000
    return {
        "ok": ok,
        "diff": diff,
        "threshold": int(threshold),
        "current_hash": str(curr_hash),
        "anchor_hash": str(anchor_obj),
        "verification_latency_ms": verify_ms,
    }
