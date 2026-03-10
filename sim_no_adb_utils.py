import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Tuple


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


def parse_action_from_llm_text(llm_text: str) -> Dict[str, Any]:
    raw = (llm_text or "").strip()
    if not raw:
        return {"action_type": "wait", "action_inputs": {}}

    segment = raw
    m = re.search(r"<tool_call>(.*?)</tool_call>", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        segment = m.group(1).strip()

    json_block = _extract_first_json_object(segment)
    if not json_block:
        json_block = _extract_first_json_object(raw)

    if json_block:
        try:
            obj = json.loads(json_block)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            args = {}
            if isinstance(obj.get("arguments"), dict):
                args = obj["arguments"]
            elif "action" in obj or "action_type" in obj:
                args = obj

            action = (args.get("action") or args.get("action_type") or "wait").strip().lower()
            action_inputs = {k: v for k, v in args.items() if k not in {"action", "action_type"}}
            return {"action_type": action, "action_inputs": action_inputs}

    lower = raw.lower()
    if "terminate" in lower or '"status":"success"' in lower:
        return {"action_type": "terminate", "action_inputs": {"status": "success"}}
    if '"action":"type"' in lower or " type " in lower:
        return {"action_type": "type", "action_inputs": {"text": ""}}
    if '"action":"click"' in lower or "click" in lower:
        return {"action_type": "click", "action_inputs": {}}
    if "wait" in lower:
        return {"action_type": "wait", "action_inputs": {}}
    return {"action_type": "wait", "action_inputs": {}}

