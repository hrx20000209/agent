# MobileAgentE/agents/mai_agent/mai_parser.py

import json
import re
from typing import Dict, Any, Optional

SCALE_FACTOR = 999


def normalize_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize tool_call into MAI schema:
    {"name":"mobile_use","arguments":{"action":..., ...}}
    """
    if not isinstance(tool_call, dict):
        return {}

    # Already normalized
    if "arguments" in tool_call and isinstance(tool_call["arguments"], dict):
        return tool_call

    # If model outputs {"name":"click","coordinate":[...]}
    name = (tool_call.get("name") or "").lower()

    # Treat "name" as action
    if name in ["click", "long_press", "type", "swipe", "drag", "open", "wait", "terminate", "system_button", "answer"]:
        args = dict(tool_call)
        args.pop("name", None)
        args["action"] = name
        return {"name": "mobile_use", "arguments": args}

    # If model outputs directly {"action":"click", ...}
    if "action" in tool_call:
        return {"name": "mobile_use", "arguments": tool_call}

    return tool_call


def parse_tagged_text(text: str) -> Dict[str, Any]:
    """
    Parse <thinking> and <tool_call> blocks.
    Compatible with some models that output </think>.
    """
    text = text.strip()

    if "</think>" in text and "</thinking>" not in text:
        text = text.replace("</think>", "</thinking>")
        text = "<thinking>" + text

    pattern = r"<thinking>(.*?)</thinking>.*?<tool_call>(.*?)</tool_call>"
    match = re.search(pattern, text, re.DOTALL)

    out = {"thinking": None, "tool_call": None}
    if match:
        out["thinking"] = match.group(1).strip().strip('"')
        out["tool_call"] = match.group(2).strip().strip('"')

    if out["tool_call"]:
        out["tool_call"] = safe_json_loads(out["tool_call"])

    return out


def safe_json_loads(s: str) -> Dict[str, Any]:
    """
    A robust JSON loader for malformed tool_call output.
    """
    s = s.strip()

    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.startswith("json"):
            s = s[len("json"):].lstrip()

    # extract outermost {...}
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        s = s[first:last + 1]

    try:
        return json.loads(s)
    except Exception:
        # fallback: minimal recovery
        # try to extract action and coordinate
        def extract(pattern, default=None):
            m = re.search(pattern, s)
            return m.group(1) if m else default

        name = extract(r'"name"\s*:\s*"([^"]+)"', "mobile_use")
        act = extract(r'"action"\s*:\s*"([^"]+)"', "click")

        coord = None
        m = re.search(r'"coordinate"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)', s)
        if m:
            coord = [int(m.group(1)), int(m.group(2))]

        args = {"action": act}
        if coord is not None:
            args["coordinate"] = coord

        return {"name": name, "arguments": args}


def mai_toolcall_to_action(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    tool_call = normalize_tool_call(tool_call)
    if not isinstance(tool_call, dict):
        return {"action_type": None}

    args = tool_call.get("arguments", {}) or {}
    if not isinstance(args, dict):
        return {"action_type": None}

    act = (args.get("action") or "").lower()

    # terminate/finish
    if act in ["terminate", "finish", "done", "exit", "stop", "answer"]:
        return {"action_type": "finished", "action_inputs": {}, "raw": args}

    if act == "wait":
        return {"action_type": "wait", "action_inputs": {"seconds": 2}, "raw": args}

    if act == "system_button":
        btn = (args.get("button") or "").lower()
        mp = {"back": "press_back", "home": "press_home", "enter": "enter"}
        return {"action_type": mp.get(btn, "press_back"), "action_inputs": {}, "raw": args}

    if act == "type":
        return {"action_type": "type", "action_inputs": {"content": args.get("text", "")}, "raw": args}

    # click / long press
    if act in ["click", "long_press"]:
        coord = args.get("coordinate")
        if not coord:
            return {"action_type": None, "raw": args}

        if len(coord) == 2:
            x, y = coord
        elif len(coord) == 4:
            x1, y1, x2, y2 = coord
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
        else:
            return {"action_type": None, "raw": args}

        x_norm = float(x) / SCALE_FACTOR
        y_norm = float(y) / SCALE_FACTOR

        eps = 0.002
        start_box = [
            max(0.0, x_norm - eps),
            max(0.0, y_norm - eps),
            min(1.0, x_norm + eps),
            min(1.0, y_norm + eps),
        ]

        return {
            "action_type": "long_press" if act == "long_press" else "click",
            "action_inputs": {"start_box": start_box},
            "raw": args,
        }

    # swipe
    if act == "swipe":
        direction = (args.get("direction") or "down").lower()

        # if MAI provides anchor coordinate, use it
        anchor = args.get("coordinate", None)
        if anchor and len(anchor) >= 2:
            ax = float(anchor[0]) / SCALE_FACTOR
            ay = float(anchor[1]) / SCALE_FACTOR
        else:
            ax, ay = 0.5, 0.5

        dist = 0.4
        sx, sy = ax, ay
        ex, ey = ax, ay

        if direction == "up":
            ey = max(0.0, ay - dist)
        elif direction == "down":
            ey = min(1.0, ay + dist)
        elif direction == "left":
            ex = max(0.0, ax - dist)
        elif direction == "right":
            ex = min(1.0, ax + dist)

        return {
            "action_type": "drag",
            "action_inputs": {
                "start_box": [sx-0.01, sy-0.01, sx+0.01, sy+0.01],
                "end_box":   [ex-0.01, ey-0.01, ex+0.01, ey+0.01],
            },
            "raw": args,
        }

    # drag
    if act == "drag":
        sc = args.get("start_coordinate")
        ec = args.get("end_coordinate")
        if not sc or not ec:
            return {"action_type": None, "raw": args}

        sx, sy = float(sc[0]) / SCALE_FACTOR, float(sc[1]) / SCALE_FACTOR
        ex, ey = float(ec[0]) / SCALE_FACTOR, float(ec[1]) / SCALE_FACTOR

        return {
            "action_type": "drag",
            "action_inputs": {
                "start_box": [sx-0.01, sy-0.01, sx+0.01, sy+0.01],
                "end_box":   [ex-0.01, ey-0.01, ex+0.01, ey+0.01],
            },
            "raw": args,
        }

    # open: fallback to home
    if act == "open":
        # better than unsupported open_app if no XML matching
        return {"action_type": "press_home", "action_inputs": {}, "raw": args}

    return {"action_type": None, "raw": args}

