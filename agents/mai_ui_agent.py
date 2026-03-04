# MobileAgentE/agents/mai_agent/mai_one_step_agent.py

from dataclasses import dataclass
import copy
import time
import base64
import ast
import json
import re
from typing import List, Optional

from PIL import Image

from agents.mai_prompt import MAI_MOBILE_SYS_PROMPT, GELAB_PROMPT, build_user_prompt
from typing import Any, Dict, Optional, Tuple, List, Union


BoxType = Union[str, List[float], Tuple[float, ...]]


def _parse_box(box: BoxType) -> List[float]:
    """
    Accept box in formats:
      - [x, y]
      - [x1, y1, x2, y2]
      - "[]"-string that can be literal_eval
    Return python list[float]
    """
    if isinstance(box, (list, tuple)):
        return [float(x) for x in box]

    if isinstance(box, str):
        try:
            val = ast.literal_eval(box)
        except Exception as e:
            raise ValueError(f"Invalid box str: {box}, err={e}")
        if not isinstance(val, (list, tuple)):
            raise ValueError(f"Box string parsed but not list/tuple: {val}")
        return [float(x) for x in val]

    raise ValueError(f"Unsupported box type: {type(box)}")


def _center_of_box(box: List[float]) -> Tuple[float, float]:
    """
    box:
      - [x, y] => return (x,y)
      - [x1,y1,x2,y2] => return center
    """
    if len(box) == 2:
        return float(box[0]), float(box[1])
    if len(box) == 4:
        x = (float(box[0]) + float(box[2])) / 2.0
        y = (float(box[1]) + float(box[3])) / 2.0
        return x, y
    raise ValueError(f"Invalid box length: {len(box)}, box={box}")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def add_chat(role, text, chat_history, image=None):
    new = copy.deepcopy(chat_history)
    if image:
        base64_image = encode_image(image)
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]
    else:
        content = [{"type": "text", "text": text}]
    new.append([role, content])
    return new


class MAIOneStepAgent:
    """
    MAI tool_call agent wrapped into your OneStepAgent style.
    """

    def __init__(self, adb_path, coord_space: str = "auto"):
        self.adb = adb_path
        self.coord_space = str(coord_space or "auto").lower()

    def init_chat(self):
        return [["system", [{"type": "text", "text": GELAB_PROMPT}]]]

    def build_prompt(self, instruction: str, history: str):
        return build_user_prompt(instruction, history)

    def run_step(self, instruction, screenshot_img, width, height, history, llm_api_func, clues, scale=1.0):
        chat = self.init_chat()
        user_prompt = self.build_prompt(instruction, history) + ("\n\n" + clues if clues else "")

        print(user_prompt)

        chat = add_chat("user", user_prompt, chat, image=screenshot_img)

        llm_output = llm_api_func(chat)
        print(f"[LLM Output] {llm_output}")
        action_obj = self.parse_action(llm_output)
        return action_obj


    def parse_action(self, llm_output: str):
        """
        Robust action parser that supports:
        - internal reasoning dict
        - MAI <tool_call>
        - raw JSON
        - partially broken JSON
        """

        llm_output = llm_output.strip()

        # ===============================
        # 1️⃣ Try: model already gave dict
        # ===============================
        try:
            if llm_output.startswith("{") and "action_type" in llm_output:
                obj = ast.literal_eval(llm_output)
                canon = _canonicalize_action(obj)
                if canon:
                    canon["coord_space"] = self.coord_space
                    return canon
        except Exception:
            pass

        # ===============================
        # 2️⃣ Extract JSON inside <tool_call>
        # ===============================
        if "<tool_call>" in llm_output:
            try:
                tool_block = llm_output.split("<tool_call>")[1].split("</tool_call>")[0]
            except Exception:
                tool_block = None

            if tool_block:
                tool_block = tool_block.strip()

                # remove code fences
                if tool_block.startswith("```"):
                    tool_block = tool_block.strip("`").strip()
                    if tool_block.startswith("json"):
                        tool_block = tool_block[len("json"):].lstrip()

                try:
                    obj = json.loads(tool_block)
                except Exception:
                    obj = None

                if obj:
                    canon = _canonicalize_action(obj)
                    if canon:
                        canon["coord_space"] = self.coord_space
                        return canon

        # ===============================
        # 3️⃣ Try: raw JSON in text
        # ===============================
        try:
            first = llm_output.find("{")
            last = llm_output.rfind("}")
            if first != -1 and last != -1:
                obj = json.loads(llm_output[first:last+1])
                canon = _canonicalize_action(obj)
                if canon:
                    canon["coord_space"] = self.coord_space
                    return canon
        except Exception:
            pass

        # ===============================
        # 4️⃣ Heuristic fallback (safe_json style)
        # ===============================
        try:
            act_match = re.search(r'"action"\s*:\s*"([^"]+)"', llm_output)
            text_match = re.search(r'"text"\s*:\s*"([^"]*)"', llm_output)
            coord_match = re.search(r'\[\s*(-?\d+)\s*,\s*(-?\d+)', llm_output)

            act = act_match.group(1).lower() if act_match else "wait"

            obj = {
                "action_type": act,
                "action_inputs": {},
                "raw": {"recovered": True},
            }

            if text_match:
                obj["action_inputs"]["content"] = text_match.group(1)

            if coord_match:
                obj["action_inputs"]["coordinate"] = [
                    int(coord_match.group(1)),
                    int(coord_match.group(2))
                ]

            obj["coord_space"] = self.coord_space
            return obj
        except Exception:
            pass

        print("⚠️ parse_action failed, fallback to wait")
        return {
            "action_type": "wait",
            "action_inputs": {"seconds": 1},
            "coord_space": self.coord_space,
            "raw": {"failed_parse": True},
        }


def _canonicalize_action(obj: dict):
    """
    Convert any action schema into canonical MobileAgentE schema.
    """

    # ---------- Case 1: already internal agent format ----------
    if "action_type" in obj:
        act = obj.get("action_type", "").lower()
        inp = obj.get("action_inputs", {}) or {}
        thinking = obj.get("thinking")

    # ---------- Case 2: MAI tool_call ----------
    elif "tool_call" in obj:
        args = obj["tool_call"].get("arguments", {})
        act = args.get("action", "").lower()
        inp = args
        thinking = obj.get("thinking")

    # ---------- Case 3: raw MAI JSON ----------
    elif "arguments" in obj:
        args = obj.get("arguments", {})
        # Support both:
        # 1) {"name":"mobile_use","arguments":{"action":"click",...}}
        # 2) {"name":"click","arguments":{...}}
        if str(obj.get("name", "")).lower() == "mobile_use":
            act = str(args.get("action", "")).lower()
        else:
            act = str(args.get("action") or obj.get("name") or "").lower()
        inp = args
        thinking = None

    # ---------- Case 4: flat JSON ----------
    # e.g. {"action":"click","coordinate":[x,y]}
    elif "action" in obj:
        act = str(obj.get("action", "")).lower()
        inp = obj
        thinking = obj.get("thinking")

    else:
        return None

    # Some models nest actual params under arguments.arguments
    if isinstance(inp, dict) and isinstance(inp.get("arguments"), dict):
        merged_inp = dict(inp)
        for k, v in inp.get("arguments", {}).items():
            if k not in merged_inp:
                merged_inp[k] = v
        inp = merged_inp

    # ----- normalize system_button -----
    if act == "system_button":
        act = f"press_{inp.get('button', '').lower()}"

    # ----- normalize open -----
    if act == "open":
        act = "open_app"

    canon = {
        "action_type": act,
        "action_inputs": {},
        "thinking": thinking,
        "raw": obj,
    }

    # ---------- coordinate ----------
    if "coordinate" in inp:
        canon["action_inputs"]["coordinate"] = inp["coordinate"]

    if "start_coordinate" in inp:
        canon["action_inputs"]["start_coordinate"] = inp["start_coordinate"]

    if "end_coordinate" in inp:
        canon["action_inputs"]["end_coordinate"] = inp["end_coordinate"]

    # ---------- text ----------
    if "text" in inp:
        canon["action_inputs"]["content"] = inp["text"]
    elif "value" in inp:
        canon["action_inputs"]["content"] = inp["value"]
    elif "app_name" in inp:
        canon["action_inputs"]["content"] = inp["app_name"]

    # ---------- swipe direction ----------
    if "direction" in inp:
        canon["action_inputs"]["direction"] = inp["direction"]

    # ---------- status ----------
    if "status" in inp:
        canon["action_inputs"]["status"] = inp["status"]

    return canon
