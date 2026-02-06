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

from agents.mai_parser import parse_tagged_text, mai_toolcall_to_action
from agents.mai_prompt import MAI_MOBILE_SYS_PROMPT, build_user_prompt
from typing import Any, Dict, Optional, Tuple, List, Union

from MobileAgentE.controller import (
    tap, swipe, type as adb_type, back, home, switch_app, enter, launch_app
)

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


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _norm_to_pixel(x_norm: float, y_norm: float, width: int, height: int) -> Tuple[int, int]:
    x_norm = _clamp01(x_norm)
    y_norm = _clamp01(y_norm)
    x = int(x_norm * width)
    y = int(y_norm * height)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@dataclass
class InfoPool:
    instruction: str = ""
    width: int = 1080
    height: int = 2340
    tree: object = None


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

    def __init__(self, adb_path):
        self.adb = adb_path

    def init_chat(self):
        return [["system", [{"type": "text", "text": MAI_MOBILE_SYS_PROMPT}]]]

    def build_prompt(self, info_pool: InfoPool, history: str):
        return build_user_prompt(info_pool.instruction, history)

    def run_step(self, instruction, screenshot_img, width, height, history, llm_api_func, clues, scale=1.0):
        info = InfoPool(instruction=instruction, width=width, height=height)

        orig_width, orig_height = width, height

        if scale != 1.0:
            new_w = int(width / scale)
            new_h = int(height / scale)
            width, height = new_w, new_h

        chat = self.init_chat()
        user_prompt = self.build_prompt(info, history) + ("\n\n" + clues if clues else "")

        print(user_prompt)

        chat = add_chat("user", user_prompt, chat, image=screenshot_img)

        llm_output = llm_api_func(chat)
        action_obj = self.parse_action(llm_output, width, height)
        return action_obj

    def execute_action(self, action_obj: Dict[str, Any], info_pool) -> Optional[str]:
        """
        Execute canonical action schema.

        action_obj format:
        {
            "action_type": "...",
            "action_inputs": {...}
        }
        """

        if not action_obj:
            print("⚠️ No valid action.")
            return None

        action_type = action_obj.get("action_type", "").lower()
        args = action_obj.get("action_inputs", {}) or {}

        print(f"[EXEC] {action_type} → {args}")

        W, H = info_pool.width, info_pool.height

        # ==============================
        # 🟢 TERMINATE
        # ==============================
        if action_type in ["finished", "done", "terminate"]:
            return "finished"

        # ==============================
        # 🟡 WAIT
        # ==============================
        if action_type == "wait":
            sec = float(args.get("seconds", 1))
            time.sleep(sec)
            return f"wait({sec})"

        # ==============================
        # 🔵 SYSTEM BUTTONS
        # ==============================
        if action_type == "press_back":
            back(self.adb)
            return "press_back"

        if action_type == "press_home":
            home(self.adb)
            return "press_home"

        if action_type == "press_enter":
            enter(self.adb)
            return "press_enter"

        # ==============================
        # 🟣 TYPE
        # ==============================
        if action_type == "type":
            text = args.get("content", "")
            adb_type(self.adb, text)
            return f"type({text})"

        # ==============================
        # 🔴 CLICK
        # ==============================
        if action_type == "click":
            coord = args.get("coordinate")
            if coord is None:
                print("⚠️ click missing coordinate")
                return None

            x_norm, y_norm = coord
            x, y = _norm_to_pixel(x_norm, y_norm, W, H)
            tap(self.adb, x, y)
            return f"click({x},{y})"

        # ==============================
        # 🟠 LONG PRESS
        # ==============================
        if action_type == "long_press":
            coord = args.get("coordinate")
            if coord is None:
                return None

            x_norm, y_norm = coord
            x, y = _norm_to_pixel(x_norm, y_norm, W, H)
            duration_ms = int(args.get("duration_ms", 600))
            swipe(self.adb, x, y, x, y)
            return f"long_press({x},{y})"

        # ==============================
        # 🟡 SWIPE
        # ==============================
        if action_type in ["swipe", "drag"]:
            s = args.get("start_coordinate") or args.get("coordinate")
            e = args.get("end_coordinate")

            if not s:
                print("⚠️ swipe missing start coordinate")
                return None

            if not e:
                # directional swipe fallback
                direction = args.get("direction", "down")
                sx_norm, sy_norm = s
                sx, sy = _norm_to_pixel(sx_norm, sy_norm, W, H)

                if direction == "down":
                    ex, ey = sx, min(H - 1, sy + 400)
                elif direction == "up":
                    ex, ey = sx, max(0, sy - 400)
                elif direction == "left":
                    ex, ey = max(0, sx - 400), sy
                else:
                    ex, ey = min(W - 1, sx + 400), sy
            else:
                sx_norm, sy_norm = s
                ex_norm, ey_norm = e
                sx, sy = _norm_to_pixel(sx_norm, sy_norm, W, H)
                ex, ey = _norm_to_pixel(ex_norm, ey_norm, W, H)

            swipe(self.adb, sx, sy, ex, ey)
            return f"swipe({sx},{sy}→{ex},{ey})"

        # ==============================
        # 🟢 OPEN APP
        # ==============================
        if action_type == "open_app":
            app_name = args.get("content")
            app = launch_app(self.adb, app_name)
            return app

        print(f"⚠️ Unsupported action_type={action_type}")
        return None


    def parse_action(self, llm_output: str, width: int, height: int):
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
                return _canonicalize_action(obj)
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
                    return _canonicalize_action(obj)

        # ===============================
        # 3️⃣ Try: raw JSON in text
        # ===============================
        try:
            first = llm_output.find("{")
            last = llm_output.rfind("}")
            if first != -1 and last != -1:
                obj = json.loads(llm_output[first:last+1])
                return _canonicalize_action(obj)
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

            return obj
        except Exception:
            pass

        print("⚠️ parse_action failed, fallback to wait")
        return {
            "action_type": "wait",
            "action_inputs": {"seconds": 1},
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
        act = args.get("action", "").lower()
        inp = args
        thinking = None

    else:
        return None

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

    # ---------- swipe direction ----------
    if "direction" in inp:
        canon["action_inputs"]["direction"] = inp["direction"]

    # ---------- status ----------
    if "status" in inp:
        canon["action_inputs"]["status"] = inp["status"]

    return canon
