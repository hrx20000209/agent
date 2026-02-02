# MobileAgentE/agents/mai_agent/mai_one_step_agent.py

from dataclasses import dataclass
import copy
import time
import base64
from typing import List, Optional

from PIL import Image

from agents.mai_parser import parse_tagged_text, mai_toolcall_to_action
from agents.mai_prompt import MAI_MOBILE_SYS_PROMPT, build_user_prompt
from typing import Any, Dict, Optional, Tuple, List, Union

from MobileAgentE.controller import (
    tap, swipe, type as adb_type, back, home, switch_app, enter
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

    def parse_action(self, llm_output: str, width: int, height: int):
        llm_output = llm_output.strip()
        parsed = parse_tagged_text(llm_output)
        tool_call = parsed.get("tool_call", None)

        if tool_call is None:
            return None

        action = mai_toolcall_to_action(tool_call)
        action["thinking"] = parsed.get("thinking", "")
        action["tool_call"] = tool_call
        return action

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
        Execute normalized action output.

        action_obj format:
          {
            "action_type": "click"/"drag"/"type"/...,
            "action_inputs": {...},
            "raw": {...}   # optional
          }

        info_pool needs:
          info_pool.width
          info_pool.height
        """
        if not action_obj:
            print("⚠️ No valid action parsed.")
            return None

        action_type = (action_obj.get("action_type") or "").lower()
        args = action_obj.get("action_inputs", {}) or {}

        print(f"[EXEC] {action_type}: {args}")

        # --------- finished ----------
        if action_type in ["finished", "finish", "done", "terminate"]:
            return "finished"

        # --------- wait ----------
        if action_type == "wait":
            sec = float(args.get("seconds", 2))
            time.sleep(sec)
            return f"wait({sec})"

        # --------- system button ----------
        if action_type == "press_back":
            back(self.adb)
            return "press_back"

        if action_type == "press_home":
            home(self.adb)
            return "press_home"

        if action_type == "switch_app":
            switch_app(self.adb)
            return "switch_app"

        if action_type == "enter":
            enter(self.adb)
            return "enter"

        # --------- type ----------
        if action_type == "type":
            text = args.get("content", "")
            adb_type(self.adb, text)
            return f"type({text})"

        # --------- click ----------
        if action_type == "click":
            if "start_box" not in args:
                print("⚠️ click missing start_box")
                return None

            box = _parse_box(args["start_box"])
            x_norm, y_norm = _center_of_box(box)
            x, y = _norm_to_pixel(x_norm, y_norm, info_pool.width, info_pool.height)

            tap(self.adb, x, y)
            return f"click({x},{y})"

        # --------- long press ----------
        if action_type == "long_press":
            if "start_box" not in args:
                print("⚠️ long_press missing start_box")
                return None

            box = _parse_box(args["start_box"])
            x_norm, y_norm = _center_of_box(box)
            x, y = _norm_to_pixel(x_norm, y_norm, info_pool.width, info_pool.height)

            # ADB long press: swipe from (x,y)->(x,y) with duration
            duration_ms = int(args.get("duration_ms", 600))
            # If your controller has long_press(), replace this line.
            swipe(self.adb, x, y, x, y)  # fallback: might not hold duration in your wrapper
            # Better: implement swipe_with_duration in controller
            return f"long_press({x},{y},{duration_ms}ms)"

        # --------- drag / swipe ----------
        if action_type in ["drag", "swipe", "select"]:
            if "start_box" not in args or "end_box" not in args:
                print("⚠️ drag/swipe missing start_box or end_box")
                return None

            sbox = _parse_box(args["start_box"])
            ebox = _parse_box(args["end_box"])

            sx_norm, sy_norm = _center_of_box(sbox)
            ex_norm, ey_norm = _center_of_box(ebox)

            sx, sy = _norm_to_pixel(sx_norm, sy_norm, info_pool.width, info_pool.height)
            ex, ey = _norm_to_pixel(ex_norm, ey_norm, info_pool.width, info_pool.height)

            swipe(self.adb, sx, sy, ex, ey)
            return f"{action_type}({sx},{sy})->({ex},{ey})"

        # --------- open_app (no XML matching) ----------
        if action_type == "open_app":
            # 你现在已经不匹配 XML tree，因此 open_app 没法直接点 app icon
            # 最合理的降级策略：回到 Home，让模型下一步 click app icon
            print("⚠️ open_app is not supported without XML matching. Fallback to HOME.")
            home(self.adb)
            return "press_home"

        print(f"⚠️ Unsupported action_type={action_type}")
        return None