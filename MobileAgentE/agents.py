from dataclasses import dataclass, field

from PIL import Image
import time
import os
import copy
import re
import json

from MobileAgentE.tree import Node
from MobileAgentE.api import encode_image
from MobileAgentE.tree import find_app_icon
from MobileAgentE.controller import tap, swipe, type, back, home, switch_app, enter, save_screenshot_to_file
from MobileAgentE.action_parser import parse_action_to_structure_output
from MobileAgentE.prompt import MOBILE_USE_PROMPT


# -----------------------------------------
# InfoPool（保留你原始结构，不修改）
# -----------------------------------------
@dataclass
class InfoPool:
    instruction: str = ""
    width: int = 1080
    height: int = 2340
    tree: Node = None


# -----------------------------------------
# 工具函数
# -----------------------------------------
def add_chat(role, text, chat_history, image=None):
    """把文本+图片加入消息队列"""
    new = copy.deepcopy(chat_history)
    if image:
        base64_image = encode_image(image)
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    else:
        content = [{"type": "text", "text": text}]
    new.append([role, content])
    return new


# -----------------------------------------
# 核心 Agent（统一版，无继承）
# -----------------------------------------
class OneStepAgent:

    def __init__(self, adb_path):
        self.adb = adb_path

    # -----------------------------
    # 初始化 system prompt
    # -----------------------------
    def init_chat(self):
        system_prompt = (
            "You are a mobile UI control agent. "
            "Each step, you receive a user instruction and a screenshot. "
            "Based on visual understanding, decide ONE atomic action. "
            "Output a JSON ONLY."
        )
        return [["system", [{"type": "text", "text": system_prompt}]]]

    # -----------------------------
    # 构建 prompt（你要求的一步 agent）
    # -----------------------------
    def build_prompt(self, info_pool: InfoPool):
        return MOBILE_USE_PROMPT.format(language="English", instruction=info_pool.instruction)

    # -----------------------------
    # 解析 LLM 输出 → action
    # -----------------------------
    def parse_action(self, llm_output: str, width: int, height: int):
        clean = llm_output.strip()

        parsed = parse_action_to_structure_output(
            clean,
            factor=28,
            origin_resized_height=height,
            origin_resized_width=width,
            model_type="qwen25vl"
        )

        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        return None

    # -----------------------------
    # 执行动作
    # -----------------------------
    def execute_action(self, action_obj, info_pool: InfoPool):

        if not action_obj:
            print("⚠️ No valid action parsed.")
            return

        action_type = action_obj.get("action_type")
        args = action_obj.get("action_inputs", {})

        print(f"[EXEC] {action_type}: {args}")

        if action_type == "open_app":
            start_time = time.time()
            app_name = action_obj.get("app_name") or action_obj.get("text") or action_obj.get("target", "")
            print(f"[Matcher] Trying to open app: {app_name}")

            node = find_app_icon(info_pool.tree, app_name)

            if node:
                # bounds 格式: "[x1,y1][x2,y2]"
                b = node.bounds
                x = (b[0] + b[2]) // 2
                y = (b[1] + b[3]) // 2
                print(f"[Matcher] Found app icon at bounds: {b}, tap=({x},{y})")
                end_time = time.time()
                searching_latency = (end_time - start_time) * 1000
                print(f"[LOG] searching latency: {searching_latency:.3f} ms")
                return self.tap(x, y)

            print("[Matcher] No matching app icon found in XML tree.")
            end_time = time.time()
            searching_latency = (end_time - start_time) * 1000
            print(f"[LOG] searching latency: {searching_latency:.3f} ms")
            return None

        # ---------- Tap ----------
        elif action_type == "click":
            x, y = eval(args["start_box"])
            tap(self.adb, int(x * info_pool.width), int(y * info_pool.height))
            # time.sleep(2)

        # ---------- type ----------
        elif action_type == "type":
            text = args.get("content", "")
            type(self.adb, text)
            # time.sleep(1)

        # ---------- swipe / drag ----------
        elif action_type in ["drag", "select"]:
            sx = eval(args["start_box"])
            ex = eval(args["end_box"])
            tap_x1 = int(((sx[0] + sx[2]) / 2) * info_pool.width)
            tap_y1 = int(((sx[1] + sx[3]) / 2) * info_pool.height)
            tap_x2 = int(((ex[0] + ex[2]) / 2) * info_pool.width)
            tap_y2 = int(((ex[1] + ex[3]) / 2) * info_pool.height)
            swipe(self.adb, tap_x1, tap_y1, tap_x2, tap_y2)
            # time.sleep(2)

        # ---------- Keyboard Enter ----------
        elif action_type == "enter":
            enter(self.adb)
            # time.sleep(1)

        # ---------- Back ----------
        elif action_type == "press_back":
            back(self.adb)
            # time.sleep(1)

        # ---------- Home ----------
        elif action_type == "press_home":
            home(self.adb)
            # time.sleep(1)

        # ---------- Switch app ----------
        elif action_type == "switch_app":
            switch_app(self.adb)
            # time.sleep(1)

        # ---------- Wait ----------
        elif action_type == "wait":
            time.sleep(2)

        else:
            print(f"⚠️ Unsupported action_type={action_type}")

        return action_obj

    # -----------------------------
    # 主入口：单步代理
    # -----------------------------
    def run_step(self, instruction, screenshot_img, width, height, llm_api_func):
        """
        llm_api_func(prompt, image) → LLM输出字符串
        """
        info = InfoPool(instruction=instruction, width=width, height=height)

        chat = self.init_chat()
        user_prompt = self.build_prompt(info)

        chat = add_chat("user", user_prompt, chat, image=screenshot_img)

        # 1) 调 LLM
        llm_output = llm_api_func(chat)

        # 2) 解析
        action = self.parse_action(llm_output, info.width, info.height)

        return action
