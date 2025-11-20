import os
import time
import json
import requests

from MobileAgentE.text_localization import ocr
from MobileAgentE.icon_localization import det
from MobileAgentE.controller import get_screenshot, type as type_text, tap, enter, swipe, back, home, switch_app
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tools import SimplePerceptor, SimpleLLMWrapper
from MobileAgentE.agents import extract_json_object


# ================= 基础配置 =================
ADB_PATH = os.environ.get("ADB_PATH", default="adb")

LLM_MODEL = "qwen-vl-max"   # 可改成 qwen-vl-max
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-1aa21ba323d044a092e3579753ec1548"   # 填入真实 key

SCREENSHOT_DIR = "screenshot"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# ================= Planning (LLM) =================
def planning(instruction, llm_wrapper, elements):
    # 构建 prompt，要求严格输出 JSON
    content = f"Task: {instruction}\nScreen Elements:\n"
    for i, e in enumerate(elements):
        text = e.get("text", "")
        coords = e.get("coordinates", [])
        content += f"{i}. {text} at {coords}\n"
    content += (
        "\n### Action ###\n"
        "Choose ONLY ONE action. "
        "You must return a VALID JSON object, nothing else, no explanation, no code block.\n"
        "Format: {\"name\": \"Tap|Swipe|Type|Home|Back|Enter|Wait|Switch_App|Open_App\", \"arguments\": {...}}\n"
        "Examples:\n"
        "{\"type\":\"Tap\", \"arguments\":{\"x\":100, \"y\":200}}\n"
        "{\"type\":\"Swipe\", \"arguments\":{\"x1\":500, \"y1\":1500, \"x2\":500, \"y2\":500}}\n"
        "{\"type\":\"Type\", \"arguments\":{\"text\":\"hello\"}}\n"
        "{\"type\":\"Home\", \"arguments\":null}\n"
    )

    # 调用 LLM
    result, _, raw = llm_wrapper.predict(content)
    result = result.strip()

    # 尝试解析 JSON
    action_object = extract_json_object(result, json_type="dict")
    if action_object is None:
        print("⚠️ Failed to parse JSON, fallback. LLM output:", result)
        action_object = {"type": "Tap", "arguments": {"x": 100, "y": 100}}

    return action_object


# ================= Operation =================
def operation(action, adb_path="adb", **kwargs):
    """
    执行 LLM 规划的动作
    action: dict, e.g.
      {"type": "tap", "x": 200, "y": 300}
      {"type": "text", "content": "hello"}
      {"type": "swipe", "x1": 100, "y1": 200, "x2": 300, "y2": 200}
      {"type": "back"}
      {"type": "home"}
      {"type": "open_app", "app_name": "Settings"}
    """
    action_type = action.get("type", "").lower()

    if action_type == "tap":
        x, y = int(action["x"]), int(action["y"])
        tap(adb_path, x, y)
        time.sleep(1)

    elif action_type == "text":
        text = action.get("content", "")
        type_text(adb_path, text)
        time.sleep(1)

    elif action_type == "swipe":
        swipe(adb_path,
              int(action["x1"]), int(action["y1"]),
              int(action["x2"]), int(action["y2"]))
        time.sleep(1)

    elif action_type == "back":
        back(adb_path)
        time.sleep(1)

    elif action_type == "home":
        home(adb_path)
        time.sleep(1)

    elif action_type == "enter":
        enter(adb_path)
        time.sleep(1)

    elif action_type == "switch_app":
        switch_app(adb_path)
        time.sleep(1)

    elif action_type == "open_app":
        screenshot_file = kwargs.get("screenshot_file", "./screenshot/screenshot.jpg")
        ocr_detection = kwargs.get("ocr_detection", None)
        ocr_recognition = kwargs.get("ocr_recognition", None)
        app_name = action.get("app_name", "").strip()

        if not app_name or ocr_detection is None or ocr_recognition is None:
            print("⚠️ Missing parameters for Open_App:", action)
            return

        text, coordinate = ocr(screenshot_file, ocr_detection, ocr_recognition)
        for ti in range(len(text)):
            if app_name == text[ti]:
                name_coordinate = [
                    int((coordinate[ti][0] + coordinate[ti][2]) / 2),
                    int((coordinate[ti][1] + coordinate[ti][3]) / 2)
                ]
                tap(adb_path, name_coordinate[0],
                    name_coordinate[1] - int(coordinate[ti][3] - coordinate[ti][1]))
                break
        if app_name in ['Fandango', 'Walmart', 'Best Buy']:
            time.sleep(10)  # 特殊 app 给更多加载时间
        time.sleep(10)

    else:
        print("⚠️ Unknown action:", action)


# ================= 主循环 =================
def run_task(instruction, perceptor, llm_wrapper, max_steps=10):
    for step in range(max_steps):
        print(f"\n=== Step {step} ===")
        screenshot_file = f"./screenshot/screenshot.jpg"
        get_screenshot(ADB_PATH)

        elements, w, h = perceptor.get_perception_infos()
        print("Perception:", elements)

        action = planning(instruction, llm_wrapper, elements)
        print("Planned Action:", action)

        operation(action)
        time.sleep(3)


if __name__ == "__main__":
    perceptor = SimplePerceptor(adb_path=ADB_PATH)
    llm_wrapper = SimpleLLMWrapper(
        api_key=API_KEY,
        base_url=API_URL,
        model=LLM_MODEL
    )
    Task = "Open settings and turn on airplane mode"
    run_task(Task, perceptor, llm_wrapper, max_steps=10)
