"""Differential check against pinned official methods, plus a 3-image smoke call.

Usage: python mai_verify_official.py /path/to/official_navigation.py
Only the official message formatter and response serializer are compiled; no
official agent entrypoint or device operation is executed by this script.
"""
import ast
import base64
import io
import json
from pathlib import Path
import sys
import types
from typing import Any, Dict, List
import copy

from PIL import Image, ImageDraw
import numpy as np
import mai_run as M


def pil_to_base64(image):
  buf = io.BytesIO()
  image.save(buf, format="PNG")
  return base64.b64encode(buf.getvalue()).decode()


tree = ast.parse(Path(sys.argv[1]).read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MAIUINaivigationAgent")
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"_build_messages", "mem2response"}]
namespace = dict(List=List, Dict=Dict, Any=Any, Image=Image, json=json, copy=copy,
                 TrajStep=types.SimpleNamespace, SCALE_FACTOR=999, pil_to_base64=pil_to_base64)
exec(compile(ast.Module(body=methods, type_ignores=[]), "official-formatters", "exec"), namespace)
image = Image.new("RGB", (1080, 2400), "white")
d = ImageDraw.Draw(image)
d.rectangle((280, 1000, 800, 1240), fill="lightblue", outline="black", width=6)
d.text((470, 1100), "START", fill="black", font_size=48)
goal = "Tap the START button."

for count in (0, 1, 2, 5):
  native = M.M.NativeHistory(3)
  steps = []
  for i in range(count):
    cases = [{"action": "wait"}, {"action": "click", "coordinate": [493, 473]},
             {"action": "drag", "start_coordinate": [100, 200], "end_coordinate": [800, 900]},
             {"action": "click", "coordinate": [100, 200, 900, 800]}, {"action": "type", "text": "你好"}]
    args = cases[i]
    native.append(args, f"Step {i}.", image)
    normalized = copy.deepcopy(args)
    for key in ("coordinate", "start_coordinate", "end_coordinate"):
      if key in normalized:
        value = normalized[key]
        if len(value) == 4:
          value = [(value[0] + value[2]) / 2, (value[1] + value[3]) / 2]
        normalized[key] = [v / 999 for v in value]
    steps.append(types.SimpleNamespace(thought=f"Step {i}.", structured_action={"action_json": normalized}))
  official = types.SimpleNamespace(system_prompt=M.PROMPT, history_n=3,
      traj_memory=types.SimpleNamespace(steps=steps), mem2ask_user_response=lambda step: None,
      mem2mcp_response=lambda step: None)
  official.mem2response = types.MethodType(namespace["mem2response"], official)
  images = [image] * min(count + 1, 3)
  expected = namespace["_build_messages"](official, goal, images)
  actual = native.messages(M.PROMPT, goal, image)
  assert actual == expected, f"Official message structure mismatch at {count} history actions"
print("PASS: byte-equivalent messages to official formatters for 0/1/2/5 history actions")
if "--no-call" in sys.argv:
  raise SystemExit(0)

root = Path("/home/pxchen/workspace/services/mai_ui_8b_gpu4_20261024")
logs = []
ep = types.SimpleNamespace(_mai_history=native, env=types.SimpleNamespace(logical_screen_size=image.size),
                          log=lambda name, row: logs.append({"name": name, **row}))
agent = types.SimpleNamespace(vllm=M.OfficialClient(M.CONFIG["api_url"], M.CONFIG["model"], M.CONFIG))
result = M.decide(ep, agent, goal, np.asarray(image))
action = result["action"]
assert action.action_type == "click", action
assert 280 <= action.x <= 800 and 1000 <= action.y <= 1240, action
report = {"official_message_equivalence": True, "image_count": 3,
          "resolution": list(image.size), "history_actions": len(native.steps),
          "action": action.__dict__, "response": result["response"],
          "latency_s": result["model_latency_s"], "requests": logs}
(root / "official_3img_smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "requests"}, ensure_ascii=False))
print("PASS: real MAI request with 3 full-resolution images; no device action executed")
