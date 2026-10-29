"""MAI-UI native tool calls -> the existing runner's absolute action contract.

This module has no Android imports, so the wire protocol can be tested without
an emulator. Coordinates follow the repository MAI adapter's 0..999 convention.
"""
import ast
import base64
import copy
import io
import json
import math
import re


def load_system_prompt(path):
  """Read the exact repository MAI prompt without importing Android modules."""
  values = {}
  for node in ast.parse(path.read_text()).body:
    if not isinstance(node, ast.Assign):
      continue
    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
    if not set(names) & {"AVAILABLE_APPS", "MAI_MOBILE_SYS_PROMPT"}:
      continue
    values[names[0]] = eval(compile(ast.Expression(node.value), str(path), "eval"),
                            {"__builtins__": {}, "json": json}, values)
  return values["MAI_MOBILE_SYS_PROMPT"]


def image_content(image):
  buf = io.BytesIO()
  image.convert("RGB").save(buf, format="PNG")
  return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


class NativeHistory:
  """Official navigation message layout: all actions, last N-1 images + current."""
  def __init__(self, history_n=3):
    if history_n < 1:
      raise ValueError("history_n must be positive")
    self.history_n = history_n
    self.steps = []

  def append(self, args, thinking, image, origin="model"):
    # Match parse_action_to_structure_output + mem2response in the pinned
    # official client, including its treatment of drag endpoint history.
    history_args = copy.deepcopy(args)
    for key in ("coordinate", "start_coordinate", "end_coordinate"):
      if key in history_args:
        value = history_args[key]
        if len(value) == 4:
          value = [(value[0] + value[2]) / 2, (value[1] + value[3]) / 2]
        history_args[key] = [v / 999 for v in value]
    if "coordinate" in history_args:
      history_args["coordinate"] = [int(v * 999) for v in history_args["coordinate"]]
    response = "<thinking>\n" + thinking + "\n</thinking>\n<tool_call>\n" + json.dumps(
        {"name": "mobile_use", "arguments": history_args}, separators=(",", ":")) + "\n</tool_call>"
    self.steps.append({"response": response, "image": image_content(image), "origin": origin, "results": []})
    for step in self.steps[:max(0, len(self.steps) - (self.history_n - 1))]:
      step["image"] = None

  def messages(self, prompt, goal, image):
    messages = [{"role": "system", "content": [{"type": "text", "text": prompt}]},
                {"role": "user", "content": [{"type": "text", "text": goal}]}]
    start = max(0, len(self.steps) - (self.history_n - 1))
    for i, step in enumerate(self.steps):
      if i >= start:
        if step["image"] is None:
          raise RuntimeError("Missing screenshot for recent executed action")
        messages.append({"role": "user", "content": [step["image"]]})
      messages.append({"role": "assistant", "content": [{"type": "text", "text": step["response"]}]})
      for result in step["results"]:
        messages.append({"role": "user", "content": [{"type": "text", "text": result}]})
    messages.append({"role": "user", "content": [image_content(image)]})
    return messages


def parse_response(response, screen_size):
  """Require one complete native call; reject malformed or ambiguous decisions."""
  if "</think>" in response and "</thinking>" not in response:
    response = "<thinking>" + response.replace("</think>", "</thinking>")
  calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.S)
  if len(calls) != 1:
    raise ValueError("Expected exactly one complete MAI tool_call")
  call = json.loads(calls[0])
  if not isinstance(call, dict) or call.get("name") != "mobile_use":
    raise ValueError("Expected mobile_use")
  args = copy.deepcopy(call.get("arguments"))
  if not isinstance(args, dict):
    raise ValueError("Tool arguments must be an object")
  action = args.get("action")
  allowed = {"click", "long_press", "type", "swipe", "open", "drag", "system_button", "wait", "terminate", "answer"}
  if action not in allowed:
    raise ValueError(f"Unsupported MAI action: {action!r}")
  width, height = screen_size
  native_args = copy.deepcopy(args)

  def point(value):
    if not isinstance(value, list) or len(value) not in (2, 4):
      raise ValueError("Expected a coordinate pair or bbox")
    if not all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 999 for x in value):
      raise ValueError("MAI coordinates must be finite numbers in 0..999")
    if len(value) == 4:
      value = [(value[0] + value[2]) / 2, (value[1] + value[3]) / 2]
    return [min(width - 1, int(value[0] / 999 * width)),
            min(height - 1, int(value[1] / 999 * height))]

  if action in {"click", "long_press"} and "coordinate" not in args:
    raise ValueError("Pointer action missing coordinate")
  for key in ("coordinate", "start_coordinate", "end_coordinate"):
    if key in args:
      args[key] = point(args[key])
  if action == "open":
    if not isinstance(args.get("text"), str) or not args["text"].strip():
      raise ValueError("Open action missing app name")
    args["action"] = "open_app"
  if action in {"type", "answer"} and not isinstance(args.get("text"), str):
    raise ValueError("Text action missing text")
  if action == "terminate" and args.get("status") not in {"success", "fail"}:
    raise ValueError("Terminate requires success or fail")
  if action == "system_button" and args.get("button") not in {"back", "home", "menu", "enter"}:
    raise ValueError("Unsupported system button")
  if action == "drag":
    if not all(k in args for k in ("start_coordinate", "end_coordinate")):
      raise ValueError("Drag missing endpoints")
    args["action"] = "swipe"
  if action == "swipe":
    direction = args.get("direction")
    if direction not in {"up", "down", "left", "right"}:
      raise ValueError("Invalid swipe direction")
    if "coordinate" in args:
      # Preserve targeted swipes; the generic converter otherwise drops the start.
      x, y = args.pop("coordinate")
      dx, dy = {"up": (0, -height * .4), "down": (0, height * .4),
                "left": (-width * .4, 0), "right": (width * .4, 0)}[direction]
      args["start_coordinate"] = [x, y]
      args["end_coordinate"] = [max(0, min(width - 1, int(x + dx))), max(0, min(height - 1, int(y + dy)))]
  thinking = re.search(r"<thinking>(.*?)</thinking>", response, re.S)
  summary = json.dumps(args, ensure_ascii=False)
  return {"action": "__TOOL_CALL__", "tool_call": {"name": "mobile_use", "arguments": args},
          "summary": summary, "explain": thinking.group(1).strip() if thinking else "",
          "native_arguments": native_args}


def native_from_action(action, extras, screen_size):
  """Encode actions actually executed by FULL45 into MAI's native history."""
  width, height = screen_size
  def point(x, y):
    return [max(0, min(999, int(x / width * 999))), max(0, min(999, int(y / height * 999)))]
  kind = action.action_type
  if extras.get("menu_keycode") or (kind == "unknown" and action.keycode == "KEYCODE_MENU"):
    return {"action": "system_button", "button": "menu"}
  if kind in {"click", "long_press", "double_tap"}:
    if kind == "double_tap":
      raise ValueError("Official AndroidWorld prompt does not advertise double_tap")
    return {"action": kind, "coordinate": point(action.x, action.y)}
  if kind == "input_text":
    return {"action": "type", "text": action.text}
  if kind == "open_app":
    return {"action": "open", "text": action.app_name}
  if kind in {"navigate_back", "navigate_home", "keyboard_enter"}:
    return {"action": "system_button", "button": {"navigate_back": "back", "navigate_home": "home", "keyboard_enter": "enter"}[kind]}
  if kind in {"swipe", "scroll"}:
    if extras.get("start_coordinate") and extras.get("end_coordinate"):
      return {"action": "drag", "start_coordinate": point(*extras["start_coordinate"]), "end_coordinate": point(*extras["end_coordinate"])}
    return {"action": "swipe", "direction": action.direction}
  if kind == "status":
    return {"action": "terminate", "status": "fail" if action.goal_status == "infeasible" else "success"}
  if kind == "answer":
    return {"action": "answer", "text": action.text or ""}
  if kind == "wait":
    return {"action": "wait"}
  raise ValueError(f"Cannot represent executed action in MAI history: {kind}")
