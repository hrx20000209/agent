"""AndroidWorld BASE/FULL45 with pinned official MAI navigation configuration.

Native structured action history and the last two historical images plus the
current image follow the official client. No GELab prompt/parser is used for
model decisions. FULL45 graph executions are also journaled as native actions.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ["STREAM_ACT"] = "0"
os.environ["STREAM_IO"] = "0"
os.environ.setdefault("SEQ_APPDIR", "/tmp/mai_ui_8b_graph")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import full_run as F
import owl_protocol as M
from mai_client import OfficialClient
from PIL import Image

# Reuse FULL45's "decision ready" scheduling event, but never provide an early
# partial decision. Official GUI-Owl transport remains non-streaming.
F.STREAM = F.ARM == "FULL45"
F.STREAM_IO = False

def configure_empty_statistics():
  """Explicit smoke mode: neutral weights and no learned offline priors."""
  F.EV.w_app = lambda package: lambda token: 1.0
  F.EV.w_ui = lambda token: 1.0
  F.full9.goal_idf = lambda: lambda token: 1.0
  F.full9.rev_prior = lambda: {}

if os.environ.get("MAI_SMOKE_EMPTY_STATS") == "1":
  configure_empty_statistics()

CONFIG_PATH = Path(os.environ.get("MAI_CONFIG", str(F.HERE.parent / "configs/guiowl_8b_official.json")))
CONFIG = json.loads(CONFIG_PATH.read_text())
PROMPT = (CONFIG_PATH.parent / CONFIG["system_prompt_file"]).read_text().rstrip("\n")
if hashlib.sha256(PROMPT.encode()).hexdigest() != CONFIG["system_prompt_sha256"]:
  raise ValueError("Official GUI-Owl system prompt does not match the pinned configuration")
_setup, _make_agent, _commit = F.FullEpisode.setup, F.FullEpisode.make_agent, F.FullEpisode.commit
_run_agent = F.FullEpisode.run_agent


def record(self, action, extras, image=None):
  if not getattr(self, "_mai_active", False) or getattr(self, "_mai_exploring", False) or getattr(self, "_mai_in_execute", False):
    return
  cap = getattr(self, "_mai_cap", None)
  if image is None:
    if cap is None:
      raise RuntimeError("Cannot journal an action without a preceding screenshot")
    image = Image.fromarray(cap["pixels"])
  decision = getattr(self, "_mai_model_decision", None)
  unchanged = decision is not None and action.__dict__ == decision["mai_initial_action"]
  args = decision["parsed"]["native_arguments"] if unchanged else M.native_from_action(action, extras, self.env.logical_screen_size)
  thought = decision["parsed"].get("explain", "") if decision else "Executed by verified graph/system control."
  if unchanged:
    image = decision["mai_image"]
  self._mai_history.append(args, thought, image, "model" if decision else "graph", response=decision["response"] if unchanged else None)
  self.log("mai_executed", {"origin": "model" if decision else "graph", "native_arguments": args,
                            "action": action.__dict__, "capture": cap["cid"] if cap else None})


def make_agent(self):
  self._mai_active = True
  agent = _make_agent(self)
  agent.vllm = OfficialClient(self.args.api_url, self.args.model, CONFIG)
  execute = agent._execute_action

  def execute_native(action, extras):
    self._mai_in_execute = True
    try:
      if extras.get("owl_keycode"):
        result = F.adb_utils.press_keyboard_generic(str(extras["owl_keycode"]), self.env.controller)
        if result.status != F.adb_utils.adb_pb2.AdbResponse.Status.OK:raise RuntimeError("Android rejected key")
      elif extras.get("owl_long_press_seconds") is not None:
        ms=int(extras["owl_long_press_seconds"]*1000)
        F.adb_utils.issue_generic_request(["shell","input","swipe",str(action.x),str(action.y),str(action.x),str(action.y),str(ms)],self.env.controller)
      elif extras.get("menu_keycode") or (action.action_type == F.json_action.UNKNOWN and action.keycode == "KEYCODE_MENU"):
        result = F.adb_utils.press_keyboard_generic("82", self.env.controller)
        if result.status != F.adb_utils.adb_pb2.AdbResponse.Status.OK:
          raise RuntimeError("Android rejected KEYCODE_MENU")
      elif action.action_type == F.json_action.WAIT and "wait_seconds" in extras:
        time.sleep(float(extras["wait_seconds"]))
      else:
        execute(action, extras)
    finally:
      self._mai_in_execute = False
    record(self, action, extras)

  agent._execute_action = execute_native
  class Summaries(list):
    def append(summaries, summary):
      super().append(summary)
      if summary.startswith("[graph] Not filled yet:") and self._mai_history.steps:
        self._mai_history.steps[-1]["results"].append("Tool call result: " + summary)
  agent._summaries = Summaries(agent._summaries)
  return agent


def recovery_feedback(self, text):
  notes = getattr(self, "_mai_recovery_notes", [])
  if not notes or notes[-1] != text:
    notes.append(text)
  self._mai_recovery_notes = notes[-2:]
  self.log("recovery_feedback", {"text": text})


def decide_native(self, agent, goal, pixels):
  image = Image.fromarray(pixels)
  messages = self._mai_history.messages(PROMPT, goal, image)
  if F.ARM == "FULL45" and getattr(self, "_mai_recovery_notes", None):
    messages[-1]["content"].insert(0, {"type": "text", "text": "Runtime observation: " + "\n".join(self._mai_recovery_notes)})
  image_count = sum(1 for msg in messages for item in msg.get("content", []) if item.get("type") == "image_url")
  self.log("mai_requests", {"message_roles": [msg["role"] for msg in messages],
                            "history_actions": len(self._mai_history.steps), "images": image_count, "message_sha256": hashlib.sha256(json.dumps(messages).encode()).hexdigest(),
                            "config": CONFIG, "graph_history_actions": sum(s["origin"] == "graph" for s in self._mai_history.steps)})
  start = time.time()
  errors = []
  for attempt in range(CONFIG["max_retries"]):
    try:
      self.log("mai_attempts", {"attempt": attempt + 1, "t": time.time()})
      response, _, raw = agent.vllm.predict_mm("", [], messages=messages)
      self.log("mai_raw_responses", {"attempt": attempt + 1, "response": response, "usage": raw.get("usage")})
      parsed = M.parse_response(response, self.env.logical_screen_size)
      if parsed["native_arguments"].get("action") == "key":
        code = M.keycode(parsed["native_arguments"].get("text"))
        action = F.json_action.JSONAction(action_type=F.json_action.UNKNOWN, keycode=code)
        call, extras = parsed["tool_call"], {"owl_keycode": code}
      elif parsed["native_arguments"] == {"action": "system_button", "button": "Menu"}:
        action = F.json_action.JSONAction(action_type=F.json_action.UNKNOWN, keycode="KEYCODE_MENU")
        call, extras = parsed["tool_call"], {"menu_keycode": 82}
      else:
        action, call, extras = F.R.gelab_agent.gelab_action_to_json_action(parsed, self.env.logical_screen_size)
      if parsed["native_arguments"].get("action") == "wait":
        extras["wait_seconds"] = float(parsed["native_arguments"].get("time",1))
      if parsed["native_arguments"].get("action") == "long_press" and "time" in parsed["native_arguments"]:
        extras["owl_long_press_seconds"] = float(parsed["native_arguments"]["time"])
      if action.action_type == "answer":
        extras["return_text"] = action.text
      latency = time.time() - start
      F._T_ACTION.append(latency)
      self.log("mai_decisions", {"response": response, "parsed": parsed, "action": action.__dict__,
                                 "latency_s": latency, "attempt": attempt + 1, "usage": raw.get("usage"), "parse_error": None})
      return {"response": response, "parsed": parsed, "action": action, "tool_call": call,
              "extras": extras, "model_latency_s": round(latency, 3), "parse_error": None,
              "mai_image": image, "mai_initial_action": dict(action.__dict__)}
    except Exception as exc:
      errors.append(str(exc))
      self.log("mai_errors", {"attempt": attempt + 1, "error": str(exc)})
  raise RuntimeError("GUI-Owl failed after 3 attempts; no guessed action executed: " + " | ".join(errors))


def decide(self, agent, goal, pixels):
  early = getattr(self, "_early", None)
  try:
    return decide_native(self, agent, goal, pixels)
  finally:
    if early is not None:
      early["ev"].set()


def commit(self, agent, decision, goal):
  self._mai_model_decision = decision
  try:
    return _commit(self, agent, decision, goal)
  finally:
    self._mai_model_decision = None


def setup(self):
  meta = _setup(self)
  self._mai_history = M.NativeHistory(CONFIG["history_n"])
  self._mai_recovery_notes = []
  self._mai_cap = None
  self._mai_active = False
  self._mai_exploring = self._mai_in_execute = False
  capture, tap, back = self.dev.capture, self.dev.tap, self.dev.back

  def capture_native(*args, **kwargs):
    cap = capture(*args, **kwargs)
    self._mai_cap = cap
    return cap
  def tap_native(x, y):
    result = tap(x, y)
    record(self, F.json_action.JSONAction(action_type="click", x=x, y=y), {})
    return result
  def back_native():
    cap = self._mai_cap
    result = back()
    record(self, F.json_action.JSONAction(action_type="navigate_back"), {},
           Image.fromarray(cap["pixels"]) if cap else None)
    return result
  self.dev.capture, self.dev.tap, self.dev.back = capture_native, tap_native, back_native

  launch = F.full7.launch
  def launch_native(env, app):
    result = launch(env, app)
    if env is self.env:
      record(self, F.json_action.JSONAction(action_type="open_app", app_name=app), {})
    return result
  F.full7.launch = launch_native

  generic = F.adb_utils.issue_generic_request
  def generic_native(*pos, **kwargs):
    command = pos[0] if pos else kwargs["args"]
    env = pos[1] if len(pos) > 1 else kwargs["env"]
    result = generic(*pos, **kwargs)
    if env is self.env.controller and list(command[:3]) == ["shell", "input", "swipe"]:
      sx, sy, ex, ey = map(int, command[3:7])
      direction = "right" if ex > sx else "left"
      if abs(ey - sy) > abs(ex - sx):
        direction = "down" if ey > sy else "up"
      record(self, F.json_action.JSONAction(action_type="swipe", direction=direction),
             {"start_coordinate": [sx, sy], "end_coordinate": [ex, ey]})
    return result
  F.adb_utils.issue_generic_request = generic_native

  meta.update(decision_backend="gui-owl-1.5-official", early_action=False, decision_ready_signal="full_response", coordinate_format="qwen-vl-999",
              model=self.args.model, mai_config=CONFIG, owl_config=CONFIG)
  meta["upstream_commit"] = "81a1531758da249c2acb74baa761d2da65e01b1c"
  meta["experiment_runtime"] = {"algorithm": "upstream FULL45 plus explicit budget boundary guard", "transport": "GUI-Owl official-format nonstreaming adapter", "custom_recovery_guard": False, "budget": "int(10*complexity)"}
  meta["graph_execution_enabled"] = F.ARM.startswith("FULL") and not F.EXEC_OFF
  meta["exploration_enabled"] = F.ARM.startswith("FULL") and not F.NOEXP
  meta["allocated_model_calls"] = int(10 * self.task.complexity)
  (self.out / "meta.json").write_text(F.R.jdump(meta))
  return meta


@contextlib.contextmanager
def exploration(ep):
  previous = getattr(ep, "_mai_exploring", False)
  ep._mai_exploring = True
  try:
    yield
  finally:
    ep._mai_exploring = previous


_window, _stuck_window = F.FullEpisode.window, F.full7.StuckExplorer.window
def window(self, *args, **kwargs):
  with exploration(self):
    return _window(self, *args, **kwargs)
def stuck_window(self, *args, **kwargs):
  with exploration(self.ep):
    return _stuck_window(self, *args, **kwargs)


def run_agent(self, *args, **kwargs):
  try:
    return _run_agent(self, *args, **kwargs)
  finally:
    self._mai_active = False


F.FullEpisode.make_agent, F.FullEpisode.decide = make_agent, decide
F.FullEpisode.commit, F.FullEpisode.setup, F.FullEpisode.window = commit, setup, window
F.FullEpisode.run_agent = run_agent
F.FullEpisode.max_steps = lambda self: int(10 * self.task.complexity)
F.full7.StuckExplorer.window = stuck_window
F.ADB = F.R.ADB = shutil.which("adb") or F.R.ADB

if __name__ == "__main__":
  if "--model" not in sys.argv:
    sys.argv += ["--model", CONFIG["model"]]
  if "--api_url" not in sys.argv:
    sys.argv += ["--api_url", CONFIG["api_url"]]
  F.R.main()
