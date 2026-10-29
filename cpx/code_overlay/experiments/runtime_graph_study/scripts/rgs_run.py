"""Runtime-graph study runner: one episode per invocation.

Modes
  agent          GELAB-ZERO-4B (LabServer vLLM) solves the task. If --explore,
                 the model request runs in a background thread while a
                 model-free explorer probes safe clickable slots of the SAME
                 screen the model is looking at, rolls back, and verifies the
                 restore before the model's action is executed.
  replay_strict  Re-execute a reference trajectory with recorded coordinates.
  replay_struct  Re-execute it relocating each target by exact selector tiers
                 (no weighted similarity); optional --recover hands the episode
                 to the agent at the first divergence (separately labelled).

Every capture is stored as a JPEG + gzipped raw a11y dump; every executed
action, probe, rollback and replay decision is a JSONL row. The Explorer never
calls a model. Task success is read only after the episode, for evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
import random
import sys
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("ANDROID_WORLD_A11Y_METHOD", "fast_provider")
os.environ.setdefault("ANDROID_WORLD_FAST_A11Y_SOCKET_PORT", "8765")
os.environ.setdefault("ANDROID_WORLD_LLM_NO_PREFIX_CACHE", "1")

from android_world import registry, suite_utils, shared_state as SS  # noqa: E402
from android_world.agents import gelab_agent, infer, seeact_utils  # noqa: E402
from android_world.env import adb_utils, env_launcher, json_action  # noqa: E402

import rgs_ui as U  # noqa: E402

ADB = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")


def jdump(obj: Any) -> str:
  def default(o):
    if dataclasses.is_dataclass(o):
      return dataclasses.asdict(o)
    if isinstance(o, (np.integer,)):
      return int(o)
    if isinstance(o, (np.floating,)):
      return float(o)
    if isinstance(o, np.ndarray):
      return o.tolist()
    return repr(o)
  return json.dumps(obj, ensure_ascii=False, default=default)


# ======================================================================
class Device:
  """Thin capture/act layer over the existing AndroidWorld env."""

  def __init__(self, env, out: Path):
    self.env = env
    self.ctrl = env.controller
    self.out = out
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "a11y").mkdir(parents=True, exist_ok=True)
    self._last_payload: str | None = None
    orig = self.ctrl._fast_a11y_provider_dump

    def robust():
      try:
        return orig()
      except RuntimeError:
        # AndroidEnv setup / task init can drop adb forwards; restore and retry.
        subprocess_forward()
        time.sleep(0.2)
        return orig()

    def wrapped():
      p = robust()
      self._last_payload = p
      return p
    self.ctrl._fast_a11y_provider_dump = wrapped
    self._dump = robust
    self.n = 0
    self.screen = tuple(env.logical_screen_size)

  def activity(self) -> str:
    try:
      return adb_utils.get_current_activity(self.ctrl)[0] or ""
    except Exception:  # pylint: disable=broad-except
      return ""

  def quick_ui(self) -> U.UIState:
    payload = json.loads(self._dump())
    return U.parse_flat(payload, activity="", screen=self.screen)

  def settle(self, timeout: float = 3.0, interval: float = 0.25) -> float:
    """Bounded observation: wait until two consecutive dumps agree."""
    t0 = time.time()
    prev = None
    while time.time() - t0 < timeout:
      try:
        ui = self.quick_ui()
        sig = (ui.exact_sig, ui.content_sig)
      except Exception:  # pylint: disable=broad-except
        sig = None
      if sig is not None and sig == prev:
        break
      prev = sig
      time.sleep(interval)
    return time.time() - t0

  def capture(self, tag: str, settle: bool = True, pause: float | None = None,
              full_res: bool = False) -> dict[str, Any]:
    t0 = time.time()
    t_settle = 0.0
    if pause:
      time.sleep(pause)
    if settle:
      t_settle = self.settle()
    t1 = time.time()
    state = self.env.get_state(wait_to_stabilize=False)
    payload_s = self._last_payload or self._dump()
    payload = json.loads(payload_s)
    act = self.activity()
    ui = U.parse_flat(payload, activity=act, screen=self.screen)
    self.n += 1
    cid = f"{self.n:04d}_{tag}"
    img = Image.fromarray(state.pixels)
    if not full_res:
      img = img.resize((img.width // 2, img.height // 2))
    img.convert("RGB").save(self.out / "img" / f"{cid}.jpg", quality=82)
    with gzip.open(self.out / "a11y" / f"{cid}.json.gz", "wt", encoding="utf-8") as f:
      json.dump({"activity": act, "payload": payload}, f, ensure_ascii=False)
    return {
        "cid": cid, "t": t0, "settle_s": round(t_settle, 3),
        "capture_s": round(time.time() - t1, 3), "pixels": state.pixels, "ui": ui,
        "summary": ui.summary(),
    }

  def tap(self, x: int, y: int):
    adb_utils.tap_screen(int(x), int(y), self.ctrl)

  def back(self):
    adb_utils.press_back_button(self.ctrl)


def strip(cap: dict[str, Any]) -> dict[str, Any]:
  return {k: v for k, v in cap.items() if k not in ("pixels", "ui")}


def selected_sibling(ui: U.UIState, node: U.Node) -> U.Node | None:
  """A different, currently selected clickable node under the same parent or
  grandparent container as the probed node (tabs, bottom navigation)."""
  anc = []
  cur = node.parent
  while cur is not None and len(anc) < 2:
    anc.append(cur)
    cur = ui.nodes[cur].parent
  for n in ui.app_nodes():
    if n.idx == node.idx or not n.selected or not n.visible:
      continue
    # The selected tab is often not clickable itself (it becomes clickable
    # once another tab is active): accept the outermost selected node whose
    # parent is one of the probed node's containers.
    m = n
    while m.parent is not None and ui.nodes[m.parent].selected:
      m = ui.nodes[m.parent]
    if m.idx == node.idx:
      continue
    if m.parent in anc:
      return m
  return None


def same_state(a: U.UIState, b: U.UIState) -> bool:
  return a.struct_sig == b.struct_sig and a.content_sig == b.content_sig


# ======================================================================
class Graph:
  """Online view of the persistent observation graph.

  Raw transition observations are append-only (observations.jsonl); this
  index is rebuilt from them at start, so later matching changes can rebuild
  the graph offline from the same evidence.
  """

  def __init__(self, path: Path | None, app_memory: Path | None = None):
    self.path = path
    self.app_memory = app_memory
    self.app_unrestorable: set[tuple] = set()
    self.nodes: dict[str, dict[str, Any]] = {}
    self.slots: dict[tuple[str, str], list[dict[str, Any]]] = {}
    self.layout: dict[str, set[str]] = {}
    self.unrestorable_families: set[tuple] = set()
    self._offset = 0
    self.refresh()

  def refresh(self):
    if self.app_memory:
      self.app_unrestorable.update(tuple(row["key"]) for row in SS.rows(self.app_memory))
    if self.path:
      rows, offset = SS.read_delta(self.path, self._offset)
      for row in rows:
        self._index(row)
      self._offset = offset

  def _node(self, s: dict[str, Any], instance: str):
    n = self.nodes.setdefault(s["struct_sig"], {
        "struct_sig": s["struct_sig"], "layout_sig": s["layout_sig"],
        "activity": s["activity"], "instances": set()})
    n["instances"].add(instance)
    self.layout.setdefault(s["layout_sig"], set()).add(s["struct_sig"])

  def _index(self, obs: dict[str, Any]):
    self._node(obs["src"], obs["instance"])
    if obs.get("dst"):
      self._node(obs["dst"], obs["instance"])
    self.slots.setdefault((obs["src"]["struct_sig"], obs["slot_id"]), []).append(obs)
    sel = obs.get("selector") or {}
    if obs.get("origin") == "explore" and not obs.get("rollback_ok", True) and sel.get("in_list") and sel.get("rid"):
      self.unrestorable_families.add((obs["src"]["struct_sig"], sel["cls"], sel["rid"]))

  @staticmethod
  def app_key(pkg: str, sel: dict[str, Any]) -> tuple:
    return (pkg, sel.get("cls", ""), sel.get("rid", ""), sel.get("label", ""))

  def add(self, obs: dict[str, Any]):
    if obs.get("origin") == "explore" and not obs.get("rollback_ok", True) and self.app_memory:
      k = self.app_key(obs["src"]["package"], obs.get("selector") or {})
      if (k[2] or k[3]) and k not in self.app_unrestorable:
        self.app_unrestorable.add(k)
        SS.append_json(self.app_memory, {"key": list(k), "instance": obs["instance"], "src_cid": obs.get("src_cid")})
    if self.path:
      SS.append_json(self.path, obs)
      self.refresh()
    else:
      self._index(obs)

  def match(self, ui: U.UIState) -> tuple[str, list[str]]:
    if ui.struct_sig in self.nodes:
      return "struct", [ui.struct_sig]
    others = sorted(self.layout.get(ui.layout_sig, set()))
    if others:
      return "ambiguous", others
    return "new", []


# ======================================================================
@dataclasses.dataclass
class ExploreCfg:
  enabled: bool = False
  budget_s: float = 6.0
  max_probes: int = 6
  max_depth: int = 1
  revalidate: bool = True
  graph_restore: bool = True


class Explorer:
  def __init__(self, dev: Device, graph: Graph, cfg: ExploreCfg, instance: str, log):
    self.dev, self.graph, self.cfg, self.instance, self.log = dev, graph, cfg, instance, log
    self.disabled_reason = ""
    self.n_probes = 0

  def plan(self, ui: U.UIState) -> tuple[list[tuple[int, U.Slot]], dict[str, int]]:
    slots = U.enumerate_slots(ui)
    stats = {"slots": len(slots), "safe": 0, "skip_known_same_instance": 0,
             "discover": 0, "revalidate": 0}
    plan = []
    for s in slots:
      if not s.safe:
        continue
      stats["safe"] += 1
      obs = self.graph.slots.get((ui.struct_sig, s.slot_id), [])
      if Graph.app_key(ui.package, s.sel) in self.graph.app_unrestorable:
        # learned in another task of the same app
        stats["skip_app_unrestorable"] = stats.get("skip_app_unrestorable", 0) + 1
        continue
      fam = (ui.struct_sig, s.sel["cls"], s.sel["rid"]) if s.sel["in_list"] and s.sel["rid"] else None
      if fam is not None and fam in self.graph.unrestorable_families:
        # A sibling row with the same handler (class + resource-id in the
        # same list) was observed to be unrestorable: generalize.
        stats["skip_family_unrestorable"] = stats.get("skip_family_unrestorable", 0) + 1
        continue
      if any(o.get("origin") == "explore" and not o.get("rollback_ok", True) for o in obs):
        # Memory makes exploration safer: a slot whose rollback was observed
        # to fail is never probed again.
        stats["skip_known_unrestorable"] = stats.get("skip_known_unrestorable", 0) + 1
        continue
      if not obs:
        plan.append((0, s))
        stats["discover"] += 1
      elif any(o["instance"] == self.instance for o in obs):
        stats["skip_known_same_instance"] += 1
      elif self.cfg.revalidate:
        plan.append((1, s))
        stats["revalidate"] += 1
    plan.sort(key=lambda p: p[0])  # stable: keeps (y, x) order inside a tier
    return plan, stats

  def run_window(self, pre: dict[str, Any], step: int, deadline_s: float) -> dict[str, Any]:
    """Probe from the state the model is reasoning about, within budget."""
    t0 = time.time()
    rec = {"step": step, "probes": [], "stop": ""}
    if self.disabled_reason:
      rec["stop"] = "disabled:" + self.disabled_reason
      rec["window_s"] = 0.0
      return rec
    ui0 = pre["ui"]
    plan, stats = self.plan(ui0)
    rec["plan_stats"] = stats
    rec["graph_match"] = self.graph.match(ui0)[0]
    for prio, slot in plan:
      if time.time() - t0 >= deadline_s:
        rec["stop"] = "budget"
        break
      if len(rec["probes"]) >= self.cfg.max_probes:
        rec["stop"] = "max_probes"
        break
      p = self.probe(pre, slot, prio, step, depth=1)
      rec["probes"].append(p)
      if not p["rollback_ok"]:
        rec["stop"] = "rollback_failed"
        self.disabled_reason = "rollback_failed@step%d" % step
        break
    else:
      rec["stop"] = rec["stop"] or "plan_exhausted"
    rec["window_s"] = round(time.time() - t0, 3)
    return rec

  def _edges_from(self, src: str):
    for (s, sid), lst in self.graph.slots.items():
      if s != src:
        continue
      for o in lst:
        sel = o.get("selector")
        act = (o.get("action") or {}).get("action_type", "click")
        if not sel or act != "click" or not o.get("dst"):
          continue
        if o.get("origin") == "explore" and not o.get("rollback_ok", True):
          pass  # the forward edge itself was observed; usable
        yield sid, sel, o["dst"]["struct_sig"]

  def restore_via_graph(self, cur: U.UIState, target: U.UIState, step: int):
    """Model-free restore: replay an already observed edge path (<=2 hops)
    from the current state back to the source state; every hop relocates
    its target by exact selector tiers and is verified."""
    tgt = target.struct_sig
    paths = []
    for sid, sel, d in self._edges_from(cur.struct_sig):
      if d == tgt:
        paths.append([(sid, sel)])
    if not paths:
      for sid, sel, d in self._edges_from(cur.struct_sig):
        for sid2, sel2, d2 in self._edges_from(d):
          if d2 == tgt:
            paths.append([(sid, sel), (sid2, sel2)])
    if not paths:
      return False, None, None
    path = paths[0]
    live = cur
    chk = None
    for sid, sel in path:
      if U.UNSAFE_LABEL_RE.search(" ".join([sel.get("label", ""), sel.get("desc", ""), sel.get("rid", "").replace("_", " ")])):
        return False, chk, [p[0] for p in path] + ["unsafe_hop"]
      cat, node, _ = U.match_element(sel, live)
      if node is None or cat not in ("exact_selector", "stable_hierarchy"):
        return False, chk, [p[0] for p in path] + ["hop_unresolved:" + cat]
      self.dev.tap(*node.center)
      chk = self.dev.capture(f"s{step}_probe{self.n_probes}_rbgraph")
      live = chk["ui"]
    return same_state(live, target), chk, [p[0] for p in path]

  def probe(self, pre: dict[str, Any], slot: U.Slot, prio: int, step: int, depth: int) -> dict[str, Any]:
    ui0 = pre["ui"]
    node = ui0.nodes[slot.node_idx]
    x, y = node.center
    t0 = time.time()
    self.n_probes += 1
    self.dev.tap(x, y)
    t_act = time.time()
    post = self.dev.capture(f"s{step}_probe{self.n_probes}_post")
    t_obs = time.time()
    ui1 = post["ui"]
    if ui1.exact_sig == ui0.exact_sig and ui1.content_sig == ui0.content_sig and ui1.activity == ui0.activity:
      outcome = "no_change"
    elif ui1.struct_sig == ui0.struct_sig and ui1.content_sig == ui0.content_sig:
      outcome = "minor_change"   # focus/selection/bounds only
    elif ui1.struct_sig == ui0.struct_sig:
      outcome = "in_place_content_change"
    elif ui1.package != ui0.package:
      outcome = "left_app"
    else:
      outcome = "new_screen"
    gm, gm_nodes = self.graph.match(ui1)
    # ---- rollback ---------------------------------------------------
    rb = {"actions": [], "ok": True, "method": "none"}
    t_rb0 = time.time()
    if outcome in ("no_change", "minor_change"):
      rb["method"] = "none_needed"
    else:
      ok = False
      if outcome == "in_place_content_change":
        self.dev.tap(x, y)
        rb["actions"].append("retap")
        chk = self.dev.capture(f"s{step}_probe{self.n_probes}_rb_retap")
        ok = same_state(chk["ui"], ui0)
        rb["method"] = "retap"
        rb["check"] = chk["cid"]
      else:
        sib = selected_sibling(ui0, node)
        if sib is not None:
          hit = U.element_at(ui1, *sib.center)
          if hit is None or U.label_of(ui1, hit) != U.label_of(ui0, sib) or hit.short_rid != sib.short_rid:
            sib = None   # the tab bar is not on the new screen: do not tap blindly
        if sib is not None:
          # Tab-like control: switching is undone by re-selecting the
          # previously selected sibling, not by BACK.
          self.dev.tap(*sib.center)
          rb["actions"].append("reselect")
          chk = self.dev.capture(f"s{step}_probe{self.n_probes}_rb_reselect")
          rb["check"] = chk["cid"]
          ok = same_state(chk["ui"], ui0)
          rb["method"] = "reselect"
        for i in range(0 if ok else 2):
          self.dev.back()
          rb["actions"].append("back")
          chk = self.dev.capture(f"s{step}_probe{self.n_probes}_rb{i}")
          rb["check"] = chk["cid"]
          if same_state(chk["ui"], ui0):
            ok = True
            break
          # Only press BACK again if the first BACK merely closed the IME or
          # did nothing; never walk further up an unknown back stack.
          if not (chk["ui"].struct_sig == ui1.struct_sig or
                  (ui1.ime_visible() and not chk["ui"].ime_visible() and chk["ui"].layout_sig != ui0.layout_sig)):
            break
        if not ok or rb["method"] == "none":
          rb["method"] = "back" if sib is None else "reselect+back"
        if not ok and self.cfg.graph_restore:
          ok, chk2, path = self.restore_via_graph(chk["ui"], ui0, step)
          rb["graph_restore"] = path
          if path:
            rb["method"] = "back+graph"
          if chk2 is not None:
            chk = chk2
            rb["check"] = chk["cid"]
      rb["ok"] = ok
      if ok:
        rb["struct_ok"] = True
        rb["exact_restored"] = chk["ui"].exact_sig == ui0.exact_sig
      else:
        rb["struct_ok"] = chk["ui"].struct_sig == ui0.struct_sig
        rb["content_ok"] = chk["ui"].content_sig == ui0.content_sig
    t_end = time.time()
    obs = {
        "kind": "transition", "origin": "explore", "instance": self.instance,
        "step": step, "depth": depth, "priority": "discover" if prio == 0 else "revalidate",
        "slot_id": slot.slot_id, "selector": slot.sel, "tap": [x, y],
        "src": ui0.summary(), "src_cid": pre["cid"],
        "dst": ui1.summary(), "dst_cid": post["cid"], "outcome": outcome,
        "dst_graph_match": gm, "dst_graph_nodes": gm_nodes[:5],
        "rollback": rb, "rollback_ok": rb["ok"], "t": t0,
        "t_act_s": round(t_act - t0, 3), "t_observe_s": round(t_obs - t_act, 3),
        "t_rollback_s": round(t_end - t_rb0, 3), "t_total_s": round(t_end - t0, 3),
    }
    self.graph.add(obs)
    self.log("probes", obs)
    return obs


# ======================================================================
def action_slot(ui: U.UIState, action: dict[str, Any], extras: dict[str, Any]) -> tuple[str, dict | None]:
  at = action.get("action_type")
  if at in ("click", "long_press", "double_tap", "input_text") and action.get("x") is not None:
    n = U.element_at(ui, int(action["x"]), int(action["y"]))
    if n is None:
      return f"{at.upper()}:@{action['x']},{action['y']}", None
    for s in U.enumerate_slots(ui):
      if s.node_idx == n.idx:
        sid = s.slot_id
        break
    else:
      sel = U.selector(ui, n)
      sid = "|".join([sel["cls"], sel["rid"], sel["desc"], sel["label"]])
    prefix = "" if at == "click" else at.upper() + ":"
    return prefix + sid, U.selector(ui, n)
  if at in ("scroll", "swipe"):
    return f"SWIPE:{action.get('direction') or extras.get('direction') or ''}", None
  if at == "open_app":
    return f"OPEN_APP:{action.get('app_name')}", None
  if at in ("navigate_back", "navigate_home", "keyboard_enter", "wait", "status", "answer"):
    return at.upper(), None
  return str(at).upper(), None


class Episode:
  def __init__(self, args):
    self.args = args
    self.out = Path(args.out)
    if (self.out / "result.json").exists():
      raise SystemExit(f"already done: {self.out}")
    self.out.mkdir(parents=True, exist_ok=True)
    self.logs: dict[str, Any] = {}

  def log(self, name: str, row: dict[str, Any]):
    with (self.out / f"{name}.jsonl").open("a") as f:
      f.write(jdump(row) + "\n")

  # ------------------------------------------------------------------
  def setup(self):
    a = self.args
    subprocess_forward()
    self.env = env_launcher.load_and_setup_env(console_port=5554, emulator_setup=False, adb_path=ADB)
    reg = registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)
    self.task_type = reg[a.task]
    self.task = suite_utils._instantiate_task(self.task_type, seed=a.seed, env=self.env)  # pylint: disable=protected-access
    self.dev = Device(self.env, self.out)
    meta = {
        "task": a.task, "seed": a.seed, "variant": a.variant, "mode": a.mode,
        "goal": self.task.goal, "params": self.task.params,
        "complexity": self.task.complexity, "start_on_home": self.task.start_on_home_screen,
        "explore": dataclasses.asdict(self.explore_cfg()), "api_url": a.api_url,
        "model": a.model, "ref": a.ref, "started": time.time(),
        "screen": list(self.dev.screen),
    }
    (self.out / "meta.json").write_text(jdump(meta))
    return meta

  def explore_cfg(self) -> ExploreCfg:
    a = self.args
    return ExploreCfg(enabled=a.explore, budget_s=a.budget, max_probes=a.max_probes,
                      max_depth=1, revalidate=not a.no_revalidate,
                      graph_restore=not a.no_graph_restore)

  def max_steps(self) -> int:
    if self.args.max_steps:
      return self.args.max_steps
    return min(gelab_agent.MAX_AGENT_STEPS, suite_utils._allocate_step_budget(self.task.complexity))  # pylint: disable=protected-access

  # ------------------------------------------------------------------
  def run(self):
    meta = self.setup()
    res: dict[str, Any] = {"task": self.args.task, "variant": self.args.variant, "mode": self.args.mode}
    t0 = time.time()
    try:
      self.task.initialize_task(self.env)
      # agent.reset(go_home=start_on_home_screen) equivalent
      self.env.reset(go_home=self.task.start_on_home_screen)
      self.env.hide_automation_ui()
      if self.args.mode == "agent":
        r = self.run_agent()
      else:
        r = self.run_replay(strict=self.args.mode == "replay_strict")
      res.update(r)
      time.sleep(1.0)
      try:
        res["is_successful"] = float(self.task.is_successful(self.env))
      except Exception as e:  # pylint: disable=broad-except
        res["is_successful"] = 0.0
        res["eval_error"] = repr(e)
    except Exception as e:  # pylint: disable=broad-except
      res["error"] = repr(e)
      res["traceback"] = traceback.format_exc()
    finally:
      try:
        self.task.tear_down(self.env)
      except Exception as e:  # pylint: disable=broad-except
        res["teardown_error"] = repr(e)
      res["wall_s"] = round(time.time() - t0, 2)
      (self.out / "result.json").write_text(jdump(res))
      try:
        self.env.close()
      except Exception:  # pylint: disable=broad-except
        pass
    return res

  # ------------------------------------------------------------------
  def make_agent(self):
    llm = infer.LlamaCppWrapper(api_url=self.args.api_url, temperature=0.0, model_name=self.args.model)
    agent = gelab_agent.GELABAgent(self.env, llm, output_path="")
    agent._actions.clear()  # pylint: disable=protected-access
    return agent

  def decide(self, agent, goal: str, pixels) -> dict[str, Any]:
    """Exactly GELABAgent.step's decision part (prompt, parse, translate)."""
    screenshot = Image.fromarray(pixels)
    messages = gelab_agent.build_gelab_messages(goal, agent._history_text(), screenshot)  # pylint: disable=protected-access
    t0 = time.time()
    response, _, _ = agent.vllm.predict_mm("", [], messages=messages)
    lat = time.time() - t0
    parse_error = None
    try:
      parsed = gelab_agent.parse_gelab_response(response)
      action, tool_call, extras = gelab_agent.gelab_action_to_json_action(parsed, self.env.logical_screen_size)
    except seeact_utils.ParseActionError as e:
      parse_error = str(e)
      parsed = OrderedDict(cot="", action="WAIT", value="1", summary="Parser fallback wait", parse_error=parse_error)
      action = json_action.JSONAction(action_type=json_action.WAIT)
      tool_call = {"name": "mobile_use", "arguments": {"action": "wait", "value": 1}}
      extras = {"wait_seconds": 1, "parse_error": parse_error, "fallback": "parse_error_wait"}
    return {"response": response, "parsed": dict(parsed), "action": action, "tool_call": tool_call,
            "extras": extras, "model_latency_s": round(lat, 3), "parse_error": parse_error}

  def commit(self, agent, d: dict[str, Any], goal: str):
    summary = gelab_agent._normalize_space(d["parsed"].get("summary")) or gelab_agent._normalize_space(d["parsed"].get("explain"))  # pylint: disable=protected-access
    if not summary:
      summary = str(d["tool_call"].get("arguments") or d["tool_call"])
    if d["extras"].get("return_text"):
      self.env.interaction_cache = str(d["extras"]["return_text"])
    t0 = time.time()
    agent._execute_action(d["action"], d["extras"])  # pylint: disable=protected-access
    exec_s = time.time() - t0
    agent._actions.append({"summary": summary})  # pylint: disable=protected-access
    agent._summaries.append(summary)  # pylint: disable=protected-access
    return summary, exec_s

  def run_agent(self, start_step: int = 0, prior_summaries: list[str] | None = None,
                label: str = "agent") -> dict[str, Any]:
    goal = self.task.goal
    agent = self.make_agent()
    if prior_summaries:
      agent._summaries.extend(prior_summaries)  # pylint: disable=protected-access
      agent._actions.extend({"summary": s} for s in prior_summaries)  # pylint: disable=protected-access
    cfg = self.explore_cfg()
    graph = Graph(Path(self.args.graph) if self.args.graph else None,
                  Path(self.args.app_memory) if self.args.app_memory else None)
    instance = f"{self.args.task}/{self.args.variant}"
    explorer = Explorer(self.dev, graph, cfg, instance, self.log)
    steps = []
    done = False
    n_calls = 0
    prev = None
    for k in range(start_step, self.max_steps()):
      pre = self.dev.capture(f"s{k}_pre", settle=True, pause=1.0, full_res=True)
      if prev is not None:
        self.ingest_agent(graph, instance, prev, pre)
      box: dict[str, Any] = {}

      def work():
        try:
          box["d"] = self.decide(agent, goal, pre["pixels"])
        except Exception as e:  # pylint: disable=broad-except
          box["err"] = e
      t_req = time.time()
      th = threading.Thread(target=work, daemon=True)
      th.start()
      xrec = None
      if cfg.enabled:
        xrec = explorer.run_window(pre, k, cfg.budget_s)
        xrec["model_done_at_explore_end"] = not th.is_alive()
        self.log("windows", xrec)
      th.join()
      t_join = time.time()
      if "err" in box:
        raise box["err"]
      d = box["d"]
      n_calls += 1
      requery = False
      verify = None
      if xrec is not None and xrec["probes"]:
        verify = self.dev.capture(f"s{k}_verify", settle=True)
        if not same_state(verify["ui"], pre["ui"]):
          # Rollback could not restore the screen the model saw: never act on
          # a stale screenshot. Re-observe and ask again (counted as overhead).
          requery = True
          pre = self.dev.capture(f"s{k}_pre_requery", settle=True, full_res=True)
          d = self.decide(agent, goal, pre["pixels"])
          n_calls += 1
      slot_id, sel = action_slot(pre["ui"], d["action"].__dict__, d["extras"])
      summary, exec_s = self.commit(agent, d, goal)
      row = {
          "step": k, "label": label, "pre": strip(pre), "slot_id": slot_id, "target_sel": sel,
          "response": d["response"], "parsed": d["parsed"], "tool_call": d["tool_call"],
          "action": d["action"].__dict__, "extras": d["extras"], "summary": summary,
          "model_latency_s": d["model_latency_s"], "exec_s": round(exec_s, 3),
          "explore_window_s": xrec["window_s"] if xrec else 0.0,
          "n_probes": len(xrec["probes"]) if xrec else 0,
          "wait_after_explore_s": round(t_join - t_req, 3),
          "verify_cid": verify["cid"] if verify else None, "requery": requery,
          "parse_error": d["parse_error"],
      }
      steps.append(row)
      self.log("steps", row)
      prev = {"pre": pre, "slot_id": slot_id, "sel": sel, "action": d["action"].__dict__, "step": k}
      if d["action"].action_type in (json_action.STATUS, json_action.ANSWER):
        done = True
        break
    final = self.dev.capture("final", settle=True, pause=1.0, full_res=True)
    if prev is not None and not done:
      self.ingest_agent(graph, instance, prev, final)
    elif prev is not None:
      self.ingest_agent(graph, instance, prev, final)
    return {"steps": len(steps) + start_step, "agent_steps": len(steps), "done": done,
            "model_calls": n_calls, "final_cid": final["cid"],
            "explore_disabled": explorer.disabled_reason, "n_probes": explorer.n_probes}

  def ingest_agent(self, graph: Graph, instance: str, prev: dict[str, Any], post: dict[str, Any]):
    ui0, ui1 = prev["pre"]["ui"], post["ui"]
    obs = {
        "kind": "transition", "origin": self.args.mode if self.args.mode != "agent" else "agent",
        "instance": instance, "step": prev["step"], "depth": 0, "priority": "",
        "slot_id": prev["slot_id"], "selector": prev["sel"], "action": prev["action"],
        "src": ui0.summary(), "src_cid": prev["pre"]["cid"],
        "dst": ui1.summary(), "dst_cid": post["cid"],
        "outcome": ("no_change" if same_state(ui0, ui1) else "new_screen"),
        "dst_graph_match": graph.match(ui1)[0], "rollback_ok": True, "t": time.time(),
    }
    graph.add(obs)
    self.log("agent_transitions", obs)

  # ------------------------------------------------------------------
  def run_replay(self, strict: bool) -> dict[str, Any]:
    ref_dir = Path(self.args.ref)
    ref_meta = json.loads((ref_dir / "meta.json").read_text())
    ref_steps = [json.loads(l) for l in (ref_dir / "steps.jsonl").read_text().splitlines() if l.strip()]
    ref_steps = [s for s in ref_steps if s.get("label", "agent") == "agent"]
    subst = param_substitution(ref_meta, json.loads((self.out / "meta.json").read_text()))
    graph = Graph(Path(self.args.graph) if self.args.graph else None)
    instance = f"{self.args.task}/{self.args.variant}"
    divergence = None
    executed = 0
    prev = None
    for i, rs in enumerate(ref_steps):
      pre = self.dev.capture(f"r{i}_pre", settle=True, pause=1.0, full_res=True)
      if prev is not None:
        self.ingest_agent(graph, instance, prev, pre)
        self.compare_outcome(i - 1, prev, pre, ref_dir, ref_steps)
      ref_ui = load_ui(ref_dir, rs["pre"]["cid"])
      dec = self.replay_decide(rs, ref_ui, pre["ui"], strict, subst)
      dec.update({"ref_step": i, "pre": strip(pre), "ref_pre_cid": rs["pre"]["cid"]})
      if not dec["execute"]:
        divergence = dec
        self.log("replay", dec)
        break
      action = json_action.JSONAction(**{k: v for k, v in dec["action"].items() if v is not None})
      extras = dict(rs.get("extras") or {})
      t0 = time.time()
      agent = self.make_agent()
      if action.action_type == json_action.ANSWER and action.text:
        self.env.interaction_cache = action.text
      agent._execute_action(action, extras)  # pylint: disable=protected-access
      dec["exec_s"] = round(time.time() - t0, 3)
      self.log("replay", dec)
      executed += 1
      slot_id, sel = action_slot(pre["ui"], dec["action"], extras)
      prev = {"pre": pre, "slot_id": slot_id, "sel": sel, "action": dec["action"], "step": i}
    final = self.dev.capture("final", settle=True, pause=1.0, full_res=True)
    if prev is not None:
      self.ingest_agent(graph, instance, prev, final)
      self.compare_outcome(len(ref_steps) - 1 if divergence is None else prev["step"], prev, final, ref_dir, ref_steps)
    out = {"replay_executed": executed, "ref_steps": len(ref_steps),
           "divergence": {k: v for k, v in (divergence or {}).items() if k != "pre"} if divergence else None,
           "final_cid": final["cid"], "subst": subst}
    if divergence is not None and self.args.recover and self.args.mode == "replay_struct":
      try:
        res = self.task.is_successful(self.env)
      except Exception:  # pylint: disable=broad-except
        res = None
      out["success_before_recovery"] = res
      summaries = [s["summary"] for s in ref_steps[:executed]]
      r = self.run_agent(start_step=executed, prior_summaries=summaries, label="recovery")
      out["recovery"] = r
    return out

  def compare_outcome(self, i, prev, post, ref_dir, ref_steps):
    """Transition-level comparison with the reference outcome of step i."""
    if i + 1 < len(ref_steps):
      ref_post_cid = ref_steps[i + 1]["pre"]["cid"]
    else:
      ref_post_cid = json.loads((ref_dir / "result.json").read_text()).get("final_cid")
    if not ref_post_cid:
      return
    ref_post = load_ui(ref_dir, ref_post_cid)
    ui = post["ui"]
    if ui.exact_sig == ref_post.exact_sig:
      cat = "same_exact"
    elif ui.struct_sig == ref_post.struct_sig:
      cat = "same_struct_dynamic_content"
    elif ui.layout_sig == ref_post.layout_sig:
      cat = "same_layout_ambiguous"
    elif same_state(ui, prev["pre"]["ui"]):
      cat = "no_observable_transition"
    else:
      cat = "different_state"
    self.log("outcomes", {"ref_step": i, "ref_post_cid": ref_post_cid, "live_post_cid": post["cid"],
                          "category": cat, "slot_id": prev["slot_id"]})

  def replay_decide(self, rs, ref_ui: U.UIState, live: U.UIState, strict: bool, subst) -> dict[str, Any]:
    act = dict(rs["action"])
    at = act.get("action_type")
    dec: dict[str, Any] = {"mode": "strict" if strict else "struct", "action_type": at,
                           "execute": False, "action": act}
    # 1. state correspondence with the reference pre-state
    if live.struct_sig == ref_ui.struct_sig:
      dec["state_match"] = "struct"
    elif live.layout_sig == ref_ui.layout_sig:
      dec["state_match"] = "layout"
    elif live.activity == ref_ui.activity and live.package == ref_ui.package:
      dec["state_match"] = "activity_only"
    else:
      dec["state_match"] = "none"
    diff = U.diff_elements(ref_ui, live)
    dec["a11y_diff"] = diff
    if dec["state_match"] in ("none", "activity_only"):
      dec["reason"] = "state_mismatch"
      dec["cause"] = classify_state_mismatch(ref_ui, live)
      return dec
    # 2. parameter-bound actions
    text = act.get("text")
    bound = False
    if text:
      new = apply_subst(text, subst)
      if new != text:
        bound = True
        act["text"] = new
        dec["substituted_text"] = [text, new]
    if at == json_action.ANSWER:
      if strict:
        dec["reason"] = "answer_is_data_bound"
        dec["cause"] = "expected_parameter_divergence"
        return dec
      dec["reason_note"] = "answer_replayed_verbatim"
    # 3. target grounding
    if act.get("x") is not None and at in ("click", "long_press", "double_tap", "input_text"):
      # Re-derive the reference target from the saved reference a11y (the
      # stored selector came from an older hit-test).
      ref_node = U.element_at(ref_ui, int(rs["action"]["x"]), int(rs["action"]["y"]))
      ref_sel = U.selector(ref_ui, ref_node) if ref_node is not None else rs.get("target_sel")
      if not ref_sel:
        dec["reason"] = "ref_target_unresolved"
        dec["cause"] = "unknown"
        return dec
      ref_label = ref_sel.get("label") or ""
      want = dict(ref_sel)
      lab_new = apply_subst(ref_label, subst) if ref_label else ref_label
      if lab_new != ref_label:
        bound = True
        for k in ("label", "text", "desc"):
          if want.get(k):
            want[k] = apply_subst(want[k], subst)
        dec["substituted_target"] = [ref_label, lab_new]
      x, y = int(act["x"]), int(act["y"])
      hit = U.element_at(live, x, y)
      hit_sel = U.selector(live, hit) if hit is not None else None
      same_hit = hit_sel is not None and all(hit_sel[k] == want[k] for k in ("cls", "rid", "text", "desc", "label"))
      dec["coord_hit_same_target"] = same_hit
      cat, node, ncand = U.match_element(want, live)
      dec["match_category"] = cat
      dec["n_candidates"] = ncand
      if strict:
        if not same_hit:
          dec["reason"] = "coordinate_target_changed"
          dec["cause"] = ("coordinate_drift" if cat in ("exact_selector", "stable_hierarchy", "structural_unique")
                          else "ambiguous_target" if cat == "ambiguous" else "missing_target")
          return dec
      else:
        if cat in ("none", "ambiguous"):
          dec["reason"] = "target_" + cat
          dec["cause"] = "ambiguous_target" if cat == "ambiguous" else "missing_target"
          return dec
        if cat == "structural_unique" and ref_label and node is not None and U.label_of(live, node) != want.get("label"):
          dec["reason"] = "target_label_differs"
          dec["cause"] = "changed_app_data"
          return dec
        nx, ny = node.center
        if U.element_at(live, x, y) is not None and U.element_at(live, x, y).idx == node.idx:
          nx, ny = x, y   # keep the original point when it still hits the target
        act["x"], act["y"] = nx, ny
        dec["relocated"] = [x, y] != [nx, ny]
    dec["parameter_bound"] = bound
    dec["execute"] = True
    dec["action"] = act
    return dec


# ======================================================================
def load_ui(ep_dir: Path, cid: str) -> U.UIState:
  with gzip.open(ep_dir / "a11y" / f"{cid}.json.gz", "rt", encoding="utf-8") as f:
    d = json.load(f)
  return U.parse_flat(d["payload"], activity=d["activity"])


def classify_state_mismatch(ref: U.UIState, live: U.UIState) -> str:
  if live.package != ref.package:
    return "different_app_or_screen"
  if live.ime_visible() != ref.ime_visible():
    return "keyboard_state"
  rnew = set(n.window for n in live.app_nodes()) - set(n.window for n in ref.app_nodes())
  if len(set(n.window for n in live.app_nodes())) > len(set(n.window for n in ref.app_nodes())):
    return "unexpected_dialog_or_overlay"
  if live.activity != ref.activity:
    return "different_destination_screen"
  return "changed_ui_hierarchy"


def _flat_params(p: Any, prefix="") -> dict[str, str]:
  out = {}
  if isinstance(p, dict):
    for k, v in p.items():
      out.update(_flat_params(v, f"{prefix}{k}."))
  elif isinstance(p, (list, tuple)):
    for i, v in enumerate(p):
      out.update(_flat_params(v, f"{prefix}{i}."))
  elif isinstance(p, (str, int, float)) and not isinstance(p, bool):
    out[prefix.rstrip(".")] = str(p)
  return out


def param_substitution(ref_meta, var_meta) -> list[list[str]]:
  """Map reference parameter values to the variant's values, only for values
  that literally appear in both goals (the agent sees the goal; nothing
  privileged is used)."""
  rp, vp = _flat_params(ref_meta.get("params")), _flat_params(var_meta.get("params"))
  pairs = []
  for k, rv in rp.items():
    vv = vp.get(k)
    if k == "seed" or vv is None or rv == vv or len(rv) < 2:
      continue
    if rv in ref_meta["goal"] and vv in var_meta["goal"]:
      pairs.append([rv, vv])
      # file-name stems: agents type "name" and pick the extension separately
      if "." in rv and "." in vv:
        rs, vs = rv.rsplit(".", 1)[0], vv.rsplit(".", 1)[0]
        if len(rs) >= 3 and rs != vs:
          pairs.append([rs, vs])
  pairs.sort(key=lambda p: -len(p[0]))
  return pairs


def apply_subst(text: str, pairs) -> str:
  for a, b in pairs:
    if a in text:
      text = text.replace(a, b)
  return text


def subprocess_forward():
  import subprocess
  subprocess.run([ADB, "-s", "emulator-5554", "forward", "tcp:8765", "localabstract:androidworld_fast_a11y"],
                 check=True, timeout=10, capture_output=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--variant", required=True)
  ap.add_argument("--mode", choices=["agent", "replay_strict", "replay_struct"], default="agent")
  ap.add_argument("--out", required=True)
  ap.add_argument("--ref")
  ap.add_argument("--graph")
  ap.add_argument("--explore", action="store_true")
  ap.add_argument("--budget", type=float, default=6.0)
  ap.add_argument("--max_probes", type=int, default=6)
  ap.add_argument("--no_revalidate", action="store_true")
  ap.add_argument("--recover", action="store_true")
  ap.add_argument("--no_graph_restore", action="store_true")
  ap.add_argument("--app_memory")
  ap.add_argument("--max_steps", type=int, default=0)
  ap.add_argument("--api_url", default="http://127.0.0.1:18095/v1/chat/completions")
  ap.add_argument("--model", default="GELAB-ZERO-4B")
  args = ap.parse_args()
  random.seed(args.seed)
  res = Episode(args).run()
  print("RESULT", jdump({k: v for k, v in res.items() if k != "traceback"}))
  if "traceback" in res:
    print(res["traceback"])


if __name__ == "__main__":
  main()
