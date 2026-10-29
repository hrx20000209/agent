"""FULL33: every graph reuse is the execution of an out-edge of the current page.

Edge types (all on the same page-keyed graph: profiling graph + this run's runtime graph):
  LAUNCH   launcher --open_app(pkg)--> app entry          evidence: observed launch transitions into pkg
  CONFIRM  commit dialog --button1--> app page             evidence: observed button1 transitions on (pkg, activity)
  PARAM    entry page --[nav | field select | field type]*--> page before commit   (parameterized function)
           evidence: successful trajectories the function was extracted from
  PATTERN  page with empty labelled fields --fill(record)--> same page               (form function)
           evidence: the page's own field labels matching record fields
  ACTION   (page, control family) --> page                 evidence: repeated own transitions / cross-task edges

Unified trigger: precondition(type) AND evidence(edge) >= 1 AND the edge grounds on the live screen AND the edge is
not braked. Every execution is written back as an edge with origin "graph" (reinforcing its evidence); a model
action that immediately undoes it (Back, or opening another app after a launch) counts one misuse, and an edge
with misuse >= max(1, valid) is braked.
"""
from __future__ import annotations

import json
import time
from collections import Counter
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from android_world import shared_state as SS

HERE = Path(__file__).resolve().parent
PROFILING_OBS = HERE.parent.parent / "runtime_graph_exploration" / "graph" / "observations.jsonl"
TYPES = ("LAUNCH", "CONFIRM", "PARAM", "PATTERN", "ACTION")
KIND_TYPE = {"bootstrap": "LAUNCH", "confirm": "CONFIRM", "function": "PARAM", "function_route": "PARAM",
             "function_entry": "PARAM", "form_function": "PATTERN", "form_route": "PATTERN", "graph_reuse": "ACTION", "graph_macro": "ACTION"}


def _pkg(s):
  return (s or {}).get("package") or (s or {}).get("pkg") or ""


def _act(s):
  return (s or {}).get("activity") or (s or {}).get("act") or ""


def pkg_of_app(alias):
  """Package of an app alias as the benchmark resolves it."""
  from android_world.env import adb_utils  # pylint: disable=import-outside-toplevel
  a = alias.lower()
  for pattern, activity in adb_utils._PATTERN_TO_ACTIVITY.items():  # pylint: disable=protected-access
    if a in [p.strip().lower() for p in pattern.split("|")]:
      return activity.split("/")[0]
  return ""


def _is_confirm(o):
  fam = str(o.get("slot_id") or o.get("fam") or "")
  sel = o.get("selector") or o.get("sel") or {}
  if isinstance(sel, str):
    return "button1" in sel
  return sel.get("rid") == "button1" or "|button1|" in fam


class GraphReuse:
  def __init__(self, gdir: Path, use_profiling: bool = False):
    self.path = gdir / "graph_exec.jsonl"
    self.runtime = gdir / "graph.jsonl"
    self.ev = Counter()          # (type, key) -> evidence count
    self.valid = Counter()       # (type, key) -> graph executions that landed as expected
    self.misuse = Counter()      # (type, key) -> executions the model undid
    self._offset = 0
    self._exec_offset = 0
    if use_profiling:            # off in FULL33: evidence comes only from this run (no offline trajectories)
      self._load_profiling()
    self.refresh()
    self.last = None             # the latest execution in this episode (for the misuse check)

  # ---- evidence --------------------------------------------------------------------------------------------
  def _edge(self, o):
    s, d = o.get("src") or {}, o.get("dst") or {}
    if not d:
      return
    if _pkg(d) and _pkg(s) != _pkg(d) and ("launcher" in _pkg(s) or str(o.get("slot_id", "")).startswith("OPEN_APP")):
      self.ev[("LAUNCH", _pkg(d))] += 1
    if _is_confirm(o) and str(o.get("episode_success", "True")) != "False":
      self.ev[("CONFIRM", (_pkg(s), _act(s)))] += 1

  def _load_profiling(self):
    if PROFILING_OBS.exists():
      for line in PROFILING_OBS.open():
        if line.strip():
          self._edge(json.loads(line))

  def refresh(self):
    """Fold in the runtime graph's new agent / exploration edges (cross-task evidence of this run)."""
    rows, self._offset = SS.read_delta(self.runtime, self._offset)
    for o in rows:
      if o.get("origin") == "agent":
        self._edge(o)
    rows, self._exec_offset = SS.read_delta(self.path, self._exec_offset)
    for row in rows:
      self._apply(row)

  def _apply(self, r):
    k = (r["type"], tuple(r["key"]) if isinstance(r["key"], list) else r["key"])
    if r.get("event") == "misuse":
      self.misuse[k] += 1
    elif r.get("ok"):
      self.valid[k] += 1
      self.ev[k] += 1

  # ---- unified trigger -------------------------------------------------------------------------------------
  def braked(self, typ, key):
    k = (typ, key)
    return self.misuse[k] >= max(1, self.valid[k])

  def evidence(self, typ, key):
    return self.ev[(typ, key)]

  def can(self, typ, key):
    self.refresh()
    return self.evidence(typ, key) >= 1 and not self.braked(typ, key)

  # ---- write-back ------------------------------------------------------------------------------------------
  def record(self, typ, key, instance, step, ok, n_actions=1, src=None, dst=None):
    r = {"event": "exec", "type": typ, "key": list(key) if isinstance(key, tuple) else key, "instance": instance,
         "step": step, "ok": bool(ok), "n_actions": n_actions, "t": time.time(),
         "src": {"package": _pkg(src), "activity": _act(src)} if src else None,
         "dst": {"package": _pkg(dst), "activity": _act(dst)} if dst else None}
    SS.append_json(self.path, r)
    self.refresh()
    self.last = (typ, key, step)

  def check_undo(self, instance, step, action):
    """The model's first action after a graph execution undoes it: one misuse for that edge."""
    if self.last is None:
      return False
    typ, key, st = self.last
    self.last = None
    if step != st + 1:
      return False
    at = action.get("action_type")
    undo = at == "navigate_back" or (typ == "LAUNCH" and at == "open_app")
    if undo:
      r = {"event": "misuse", "type": typ, "key": list(key) if isinstance(key, tuple) else key,
           "instance": instance, "step": step, "t": time.time()}
      SS.append_json(self.path, r)
      self.refresh()
    return undo
