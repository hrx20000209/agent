"""FULL7 additions to the FULL5 controller.

1. Stuck-driven exploration (validated as GX4 + GX5 in the runtime-dynamics study):
   - the effect of every model tap is recorded: did the screen change?
   - a screen is STUCK when a control was tapped without effect; on a screen the
     graph has never seen (runtime-novel state) one no-op suffices, on a known
     screen the same no-op must repeat (>= 2), which removes false triggers
   - while stuck, the inference window probes an untried safe control
     (verified rollback) instead of E1; its effect and reversibility are written
     to a persistent effect memory and never probed again
   - the prompt gets a directive hint backed by runtime evidence:
     "Recommended next action: tap X at (x, y)" + the observed no-ops
2. First-run screens (separate module, outside the main algorithm): on a
   runtime-novel screen whose activity is an intro / onboarding / welcome /
   tutorial screen, the system taps skip / done / next until it leaves it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from android_world import shared_state as SS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "runtime_graph_exploration" / "scripts"))
import memory as MEM  # noqa: E402

C, U, R = MEM.C, MEM.U, MEM.R
from android_world.agents import gelab_agent  # noqa: E402

EXTRA = {"text": ""}
_orig_build = gelab_agent.build_gelab_messages


def _build(goal, hist, img):
  msgs = _orig_build(goal, hist, img)
  if EXTRA["text"]:
    msgs[1]["content"][0]["text"] += "\n\n" + EXTRA["text"]
  return msgs


gelab_agent.build_gelab_messages = _build

ONBOARD_ACT = re.compile(r"(intro|onboard|welcome|tutorial|wizard)", re.I)
ONBOARD_ORDER = ("skip", "done", "get started", "got it", "start", "continue", "next")
_KNOWN_CACHE = {}


def _known_from_history(task):
  tk = MEM.template_key(task)
  if tk not in _KNOWN_CACHE:
    snap = MEM.Snapshot(task)        # leave-one-template-out: structure only, no actions of this template
    ks = set()
    for o in snap.obs:
      ks.add(o["src"]["key"])
      if o["dst"]:
        ks.add(o["dst"]["key"])
    _KNOWN_CACHE[tk] = ks
  return _KNOWN_CACHE[tk]


class StuckExplorer:
  def __init__(self, episode, goal, gdir, hard=False, online_only=False):
    self.hard = hard          # FULL9: hard-coded stuck rules, no hint injection
    self.last_noop = None     # (state key, fam, consecutive count)
    self.ep, self.goal, self.gdir = episode, goal, Path(gdir)
    # Online arms learn page novelty only within this run, just like their graph.
    self.known = set() if online_only else set(_known_from_history(episode.args.task))
    kf = self.gdir / "known_states.txt"
    self.known |= set(SS.read_text(kf).split())
    self.visited = set()
    self.effect = {}
    self.mem_path = self.gdir / "effect_memory.jsonl"
    self.mem = {}
    self._mem_offset = 0
    self.refresh_shared()
    self.onboarded = set()

  def refresh_shared(self):
    self.known |= set(SS.read_text(self.gdir / "known_states.txt").split())
    rows, self._mem_offset = SS.read_delta(self.mem_path, self._mem_offset)
    for row in rows:
      self.mem[(row["key"], row["fam"])] = row

  # ------------------------------------------------------------ effects
  def _record(self, key, fam, label, changed, ui_dst):
    e = self.effect.setdefault((key, fam), {"label": label or fam.split("|")[-1], "changed": False, "n": 0,
                                            "title": "", "known": False})
    e["n"] += 1
    if changed:
      e["changed"] = True
      # "moves": the control leads to a different screen state (layout / activity), not an in-place value change
      # (digit keys, toggles, value pickers change content in place and are never recommended as a way out)
      if ui_dst is not None and MEM.state_key(ui_dst) != key:
        e["moves"] = True
      labs = MEM.page_labels(ui_dst)
      e["title"] = labs[0][:40] if labs else ""
      e["known"] = MEM.state_key(ui_dst) in self.known

  def observe_model(self, prev, ui_now):
    ui0 = prev["pre"]["ui"]
    k0 = MEM.state_key(ui0)
    self.visited.add(k0)
    fam = prev.get("fam")
    if not fam:
      self.last_noop = None
      return
    changed = not (MEM.state_key(ui_now) == k0 and ui_now.exact_sig == ui0.exact_sig)
    if self.hard:
      # only navigation controls can be "stuck": fields, toggles, value pickers and commits (Save, OK, ...)
      # legitimately leave the screen unchanged
      slot = next((sl for sl in U.enumerate_slots(ui0) if C.family_of_sel(sl.sel) == fam), None)
      if slot is None or not slot.safe:
        self.last_noop = None
        return
      if changed:
        self.last_noop = None
      elif self.last_noop and self.last_noop[:2] == (k0, fam):
        self.last_noop = (k0, fam, self.last_noop[2] + 1)
      else:
        self.last_noop = (k0, fam, 1)
    self._record(k0, fam, prev.get("label", ""), changed, ui_now)

  def needs_back(self, ui):
    if not self.hard or not self.last_noop:
      return False
    if self.last_noop[0] == MEM.state_key(ui) and self.last_noop[2] >= 3:
      self.last_noop = None
      return True
    return False

  def novel(self, ui):
    return MEM.state_key(ui) not in self.known

  def is_stuck(self, ui):
    if self.hard:
      return False
    key = MEM.state_key(ui)
    need = 1 if self.novel(ui) else 2
    return any(k == key and not e["changed"] and e["n"] >= need for (k, _), e in self.effect.items())

  def hint(self, ui):
    if self.hard or not self.is_stuck(ui):
      return ""
    key = MEM.state_key(ui)
    for (k, f), r in self.mem.items():           # remembered probe effects for this screen
      if k == key and r["changed"] and (k, f) not in self.effect:
        self.effect[(k, f)] = {"label": r["label"], "changed": True, "n": 0, "title": r["title"], "known": r["known"],
                               "moves": r.get("moves", False)}
    live = {C.family_of_sel(sl.sel): sl for sl in U.enumerate_slots(ui)}
    lines, eff = [], []
    for (k, f), e in self.effect.items():
      if k != key:
        continue
      if not e["changed"]:
        lines.append(f"- Tapping \"{e['label']}\" did not change the screen ({e['n']} time{'s' if e['n'] > 1 else ''}).")
      elif f in live and e.get("moves"):
        eff.append((f, e))
        lines.append(f"- Tapping \"{e['label']}\" changed the screen" + (f" to \"{e['title']}\"" if e["title"] else "") + ".")
    text = "Observed on THIS screen in the current run:\n" + "\n".join(lines)
    if eff:
      eff.sort(key=lambda x: (not x[1]["known"], x[0]))
      f, e = eff[0]
      cx, cy = ui.nodes[live[f].node_idx].center
      W, H = ui.screen
      text = (f"Recommended next action: tap \"{e['label']}\" at point ({round(cx * 1000 / W)}, {round(cy * 1000 / H)}). "
              "It was observed to change this screen, while the elements below were observed to have no effect.\n" + text)
    return text

  def window(self, ex, pre, step):
    ui = pre["ui"]
    key = MEM.state_key(ui)
    rec = {"step": step, "probes": [], "mode": "STUCK"}
    for sl in U.enumerate_slots(ui):
      fam = C.family_of_sel(sl.sel)
      if not sl.safe or (key, fam) in self.effect or (key, fam) in self.mem:
        continue
      p = ex.probe(pre, sl, 0, step, depth=1)
      try:
        dst = R.load_ui(self.ep.out, p["dst_cid"])
      except Exception:  # pylint: disable=broad-except
        dst = None
      label = U.label_of(ui, ui.nodes[sl.node_idx]) or sl.sel.get("desc") or ""
      changed = dst is not None and not (MEM.state_key(dst) == key and dst.exact_sig == ui.exact_sig)
      self._record(key, fam, label, changed, dst)
      e = self.effect[(key, fam)]
      r = {"key": key, "fam": fam, "label": e["label"], "changed": e["changed"], "moves": e.get("moves", False), "title": e["title"],
           "known": e["known"], "rollback_ok": p["rollback_ok"], "episode": self.ep.args.variant}
      self.mem[(key, fam)] = r
      SS.append_json(self.mem_path, r)
      rec["probes"].append({"why": "STUCK", "slot": sl.slot_id, "family": fam, "outcome": p["outcome"],
                            "rollback_ok": p["rollback_ok"], "t": p["t_total_s"], "dst_ui": dst})
      if not p["rollback_ok"]:
        ex.disabled_reason = f"rollback_failed@{step}"
      break
    return rec

  # -------------------------------------------------- first-run screens
  def onboarding(self, ui, agent, steps, k):
    if "launcher" in ui.package or not ONBOARD_ACT.search(ui.activity) or not self.novel(ui) or ui.package in self.onboarded:
      return 0
    self.onboarded.add(ui.package)
    n = 0
    for _ in range(8):
      cap = self.ep.dev.capture(f"s{k + n}_onboard", settle=True, pause=1.0)
      cur = cap["ui"]
      if not ONBOARD_ACT.search(cur.activity):
        break
      target = None
      for word in ONBOARD_ORDER:
        for sl in U.enumerate_slots(cur):
          lab = (sl.sel.get("label") or sl.sel.get("desc") or "").strip().lower()
          rid = (sl.sel.get("rid") or "").lower()
          if lab == word or rid == word.replace(" ", "_"):
            target = sl
            break
        if target is not None:
          break
      if target is None:
        break
      cx, cy = cur.nodes[target.node_idx].center
      self.ep.dev.tap(cx, cy)
      lab = target.sel.get("label") or target.sel.get("desc") or target.sel.get("rid")
      summ = f"[system] Tapped \"{lab}\" on the app's first-run screen."
      agent._actions.append({"summary": summ})  # pylint: disable=protected-access
      agent._summaries.append(summ)  # pylint: disable=protected-access
      row = {"step": k + n, "kind": "onboarding", "tok": C.family_of_sel(target.sel), "summary": summ,
             "pre": R.strip(cap)}
      steps.append(row)
      self.ep.log("steps", row)
      n += 1
    return n

  def finish(self):
    new = self.visited - self.known
    if new:
      SS.append_text(self.gdir / "known_states.txt", "\n".join(sorted(new)) + "\n")


# ------------------------------------------------------------------ FULL8 helpers
_KNOWN_PKGS = None


def _known_pkgs():
  """Packages whose screens the graph has observed (historical transitions)."""
  global _KNOWN_PKGS
  if _KNOWN_PKGS is None:
    _KNOWN_PKGS = set()
    for l in open(MEM.OBS):
      o = json.loads(l)
      _KNOWN_PKGS.add(o["src"]["pkg"])
      if o["dst"]:
        _KNOWN_PKGS.add(o["dst"]["pkg"])
  return _KNOWN_PKGS


def resolve_app(goal, ui, require_known=True):
  """App named in the instruction (earliest mention, longest alias) whose package the graph has observed.
  Only on the launcher: if the task starts inside an app nothing is launched."""
  from android_world.env import adb_utils
  if "launcher" not in ui.package and "launcher" not in ui.activity.lower():
    return None
  g = goal.lower()
  best = None
  for pattern, activity in adb_utils._PATTERN_TO_ACTIVITY.items():  # pylint: disable=protected-access
    pkg = activity.split("/")[0]
    if require_known and pkg not in _known_pkgs():
      continue
    for alias in pattern.split("|"):
      mm = re.search(r"(?<![\w])" + re.escape(alias) + r"(?![\w])", g)
      if mm:
        cand = (mm.start(), -len(alias), alias)
        if best is None or cand < best:
          best = cand
  return best[2] if best else None


def launch(env, app):
  from android_world.env import adb_utils
  adb_utils.launch_app(app, env.controller)


def capture_stable(dev, tag, limit_s=2.0):
  """Capture once the layout stops changing (bounded), so the first decision sees a drawn screen."""
  import time as _t
  t0 = _t.time()
  cap = dev.capture(tag, settle=True, pause=0.4, full_res=True)
  while _t.time() - t0 < limit_s:
    nxt = dev.capture(tag + "_s", settle=True, pause=0.4, full_res=True)
    if nxt["ui"].layout_sig == cap["ui"].layout_sig:
      return nxt
    cap = nxt
  return cap


def confirm_own_commit(ep, ui, agent, steps, k, own, pure=False):
  """After the model's own commit, confirm the dialog that follows with its positive button (button1)."""
  slot = next((sl for sl in U.enumerate_slots(ui)
               if sl.sel.get("rid") == "button1" and sl.unsafe_reason in ("dialog_action_button", "unsafe_label")), None)
  if slot is None:
    return 0
  if pure:
    # only a pure confirmation (message + yes/no buttons): pickers, lists and fields are choices for the model
    others = [sl for sl in U.enumerate_slots(ui) if sl.sel.get("rid") not in ("button1", "button2", "button3")]
    if others or any(n.editable for n in ui.app_nodes()):
      return 0
  node = ui.nodes[slot.node_idx]
  ep.dev.tap(*node.center)
  lab = U.label_of(ui, node) or "OK"
  summ = f"[system] Confirmed \"{lab}\" for the action just chosen."
  agent._actions.append({"summary": summ})  # pylint: disable=protected-access
  agent._summaries.append(summ)  # pylint: disable=protected-access
  fam = C.family_of_sel(slot.sel)
  import seq_run as S  # noqa: E402  (FULL runs already import it)
  own.append(S.own_entry(ui, fam, fam))
  row = {"step": k, "kind": "confirm", "tok": fam, "summary": summ}
  steps.append(row)
  ep.log("steps", row)
  return 1
