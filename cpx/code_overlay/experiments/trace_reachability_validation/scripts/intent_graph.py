"""FULL23 (design review P0-1 / P0-2): the intent layer of the unified graph and the navigation macro.

Reachability comes from the shared transition graph (agent and exploration observations, any task).
Intent per edge role (package, activity, control family):
  use     - goal-term vectors of SUCCESSFUL episodes that traversed the edge (execution validation, persisted
            in edge_use.jsonl, leave-one-task-out);
  unlock  - TF-IDF relevance of the edge's destination captures to the goal (+ its parameter values).
A navigation edge is executed without inference when (calibrated offline, scripts/intent_pred.py:
theta 0.5 / margin 0.3 -> 11 fires per run at 0.91-1.00 precision on FULL21/FULL21R):
  n_obs >= 2, landing-activity share >= 0.9, n_valid >= 1, cos(g, use) >= THETA, intent margin >= MARGIN,
  safe navigation control on the live screen, not used before from this page in this episode.
At most MAX_HOPS hops per counted step; each hop verifies the landing activity; mismatch -> misuse + brake.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from android_world import shared_state as SS

import full7
import full9
import explore_value as EV
import joint_graph as JG
from android_world.env import json_action

R, U, C = full9.R, full9.U, full9.C
LAM, THETA, MARGIN, MAX_HOPS = 0.5, 0.5, 0.3, 2


def cos(a, b):
  if not a or not b:
    return 0.0
  dot = sum(v * b.get(t, 0.0) for t, v in a.items())
  if dot == 0:
    return 0.0
  return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


class EdgeIntent:
  """Execution-validated use profiles of edge roles, shared across tasks (leave-one-task-out)."""

  def __init__(self, path, task):
    self.path, self.task = path, task
    self.use, self.n_valid, self.n_misuse = {}, Counter(), Counter()
    self._offset = 0
    self.refresh()

  def refresh(self):
    rows, self._offset = SS.read_delta(self.path, self._offset)
    for e in rows:
      key = tuple(e["key"])
      if e.get("misuse"):
        self.n_misuse[key] += 1
      elif e.get("task") != self.task:
        self.use.setdefault(key, Counter()).update(e["g"])
        self.n_valid[key] += 1

  def misuse(self, key):
    if key is None:
      return
    SS.append_json(self.path, {"key": list(key), "misuse": 1, "task": self.task})
    self.refresh()

  def record_episode(self, steps_path, gvec, success):
    if not success or not steps_path.exists() or not gvec:
      return
    rows = [json.loads(l) for l in steps_path.open()]
    seen = set()
    for i, r in enumerate(rows):
      tok = r.get("tok") or ""
      if not r.get("pre") or not tok.startswith("E|"):
        continue
      nxt = next((x for x in rows[i + 1:] if x.get("pre")), None)
      if nxt is None:
        continue
      s, d = r["pre"]["summary"], nxt["pre"]["summary"]
      if (s["activity"], s["layout_sig"]) == (d["activity"], d["layout_sig"]):
        continue
      key = (s["package"], s["activity"], tok)
      if key in seen:
        continue
      seen.add(key)
      SS.append_json(self.path, {"key": list(key), "task": self.task, "g": gvec})


def _candidates(ep, ui, graph):
  """Edge roles of this activity known in the graph (any task, any origin) with their reachability stats."""
  out = {}
  for (struct, sid), obs in graph.slots.items():
    for o in obs:
      s, d = o.get("src") or {}, o.get("dst")
      if not d or s.get("package") != ui.package or s.get("activity") != ui.activity:
        continue
      if (d.get("activity"), d.get("layout_sig")) == (s.get("activity"), s.get("layout_sig")):
        continue
      sel = o.get("selector") or {}
      fam = sid if sid.startswith("E|") else (C.family_of_sel(sel) if sel else "E|" + sid)
      c = out.setdefault(fam, {"n": 0, "acts": Counter(), "ents": []})
      c["n"] += 1
      c["acts"][d.get("activity")] += 1
      if o.get("dst_cid"):
        c["ents"].append((o["instance"], o["dst_cid"]))
  return out


def pick(ep, ui, graph):
  intent = ep._intent  # pylint: disable=protected-access
  g = ep._gvec  # pylint: disable=protected-access
  gp = dict(g)
  for t, v in ep._pvec.items():  # pylint: disable=protected-access
    gp[t] = gp.get(t, 0.0) + v
  raw = ep.out.parents[1]
  cands = _candidates(ep, ui, graph)
  live = {C.family_of_sel(t.sel) for t in U.enumerate_slots(ui)}
  scored = []
  for fam, c in cands.items():
    if fam not in live:
      continue                     # the competitors are the controls the model actually faces on this screen
    key = (ui.package, ui.activity, fam)
    cu = cos(g, intent.use.get(key))
    unlock = 0.0
    for inst, cid in c["ents"][-3:]:
      dui = EV.load_cap(raw, inst, cid)
      if dui is not None:
        unlock = max(unlock, EV.rel(gp, EV.page_tokens(dui)))
    scored.append((LAM * cu + (1 - LAM) * unlock, fam, cu, c, key))
  if not scored:
    return None
  scored.sort(key=lambda x: -x[0])
  top, fam, cu, c, key = scored[0]
  second = scored[1][0] if len(scored) > 1 else 0.0
  act, n_act = c["acts"].most_common(1)[0]
  if not (c["n"] >= 2 and n_act / c["n"] >= 0.9 and intent.n_valid[key] >= 1 and intent.n_misuse[key] == 0
          and cu >= THETA and top - second >= MARGIN):
    return None
  if (JG.page_key(ui), fam) in ep._macro_used:  # pylint: disable=protected-access
    return None
  slots = U.enumerate_slots(ui)
  same = [t for t in slots if C.family_of_sel(t.sel) == fam]
  # a control family repeated on the screen is a set of entity instances (cards, rows): which one is a task
  # parameter, never a navigation decision the graph may take
  if len(same) != 1:
    return None
  sl = same[0]
  if not sl.safe or sl.sel.get("in_list") or full9.effectful(sl.sel) or sl.unsafe_reason in ("unsafe_label", "dialog_action_button"):
    return None
  return {"slot": sl, "fam": fam, "key": key, "pred_act": act, "intent": round(top, 3), "margin": round(top - second, 3),
          "cos_use": round(cu, 3), "n_valid": intent.n_valid[key]}


def try_macro(ep, ui, pre, agent, steps, k, own, m, graph, picker=None):
  """Execute up to MAX_HOPS validated navigation hops without inference; one counted step."""
  import seq_run as S  # pylint: disable=import-outside-toplevel
  hops, cur = 0, pre
  for _ in range(MAX_HOPS):
    live = cur["ui"]
    if full9.transient(live):
      break
    p = (picker or pick)(ep, live, graph)
    if p is None:
      if hops == 0:
        m["macro_none"] += 1
      break
    node = live.nodes[p["slot"].node_idx]
    action = json_action.JSONAction(action_type="click", x=node.center[0], y=node.center[1])
    agent._execute_action(action, {})  # pylint: disable=protected-access
    ep._exec_log.append({"action": {kk: vv for kk, vv in action.__dict__.items() if vv is not None}, "commit": False})  # pylint: disable=protected-access
    lab = U.label_of(live, node) or p["slot"].sel.get("desc") or p["slot"].sel.get("rid")
    summ = f"[graph] Tapped \"{lab}\"."
    agent._actions.append({"summary": summ})  # pylint: disable=protected-access
    agent._summaries.append(summ)  # pylint: disable=protected-access
    own.append(S.own_entry(live, p["fam"], p["fam"]))
    row = {"step": k, "kind": "graph_macro", "hop": hops, "tok": p["fam"], "summary": summ,
           "evidence": {kk: vv for kk, vv in p.items() if kk not in ("slot",)}, "pre": R.strip(cur)}
    steps.append(row)
    ep.log("steps", row)
    ep._macro_used.add((JG.page_key(live), p["fam"]))  # pylint: disable=protected-access
    ep._last_macro_key = p["key"]  # pylint: disable=protected-access
    hops += 1
    m["macro_hops"] += 1
    cur = full7.capture_stable(ep.dev, f"s{k}_macro{hops}")
    if cur["ui"].activity != p["pred_act"]:
      ep._macro_brake = True  # pylint: disable=protected-access
      m["macro_brake_landing"] += 1
      ep._intent.misuse(p["key"])  # pylint: disable=protected-access
      break
  if hops:
    m["macro_steps"] += 1
    ep._last_macro_step = k  # pylint: disable=protected-access
  return hops
