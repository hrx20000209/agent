"""FULL9 additions: function-graph execution, TF-IDF-ordered exploration, hard stuck rules.

FuncExec
  Library = functions (parameterized sub-trajectories) of OTHER templates
  (history, leave-one-template-out) + functions extracted from the current
  episode as soon as the model commits a record itself.
  Authorization: same operation verb, every slot field of the function is
  present in the goal, first step groundable on the live screen.
  Execution: bind slots to the goal values of the next unprocessed record (goal
  order), run the steps with verification (target present; typed field shows
  the value), stop before the commit (the model commits), one counted step.
  Each record is attempted at most once by a function.
"""

from __future__ import annotations

import json
import os
import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "runtime_graph_exploration" / "scripts"))
import fgraph as FG  # noqa: E402
import memory as MEM  # noqa: E402

C, U, R = MEM.C, MEM.U, MEM.R
from android_world.env import json_action  # noqa: E402

_GOAL_IDF = None


def goal_idf():
  """Word IDF over all AndroidWorld goals seen in history (documents = goals)."""
  global _GOAL_IDF
  if _GOAL_IDF is None:
    df, n = Counter(), 0
    seen = set()
    for meta in (MEM.TRV / "raw/full/BASE").glob("*/p*/meta.json"):
      g = json.loads(meta.read_text())["goal"]
      if g in seen:
        continue
      seen.add(g)
      n += 1
      df.update(set(re.findall(r"[a-z]+", g.lower())))
    _GOAL_IDF = (df, n)
  df, n = _GOAL_IDF
  return lambda w: math.log((1 + n) / (1 + df.get(w, 0)))


def op_similarity(label, ops):
  idf = goal_idf()
  lw = set(w for w in re.findall(r"[a-z]+", (label or "").lower()) if w not in FG.STOP)
  ow = Counter(ops)
  if not lw or not ow:
    return 0.0
  num = sum(ow[w] * idf(w) ** 2 for w in lw if w in ow)
  den = math.sqrt(sum((c * idf(w)) ** 2 for w, c in ow.items())) * math.sqrt(sum(idf(w) ** 2 for w in lw))
  return num / den if den else 0.0


def _base_arm():
  """Arm behaviour name: replicate suffixes (R, R2, W, X) share the base arm's behaviour."""
  a = os.environ.get("FULL_ARM", "")
  return re.sub(r"(R\d*|W|X|D)$", "", a) if re.match(r"^FULL\d+(R\d*|W|X|D)$", a) else a


def field_role(fd):
  """FULL35: a field is an ENTITY slot (the thing the record is about) when the parser named it from the sentence
  (enumeration items, file names, phrase keys); table / key-value columns keep their own name."""
  return "ENTITY" if fd.startswith(("enum:", "file:")) or " " in fd else fd


def _role_adapted(funcs, recs):
  """FULL35: re-bind a function's single ENTITY slot to the current task's single ENTITY field, so a procedure
  learned on "delete recipe X" applies to "delete recipes A, B, C" (slot names differ, the role is the same)."""
  cur = [fd for fd in recs if field_role(fd) == "ENTITY"]
  if len(cur) != 1:
    return []
  out = []
  for f in funcs:
    ent = [fd for fd in f["fields"] if field_role(fd) == "ENTITY"]
    if len(ent) != 1 or ent[0] == cur[0]:
      continue
    g = json.loads(json.dumps(f))
    g["fields"] = sorted(cur[0] if fd == ent[0] else fd for fd in g["fields"])
    for st in g["steps"]:
      if st.get("field") == ent[0]:
        st["field"] = cur[0]
    g["src"] = g.get("src", "") + "#role"
    out.append(g)
  return out


def content_choice(ui, node, fam):
  """NAV_UNAMBIG: the control is picked by content, not by role - one of several same-family controls with no
  label (day cells, list rows) or a label carrying digits (dates, times, counts). A recorded step on such a control
  names the demo's content and must not be replayed on another task's screen."""
  lab = FG.norm(U.label_of(ui, node) or "")
  if re.search(r"\d", lab):
    return True
  same = sum(1 for s_ in U.enumerate_slots(ui) if C.family_of_sel(s_.sel) == fam)
  return same > 1 and not lab


COMMIT_WORDS = {"save", "done", "ok", "add", "create", "send", "apply", "update", "submit", "confirm", "set", "finish"}


class FuncExec:
  def __init__(self, episode, goal):
    self.ep, self.goal = episode, goal
    self.P = FG.parse_goal(goal)
    self.recs = FG.records(self.P)
    if _base_arm() in ("FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and getattr(episode, "out", None) is not None:
      # FULL33: online only - the library holds functions of this run's earlier self-completed episodes
      cache = Path(os.environ.get("SEQ_APPDIR", "/tmp/full")) / os.environ["FULL_ARM"] / "function_cache_online.jsonl"
      cache.parent.mkdir(parents=True, exist_ok=True)
      FG.build_online(Path(episode.out).parents[1], cache)
      self.lib = FG.Library(episode.args.task, cache=cache)
      if _base_arm() in ("FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
        self.lib.funcs = self.lib.funcs + _role_adapted(self.lib.funcs, self.recs)
    else:
      self.lib = FG.Library(episode.args.task)
    self.done = set()             # records finished (closed by a commit)
    self.tried = set()            # records already attempted by a function
    self.applied_rec = None       # record of the last function application, closed by the model's commit
    self.model_rec = None         # record the model is working on (it typed / tapped one of its values)
    self.vals = {}                # normalized value -> record index (multi-record goals)
    for f, vs in self.recs.items():
      if len(vs) > 1:
        for i, v in enumerate(vs):
          self.vals.setdefault(FG.norm(v), i)
    self.own = []

  def _n_recs(self):
    return max([len(v) for v in self.recs.values()] + [1])

  def _value(self, field, rec):
    vals = self.recs.get(field, [])
    if not vals:
      return None
    return vals[rec] if rec < len(vals) else (vals[0] if len(vals) == 1 else None)

  def _ground_first(self, f, ui, rec):
    s0 = f["steps"][0]
    if s0["op"] == "nav":
      return any(C.family_of_sel(sl.sel) == s0["fam"] for sl in U.enumerate_slots(ui))
    if s0["op"] == "type":
      return any(m.editable and m.visible for m in ui.app_nodes())
    val = self._value(s0["field"], rec)
    return val is not None and self._row_with(ui, s0["row"], val) is not None

  @staticmethod
  def _row_with(ui, row, val):
    v = FG.norm(val)
    for n in ui.app_nodes():
      if not n.visible or f"{n.short_cls}|{n.short_rid}" != row:
        continue
      inner = " ".join(FG.norm(x.text or x.desc) for x in ui.app_nodes() if (x.text or x.desc)
                       and n.bounds[0] <= x.center[0] < n.bounds[2] and n.bounds[1] <= x.center[1] < n.bounds[3])
      if v and v in inner:
        return n
    return None

  @property
  def next_rec(self):
    return next((r for r in range(self._n_recs()) if r not in self.done), self._n_recs())

  def observe_model_action(self, ui, action):
    """Track which record the model itself is working on (typed or tapped one of its values)."""
    v = None
    if action.get("action_type") == "input_text":
      v = FG.norm(action.get("text"))
    elif action.get("x") is not None:
      n = U.element_at(ui, int(action["x"]), int(action["y"]))
      if n is not None:
        v = FG.norm(U.label_of(ui, n))
    if v in self.vals:
      self.model_rec = self.vals[v]

  def try_apply(self, pre, agent, steps, k, own):
    rec = self.next_rec
    if rec >= self._n_recs() or rec in self.tried or self.applied_rec is not None or self.model_rec is not None:
      return 0
    ui = pre["ui"]
    multi = {fd for fd, vs in self.recs.items() if len(vs) > 1}
    for f in self.lib.candidates(self.P, ui.activity):
      # a function must be a complete record procedure: >= 2 actions, >= 1 slot, and on multi-record goals it
      # must fill every per-record field (a half-filled record confuses the model more than it helps)
      if len(f["steps"]) < 2 or not f["fields"] or (multi and not multi <= set(f["fields"])):
        continue
      if all(self._value(fd, rec) is not None for fd in f["fields"]) and self._ground_first(f, ui, rec):
        self.tried.add(rec)
        n = self._execute(f, rec, pre, agent, steps, k, own)
        if n:
          self.applied_rec = rec
        return n
    return 0

  def _execute(self, f, rec, pre, agent, steps, k, own):
    cap, n = pre, 0
    for j, st in enumerate(f["steps"]):
      ui = cap["ui"]
      if j > 0:
        cap = self.ep.dev.capture(f"s{k}_fn{j}", settle=True, pause=0.6, full_res=True)
        ui = cap["ui"]
      action, summ = None, ""
      if st["op"] == "nav":
        sl = next((s for s in U.enumerate_slots(ui) if C.family_of_sel(s.sel) == st["fam"]
                   and s.unsafe_reason not in ("unsafe_label", "dialog_action_button")), None)
        if sl is None:
          break
        if os.environ.get("NAV_UNAMBIG") == "1" and content_choice(ui, ui.nodes[sl.node_idx], st["fam"]):
          break
        x, y = ui.nodes[sl.node_idx].center
        lab = U.label_of(ui, ui.nodes[sl.node_idx]) or st.get("label") or sl.sel.get("rid")
        action = json_action.JSONAction(action_type="long_press" if st.get("kind") == "long_press" else "click", x=x, y=y)
        summ = f"[graph] Tapped \"{lab}\"."
      elif st["op"].startswith("slot_"):
        val = self._value(st["field"], rec)
        node = self._row_with(ui, st["row"], val) if val else None
        if node is None:
          break
        x, y = node.center
        action = json_action.JSONAction(action_type="long_press" if st["op"] == "slot_long_press" else "click", x=x, y=y)
        summ = f"[graph] Tapped \"{val}\"."
      elif st["op"] == "type":
        val = self._value(st["field"], rec)
        edits = sorted([m for m in ui.app_nodes() if m.editable and m.visible], key=lambda m: (m.bounds[1], m.bounds[0]))
        sel = st.get("sel") or {}
        # relocation: same resource-id > same position among the screen's fields > the focused field
        node = next((m for m in edits if sel.get("rid") and m.short_rid == sel["rid"]), None)
        if node is None and sel.get("k") is not None and sel["k"] < len(edits):
          node = edits[sel["k"]]
        if node is None:
          node = next((m for m in edits if m.focused), None)
        if node is None or val is None:
          break
        x, y = node.center
        action = json_action.JSONAction(action_type="input_text", text=val, x=x, y=y)
        summ = f"[graph] Typed \"{val}\"."
      if action is None:
        break
      agent._execute_action(action, {})  # pylint: disable=protected-access
      if getattr(self, "exec_log", None) is not None:
        self.exec_log.append({"action": {kk: vv for kk, vv in action.__dict__.items() if vv is not None}, "commit": False})
      agent._actions.append({"summary": summ})  # pylint: disable=protected-access
      agent._summaries.append(summ)  # pylint: disable=protected-access
      tok = C.action_token(ui, action.__dict__)[0]
      import seq_run as S  # noqa: E402
      own.append(S.own_entry(ui, tok, tok))
      row = {"step": k, "kind": "function", "tok": tok, "summary": summ, "src": f.get("src"), "rec": rec,
             "pre": R.strip(cap)}
      steps.append(row)
      self.ep.log("steps", row)
      n += 1
    else:
      # AGGR_COMMIT (seen-task split): a function recorded on the SAME template also taps the commit the model
      # tapped there, so the record is closed without an inference call
      if n and f.get("complete") and f.get("template") == MEM.template_key(self.ep.args.task):
        n += self._aggr_commit(f.get("commit"), f.get("src"), rec, k, agent, steps, own)
    return n

  def _aggr_commit(self, label, src, rec, k, agent, steps, own):
    if os.environ.get("AGGR_COMMIT") != "1" or not label or label.split()[0] not in COMMIT_WORDS:
      return 0                    # only a real commit (a time / share / format control is not one)
    cap = self.ep.dev.capture(f"s{k}_fnc", settle=True, pause=0.6, full_res=True)
    ui = cap["ui"]
    if ui.ime_visible():
      agent._execute_action(json_action.JSONAction(action_type="navigate_back"), {})  # pylint: disable=protected-access
      cap = self.ep.dev.capture(f"s{k}_fnc_noime", settle=True, pause=0.6, full_res=True)
      ui = cap["ui"]
    sl = next((s_ for s_ in U.enumerate_slots(ui) if s_.unsafe_reason in ("unsafe_label", "dialog_action_button")
               and FG.norm(U.label_of(ui, ui.nodes[s_.node_idx])) == label), None)
    if sl is None:
      return 0
    x, y = ui.nodes[sl.node_idx].center
    action = json_action.JSONAction(action_type="click", x=x, y=y)
    summ = f"[graph] Tapped \"{label}\"."
    agent._execute_action(action, {})  # pylint: disable=protected-access
    if getattr(self, "exec_log", None) is not None:
      self.exec_log.append({"action": {kk: vv for kk, vv in action.__dict__.items() if vv is not None}, "commit": True})
    agent._actions.append({"summary": summ})  # pylint: disable=protected-access
    agent._summaries.append(summ)  # pylint: disable=protected-access
    tok = C.action_token(ui, action.__dict__)[0]
    import seq_run as S  # noqa: E402
    own.append(S.own_entry(ui, tok, tok))
    row = {"step": k, "kind": "function", "tok": tok, "summary": summ, "src": src, "rec": rec, "commit": True,
           "pre": R.strip(cap)}
    steps.append(row)
    self.ep.log("steps", row)
    self.graph_commit = True
    return 1

  def _same_template_commit(self, pkg):
    """Commit label the model used on this template's seen episodes in this app (most frequent)."""
    tk = MEM.template_key(self.ep.args.task)
    c = Counter(f.get("commit") for f in self.lib.funcs if f.get("template") == tk and f.get("pkg") == pkg and f.get("commit") and f.get("complete"))
    return c.most_common(1)[0][0] if c else None

  def try_apply_via_edge(self, ep, graph, pre, agent, steps, k, own, m):
    """A function whose entry screen is one observed edge away (an agent edge or an edge observed only by
    exploration): take that edge, then run the function. Counted as one step."""
    rec = self.next_rec
    if rec >= self._n_recs() or rec in self.tried or self.applied_rec is not None or self.model_rec is not None:
      return 0
    ui = pre["ui"]
    multi = {fd for fd, vs in self.recs.items() if len(vs) > 1}
    entry = {f["entry_act"] for f in self.lib.funcs
             if f["verb"] is not None and f["verb"] == self.P["verb"] and f["entry_act"] != ui.activity
             and len(f["steps"]) >= 2 and f["fields"] and not (multi and not multi <= set(f["fields"]))
             and all(self._value(fd, rec) is not None for fd in f["fields"])}
    if not entry:
      m["func_edge_no_entry"] += 1
      return 0
    m["func_edge_has_entry"] += 1
    for sl in U.enumerate_slots(ui):
      if not sl.safe or sl.sel.get("in_list"):
        continue
      obs = graph.slots.get((ui.struct_sig, sl.slot_id), [])
      if not any(o.get("dst") and o["dst"].get("activity") in entry for o in obs):
        continue
      via_explore = all(o.get("origin") == "explore" for o in obs)
      node = ui.nodes[sl.node_idx]
      action = json_action.JSONAction(action_type="click", x=node.center[0], y=node.center[1])
      agent._execute_action(action, {})  # pylint: disable=protected-access
      if getattr(self, "exec_log", None) is not None:
        self.exec_log.append({"action": {kk: vv for kk, vv in action.__dict__.items() if vv is not None}, "commit": False})
      lab = U.label_of(ui, node) or sl.sel.get("desc") or sl.sel.get("rid")
      summ = f"[graph] Tapped \"{lab}\"."
      agent._actions.append({"summary": summ})  # pylint: disable=protected-access
      agent._summaries.append(summ)  # pylint: disable=protected-access
      row = {"step": k, "kind": "function_entry", "tok": C.family_of_sel(sl.sel), "summary": summ,
             "via_explored_edge": via_explore, "pre": R.strip(pre)}
      steps.append(row)
      ep.log("steps", row)
      import full7  # noqa: E402
      pre2 = full7.capture_stable(ep.dev, f"s{k}_fentry")
      n = self.try_apply(pre2, agent, steps, k, own)
      m["func_entry_edges"] += 1
      m["func_entry_via_explored_edge"] += via_explore
      return 1 + n
    m["func_edge_no_edge"] += 1
    return 0

  def try_apply_via_edge2(self, ep, graph, pre, agent, steps, k, own, m):
    """FULL12. A function can start on a screen the graph has OBSERVED one edge away from here: the destination
    capture of that edge (recorded by the agent or only by exploration) grounds the function's first step with
    this goal's values. Take the edge, re-ground on the live screen, run the function."""
    rec = self.next_rec
    if rec >= self._n_recs() or rec in self.tried or self.applied_rec is not None or self.model_rec is not None:
      return 0
    ui = pre["ui"]
    multi = {fd for fd, vs in self.recs.items() if len(vs) > 1}
    funcs = [f for f in self.lib.funcs
             if f["verb"] is not None and f["verb"] == self.P["verb"] and len(f["steps"]) >= 2 and f["fields"]
             and not (multi and not multi <= set(f["fields"])) and all(self._value(fd, rec) is not None for fd in f["fields"])]
    if not funcs:
      m["func_edge_no_func"] += 1
      return 0
    raw = ep.out.parents[1]
    best = None
    for sl in U.enumerate_slots(ui):
      if not sl.safe or sl.sel.get("in_list"):
        continue
      # agent transitions are keyed by control family, exploration transitions by slot id
      obs = graph.slots.get((ui.struct_sig, C.family_of_sel(sl.sel)), []) + graph.slots.get((ui.struct_sig, sl.slot_id), [])
      agent_seen = any(o.get("origin") != "explore" for o in obs)
      for o in obs:
        d = o.get("dst")
        if not d or d.get("package") != ui.package or d.get("struct_sig") == ui.struct_sig or not o.get("dst_cid"):
          continue
        task, var = o["instance"].split("/")
        try:
          dui = R.load_ui(raw / task / var.split("_")[0], o["dst_cid"])
        except Exception:  # pylint: disable=broad-except
          continue
        if any(f["entry_act"] == dui.activity and self._ground_first(f, dui, rec) for f in funcs):
          best = (sl, not agent_seen)
          break
      if best:
        break
    if best is None:
      m["func_edge_no_edge"] += 1
      return 0
    sl, via_explore = best
    node = ui.nodes[sl.node_idx]
    action = json_action.JSONAction(action_type="click", x=node.center[0], y=node.center[1])
    agent._execute_action(action, {})  # pylint: disable=protected-access
    if getattr(self, "exec_log", None) is not None:
      self.exec_log.append({"action": {kk: vv for kk, vv in action.__dict__.items() if vv is not None}, "commit": False})
    lab = U.label_of(ui, node) or sl.sel.get("desc") or sl.sel.get("rid")
    summ = f"[graph] Tapped \"{lab}\"."
    agent._actions.append({"summary": summ})  # pylint: disable=protected-access
    agent._summaries.append(summ)  # pylint: disable=protected-access
    row = {"step": k, "kind": "function_entry", "tok": C.family_of_sel(sl.sel), "summary": summ,
           "via_explored_edge": via_explore, "pre": R.strip(pre)}
    steps.append(row)
    ep.log("steps", row)
    import full7  # noqa: E402
    pre2 = full7.capture_stable(ep.dev, f"s{k}_fentry")
    n = self.try_apply(pre2, agent, steps, k, own)
    m["func_entry_edges"] += 1
    m["func_entry_via_explored_edge"] += via_explore
    m["func_entry_then_applied"] += bool(n)
    return 1 + n

  # ---- FULL14: schema-level form function ------------------------------------------------------------
  # A record of the goal (field -> value) is itself a parameterized procedure on any screen whose text fields are
  # named by those fields: type(field_1) ... type(field_n), stop before the commit. The entry screen of this
  # function is grounded from the screen itself (live, or a destination capture in the graph), not from history.
  def _form_plan(self, ui, rec):
    if getattr(self, "form_v2", False):
      return self._form_plan_v2(ui, rec)
    edits = [n for n in ui.app_nodes() if n.editable and n.visible]
    plan, used = [], set()
    head = next(iter(self.recs), None)
    for fd in self.recs:
      ft, val = field_tokens(fd), self._value(fd, rec)
      if not ft or val is None:
        continue
      for n in edits:
        if n.idx in used:
          continue
        lab = edit_label_tokens(ui, n)
        if field_matches(fd, ui, n, head):
          used.add(n.idx)
          cur = set(_toks(n.text))
          if FG.norm(n.text) == FG.norm(val):
            break                     # already filled with this value
          # the field's text is a hint only if the field's other name evidence (rid, description, caption)
          # explains it; anything else is content (typed by the model or by a function): never overwrite
          if n.text and not cur <= edit_label_tokens(ui, n, with_text=False):
            break
          plan.append((fd, val, n))
          break
    return plan

  def _form_plan_v2(self, ui, rec):
    """FULL28: hint-aware emptiness, field synonyms, file-name fields; a field is filled only when exactly one
    empty text field on the screen is named by it."""
    edits = [n for n in ui.app_nodes() if n.editable and n.visible]
    head = next(iter(self.recs), None)
    plan, used = [], set()
    v3 = getattr(self, "form_v3", False)
    filled = getattr(self, "form_filled", set())
    for fd in self.recs:
      val = self._value(fd, rec)
      if val is None or (v3 and (rec, fd) in filled):
        continue                           # FULL29: each field of a record is typed at most once
      cands = [n for n in edits if n.idx not in used and field_matches_v2(fd, ui, n, head, self.P.get("verb"), v3)]
      if len(cands) != 1:
        continue
      n = cands[0]
      used.add(n.idx)
      if FG.norm(n.text) == FG.norm(val) or not edit_is_empty(n):
        continue
      plan.append((fd, val, n))
    return plan

  def form_pending(self):
    rec = self.next_rec
    return (rec < self._n_recs() and rec not in getattr(self, "form_tried", set()) and self.applied_rec is None
            and sum(1 for fd in self.recs if field_tokens(fd) and self._value(fd, rec) is not None) >= 2)

  def try_form(self, pre, agent, steps, k, own, m, via=None):
    rec = self.next_rec
    self.form_tried = getattr(self, "form_tried", set())
    if getattr(self, "form_retrigger", False):
      # FULL18: the record's form is filled wherever its empty fields appear (after a library function, after the
      # model started, after scrolling), once per (record, fields still empty on this screen)
      if rec >= self._n_recs() or (self.applied_rec is not None and self.applied_rec != rec) or \
         (self.model_rec is not None and self.model_rec != rec):
        return 0
      min_fields = 1 if getattr(self, "form_v2", False) else 2
      if sum(1 for fd in self.recs if (field_tokens(fd) or fd.startswith("file:")) and self._value(fd, rec) is not None) < min_fields:
        return 0
      plan = self._form_plan(pre["ui"], rec)
      key = (rec, frozenset(fd for fd, _, _ in plan))
      if not plan or key in self.form_tried:
        return 0
      self.form_tried.add(key)
      self.form_filled = getattr(self, "form_filled", set()) | {(rec, fd) for fd, _, _ in plan}
      ui = pre["ui"]
    else:
      if rec >= self._n_recs() or rec in self.form_tried or self.applied_rec is not None or self.model_rec is not None:
        return 0
      ui = pre["ui"]
      plan = self._form_plan(ui, rec)
      if len(plan) < 2:
        return 0
      self.form_tried.add(rec)
    self.tried.add(rec)
    n, cap = 0, pre
    for j, (fd, val, _) in enumerate(plan):
      if j > 0:
        cap = self.ep.dev.capture(f"s{k}_form{j}", settle=True, pause=0.6, full_res=True)
      live = cap["ui"]
      node = next((x for x in live.app_nodes() if x.editable and x.visible and
                   ((lambda a1, a2, a3, a4: field_matches_v2(a1, a2, a3, a4, self.P.get("verb"))) if getattr(self, "form_v2", False)
                    else field_matches)(fd, live, x, next(iter(self.recs), None))), None)
      if node is None and live.ime_visible():
        agent._execute_action(json_action.JSONAction(action_type="navigate_back"), {})  # pylint: disable=protected-access
        cap = self.ep.dev.capture(f"s{k}_form{j}_noime", settle=True, pause=0.6, full_res=True)
        live = cap["ui"]
        node = next((x for x in live.app_nodes() if x.editable and x.visible and
                   ((lambda a1, a2, a3, a4: field_matches_v2(a1, a2, a3, a4, self.P.get("verb"))) if getattr(self, "form_v2", False)
                    else field_matches)(fd, live, x, next(iter(self.recs), None))), None)
      if node is None:
        break
      x, y = node.center
      action = json_action.JSONAction(action_type="input_text", text=val, x=x, y=y, clear_text=bool(node.text))
      agent._execute_action(action, {})  # pylint: disable=protected-access
      if getattr(self, "exec_log", None) is not None:
        self.exec_log.append({"action": {kk: vv for kk, vv in action.__dict__.items() if vv is not None}, "commit": False})
      summ = f"[graph] Typed \"{val}\" into {fd}."
      agent._actions.append({"summary": summ})  # pylint: disable=protected-access
      agent._summaries.append(summ)  # pylint: disable=protected-access
      tok = C.action_token(live, action.__dict__)[0]
      import seq_run as S  # noqa: E402
      own.append(S.own_entry(live, tok, tok))
      row = {"step": k, "kind": "form_function", "tok": tok, "summary": summ, "rec": rec, "field": fd,
             "via": via, "pre": R.strip(cap)}
      steps.append(row)
      self.ep.log("steps", row)
      n += 1
    if n:
      self.applied_rec = rec
      m["form_apps"] += 1
      m["form_steps"] += n
      if _base_arm() in ("FULL44", "FULL45"):
        # tell the model what the graph did NOT fill (choices, pickers, fields off screen), so a partly filled form is
        # never mistaken for a complete one
        done = {fd for fd, _, _ in plan[:n]}
        left = [(fd, self._value(fd, rec)) for fd in self.recs if fd not in done and self._value(fd, rec) is not None
                and not fd.startswith("file:")]
        if _base_arm() == "FULL45" and left:
          # FULL45: a value that is one of the screen's options (a category chip, a list entry) is selected by the
          # graph too: tap it if visible, else scroll the option strip that holds the other options (<= 4 swipes)
          picked = self._select_options(left, agent, steps, k, own, m)
          left = [(fd, v) for fd, v in left if fd not in picked]
        if left:
          summ = "[graph] Not filled yet: " + "; ".join(f"{fd} = {v}" for fd, v in left) + ". Set these before saving."
          agent._actions.append({"summary": summ})  # pylint: disable=protected-access
          agent._summaries.append(summ)  # pylint: disable=protected-access
          m["form_left_fields"] += len(left)
        elif os.environ.get("AGGR_COMMIT") == "1":
          # AGGR_COMMIT: every field of the record is filled: tap the commit this template's seen episodes used
          n += self._aggr_commit(self._same_template_commit(ui.package), "form", rec, k, agent, steps, own)
    return n

  def _select_options(self, left, agent, steps, k, own, m):
    from android_world.env import adb_utils  # noqa: E402
    picked = set()
    for fd, val in left:
      want = FG.norm(str(val))
      if not want or len(want) > 40:
        continue
      for attempt in range(5):
        cap = self.ep.dev.capture(f"s{k}_opt_{len(picked)}_{attempt}", settle=True, pause=0.3, full_res=True)
        ui = cap["ui"]
        hit = next((n for n in ui.app_nodes() if n.visible and not n.editable and FG.norm(n.text or n.desc or "") == want), None)
        if hit is not None:
          x, y = hit.center
          agent._execute_action(json_action.JSONAction(action_type="click", x=x, y=y), {})  # pylint: disable=protected-access
          summ = f"[graph] Selected \"{val}\" for {fd}."
          agent._actions.append({"summary": summ})  # pylint: disable=protected-access
          agent._summaries.append(summ)  # pylint: disable=protected-access
          row = {"step": k, "kind": "form_function", "tok": f"SELECT:{fd}", "summary": summ, "field": fd, "pre": R.strip(cap)}
          steps.append(row)
          self.ep.log("steps", row)
          m["form_options"] += 1
          picked.add(fd)
          break
        # an option strip: a horizontally scrollable container whose children are short clickable labels
        strip = next((n for n in ui.app_nodes() if n.scrollable and n.visible and n.bounds[2] - n.bounds[0] > 2 * (n.bounds[3] - n.bounds[1])), None)
        if strip is None:
          break
        x1, y1, x2, y2 = strip.bounds
        yc = (y1 + y2) // 2
        w = x2 - x1
        # swipe inside the strip, away from the screen edges (an edge swipe is the system Back gesture: FULL45 run 1
        # closed the expense editor this way)
        adb_utils.issue_generic_request(["shell", "input", "swipe", str(x1 + int(0.75 * w)), str(yc), str(x1 + int(0.25 * w)),
                                         str(yc), "400"], self.ep.env.controller)
    return picked

  def try_form_via_edge(self, ep, graph, pre, agent, steps, k, own, m):
    """The form function's entry screen is one observed edge away: the edge's destination capture (recorded by
    the agent, or only by exploration) grounds >= 2 of this record's fields. Take the edge, then fill."""
    rec = self.next_rec
    if rec >= self._n_recs() or rec in getattr(self, "form_tried", set()) or self.applied_rec is not None or self.model_rec is not None:
      return 0
    if not any(field_tokens(fd) and self._value(fd, rec) is not None for fd in self.recs):
      return 0
    ui = pre["ui"]
    raw = ep.out.parents[1]
    best = None
    for sl in U.enumerate_slots(ui):
      if not sl.safe or sl.sel.get("in_list") or effectful_v2(sl.sel):
        continue
      obs = graph.slots.get((ui.struct_sig, C.family_of_sel(sl.sel)), []) + graph.slots.get((ui.struct_sig, sl.slot_id), [])
      agent_seen = any(o.get("origin") != "explore" for o in obs)
      for o in obs:
        d = o.get("dst")
        if not d or d.get("package") != ui.package or d.get("struct_sig") == ui.struct_sig or not o.get("dst_cid"):
          continue
        task, var = o["instance"].split("/")
        try:
          dui = R.load_ui(raw / task / var.split("_")[0], o["dst_cid"])
        except Exception:  # pylint: disable=broad-except
          continue
        if len(self._form_plan(dui, rec)) >= 2:
          best = (sl, not agent_seen)
          break
      if best:
        break
    m["form_edge_lookups"] += 1
    if best is None:
      return 0
    sl, via_explore = best
    node = ui.nodes[sl.node_idx]
    action = json_action.JSONAction(action_type="click", x=node.center[0], y=node.center[1])
    agent._execute_action(action, {})  # pylint: disable=protected-access
    if getattr(self, "exec_log", None) is not None:
      self.exec_log.append({"action": {kk: vv for kk, vv in action.__dict__.items() if vv is not None}, "commit": False})
    lab = U.label_of(ui, node) or sl.sel.get("desc") or sl.sel.get("rid")
    summ = f"[graph] Tapped \"{lab}\"."
    agent._actions.append({"summary": summ})  # pylint: disable=protected-access
    agent._summaries.append(summ)  # pylint: disable=protected-access
    row = {"step": k, "kind": "form_entry", "tok": C.family_of_sel(sl.sel), "summary": summ,
           "via_explored_edge": via_explore, "pre": R.strip(pre)}
    steps.append(row)
    ep.log("steps", row)
    import full7  # noqa: E402
    pre2 = full7.capture_stable(ep.dev, f"s{k}_formentry")
    n = self.try_form(pre2, agent, steps, k, own, m, via="explore" if via_explore else "agent")
    m["form_entry_edges"] += 1
    m["form_entry_via_explored_edge"] += via_explore
    m["form_entry_then_filled"] += bool(n)
    return 1 + n

  def on_model_commit(self):
    """The model committed: the open record (function-applied or the model's own) is done (idempotent for
    two-stage commits such as Delete + Confirm); extract this episode's own functions."""
    if self.applied_rec is not None:
      self.done.add(self.applied_rec)
      self.applied_rec = None
    elif self.model_rec is not None:
      self.done.add(self.model_rec)
    elif self._n_recs() == 1 and not getattr(self, "strict_done", False):
      self.done.add(0)            # FULL14 (strict_done): a commit before any of the record's values was entered
                                  # (dismissing a popup) does not complete the record
    self.model_rec = None
    try:
      fs = FG.extract(self.ep.out, self.goal)
      for f in fs:
        f["src"] = "self"
      self.lib.add(fs)
    except Exception:  # pylint: disable=broad-except
      pass


def frontier_order(ui, slots, ops, stats_apps):
  """Unknown safe slots ordered by (goal-operation similarity desc, cross-app generality asc, screen order)."""
  def key(item):
    i, sl = item
    lab = sl.sel.get("label") or sl.sel.get("desc") or sl.sel.get("rid") or ""
    ident = FG.norm(lab)
    return (-op_similarity(lab, ops), stats_apps.get(ident, 0), i)
  return [sl for _, sl in sorted(enumerate(slots), key=key)]


def transient(ui):
  """Popup menu / dialog (app nodes in more than one window), keyboard, selection mode, open drawer."""
  app = [n for n in ui.app_nodes() if n.visible]
  if len({n.window for n in app}) > 1 or ui.ime_visible():
    return True
  if any(re.match(r"^(\d+\s+selected|selected\b)", (n.text or "").strip(), re.I) for n in app):
    return True
  if any("action_mode" in (n.rid or "") for n in app):
    return True
  return any("drawer" in (n.rid or "").lower() or "navigation_view" in (n.rid or "").lower() for n in app)


def decision_still_valid(ui_pre, ui_now, action):
  """After probing, is the model's decision (made on ui_pre) still executable as intended on ui_now?
  Same page (activity + layout) and, for targeted actions, the same control at the same point."""
  if ui_now.activity != ui_pre.activity or ui_now.layout_sig != ui_pre.layout_sig:
    return False
  if action.x is not None and action.action_type in ("click", "long_press", "double_tap", "input_text"):
    n0 = U.element_at(ui_pre, int(action.x), int(action.y))
    n1 = U.element_at(ui_now, int(action.x), int(action.y))
    if n0 is None or n1 is None:
      return False
    return (C.family_of_sel(U.selector(ui_pre, n0)) == C.family_of_sel(U.selector(ui_now, n1))
            and U.label_of(ui_pre, n0) == U.label_of(ui_now, n1))
  return True


def replayable(log):
  """The path since the last app launch can be re-executed from a fresh launch: it starts with the launch, contains
  no commit and no untracked graph execution, and is short."""
  return bool(log) and log[0]["action"] is not None and log[0]["action"].get("action_type") == "open_app" \
      and not any(e["commit"] or e["action"] is None for e in log) and len(log) <= 12


def relaunch_restore(ep, target_ui, step):
  """Graph replay restoration: kill the app, relaunch it and re-execute the path recorded since the last app
  launch. Only when that path contains no commit (re-executing a commit would repeat its side effect)."""
  import subprocess
  import time as _t
  log = getattr(ep, "_exec_log", [])
  if (not log or log[0]["action"] is None or log[0]["action"].get("action_type") != "open_app"
      or any(e["commit"] or e["action"] is None for e in log) or len(log) > 12):
    return False
  subprocess.run([R.ADB, "-s", "emulator-5554", "shell", "am", "force-stop", target_ui.package],
                 capture_output=True, timeout=30)
  for e in log:
    a = json_action.JSONAction(**e["action"])
    ep._agent._execute_action(a, {})  # pylint: disable=protected-access
    _t.sleep(1.0)
  cap = ep.dev.capture(f"s{step}_relaunch", settle=True, pause=1.0)
  ok = cap["ui"].activity == target_ui.activity and cap["ui"].layout_sig == target_ui.layout_sig
  # the screen came back; did its data? (a probe that added/changed data leaves different content behind)
  ep._relaunch_content_same = cap["ui"].content_sig == target_ui.content_sig
  ep.log("restore", {"step": step, "method": "relaunch_replay", "n_actions": len(log), "ok": ok})
  return ok


# Exploration must be side-effect free. A control whose label is an operation (verb) changes app data, and
# restoring the screen (Back / relaunch) does not undo it: "Lap", "Add to playing queue", "Start", "Delete".
EFFECT_VERBS = {"add", "new", "create", "delete", "remove", "save", "send", "start", "stop", "pause", "resume", "lap",
                "reset", "play", "record", "clear", "share", "apply", "set", "mark", "archive", "copy", "move", "rename",
                "favourite", "favorite", "bookmark", "unbookmark", "pin", "unpin", "duplicate", "import", "export",
                "install", "uninstall", "enable", "disable", "turn", "submit", "confirm", "ok", "done", "yes", "allow",
                "accept", "get", "skip", "dismiss", "undo", "redo", "restore", "empty", "download", "upload", "sync",
                "call", "dial", "reply", "forward", "snooze", "shuffle", "repeat", "like", "rate", "take", "capture",
                "shutter", "update", "trash"}


def effectful(sel):
  lab = (sel.get("label") or sel.get("desc") or sel.get("text") or "").strip().lower()
  rid = (sel.get("rid") or "").lower()
  first = re.split(r"[^a-z]+", lab)[0] if lab else ""
  rid_toks = set(re.split(r"[^a-z]+", rid)) if rid else set()
  return first in EFFECT_VERBS or (not lab and bool(rid_toks & EFFECT_VERBS)) or sel.get("cls") in ("Switch", "CheckBox", "ToggleButton", "RadioButton")


_DROP_FIELD = {"dollars", "dollar", "usd", "value", "the", "a", "of"}


def _toks(x):
  return [t for t in re.split(r"[^a-z]+", (x or "").lower()) if t]


def field_tokens(fd):
  """Name tokens of a goal field ("amount_dollars" -> {amount}, "first name" -> {first}); value-only slots
  (enumerations, file names) have no name."""
  if fd.startswith(("enum:", "file:")):
    return set()
  t = set(_toks(fd)) - _DROP_FIELD
  if len(t) > 1:
    t -= {"name"}
  return t


def edit_label_tokens(ui, n, with_text=True):
  """Name of a text field: its hint/text, description, resource-id and the closest caption right above it."""
  labs = ([n.text] if with_text else []) + [n.desc, n.short_rid.replace("_", " ")]
  best = None
  for m in ui.app_nodes():
    if m.editable or not m.visible or not m.text or m is n:
      continue
    if m.bounds[3] <= n.bounds[1] + 5 and n.bounds[1] - m.bounds[3] < 160 and abs(m.bounds[0] - n.bounds[0]) < 300:
      d = n.bounds[1] - m.bounds[3]
      if best is None or d < best[0]:
        best = (d, m.text)
  if best:
    labs.append(best[1])
  return {t for l in labs for t in _toks(l)}


# FULL14: opening an editor ("Add", "New event", "+") changes no data until it is saved, and is exactly how the
# entry screens of form functions are found; "Add to <x>" acts on an existing item and is a commit.
ENTRY_VERBS = {"add", "new", "create"}


def effectful_v2(sel):
  lab = (sel.get("label") or sel.get("desc") or sel.get("text") or "").strip().lower()
  rid_toks = set(_toks(sel.get("rid") or ""))
  t = _toks(lab)
  if t and t[0] in ENTRY_VERBS and "to" not in t:
    return False
  if not t and rid_toks & ENTRY_VERBS:
    return False
  return effectful(sel)


def opens_editor(sel):
  lab = (sel.get("label") or sel.get("desc") or sel.get("text") or "").strip().lower()
  t = _toks(lab)
  return bool((t and t[0] in ENTRY_VERBS and "to" not in t) or (not t and set(_toks(sel.get("rid") or "")) & ENTRY_VERBS))


ENTITY_NAME_FIELDS = ({"title"}, {"name"})


def field_matches(fd, ui, n, head=None):
  """A goal field names a text field when its name tokens are among the field's label tokens, or its compact
  name ("preparationtime") occurs in the field's compact resource-id/label; the record's head field (the entity
  itself, "Recipe: <x>") names the screen's title/name field."""
  ft, lab = field_tokens(fd), edit_label_tokens(ui, n)
  if not ft:
    return False
  if fd == head and not ({"title"} <= lab or {"name"} <= lab or ft <= set(_toks(n.short_rid)) | set(_toks(n.desc))):
    return False          # the record's head is the entity itself: only its own title/name field names it
  if ft <= lab:
    return True
  compact = "".join(_toks(fd))
  if len(compact) >= 6 and (compact in "".join(_toks(n.short_rid)) or compact in "".join(_toks(n.text))):
    return True
  return fd == head and any(e <= lab for e in ENTITY_NAME_FIELDS) and not any(f <= lab for f in (ft,))



# ---- FULL15: graph-aware exploration -----------------------------------------------------------------------
def page_family_index(graph):
  """(package, activity, control family) of every transition the graph holds (agent or exploration), and the
  subset exploration has seen reversed by Back."""
  known, reversible = set(), set()
  by_page = {}
  for (struct, sid), obs in graph.slots.items():
    for o in obs:
      fam = sid if sid.startswith("E|") else C.family_of_sel(o.get("selector") or {}) if o.get("selector") else "E|" + sid
      k = (o["src"]["package"], o["src"]["activity"], fam)
      known.add(k)
      if o.get("dst"):
        by_page.setdefault(k, []).append(o)
      if o.get("origin") == "explore" and o.get("rollback_ok"):
        reversible.add(k)
  graph._by_page = by_page
  return known, reversible


def entry_tokens(fx, ui):
  """Information need of a pending library function that does not ground here: tokens of its first control and
  of its entry screen's name."""
  if fx is None:
    return set()
  rec = fx.next_rec
  if rec >= fx._n_recs() or fx.applied_rec is not None or fx.model_rec is not None:
    return set()
  toks = set()
  for f in fx.lib.funcs:
    if f["verb"] is None or f["verb"] != fx.P["verb"] or len(f["steps"]) < 2:
      continue
    if not all(fx._value(fd, rec) is not None for fd in f["fields"]):
      continue
    if f["entry_act"] == ui.activity and fx._ground_first(f, ui, rec):
      return set()                       # it grounds here: nothing to look for
    s0 = f["steps"][0]
    toks |= set(_toks((s0.get("fam") or "").split("|", 1)[-1]))
    act = f["entry_act"].rsplit(".", 1)[-1]
    toks |= set(_toks(re.sub(r"([a-z])([A-Z])", r"\1 \2", act))) - {"activity", "fragment", "main"}
  return toks - {"e", "button", "imagebutton", "textview", "view", "layout", "linearlayout", "framelayout"}



def container_key(package, activity, sel):
  """Role of a control = the nearest named container it sits in (tab bar, keypad, toolbar). Controls of one
  container share their back-stack behaviour: a tab switch or a keypad digit is undone by Back for none of them."""
  anc = sel.get("ancestry") or []
  named = next((a for a in anc if a.split("|", 1)[-1]), None)
  if named is None or named.split("|", 1)[-1].lower() in _GENERIC_CONTAINERS:
    return (package, activity, C.family_of_sel(sel))       # no specific role container: the control itself
  return (package, activity, named)


_GENERIC_CONTAINERS = {"content", "action_bar_root", "decor_content_parent", "root", "main", "container",
                       "main_content", "fragment_container", "nav_host_fragment", "coordinator", "parentpanel"}



# ---- reversibility prediction (FULL15) -----------------------------------------------------------------------
_REV_PRIOR = None


def rev_prior():
  global _REV_PRIOR
  if _REV_PRIOR is None:
    try:
      _REV_PRIOR = json.loads((HERE.parent.parent / "runtime_graph_exploration" / "graph" / "rev_prior.json").read_text())
    except Exception:  # pylint: disable=broad-except
      _REV_PRIOR = {}
  return _REV_PRIOR


def container_evidence(graph):
  """Per (package, activity, container): exploration probes Back did / did not undo, from the graph."""
  ev = {}
  for (struct, sid), obs in graph.slots.items():
    for o in obs:
      if o.get("origin") != "explore" or not o.get("selector") or o.get("depth", 1) != 1:
        continue
      k = container_key(o["src"]["package"], o["src"]["activity"], o["selector"])
      e = ev.setdefault(k, [0, 0])
      e[0 if o.get("rollback_ok") else 1] += 1
  return ev


def app_kind_evidence(graph, split_rows=False):
  """FULL34: per (package, probe kind) - how often Back undid this run's depth-1 probes of that kind in the app
  (kind: "discover" = a control never seen before, "revalidate" = re-observing a known edge)."""
  ev = {}
  for (struct, sid), obs in graph.slots.items():
    for o in obs:
      if o.get("origin") != "explore" or o.get("depth", 1) != 1 or not o.get("src"):
        continue
      kind = o.get("priority") or "discover"
      if split_rows and (o.get("selector") or {}).get("in_list"):
        kind += "_row"                   # FULL35: entity rows and controls are different populations
      k = (o["src"].get("package"), kind)
      e = ev.setdefault(k, [0, 0])
      e[0 if o.get("rollback_ok") else 1] += 1
  return ev


def p_reversible_h(ui, sel, evidence, app_ev, kind, strength=5.0, app_strength=10.0):
  """FULL34: hierarchical Beta posterior. Level 1: cross-app role x zone prior p0. Level 2: this app's probes of
  the same kind in this run, mean mu = (s0 p0 + ok_app) / (s0 + n_app). Level 3: the control's container evidence
  with prior mu. A new container in an app whose discoveries keep failing inherits that app's failure rate."""
  import rev_prior as RP  # noqa: E402
  ft = RP.features(ui, sel)
  pri = rev_prior()
  p0 = 0.75
  if ft is not None:
    for k in ("role_zone|" + ft["role"] + "/" + ft["zone"], "role|" + ft["role"]):
      v = pri.get(k)
      if v and v["n"] >= 10:
        p0 = (v["ok"] + 1) / (v["n"] + 2)
        break
  ok_a, fail_a = app_ev.get((ui.package, kind), (0, 0))
  mu = (app_strength * p0 + ok_a) / (app_strength + ok_a + fail_a)
  ok, fail = evidence.get(container_key(ui.package, ui.activity, sel), (0, 0))
  return (strength * mu + ok) / (strength + ok + fail)


def p_reversible(ui, sel, evidence, strength=5.0):
  """Beta posterior: cross-app prior of the control's role (sibling set / selected sibling set / single) and
  screen zone, updated with this app's evidence for the control's container."""
  import rev_prior as RP  # noqa: E402
  ft = RP.features(ui, sel)
  pri = rev_prior()
  p0 = 0.75
  if ft is not None:
    for k in ("role_zone|" + ft["role"] + "/" + ft["zone"], "role|" + ft["role"]):
      v = pri.get(k)
      if v and v["n"] >= 10:
        p0 = (v["ok"] + 1) / (v["n"] + 2)
        break
  ok, fail = evidence.get(container_key(ui.package, ui.activity, sel), (0, 0))
  return (strength * p0 + ok) / (strength + ok + fail)



def edits_changed(ui0, ui1):
  """Did the text content of the page's editable fields change (same page, a field's text differs)?"""
  if ui1.activity != ui0.activity:
    return False
  e0 = sorted((n.short_rid, n.text or "") for n in ui0.app_nodes() if n.editable)
  e1 = sorted((n.short_rid, n.text or "") for n in ui1.app_nodes() if n.editable)
  if not e0 or not e1:
    return False
  rid0 = {r for r, _ in e0}
  return any(r in rid0 and t != dict(e0).get(r) for r, t in e1)



def relocate_decision(ui_pre, ui_now, action):
  """FULL21: the model's targeted decision survives a changed screen if the same activity still shows the same
  control (family + label, unique on the screen); the action is moved to that control's current centre in place."""
  if ui_now.activity != ui_pre.activity or action.x is None:
    return False
  if action.action_type not in ("click", "long_press", "double_tap", "input_text"):
    return False
  n0 = U.element_at(ui_pre, int(action.x), int(action.y))
  if n0 is None:
    return False
  fam, lab = C.family_of_sel(U.selector(ui_pre, n0)), U.label_of(ui_pre, n0)
  same = [n for n in ui_now.app_nodes() if n.visible and C.family_of_sel(U.selector(ui_now, n)) == fam
          and U.label_of(ui_now, n) == lab]
  if len(same) != 1:
    return False
  action.x, action.y = same[0].center
  return True



FIELD_SYNONYMS = {"number": {"phone", "number", "tel"}, "phone": {"phone", "number"}, "last": {"surname", "last"},
                  "message": {"message", "body", "text", "editor", "compose", "content"},
                  "content": {"content", "editor", "body", "text", "message"},
                  "text": {"text", "editor", "body", "content"}, "txt": {"editor", "body", "content"},
                  "titled": {"name", "title"}}


def edit_is_empty(n):
  """A text field is empty when it has no text or shows only its hint."""
  return (not n.text) or bool(n.hint and FG.norm(n.text) == FG.norm(n.hint))


def field_matches_v2(fd, ui, n, head=None, verb=None, strict_head=False):
  lab = edit_label_tokens(ui, n) | set(_toks(getattr(n, "hint", "")))
  if fd == "number":
    # a bare number parsed from a date or a name is a parameter only where the field asks for a number
    return bool(lab & {"phone", "tel", "number"})
  if fd.startswith("file:"):
    # a file named in the goal is the thing to create only for creating verbs; otherwise it is a source file
    return verb in ("add", "create", "record", "save") and "name" in lab
  if field_matches(fd, ui, n, head):
    return True
  ft = field_tokens(fd)
  if not ft:
    return False
  if fd == head and not ({"title"} <= lab or {"name"} <= lab):
    # FULL29: the record's head is its title; a field that merely mentions the head word ("Recipe source") is not it
    return False if strict_head else bool(ft <= lab)
  return all(lab & FIELD_SYNONYMS.get(x, {x}) for x in ft)
