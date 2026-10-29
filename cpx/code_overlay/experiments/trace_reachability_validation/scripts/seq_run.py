"""Same-app consecutive tasks with a continuously built graph (one episode).

Arms (env SEQ_ARM):
  BASE : plain agent
  G    : graph from the agent's own executions in this app (continuous across
         tasks/passes) + interventions:
           * prefix jump: follow the stage-aware dominant continuation of
             earlier successful trajectories (all action types counted) as far
             as it is gated, then reach the deepest gated page by the shortest
             known path in the graph; every hop is executed and verified
             (page must equal the expected one, else BACK and hand to model)
           * coordinate repair: a model click that hits no clickable element
             is snapped to the nearest clickable within 80 px
  GX   : G + model-free exploration in the inference window, depth 1:
           page with trace support -> probe up to 2 slots in trace order
           gap page (no support)   -> coverage probing, unknown slots first
         probe edges join the graph, so shortest paths can use them.
State: graph_snapshots/seq/<ARM>/<app>/{store.json, graph.jsonl, unrestorable.jsonl}
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import pa_trace as PA  # noqa: E402
import rgs_run as R  # noqa: E402
from android_world.env import json_action  # noqa: E402
from android_world import shared_state as SS

U = C.U
ARM = os.environ.get("SEQ_ARM", "BASE")
APPDIR = Path(os.environ.get("SEQ_APPDIR", "/tmp/seq"))
GATE_N, GATE_D, MAX_CHAIN = 2, 0.8, 4


def own_entry(ui, tok, fam):
  return {"state": ui.struct_sig, "struct": ui.struct_sig, "layout": ui.layout_sig, "tok": tok, "fam": fam}


class Store:
  """Successful trajectories of earlier episodes in this app (continuous)."""

  def __init__(self, path: Path, only_task: str | None = None):
    self.path = path
    self.only_task = only_task
    self.data = SS.read_json(path, [])
    self.__dict__.pop("_tidx", None)
    if only_task is not None:
      self.data = [d for d in self.data if d["task"] == only_task]
    self.idx = {"struct": PA.Index(), "layout": PA.Index()}
    self.succ = {"struct": defaultdict(Counter), "layout": defaultdict(Counter)}
    for d in self.data:
      for key in ("struct", "layout"):
        s = [dict(x, state=x[key], fam=x["tok"]) for x in d["seq"]]   # stage-aware: all action types
        self.idx[key].add_seq(s)
        for i in range(len(s) - 1):
          self.succ[key][(s[i][key], s[i]["tok"])][s[i + 1][key]] += 1

  def predict(self, own, state):
    """Dominant continuation token (gated) for the current context."""
    for key in ("struct", "layout"):
      seq = [dict(x, state=x[key]) for x in own] + [{"state": state[key], "tok": "", "fam": None}]
      counts = self.idx[key].lookup(seq, len(seq) - 1)
      for k in range(PA.K_MAX, -1, -1):
        c = counts[k]
        if c and sum(c.values()) > 0:
          tok, n = c.most_common(1)[0]
          info = {"key": key, "order": k, "top": tok, "count": n, "total": sum(c.values())}
          if n >= GATE_N and n / sum(c.values()) >= GATE_D:
            return tok, key, info
          return None, key, info
    return None, None, {"order": -1}

  def task_dominants(self, own, state, key):
    """Per task template: dominant continuation (and its support) at the
    longest matched context, using the same key as the global prediction."""
    if not hasattr(self, "_tidx"):
      self._tidx = {}
      for d in self.data:
        ix = self._tidx.setdefault(d["task"], {"struct": PA.Index(), "layout": PA.Index()})
        for k2 in ("struct", "layout"):
          ix[k2].add_seq([dict(x, state=x[k2], fam=x["tok"]) for x in d["seq"]])
    out = {}
    seq = [dict(x, state=x[key]) for x in own] + [{"state": state[key], "tok": "", "fam": None}]
    for task, ix in self._tidx.items():
      counts = ix[key].lookup(seq, len(seq) - 1)
      for k in range(PA.K_MAX, -1, -1):
        c = counts[k]
        if c and sum(c.values()) > 0:
          tok, n = c.most_common(1)[0]
          out[task] = (tok, n, sum(c.values()))
          break
    return out

  def successor(self, key, sig, tok):
    c = self.succ[key].get((sig, tok))
    return c.most_common(1)[0][0] if c else None

  def append(self, seq, task):
    SS.update_json(self.path, lambda data: data + [{"task": task, "seq": seq}], [])
    self.refresh()

  def refresh(self):
    self.__init__(self.path, self.only_task)


class SeqEpisode(R.Episode):
  def run_agent(self, start_step=0, prior_summaries=None, label="agent"):
    goal, agent = self.task.goal, self.make_agent()
    app = C.APP_OF.get(self.args.task, self.args.task).replace(" ", "_")
    adir = APPDIR / ARM / app
    adir.mkdir(parents=True, exist_ok=True)
    store = Store(adir / "store.json") if ARM != "BASE" else None
    self._same = Store(adir / "store.json", only_task=self.args.task) if "C" in ARM else None
    graph = R.Graph(adir / "graph.jsonl", adir / "unrestorable.jsonl") if ARM != "BASE" else R.Graph(None)
    instance = f"{self.args.task}/{self.args.variant}"
    cfg = R.ExploreCfg(enabled=(ARM == "GX"), budget_s=6.0, max_probes=6, max_depth=1)
    explorer = R.Explorer(self.dev, graph, cfg, instance, self.log)
    own, steps, used = [], [], Counter()
    n_calls = n_jump_steps = n_jumps = n_jump_reject = n_repair = n_shortcut = 0
    done, prev, k = False, None, 0
    while k < self.max_steps():
      pre = self.dev.capture(f"s{k}_pre", settle=True, pause=1.0, full_res=True)
      ui = pre["ui"]
      if prev is not None:
        self.ingest_agent(graph, instance, prev, pre)
        prev = None
      # ---------------- prefix jump (graph-guided skipping)
      if store is not None:
        path = self.plan_jump(store, graph, own, ui, used)
        if path:
          n_jumps += 1
          ok_hops = self.exec_jump(path, own, agent, steps, k, explorer, graph, instance)
          n_jump_steps += ok_hops
          n_shortcut += int(bool(path and path[0].get("shortcut")))
          if ok_hops < len(path):
            n_jump_reject += 1
          if ok_hops:
            k += ok_hops
            continue
          pre = self.dev.capture(f"s{k}_pre_after_jump", settle=True, full_res=True)
          ui = pre["ui"]
      # ---------------- model step (+ exploration in the window)
      box = {}
      th = threading.Thread(target=lambda: box.update(d=self.decide(agent, goal, pre["pixels"])), daemon=True)
      t_req = time.time()
      th.start()
      xrec = None
      if cfg.enabled:
        xrec = self.explore_window(explorer, store, own, pre, k)
        self.log("windows", xrec)
      th.join()
      d = box["d"]
      n_calls += 1
      requery = False
      if xrec and xrec["probes"]:
        v = self.dev.capture(f"s{k}_verify", settle=True)
        if not R.same_state(v["ui"], ui):
          requery = True
          pre = self.dev.capture(f"s{k}_pre_requery", settle=True, full_res=True)
          ui = pre["ui"]
          d = self.decide(agent, goal, pre["pixels"])
          n_calls += 1
      # ---------------- coordinate repair
      a = d["action"]
      repaired = None
      if store is not None and a.action_type == "click" and a.x is not None:
        hit = [n for n in ui.app_nodes() if n.visible and n.clickable and n.enabled and
               n.bounds[0] <= a.x < n.bounds[2] and n.bounds[1] <= a.y < n.bounds[3]]
        if not hit:
          best = None
          for n in ui.app_nodes():
            if n.clickable and n.visible and n.enabled:
              l, t, r, b = n.bounds
              dd = (max(l - a.x, 0, a.x - r) ** 2 + max(t - a.y, 0, a.y - b) ** 2) ** 0.5
              if best is None or dd < best[0]:
                best = (dd, n)
          if best and best[0] <= 80:
            repaired = [a.x, a.y, *best[1].center]
            a.x, a.y = best[1].center
            n_repair += 1
      tok, fam, sel = C.action_token(ui, a.__dict__)
      summary, exec_s = self.commit(agent, d, goal)
      own.append(own_entry(ui, tok, fam))
      row = {"step": k, "kind": "model", "pre": R.strip(pre), "action": a.__dict__, "tok": tok, "summary": summary,
             "model_latency_s": d["model_latency_s"], "requery": requery, "repaired": repaired,
             "n_probes": len(xrec["probes"]) if xrec else 0, "window_s": xrec["window_s"] if xrec else 0}
      steps.append(row)
      self.log("steps", row)
      prev = {"pre": pre, "slot_id": tok, "sel": sel, "action": a.__dict__, "step": k}
      k += 1
      if a.action_type in (json_action.STATUS, json_action.ANSWER):
        done = True
        break
    final = self.dev.capture("final", settle=True, pause=1.0, full_res=True)
    if prev is not None:
      self.ingest_agent(graph, instance, prev, final)
    res = {"steps": len(steps), "agent_steps": len(steps), "done": done, "model_calls": n_calls,
           "jumps": n_jumps, "jump_steps": n_jump_steps, "jump_rejected": n_jump_reject, "shortcuts": n_shortcut,
           "repairs": n_repair, "n_probes": explorer.n_probes, "explore_disabled": explorer.disabled_reason,
           "final_cid": final["cid"], "store_eps": len(store.data) if store else 0}
    self._own_seq = own
    return res

  # -------------------------------------------------------------- jump
  def plan_jump(self, store, graph, own, ui, used):
    """Chain of gated dominant continuations from the current page; then try a
    shorter path in the graph to the deepest page of that chain."""
    chain, sim_own = [], list(own)
    cur = {"struct": ui.struct_sig, "layout": ui.layout_sig}
    for _ in range(MAX_CHAIN):
      tok, key, info = store.predict(sim_own, cur)
      if tok is None or not tok.startswith("E|"):
        break
      nxt = {k2: store.successor(k2, cur[k2], tok) for k2 in ("struct", "layout")}
      if nxt[key] is None:
        break
      commit_ok = False
      if self._same is not None:
        st_tok, _, _ = self._same.predict(sim_own, cur)
        commit_ok = st_tok == tok
      chain.append({"tok": tok, "key": key, "from": dict(cur), "expect": nxt, "gate": info, "commit_ok": commit_ok})
      sim_own.append({"state": cur["struct"], "struct": cur["struct"], "layout": cur["layout"], "tok": tok, "fam": tok})
      cur = {k2: (nxt[k2] or "") for k2 in ("struct", "layout")}
    if not chain:
      return []
    if used[(ui.struct_sig, chain[0]["tok"])] >= 2:
      return []
    used[(ui.struct_sig, chain[0]["tok"])] += 1
    # shortest known path (safe clicks) from here to the chain's last page (layout level)
    target = chain[-1]["expect"]["layout"]
    sp = self.shortest(graph, ui.layout_sig, target, len(chain))
    if sp is not None and len(sp) < len(chain):
      return [{"tok": t, "key": "layout", "expect": {"layout": dst, "struct": None}, "shortcut": True} for t, dst in sp]
    return chain

  def shortest(self, graph, src, dst, limit):
    adj = defaultdict(list)
    for (s, sid), lst in graph.slots.items():
      for o in lst:
        if not o.get("dst") or not o.get("rollback_ok", True) and o.get("origin") == "explore":
          continue
        sel = o.get("selector") or {}
        fam = C.family_of_sel(sel) if sel else None
        if not fam or not fam.startswith("E|"):
          continue
        adj[o["src"]["layout_sig"]].append((fam, o["dst"]["layout_sig"]))
    q, seen = deque([(src, [])]), {src}
    while q:
      node, path = q.popleft()
      if node == dst and path:
        return path
      if len(path) >= limit:
        continue
      for fam, nd in adj.get(node, []):
        if nd not in seen:
          seen.add(nd)
          q.append((nd, path + [(fam, nd)]))
    return None

  def exec_jump(self, path, own, agent, steps, k, explorer, graph, instance):
    ok = 0
    for j, hop in enumerate(path):
      cap = self.dev.capture(f"s{k + j}_jpre", settle=(j > 0), full_res=True) if j > 0 else self.dev.capture(f"s{k}_jpre", settle=False, full_res=True)
      ui = cap["ui"]
      slot = None
      for s in U.enumerate_slots(ui):
        if C.family_of_sel(s.sel) != hop["tok"]:
          continue
        # irreversible commits only with SAME-task evidence; never fields/toggles
        if s.safe or (hop.get("commit_ok") and s.unsafe_reason in ("unsafe_label", "dialog_action_button")):
          slot = s
          break
      if slot is None:
        self._jump_absent = getattr(self, "_jump_absent", 0) + 1
        break
      node = ui.nodes[slot.node_idx]
      self.dev.tap(*node.center)
      post = self.dev.capture(f"s{k + j}_jpost", settle=True)
      # Verify at layout level: struct signatures split the same page across
      # variants when titles carry data (14/16 false splits in manual review).
      exp = hop["expect"].get("layout")
      got = post["ui"].layout_sig
      is_commit = not slot.safe
      label = U.label_of(ui, node) or slot.sel.get("rid") or slot.sel.get("cls")
      row = {"step": k + j, "kind": "jump", "scope": hop.get("scope"), "tok": hop["tok"], "expected": exp, "match": got == exp,
             "shortcut": bool(hop.get("shortcut")), "pre": R.strip(cap), "post_cid": post["cid"]}
      if got != exp and is_commit:
        # The commit happened and cannot be undone: record it as an executed
        # step in the agent's history, stop the chain, hand over to the model.
        summary = f"Tapped \"{label}\"."
        agent._actions.append({"summary": summary})  # pylint: disable=protected-access
        agent._summaries.append(summary)  # pylint: disable=protected-access
        own.append(own_entry(ui, hop["tok"], hop["tok"]))
        row.update(summary=summary, rollback={"ok": False, "method": "none_after_commit"})
        self._jump_mismatch = getattr(self, "_jump_mismatch", 0) + 1
        steps.append(row)
        self.log("steps", row)
        ok += 1
        break
      if got != exp:
        row["rollback"] = _restore(explorer, ui, post["ui"], k + j)
        self._jump_mismatch = getattr(self, "_jump_mismatch", 0) + 1
        self.log("steps", row)
        break
      summary = f"Tapped \"{label}\"."
      agent._actions.append({"summary": summary})  # pylint: disable=protected-access
      agent._summaries.append(summary)  # pylint: disable=protected-access
      own.append(own_entry(ui, hop["tok"], hop["tok"]))
      self.ingest_agent(graph, instance, {"pre": cap, "slot_id": slot.slot_id, "sel": slot.sel,
                                          "action": {"action_type": "click", "x": node.center[0], "y": node.center[1]}, "step": k + j}, post)
      row["summary"] = summary
      steps.append(row)
      self.log("steps", row)
      ok += 1
    return ok

  # -------------------------------------------------------------- explore
  def explore_window(self, ex, store, own, pre, step):
    t0 = time.time()
    rec = {"step": step, "probes": [], "stop": ""}
    if ex.disabled_reason:
      return dict(rec, stop="disabled", window_s=0.0)
    ui = pre["ui"]
    plan, stats = ex.plan(ui)
    counts = Counter()
    for key in ("struct", "layout"):
      seq = [dict(x, state=x[key]) for x in own] + [{"state": ui.struct_sig if key == "struct" else ui.layout_sig, "tok": "", "fam": None}]
      cs = store.idx[key].lookup(seq, len(seq) - 1)
      for c in cs:
        if c:
          counts.update(c)
      if counts:
        break
    pos = {s.slot_id: i for i, (_, s) in enumerate(plan)}
    supported = [s for _, s in plan if counts[C.family_of_sel(s.sel)] > 0]
    if supported:
      order = sorted(supported, key=lambda s: (-counts[C.family_of_sel(s.sel)], pos[s.slot_id]))[:2]
      rec["mode"] = "trace"
    else:
      order = [s for p, s in plan if p == 0] + [s for p, s in plan if p != 0]
      rec["mode"] = "gap_coverage"
    for s in order:
      if time.time() - t0 >= ex.cfg.budget_s or len(rec["probes"]) >= ex.cfg.max_probes:
        rec["stop"] = "budget"
        break
      p = ex.probe(pre, s, 0, step, depth=1)
      rec["probes"].append({"slot": s.slot_id, "outcome": p["outcome"], "rollback_ok": p["rollback_ok"], "t": p["t_total_s"]})
      if not p["rollback_ok"]:
        ex.disabled_reason = f"rollback_failed@{step}"
        rec["stop"] = "rollback_failed"
        break
    rec["window_s"] = round(time.time() - t0, 3)
    return rec


def _restore(ex, ui_target, ui_post, step):
  if R.same_state(ui_post, ui_target):
    return {"ok": True, "method": "none_needed"}
  for i in range(2):
    ex.dev.back()
    chk = ex.dev.capture(f"s{step}_jrb{i}")
    if R.same_state(chk["ui"], ui_target) or chk["ui"].layout_sig == ui_target.layout_sig:
      return {"ok": True, "method": "back"}
    if chk["ui"].struct_sig != ui_post.struct_sig:
      break
  return {"ok": False, "method": "back"}


_orig_run = R.Episode.run


def _run(self):
  res = _orig_run(self)
  if ARM != "BASE" and res.get("is_successful") == 1.0 and getattr(self, "_own_seq", None):
    app = C.APP_OF.get(self.args.task, self.args.task).replace(" ", "_")
    Store(APPDIR / ARM / app / "store.json").append(self._own_seq, self.args.task)
  return res


_orig_setup = R.Episode.setup


def _setup(self):
  meta = _orig_setup(self)
  meta["seq_arm"] = ARM
  (self.out / "meta.json").write_text(R.jdump(meta))
  return meta


SeqEpisode.run = _run
SeqEpisode.setup = _setup
R.Episode = SeqEpisode

if __name__ == "__main__":
  R.main()
