"""Full AndroidWorld run: BASE vs FULL design (one episode per invocation).

FULL = G (global continuous graph) + E1/E2 exploration + R1 verified jumps
       + R2 pre-observation reuse. Each module's measured motivation is in
       reports/full_design_zh.md.

G   : one global store/graph for the whole run (pages carry activity in their
      signatures, so apps do not collide); append-only transitions; stage-aware
      continuation counts (all action tokens) tagged by task template.
E1  : predicted-next validation probe - the trace top-1 slot on this page, if
      its outcome was not yet observed in THIS run instance (1 probe).
E2  : gap-page completion - page with no continuation counts at all: probe
      unknown safe out-edges (<= 3). Pages that already have trace support
      get no coverage probing.
R1  : verified jump - follow gated dominant continuations; navigation clicks
      with any-scope evidence, commit clicks only with SAME-task evidence;
      every hop verified, mismatch -> back + hand to model.
R2  : pre-observation reuse - if the model picks the slot probed in this
      window and the post-action page equals the probed page, skip the fixed
      1 s pause before the next observation.
"""

from __future__ import annotations

import json
import math
import re
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from android_world import shared_state as SS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("SEQ_ARM", os.environ.get("FULL_ARM", "BASE"))
import seq_run as S  # noqa: E402  (patches rgs_run.Episode; we override below)
import common as C  # noqa: E402
import rgs_run as R  # noqa: E402
import proc_replay as P  # noqa: E402
import full7  # noqa: E402
import full9  # noqa: E402
import recovery_guard as RG  # noqa: E402
import explore_value as EV  # noqa: E402
import joint_graph as JG  # noqa: E402
import intent_graph as IG  # noqa: E402
import term_graph as TG  # noqa: E402
import graph_exec as GX  # noqa: E402
from android_world.env import adb_utils  # noqa: E402
from android_world.env import json_action  # noqa: E402

U = C.U
ARM_NAME = os.environ.get("FULL_ARM", "BASE")
# replicate arms: "FULL19R" (or "FULL19R2") behaves exactly like FULL19 but keeps its own graph directory
# "FULL23W": FULL23 behaviour, warm-started with execution-validation evidence from earlier runs (seeded files)
# "FULL29X": ablation - FULL29 behaviour with every exploration probe switched off (graph built from inference only)
# "FULL40D": FULL40 behaviour run under a simulated on-device inference load (ONDEVICE_LOAD=1)
ARM = re.sub(r"(R\d*|W|X|D)$", "", ARM_NAME) if re.match(r"^FULL\d+(R\d*|W|X|D)$", ARM_NAME) else ARM_NAME
NOEXP = bool(re.match(r"^FULL\d+X$", ARM_NAME))
APPDIR = Path(os.environ.get("SEQ_APPDIR", "/tmp/full"))


ADB = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
# benchmark hygiene (10-20): app preferences that the task initialisation does not reset but that change the first
# screen of later tasks. Simple Calendar Pro's default view drifted from 1 (month) to 7 (month + day list) during
# FULL30 and every later run inherited it; restore the original value before every episode.
HYGIENE_PREFS = [("com.simplemobiletools.calendar.pro", "Prefs.xml", "view", "1")]


def env_hygiene():
  import subprocess  # noqa: E402
  for pkg, fname, key, val in HYGIENE_PREFS:
    path = f"/data/data/{pkg}/shared_prefs/{fname}"
    cmd = (f"sed -i 's/<int name=\"{key}\" value=\"[0-9-]*\" \\/>/<int name=\"{key}\" value=\"{val}\" \\/>/' {path}; "
           f"am force-stop {pkg}")
    try:
      subprocess.run([ADB, "-s", os.environ.get("ANDROID_SERIAL", "emulator-5554"), "shell", cmd], timeout=15,
                     capture_output=True)
    except Exception:  # pylint: disable=broad-except
      pass


class FullEpisode(S.SeqEpisode):
  def run_agent(self, start_step=0, prior_summaries=None, label="agent"):
    env_hygiene()
    goal, agent = self.task.goal, self.make_agent()
    gdir = APPDIR / ARM_NAME
    gdir.mkdir(parents=True, exist_ok=True)
    full = ARM.startswith("FULL")
    store = S.Store(gdir / "store.json") if full else None
    self._same = S.Store(gdir / "store.json", only_task=self.args.task) if (full and ARM not in ("FULL2", "FULL3", "FULL4", "FULL5", "FULL7", "FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45")) else None
    self._last_model_commit = False
    # FULL9: the function graph subsumes episode-local replay (its own segments become functions)
    pr = P.ProcReplay(goal, enum_only=(ARM in ("FULL7", "FULL8"))) if ARM in ("FULL5", "FULL7", "FULL8") else None
    fx = full7.StuckExplorer(self, goal, gdir, online_only=ARM in tuple(f"FULL{i}" for i in range(33, 46)), hard=(ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"))) if ARM in ("FULL7", "FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else None
    fn = full9.FuncExec(self, goal) if ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else None
    # FULL33: the task has record fields to fill -> creation entries ("New Event", "+") may be explored
    self._create_task = bool(fn is not None and ARM in ("FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and fn.recs and fn.P.get("verb") in ENTRY_VERBS)
    if ARM in ("FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
      # creation entries are explored in every task: the editors they open become graph pages that later tasks'
      # form routes reach (cross-task reuse of exploration)
      self._create_task = True
    self._repetition_guard = RG.RepetitionGuard()
    self._runtime_stop_reason = None
    self._exec_log = []           # FULL11: actions executed since the last app launch (replayable path)
    self._agent = agent
    self._probe_states = {}       # FULL11: (layout key, slot family) first observed by exploration -> step
    if fn is not None:
      fn.exec_log = self._exec_log
    self._fn_ops = fn.P["ops"] if fn is not None else []
    self._fn = fn
    self._gvec = EV.goal_vector(goal, fn) if ARM in ("FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else {}
    self._pvec = EV.param_vector(fn) if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else {}
    self._rv_seen = set()
    self._page_acts, self._rec_pages, self._pend = {}, {}, None      # FULL18: episode-local page -> model actions
    self._reuse_brake, self._last_reuse_step = False, -9
    self._macro_brake, self._last_macro_step, self._macro_used = False, -9, set()   # FULL23
    self._intent = IG.EdgeIntent(gdir / "edge_use.jsonl", self.args.task) if ARM in ("FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else None
    if ARM == "FULL27":
      # speculation drafts: exploration contributes transitions only, intent comes from inference trajectories
      self._tmat = TG.TermMatrix(gdir / "graph.jsonl", self.out.parents[1], self.args.task, w_explore=0.0)
      self._gvec_unit = TG._unit(EV.goal_vector(goal, fn))  # pylint: disable=protected-access
    self._spec = None
    if ARM in ("FULL24", "FULL26"):
      # node x term matrix over inference + exploration trajectories of the run (cross-task, leave-own-task-out)
      self._tmat = TG.TermMatrix(gdir / "graph.jsonl", self.out.parents[1], self.args.task)
      self._gvec_unit = TG._unit(EV.goal_vector(goal, fn))  # pylint: disable=protected-access
    self._eff_path = gdir / "effectful.jsonl"
    self._learned_eff = set()
    if ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and self._eff_path.exists():
      self._learned_eff = {tuple(r["key"]) for r in SS.rows(self._eff_path)}
    self._grp_path = gdir / "unrestorable_groups.jsonl"
    self._bad_groups = set()
    if ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and self._grp_path.exists():
      self._bad_groups = {tuple(r["key"]) for r in SS.rows(self._grp_path)}
    self._d2_path = gdir / "d2_unsafe.jsonl"
    self._d2_bad = set()
    if ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and self._d2_path.exists():
      self._d2_bad = {tuple(r["key"]) for r in SS.rows(self._d2_path)}
    if fn is not None and ARM in ("FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
      fn.strict_done = True
    if fn is not None and ARM in ("FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
      fn.form_retrigger = True
    if fn is not None and ARM in ("FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
      fn.form_v2 = True          # hint-aware, synonyms, single-field records
    if fn is not None and ARM in ("FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
      fn.form_v3 = True          # each record field typed once; the head field only into a title / name field
    if fn is not None and ARM in ("FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
      fn.route_soft = True       # function routing over all layouts of the activity, through creation entries
    self._rb_fail = 0
    if ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):     # known environment drift: the SMS role must belong to the benchmark's SMS app
      import subprocess as _sp
      _sp.run([R.ADB, "-s", "emulator-5554", "shell", "cmd", "role", "add-role-holder", "android.app.role.SMS",
               "com.simplemobiletools.smsmessenger"], capture_output=True, timeout=30)
    graph = R.Graph(gdir / "graph.jsonl", gdir / "unrestorable.jsonl") if full else R.Graph(None)
    instance = f"{self.args.task}/{self.args.variant}"
    # FULL33: unified graph reuse - launch / confirm / function / form / repeat / macro are all out-edges of the
    # current page, triggered by graph evidence and braked by the model undoing them
    gx = GX.GraphReuse(gdir) if ARM in ("FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else None
    cfg = R.ExploreCfg(enabled=full, budget_s=(30.0 if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else 6.0), max_probes=3, max_depth=1)
    ex = R.Explorer(self.dev, graph, cfg, instance, self.log)
    own, steps, used = [], [], Counter()
    m = Counter()
    done, prev, k = False, None, 0
    booted = False
    reuse_next = None
    # FULL37: the step budget is the INFERENCE budget (as for BASE: 20 model calls); graph executions do not use
    # it. A separate cap on all counted steps (2x) guards against graph loops.
    while ((m["calls"] < self.max_steps() and k < 2 * self.max_steps()) if ARM in ("FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else k < self.max_steps()):
      if full and os.environ.get("MAI_SHARED_STATE") == "1":
        self.refresh_shared(graph, store, fx, fn, gx)
      if reuse_next is not None:
        # R2: model picked the probed slot; check the page without the fixed pause
        pre = self.dev.capture(f"s{k}_pre_reuse", settle=True, pause=None, full_res=True)
        if R.same_state(pre["ui"], reuse_next):
          m["reuse_hits"] += 1
        else:
          m["reuse_miss"] += 1
          pre = self.dev.capture(f"s{k}_pre", settle=True, pause=1.0, full_res=True)
        reuse_next = None
      elif ARM in ("FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and prev is not None:
        # system side: no fixed pause - wait until the screen changed from the one the action was taken on and is
        # stable, or (no change) for the stabilisation time the graph learned for this edge
        t_w = self._adaptive_wait(prev, m)
        pre = self.dev.capture(f"s{k}_pre", settle=False, pause=None, full_res=True)
        if ARM in ("FULL43", "FULL44", "FULL45"):
          pre = self._pixels_caught_up(prev, pre, k, m)
        pre["adaptive_wait_s"] = round(t_w, 3)
      else:
        pre = self.dev.capture(f"s{k}_pre", settle=True, pause=1.0, full_res=True)
      ui = pre["ui"]
      if STREAM:
        self._resolve_pending(steps)       # FULL41: the summary decoded while the action ran and the screen settled
      if fx is not None and prev is not None:
        fx.observe_model(prev, ui)
      if prev is not None:
        self.ingest_agent(graph, instance, prev, pre)
        prev = None
      if ARM in ("FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and self._pend is not None:
        P0, fam0, rec0, first0 = self._pend
        self._page_acts.setdefault(P0, []).append({"fam": fam0, "land": JG.page_key(ui), "rec": rec0, "first": first0})
        self._pend = None
      if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and k == 0 and not booted:
        booted = True
        app = full7.resolve_app(goal, ui, require_known=gx is None)
        if app and ARM == "FULL45" and _app_not_first(goal, app):
          m["bootstrap_skipped_multistep"] += 1
          app = None                       # FULL45: "turn on X and open Y": the app is not the first step
        if app and gx is not None and not gx.can("LAUNCH", GX.pkg_of_app(app)):
          m["gx_launch_no_edge"] += 1
          app = None                       # no launch edge into this app in the graph: the model opens it
        if app:
          # graph reuse before inference: the instruction names an app whose launch transition the graph has
          # observed; launch it without a model call and without a counted step (the model's first decision
          # is made on the app's first screen within the same counted step)
          full7.launch(self.env, app)
          self._exec_log[:] = [{"action": {"action_type": "open_app", "app_name": app}, "commit": False}]
          summ = f"[system] Opened the app \"{app}\"."
          agent._actions.append({"summary": summ})  # pylint: disable=protected-access
          agent._summaries.append(summ)  # pylint: disable=protected-access
          row0 = {"step": k, "kind": "bootstrap", "tok": f"OPEN_APP:{app}", "summary": summ, "pre": R.strip(pre)}
          steps.append(row0)
          self.log("steps", row0)
          m["bootstrap"] += 1
          m["env_actions_extra"] += 1
          pre = full7.capture_stable(self.dev, f"s{k}_pre_boot")
          ui = pre["ui"]
          if gx is not None:
            pk_ = GX.pkg_of_app(app)
            gx.record("LAUNCH", pk_, instance, k, ui.package == pk_, src=None, dst=ui.summary())
      if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and self._last_model_commit:
        # the model itself just chose a commit; the dialog that follows is confirmed with its positive button
        ck_ = (ui.package, ui.activity)
        if gx is not None and not gx.can("CONFIRM", ck_):
          m["gx_confirm_no_edge"] += 1
          n_c = 0
        elif ARM == "FULL45" and _destructive_dialog(ui) and _dialog_key(ui) not in self._confirmed_destructive:
          m["confirm_left_to_model"] += 1   # FULL45: first irreversible confirmation of the episode: the model's
          n_c = 0
        else:
          n_c = full7.confirm_own_commit(self, ui, agent, steps, k, own, pure=(ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45")))
        self._last_model_commit = False
        if n_c and gx is not None:
          gx.record("CONFIRM", ck_, instance, k, True, src=ui.summary())
        if n_c:
          self._exec_log.append({"action": None, "commit": True})
          m["jump_steps"] += 1
          m["jump_commit_steps"] += 1
          k += 1
          continue
      if fx is not None and ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and fx.needs_back(ui):
        # hard rule: the same navigation control tapped 3 times without any effect -> one Back
        self.dev.back()
        summ = "[system] Pressed Back: the last control was tapped 3 times without any effect."
        agent._actions.append({"summary": summ})  # pylint: disable=protected-access
        agent._summaries.append(summ)  # pylint: disable=protected-access
        steps.append({"step": k, "kind": "unstick", "tok": "NAVIGATE_BACK", "summary": summ})
        self.log("steps", steps[-1])
        m["unstick_back"] += 1
        self._exec_log.append({"action": {"action_type": "navigate_back"}, "commit": False})
        k += 1
        continue
      if fn is not None:      # FULL9: function graph (parameterized sub-trajectories), before inference
        if ARM == "FULL14":
          self.log("fn_gate", {"step": k, "rec": fn.next_rec, "tried": sorted(fn.tried), "form_tried": sorted(getattr(fn, "form_tried", set())),
                               "applied": fn.applied_rec, "model_rec": fn.model_rec, "done": sorted(fn.done), "act": ui.activity})
        n_f = fn.try_apply(pre, agent, steps, k, own)
        if not n_f and ARM == "FULL11":
          n_f = fn.try_apply_via_edge(self, graph, pre, agent, steps, k, own, m)
        elif not n_f and ARM in ("FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
          n_f = fn.try_apply_via_edge2(self, graph, pre, agent, steps, k, own, m)
        if not n_f and ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
          # function routing over the joint graph (agent + exploration pages), <= 3 safe hops, one counted step
          n_f = JG.try_route(fn, self, graph, pre, agent, steps, k, own, m)
        if not n_f and ARM in ("FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
          # schema-level form function, re-triggered wherever this record's empty fields appear
          n_f = fn.try_form(pre, agent, steps, k, own, m, via="here")
        if not n_f and ARM in ("FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
          # form routing: walk the graph (inference + exploration edges) to a known editor of this record's fields
          n_f = JG.try_form_route(fn, self, graph, pre, agent, steps, k, own, m)
        if not n_f and ARM == "FULL14":
          # schema-level form function: on this screen, else one observed edge away (agent or exploration)
          n_f = fn.try_form(pre, agent, steps, k, own, m, via="here") or fn.try_form_via_edge(self, graph, pre, agent, steps, k, own, m)
        if n_f and gx is not None:
          typ_ = GX.KIND_TYPE.get(steps[-1].get("kind"), "PARAM")
          gx.record(typ_, (ui.package, ui.activity, typ_), instance, k, True, n_actions=n_f, src=ui.summary())
        if n_f:
          m["func_apps"] += 1
          m["func_steps"] += n_f
          m["env_actions_extra"] += n_f - 1
          self._last_model_commit = False
          k += 1
          continue
      if ARM in ("FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and not self._reuse_brake and not full9.transient(ui):
        n_r = self.try_reuse(ui, pre, agent, steps, k, own, m, fn)
        if n_r and gx is not None:
          gx.record("ACTION", (ui.package, ui.activity, steps[-1].get("tok")), instance, k, True, src=ui.summary())
        if n_r:
          self._last_model_commit = False
          k += 1
          continue
      if ARM in ("FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and not self._macro_brake and not full9.transient(ui):
        # goal-conditioned navigation macro over the unified graph (design review P0-2)
        n_m = IG.try_macro(self, ui, pre, agent, steps, k, own, m, graph,
                           picker=(TG.pick if ARM in ("FULL24", "FULL26") else _pick_once if ARM in ("FULL44", "FULL45") else None))
        if n_m and gx is not None:
          gx.record("ACTION", (ui.package, ui.activity, "macro"), instance, k, True, n_actions=n_m, src=ui.summary())
        if n_m:
          self._last_model_commit = False
          k += 1
          continue
      if fx is not None:      # first-run screens: handled by a separate module, outside the main algorithm
        n_ob = fx.onboarding(ui, agent, steps, k)
        if n_ob:
          m["onboarding_steps"] += n_ob
          k += 1 if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else n_ob
          m["env_actions_extra"] += (n_ob - 1) if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else 0
          continue
      if pr is not None:      # R3: episode-local entity procedure replay
        pr.set_post(ui)
        plan = pr.plan(ui)
        if plan:
          m["replays"] += 1
          ok = self.exec_replay(plan, pr, pre, own, agent, steps, k, graph, instance)
          m["replay_steps"] += ok
          m["replay_full"] += ok == len(plan["template"])
          if ok:
            k += 1 if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else ok      # FULL8: a multi-hop graph execution is one counted step
            m["env_actions_extra"] += (ok - 1) if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else 0
            continue
      if store is not None:   # R1
        path = self.plan_jump(store, graph, own, ui, used)
        if path:
          m["jumps"] += 1
          ok = self.exec_jump(path, own, agent, steps, k, ex, graph, instance)
          m["jump_steps"] += ok
          m["jump_commit_steps"] += sum(1 for h in path[:ok] if h.get("commit_ok"))
          if ok:
            self._exec_log.append({"action": None, "commit": True})   # untracked taps: no replay restoration
          if ok < len(path):
            m["jump_rejected"] += 1
          self._last_model_commit = False
          if ok:
            k += 1 if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else ok
            m["env_actions_extra"] += (ok - 1) if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else 0
            continue
          pre = self.dev.capture(f"s{k}_pre_after_jump", settle=True, full_res=True)
          ui = pre["ui"]
      stuck_text = fx.hint(ui) if fx is not None else ""
      full7.EXTRA["text"] = stuck_text
      if stuck_text:
        m["stuck_hints"] += 1
      box = {}
      def _decide():
        try:
          box["d"] = self.decide(agent, goal, pre["pixels"])
        except Exception as e:  # pylint: disable=broad-except
          box["err"] = e
      self._early = {"ev": threading.Event()} if STREAM else None
      self._t_req = time.time()
      th = threading.Thread(target=_decide, daemon=True)
      th.start()
      self._infer_thread, self._spec = th, None
      if NOEXP:
        xrec = None
      elif fx is not None and fx.is_stuck(ui) and not ex.disabled_reason:
        xrec = fx.window(ex, pre, k)          # stuck: probe an untried control instead of E1
        m["probes_stuck"] += len(xrec["probes"])
      else:
        xrec = self.window(ex, store, own, pre, k, instance) if full else None
      if xrec is not None:
        self.log("windows", xrec)
        m["probes_e1"] += sum(p["why"] == "E1" for p in xrec["probes"])
        m["probes_e2"] += sum(p["why"] == "E2" for p in xrec["probes"])
        m["probes_fx"] += sum(p["why"] == "FX" for p in xrec["probes"])
        m["probes_d2"] += sum(p["why"] == "D2" for p in xrec["probes"])
        m["probes_rv"] += sum(p["why"] == "RV" for p in xrec["probes"])
        m["learned_effectful"] += bool(xrec.get("learned_effectful"))
        m["learned_group"] += bool(xrec.get("learned_group"))
        m["p_rev_dropped"] += xrec.get("p_rev_dropped", 0)
        if xrec.get("joint_restore"):
          m["joint_restore_ok" if xrec["joint_restore"]["ok"] else "joint_restore_fail"] += 1
        m["drift_stops"] += xrec.get("stop") == "drift"
        m["learned_edit_effect"] += bool(xrec.get("learned_edit_effect"))
        m["probes_total"] += len(xrec["probes"])
        m["windows_explored"] += bool(xrec["probes"])
        if xrec.get("value"):
          m["fx_" + xrec["value"]["level"]] += 1
          m["fx_rel_pos"] += xrec["value"]["rel"] > 0
        m["fx_entry_need"] += bool(xrec.get("entry_need"))
      if STREAM:
        self._early["ev"].wait()          # FULL41: action fields complete (or the response ended)
      if STREAM and self._early.get("d") is not None and th.is_alive():
        d = self._early["d"]              # act now; the summary keeps decoding (resolved at the next observation)
        self._pending = {"th": th, "box": box, "early": d, "step": k}
        m["early_actions"] += 1
      else:
        th.join()
        if "d" not in box:                   # the background decision failed (server outage): decide again here
          self.log("decide_retry", {"k": k, "err": repr(box.get("err"))[:300]})
          box["d"] = self.decide(agent, goal, pre["pixels"])
        d = box["d"]
      t_decision = time.time()
      m["calls"] += 1
      requery = False
      spec_hit = False
      if ARM in ("FULL27", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and self._spec is not None:
        sp = self._spec
        tok_s, fam_s, _ = C.action_token(ui, d["action"].__dict__)
        hit_dst = False
        if ARM in ("FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and d["action"].action_type == "click" and fam_s and fam_s != sp["fam"]:
          # FULL41: compare by destination - the model's edge leads (in the graph, deterministically) to the page
          # the device is already parked on
          e_m = JG.JointGraph(ex.graph).edge(JG.page_key(ui), fam_s)
          pd = JG.JointGraph.predicted(e_m)
          hit_dst = pd is not None and len(e_m["dst"]) == 1 and pd == JG.page_key(sp["dst"]["ui"])
        if d["action"].action_type == "click" and (fam_s == sp["fam"] or hit_dst):
          spec_hit = True                        # the model chose exactly the drafted action: already executed
          m["spec_hit"] += 1
          m["park_hit_dst"] += hit_dst
        else:
          m["spec_miss"] += 1
          self.dev.back()
          chk = self.dev.capture(f"s{k}_spec_undo", settle=True)
          if not R.same_state(chk["ui"], ui):
            ok_u, _ = JG.restore(self, ex.graph, ui, k)
            if not ok_u and not full9.relaunch_restore(self, ui, k):
              m["spec_undo_fail"] += 1
              if ARM in ("FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
                xrec = dict(xrec or {}, probes=list((xrec or {}).get("probes") or []) +
                            [{"why": "PARK", "family": sp["fam"], "rollback_ok": False, "dst_ui": None}])
      unresolved = ARM == "FULL45" and xrec and xrec.get("recovery_unresolved")
      if unresolved:
        spec_hit = False
        m["recovery_unresolved"] += 1
        if hasattr(self, "recovery_feedback"):
          self.recovery_feedback("Exploration changed the device state and could not restore the original state or data. Exploration is now disabled. Earlier history may no longer describe the current device. Inspect the latest screenshot, reopen the intended app if needed, and verify task data before continuing.")
      if spec_hit:
        xrec = dict(xrec or {}, probes=[])     # nothing to verify: the screen moved on by the model's own choice
      if xrec and (xrec["probes"] or unresolved):
        v = self.dev.capture(f"s{k}_verify", settle=True)
        if not unresolved and ARM in ("FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and not R.same_state(v["ui"], ui) and full9.decision_still_valid(ui, v["ui"], d["action"]):
          # the screen differs only in ways that do not affect THIS decision (same page, same target at the
          # same place): execute it without asking the model again
          m["verify_relaxed"] += 1
        elif not unresolved and ARM in ("FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and not R.same_state(v["ui"], ui) and full9.relocate_decision(ui, v["ui"], d["action"]):
          # same activity and the control the model chose is still on screen (moved): the decision stands, it is
          # executed at the control's current position
          m["verify_relocated"] += 1
        elif ARM == "FULL22" and not R.same_state(v["ui"], ui):
          # no re-inference: bring the screen back to the state the model decided on, then execute that decision;
          # if the state cannot be restored the decision is dropped (never executed on a screen it was not made for)
          ok_s, _ = JG.restore(self, ex.graph, ui, k)
          if not ok_s and full9.relaunch_restore(self, ui, k):
            ok_s = True
          v2 = self.dev.capture(f"s{k}_verify2", settle=True)
          if ok_s and (R.same_state(v2["ui"], ui) or full9.decision_still_valid(ui, v2["ui"], d["action"])
                       or full9.relocate_decision(ui, v2["ui"], d["action"])):
            m["restored_then_executed"] += 1
          else:
            m["decision_dropped"] += 1
            row = {"step": k, "kind": "dropped_decision", "action": d["action"].__dict__, "summary": d.get("summary", ""),
                   "pre": R.strip(pre)}
            steps.append(row)
            self.log("steps", row)
            k += 1
            continue
        elif ARM == "FULL35" and not R.same_state(v["ui"], ui) and self._restore_for_decision(ex, ui, d, k, m):
          pass                               # FULL35: the screen the model decided on was restored; execute
        elif unresolved or not R.same_state(v["ui"], ui):
          if m["calls"] >= self.max_steps():
            self._runtime_stop_reason = "budget_before_recovery_requery"
            break
          requery = True
          m["requery"] += 1
          pre = self.dev.capture(f"s{k}_pre_requery", settle=True, full_res=True)
          ui = pre["ui"]
          if fx is not None:
            full7.EXTRA["text"] = fx.hint(ui)
          if STREAM:
            self._drop_pending()
          d = self.decide(agent, goal, pre["pixels"])
          m["calls"] += 1
      full7.EXTRA["text"] = ""
      a = d["action"]
      tok, fam, sel = C.action_token(ui, a.__dict__)
      if ARM == "FULL45":
        # Repeated commits (e.g. deleting identical list rows) can make real data progress.
        effectful = bool(fam and is_commit_token(ui, fam)) or full9.effectful(sel or {})
        repeated = self._repetition_guard.observe(ui, a.__dict__, effectful=effectful)
        if repeated == 3 and hasattr(self, "recovery_feedback"):
          self.recovery_feedback("The same pointer action has been repeated three times with unchanged observed state. Inspect the current screenshot and choose a different recovery action; do not keep repeating the ineffective click or swipe.")
        if repeated >= 6:
          self._runtime_stop_reason = "repeated_action_without_progress"
          m["repetition_guard_stops"] += 1
          self.log("recovery_guard", {"step": k, "reason": self._runtime_stop_reason, "action": a.__dict__, "repetitions": repeated})
          break
      if gx is not None and gx.check_undo(instance, k, a.__dict__):
        m["gx_misuse"] += 1
      if xrec and fam:
        hit = [p for p in xrec["probes"] if p["family"] == fam and p.get("dst_ui") is not None and p["rollback_ok"]]
        if hit:
          m["e1_hit" if hit[0]["why"] == "E1" else "e2_hit"] += 1
          reuse_next = hit[0]["dst_ui"]
      if ARM in ("FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
        self._note_model_action(ui, a, m)
      if ARM == "FULL45":
        self._last_act_fam = fam
        if fam and "button1" in fam and _destructive_dialog(ui):
          self._confirmed_destructive.add(_dialog_key(ui))
      if ARM in ("FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and self._last_macro_step == k - 1 and a.action_type == json_action.NAVIGATE_BACK:
        self._macro_brake = True                 # the model undid the graph's navigation: misuse
        m["macro_brake_undo"] += 1
        self._intent.misuse(getattr(self, "_last_macro_key", None))
      if ARM in ("FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
        if self._last_reuse_step == k - 1 and a.action_type == json_action.NAVIGATE_BACK:
          self._reuse_brake = True               # the model undid the graph's step: stop reusing in this episode
          m["reuse_brake_undo"] += 1
        P_ = JG.page_key(ui)
        rec_ = fn.next_rec if fn is not None else 0
        first_ = P_ not in self._rec_pages.get(rec_, set())
        self._rec_pages.setdefault(rec_, set()).add(P_)
        n_ = U.element_at(ui, int(a.x), int(a.y)) if a.x is not None else None
        if (a.action_type in ("click", "long_press") and n_ is not None and fam and not n_.in_list
            and not is_commit_token(ui, fam)):
          self._pend = (P_, fam, rec_, first_)
      if spec_hit and STREAM and getattr(self, "_pending", None) is not None and self._pending["early"] is d:
        summary = self._placeholder_summary(agent)
      elif spec_hit:
        summary = (R.gelab_agent._normalize_space(d["parsed"].get("summary"))  # pylint: disable=protected-access
                   or R.gelab_agent._normalize_space(d["parsed"].get("explain")) or "tap")  # pylint: disable=protected-access
        agent._actions.append({"summary": summary})  # pylint: disable=protected-access
        agent._summaries.append(summary)  # pylint: disable=protected-access
        exec_s = 0.0
        reuse_next = self._spec["dst"]["ui"]
      else:
        summary, exec_s = self.commit(agent, d, goal)
      own.append(S.own_entry(ui, tok, fam))
      self._last_model_commit = bool(fam) and is_commit_token(ui, fam)
      if a.action_type == json_action.OPEN_APP:
        self._exec_log[:] = []
      self._exec_log.append({"action": {kk: vv for kk, vv in a.__dict__.items() if vv is not None},
                             "commit": self._last_model_commit})
      if pr is not None:
        pr.record(ui, a.__dict__, tok, sel, self._last_model_commit)
      if fn is not None:
        fn.observe_model_action(ui, a.__dict__)
        if self._last_model_commit:
          fn.on_model_commit()
      row = {"step": k, "kind": "model", "pre": R.strip(pre), "action": a.__dict__, "tok": tok, "summary": summary,
             "model_latency_s": d["model_latency_s"], "requery": requery, "t_decision": t_decision,
             "spec_hit": spec_hit,
             "probes": [{k2: v2 for k2, v2 in p.items() if k2 != "dst_ui"} for p in xrec["probes"]] if xrec else []}
      if stuck_text:
        row["stuck_hint"] = stuck_text[:300]
      steps.append(row)
      self.log("steps", row)
      prev = {"pre": pre, "slot_id": tok, "sel": sel, "action": a.__dict__, "step": k, "fam": fam,
              "label": (U.label_of(ui, U.element_at(ui, int(a.x), int(a.y))) if a.x is not None and
                        U.element_at(ui, int(a.x), int(a.y)) is not None else "")}
      k += 1
      if a.action_type in (json_action.STATUS, json_action.ANSWER):
        done = True
        break
    final = self.dev.capture("final", settle=True, pause=1.0, full_res=True)
    if STREAM:
      self._resolve_pending(steps)
    if fx is not None:
      fx.finish()
    if prev is not None:
      self.ingest_agent(graph, instance, prev, final)
    self._own_seq = own
    res = {"steps": (k if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else len(steps)), "env_actions": len(steps), "agent_steps": len(steps),
           "done": done, "model_calls": m["calls"],
           "runtime_stop_reason": self._runtime_stop_reason,
           "final_cid": final["cid"], "n_probes": ex.n_probes, "explore_disabled": ex.disabled_reason,
           "store_eps": len(store.data) if store else 0}
    m["jump_absent"] = getattr(self, "_jump_absent", 0)
    m["jump_mismatch"] = getattr(self, "_jump_mismatch", 0)
    res.update({f"m_{kk}": vv for kk, vv in m.items()})
    return res

  def refresh_shared(self, graph, store, fx, fn, gx):
    """Refresh peer evidence at a decision boundary, without holding device locks."""
    graph.refresh()
    if store is not None:
      store.refresh()
    if fx is not None:
      fx.refresh_shared()
    if fn is not None:
      fn.refresh_shared()
    if gx is not None:
      gx.refresh()
    if self._intent is not None:
      self._intent.refresh()
    for name, path in (("_learned_eff", self._eff_path), ("_bad_groups", self._grp_path), ("_d2_bad", self._d2_path)):
      getattr(self, name).update(tuple(row["key"]) for row in SS.rows(path))
    self.log("shared_refresh", {"worker": os.environ.get("MAI_WORKER_ID", "0"),
             "nodes": len(graph.nodes), "observations": sum(map(len, graph.slots.values())),
             "functions": len(fn.lib.funcs) if fn else 0,
             "successful_episodes": len(store.data) if store else 0})

  def plan_jump(self, store, graph, own, ui, used):
    if ARM in ("FULL3", "FULL4", "FULL5", "FULL7", "FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
      return self.plan_jump_invariant(store, graph, own, ui, used)
    path = S.SeqEpisode.plan_jump(self, store, graph, own, ui, used)
    if ARM == "FULL2":
      for i, hop in enumerate(path):
        # irreversible commits only to CONFIRM what the model itself just chose
        hop["commit_ok"] = (i == 0 and self._last_model_commit)
    return path

  def plan_jump_invariant(self, store, graph, own, ui, used):
    """Jump only along continuations that are (a) gated within this task
    template, or (b) unanimous across >= 2 task templates (the cross-task
    reusable prefix). Majority across templates is NOT enough: it reflects
    which tasks happened to run, not this task's intent."""
    chain, sim_own = [], list(own)
    cur = {"struct": ui.struct_sig, "layout": ui.layout_sig}
    me = self.args.task
    for i in range(S.MAX_CHAIN):
      tok, key, info = store.predict(sim_own, cur)
      top = info.get("top")
      if top is None or not top.startswith("E|") or key is None:
        break
      doms = store.task_dominants(sim_own, cur, key)
      same = doms.get(me)
      gate_n = 1 if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else S.GATE_N
      same_ok = same is not None and same[0] == top and same[1] >= gate_n and same[1] / same[2] >= S.GATE_D
      # cross-task: unanimous across >= 2 templates AND the agent has already
      # revealed its branch by an own (non open-app) action in the matched
      # context; at the task entry, templates diverge (0% cross-task top-1).
      agent_acted = any(not x["tok"].startswith(("OPEN_APP", "STATUS", "WAIT")) for x in own)
      cross_ok = (len(doms) >= 2 and all(d[0] == top for d in doms.values())
                  and info.get("order", 0) >= 1 and agent_acted)
      if ARM in ("FULL4", "FULL5", "FULL7", "FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
        cross_ok = False   # cross-task evidence is survivor-biased: only orders E1 probes
      # FULL5: confirming the model's OWN just-chosen commit needs no history
      # evidence beyond which dialog button follows (intent is the model's).
      confirm_ok = (ARM in ("FULL5", "FULL7", "FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and i == 0 and self._last_model_commit and is_dialog_button(ui, top))
      if not (same_ok or cross_ok or confirm_ok):
        break
      nxt = {k2: store.successor(k2, cur[k2], top) for k2 in ("struct", "layout")}
      if nxt["layout"] is None:
        break
      chain.append({"tok": top, "key": key, "expect": nxt, "gate": info, "scope": "same" if same_ok else ("cross" if cross_ok else "confirm"),
                    "commit_ok": (i == 0 and self._last_model_commit)})
      sim_own.append({"state": cur["struct"], "struct": cur["struct"], "layout": cur["layout"], "tok": top, "fam": top})
      cur = {k2: (nxt[k2] or "") for k2 in ("struct", "layout")}
    if not chain or used[(ui.struct_sig, chain[0]["tok"])] >= 2:
      return []
    used[(ui.struct_sig, chain[0]["tok"])] += 1
    return chain

  def exec_replay(self, plan, pr, pre, own, agent, steps, k, graph, instance):
    """Follow the model's own demonstrated procedure for the next goal entity;
    stop before its commit and on the first landing mismatch."""
    ok, cap = 0, pre
    for j, h in enumerate(plan["template"]):
      ui = cap["ui"]
      if j == 0:
        node = plan["node"]
        label = plan["entity"]
      else:
        slot = next((sl for sl in U.enumerate_slots(ui) if C.family_of_sel(sl.sel) == h["tok"] and sl.safe), None)
        if slot is None:
          self.log("steps", {"step": k + j, "kind": "replay_absent", "tok": h["tok"], "entity": plan["entity"]})
          break
        node = ui.nodes[slot.node_idx]
        label = U.label_of(ui, node) or slot.sel.get("rid") or slot.sel.get("cls")
      x, y = node.center
      if h["at"] == "long_press":
        adb_utils.long_press(int(x), int(y), self.dev.ctrl)
        summary = f"Long-pressed \"{label}\"."
      else:
        self.dev.tap(x, y)
        summary = f"Tapped \"{label}\"."
      post = self.dev.capture(f"s{k + j}_rpost", settle=True, pause=1.0 if j == 0 else None)
      match = post["ui"].activity == h["post_act"]
      action = {"action_type": h["at"], "x": int(x), "y": int(y)}
      tok, _, sel = C.action_token(ui, action)
      agent._actions.append({"summary": summary})  # pylint: disable=protected-access
      agent._summaries.append(summary)  # pylint: disable=protected-access
      own.append(S.own_entry(ui, tok, tok))
      pr.record(ui, action, tok, sel, False, origin="replay")
      pr.set_post(post["ui"])
      self.ingest_agent(graph, instance, {"pre": cap, "slot_id": tok, "sel": sel, "action": action, "step": k + j}, post)
      row = {"step": k + j, "kind": "replay", "entity": plan["entity"], "tok": tok, "match": match,
             "summary": summary, "pre": R.strip(cap), "post_cid": post["cid"]}
      steps.append(row)
      self.log("steps", row)
      ok += 1
      cap = post
      if not match:
        break
    pr.tried.add(plan["entity"])
    pr.open = plan["entity"]      # closed only by the model's own commit
    return ok

  def _note_model_action(self, ui, a, m):
    """Exploration's contribution to the graph: did the model take an edge exploration observed first?"""
    if a.action_type not in ("click", "long_press") or a.x is None:
      return
    n = U.element_at(ui, int(a.x), int(a.y))
    if n is None:
      return
    fam = C.family_of_sel(U.selector(ui, n))
    if hasattr(self, "_rv_seen"):
      self._rv_seen.add(((ui.package, ui.activity, fam), ui.content_sig))   # the agent itself observed it now
    if (ui.layout_sig, fam) in self._probe_states:
      m["model_used_explored_edge"] += 1

  def window(self, ex, store, own, pre, step, instance):
    """E1 then E2; nothing else."""
    t0 = time.time()
    rec = {"step": step, "probes": [], "stop": ""}
    if ex.disabled_reason:
      return dict(rec, stop="disabled", window_s=0.0)
    ui = pre["ui"]
    if ARM in ("FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and full9.transient(ui):
      # temporary states (popup menu / dialog / keyboard / selection mode / open drawer) are not on the back
      # stack: any probe destroys them and Back cannot bring them back
      return dict(rec, stop="transient_state", window_s=0.0)
    if ARM in ("FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
      import state_abstraction as SA  # noqa: E402
      if SA.mode(ui) == "dialog":
        # a single-window dialog is not on the back stack either (FULL29-33: 8 of 10 probes there failed Back)
        return dict(rec, stop="dialog_state", window_s=0.0)
    if ARM == "FULL13" and not full9.replayable(self._exec_log):
      # restorability guarantee: probe only while a failed Back can be repaired by relaunch + replay of the path
      return dict(rec, stop="not_restorable", window_s=0.0)
    plan, _ = ex.plan(ui)
    safe = {s.slot_id: s for _, s in plan}
    by_fam = {}
    for _, s in plan:
      by_fam.setdefault(C.family_of_sel(s.sel), s)
    tok, key, info = store.predict(own, {"struct": ui.struct_sig, "layout": ui.layout_sig})
    top = info.get("top")
    rec["gate"] = info
    targets = []
    if top and top in by_fam:
      s = by_fam[top]
      seen_here = any(o["instance"] == instance for o in graph_obs(ex.graph, ui.struct_sig, s.slot_id))
      if ARM == "FULL45" and _risky_slot(self, ui, s):
        seen_here = True                  # FULL45: the predicted control is a toggle / mode switch: never probed
      if not seen_here and not (((ARM in ("FULL12", "FULL13", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and full9.effectful(s.sel)) or (ARM == "FULL14" and full9.effectful_v2(s.sel)))):
        targets.append(("E1", s))
    if (not targets or ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45")) and ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
      # graph expansion: an unknown safe out-edge of this screen, ordered by similarity to the goal's operation
      # terms, then by cross-app generality (app-specific controls first), then screen order
      unknown = [s for p0, s in plan if p0 == 0 and (ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") or not s.sel.get("in_list"))
                 and Graph_app_key(ui, s) not in getattr(self, "_bad_fams", set())
                 and not (((ARM in ("FULL12", "FULL13", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and full9.effectful(s.sel) and not _entry_ok(self, s.sel)) or (ARM == "FULL14" and full9.effectful_v2(s.sel))))]
      if ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
        # graph-aware: a control whose transition from this page is already in the graph (seen by the agent or
        # by exploration in any task) carries no new information; nor does one learned to change app data
        known, reversible = full9.page_family_index(ex.graph)
        unknown = [s for s in unknown if (ui.package, ui.activity, C.family_of_sel(s.sel)) not in known
                   and (ui.package, C.family_of_sel(s.sel)) not in self._learned_eff
                   and full9.container_key(ui.package, ui.activity, s.sel) not in self._bad_groups]
        # predicted reversibility: probe only controls Back is expected to undo; when only risky controls are
        # left unexplored, do not explore at all (deduplication must not push probes onto them)
        evid = full9.container_evidence(ex.graph)
        if ARM in ("FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
          sr = ARM in ("FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45")
          app_ev = full9.app_kind_evidence(ex.graph, split_rows=sr)
          prev = {s.slot_id: full9.p_reversible_h(ui, s.sel, evid, app_ev, "discover" + ("_row" if sr and s.sel.get("in_list") else ""))
                  for s in unknown}
        else:
          prev = {s.slot_id: full9.p_reversible(ui, s.sel, evid) for s in unknown}
        rec["p_rev_dropped"] = sum(1 for v in prev.values() if v < P_REV_MIN)
        if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
          # risk-adjusted: a less certain control is acceptable only while relaunch + replay can repair it
          rp = full9.replayable(self._exec_log)
          unknown = [s for s in unknown if prev[s.slot_id] >= (P_REV_MIN_RV if ARM in ("FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and _entry_ok(self, s.sel) else P_REV_MIN) or
                     (ARM not in ("FULL20", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and rp and prev[s.slot_id] >= P_REV_RISKY)]
        else:
          unknown = [s for s in unknown if prev[s.slot_id] >= P_REV_MIN]
        rec["known_index"] = len(known)
        if ARM in ("FULL44", "FULL45"):
          n0 = len(unknown)
          unknown = [s for s in unknown if not (_risky_slot(self, ui, s) if ARM == "FULL45" else _risky_probe(s.sel))]
          rec["risky_dropped"] = n0 - len(unknown)
      if unknown and ARM in ("FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
        # unified graph-based exploration value: TF-IDF relevance of the best text evidence the graph holds for
        # each control (known edge > graph neighbours > the control's own text), global x graph-local IDF
        gv = EV.GraphValue(ex.graph, self.out.parents[1], self._gvec, ui.package, app_doc_idf=ARM in ("FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"), instance=instance if ARM in ("FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else None)
        if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
          gv.pvec = self._pvec
        scored = sorted(((gv.score(ui, sl), i, sl) for i, sl in enumerate(unknown)), key=lambda x: (-x[0][0], x[1]))
        if ARM in ("FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
          # expected benefit minus expected re-inference cost; only controls with positive priority are probed
          bonus = 0.0
          if ARM == "FULL26":
            # bridging: pages many tasks pass through are explored more completely (amortized over later tasks)
            bonus = LAMBDA_T * math.log1p(TG.traffic(self._tmat, ui))
            rec["traffic_bonus"] = round(bonus, 3)
          if ARM in ("FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
            # value of information: a control whose relevance is UNKNOWN (no text of its own and no graph
            # evidence, or an entity row whose parameters do not decide it) outranks one whose relevance is known
            # to be 0; LAMBDA_U = KAPPA * (1 - P_REV_MIN_RV): an unknown control is probed iff its P_rev >= 0.7
            unk = lambda sl, sc_: sc_[1] == 0 and sc_[2] == "L3" and (sl.sel.get("in_list") or not EV.own_tokens(ui, sl.sel))  # noqa: E731
            scored = [((sc[0] + (LAMBDA_U if unk(sl, sc) else 0.0),) + tuple(sc[1:]), i, sl) for sc, i, sl in scored]
          scored = sorted(((((sc[0] + bonus - KAPPA * (1 - prev.get(sl.slot_id, 1.0))),) + tuple(sc[1:]), i, sl)
                           for sc, i, sl in scored), key=lambda x: (-x[0][0], x[1]))
          rec["fx_nonpositive"] = sum(1 for sc, _, _ in scored if sc[0] <= 0)
          if ARM in ("FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
            # more exploration: reversible controls of zero relevance are queued after the relevant ones; they are
            # probed only while the decision is still pending (the loop stops when the action is known)
            scored = [x for x in scored if x[0][0] > 0] + [x for x in scored if x[0][0] <= 0]
          else:
            scored = [x for x in scored if x[0][0] > 0] or []
        if not scored:
          scored = None
      if unknown and ARM in ("FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and scored is None:
        pass
      elif unknown and ARM in ("FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
        (sc0, val0, lvl0), _, best = scored[0]
        rec["value"] = {"score": round(sc0, 3), "rel": round(val0, 3), "level": lvl0, "local_pages": gv.local_n}
        if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
          # siblings of one container share their behaviour: one probe per container per window
          taken = {t[1].slot_id for t in targets}
          conts = {full9.container_key(ui.package, ui.activity, t[1].sel) for t in targets}
          fx_cap = 2 if ARM == "FULL20" else PROBES_PER_STEP   # FULL20: FX carries 31% Back failures; RV 3%
          for _, _, sl in scored:
            if len(targets) >= PROBES_PER_STEP or sum(t[0] == "FX" for t in targets) >= fx_cap:
              break
            ck = full9.container_key(ui.package, ui.activity, sl.sel)
            if sl.slot_id in taken or (ck in conts and ck[2] != C.family_of_sel(sl.sel)):
              continue
            conts.add(ck)
            targets.append(("FX", sl))
        else:
          targets.append(("FX", best))
      elif unknown:
        order = full9.frontier_order(ui, unknown, self._fn_ops, ELEMENT_APPS)
        if ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
          need = full9.entry_tokens(self._fn, ui)
          if need:
            # information need = the entry screen of a pending function that does not ground here
            order = sorted(order, key=lambda sl: -len(need & set(full9._toks(sl.sel.get("label") or sl.sel.get("desc") or sl.sel.get("rid") or ""))))
            rec["entry_need"] = sorted(need)[:8]
        if ARM == "FULL14" and self._fn is not None and self._fn.form_pending():
          # information need = the entry screen of the pending form function: editor-opening controls first
          order = sorted(order, key=lambda sl: not full9.opens_editor(sl.sel))
        targets.append(("FX", order[0]))
      elif ARM in ("FULL15", "FULL16"):
        d2 = self.depth2_target(ex, ui, known, reversible)
        if d2 is not None:
          targets.append(("D2", d2))
      if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
        # fixed exploration effort: fill up to PROBES_PER_STEP with deeper probes behind distinct known edges
        used = set()
        while len(targets) < PROBES_PER_STEP:
          d2 = self.depth2_target(ex, ui, known, reversible, exclude=used)
          if d2 is None:
            break
          used.add(C.family_of_sel(d2.sel))
          targets.append(("D2", d2))
        if len(targets) < PROBES_PER_STEP:
          # re-observation: an edge the graph knows from this page but not in this episode. Its destination holds
          # this task's content (list rows, field values) that function routing grounds on; only controls Back has
          # been seen to undo (or predicted to)
          bp = getattr(ex.graph, "_by_page", {})
          gvr = EV.GraphValue(ex.graph, self.out.parents[1], self._gvec, ui.package, app_doc_idf=ARM in ("FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"), instance=instance if ARM in ("FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else None)
          gvr.pvec = self._pvec
          taken = {t[1].slot_id for t in targets}
          conts = {full9.container_key(ui.package, ui.activity, t[1].sel) for t in targets}
          rv = []
          for sl in U.enumerate_slots(ui):
            fam = C.family_of_sel(sl.sel)
            key = (ui.package, ui.activity, fam)
            if not sl.safe or sl.slot_id in taken or full9.effectful(sl.sel) or key not in known:
              continue
            if ARM in ("FULL44", "FULL45") and (_risky_slot(self, ui, sl) if ARM == "FULL45" else _risky_probe(sl.sel)):
              continue
            obs = bp.get(key, [])
            # the information unit is (edge, page content): re-observe once per content state of this page
            if (key, ui.content_sig) in self._rv_seen:
              continue
            ck = full9.container_key(ui.package, ui.activity, sl.sel)
            if (ck in self._bad_groups or (ui.package, fam) in self._learned_eff or Graph_app_key(ui, sl) in getattr(self, "_bad_fams", set())
                or key in self._d2_bad):
              continue
            p_rv = (full9.p_reversible_h(ui, sl.sel, evid, full9.app_kind_evidence(ex.graph, split_rows=ARM in ("FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45")),
                                         "revalidate" + ("_row" if ARM in ("FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and sl.sel.get("in_list") else "")) if ARM in ("FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45")
                    else full9.p_reversible(ui, sl.sel, evid))
            if key not in reversible and p_rv < (P_REV_MIN_RV if ARM in ("FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else P_REV_MIN):
              continue
            ents = [(o["instance"], o["dst_cid"], o["dst"].get("layout_sig")) for o in obs if o.get("dst_cid")][-3:]
            sc_rv = gvr.score(ui, sl, ents)[0]
            if ARM in ("FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and sc_rv <= 0:
              continue                       # P0-3: re-observe only edges whose content is relevant to this goal
            rv.append((sc_rv, sl, ck))
          for _, sl, ck in sorted(rv, key=lambda x: -x[0]):
            if len(targets) >= PROBES_PER_STEP:
              break
            if ck in conts and ck[2] != C.family_of_sel(sl.sel):
              continue
            conts.add(ck)
            targets.append(("RV", sl))
        for t in targets:
          self._rv_seen.add(((ui.package, ui.activity, C.family_of_sel(t[1].sel)), ui.content_sig))
    elif not targets and info.get("order", -1) < 0 and ARM not in ("FULLNE", "FULL2", "FULL3", "FULL4", "FULL5", "FULL7", "FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):   # gap page (E2 off in FULLNE/FULL2/FULL3)
      for p, s in plan:
        if p == 0 and len(targets) < 3:      # unknown out-edges only
          targets.append(("E2", s))
    rec["mode"] = targets[0][0] if targets else "none"
    for i_t, (why, s) in enumerate(targets):
      if ARM == "FULL45" and _risky_slot(self, ui, s):
        continue
      if time.time() - t0 >= ex.cfg.budget_s:
        rec["stop"] = "budget"
        break
      if STREAM and getattr(self, "_early", None) is not None and self._early["ev"].is_set():
        rec["stop"] = "decided"              # FULL41: the action is known; the window ends with the decision
        break
      if ARM in ("FULL42", "FULL43", "FULL44", "FULL45") and i_t > 0:
        # deadline-aware: the window ends where the action usually arrives (median time-to-action of this run's
        # streamed decisions); a probe started later than ~2 s before it would overrun the decision
        ta = sorted(_T_ACTION[-15:]) or [10.0]
        if time.time() - getattr(self, "_t_req", t0) > ta[len(ta) // 2] - 2.0:
          rec["stop"] = "deadline"
          break
        if rec["probes"] and rec["probes"][-1]["outcome"] == "no_change":
          time.sleep(0.8)                    # a slow UI (on-device load) may show the last probe's effect late
      if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and i_t > 0:
        # every probe must start from the origin state: verify before probing again
        chk0 = self.dev.capture(f"s{step}_x{i_t}_chk", settle=True)
        if not R.same_state(chk0["ui"], ui):
          rec["stop"] = "drift"
          ok_j, _ = JG.restore(self, ex.graph, ui, step)
          if not ok_j and not full9.relaunch_restore(self, ui, step):
            rec["drift_unrestored"] = True
            if ARM == "FULL45":
              rec["recovery_unresolved"] = True
              ex.disabled_reason = f"drift_unrestored@{step}"
          break
      if why == "D2":
        p = self.probe_depth2(ex, pre, s, step)
        if p is None:
          rec["stop"] = "d2_not_landed"
          break
      else:
        p = ex.probe(pre, s, 1 if why == "RV" else 0, step, depth=1)
      dst_ui = None
      try:
        dst_ui = R.load_ui(self.out, p["dst_cid"])
      except Exception:  # pylint: disable=broad-except
        pass
      fam_p = C.family_of_sel(s.sel)
      kp = (ui.layout_sig, fam_p)
      if kp not in self._probe_states and not any(o.get("origin") != "explore" for o in graph_obs(ex.graph, ui.struct_sig, s.slot_id)
                                                     + graph_obs(ex.graph, ui.struct_sig, fam_p)):
        self._probe_states[kp] = step
      if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and why != "D2" and dst_ui is not None and full9.edits_changed(ui, dst_ui):
        # the control wrote into a text field of this page: it edits data even if Back/retap restored the
        # screen. Its whole container is learned effectful (graph-persisted) and the window stops here
        gk = full9.container_key(ui.package, ui.activity, s.sel)
        if gk not in self._bad_groups:
          self._bad_groups.add(gk)
          SS.append_json(self._grp_path, {"key": list(gk), "family": C.family_of_sel(s.sel), "instance": instance, "step": step,
                                "reason": "edits_text"})
        rec["learned_edit_effect"] = list(gk)
      if ARM == "FULL45" and why != "D2" and p.get("outcome") == "in_place_content_change":
        key_e = (ui.package, C.family_of_sel(s.sel))
        if key_e not in self._learned_eff:
          self._learned_eff.add(key_e)
          SS.append_json(self._eff_path, {"key": list(key_e), "instance": instance, "step": step, "reason": "in_place_change"})
      rec["probes"].append({"why": why, "slot": s.slot_id, "family": C.family_of_sel(s.sel), "outcome": p["outcome"],
                            "rollback_ok": p["rollback_ok"], "t": p["t_total_s"], "dst_ui": dst_ui if why != "D2" else None})
      if not p["rollback_ok"] and ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and why != "D2":
        # Back could not undo it: the whole container (its siblings) is learned non-reversible, across tasks
        gk = full9.container_key(ui.package, ui.activity, s.sel)
        if gk not in self._bad_groups:
          self._bad_groups.add(gk)
          SS.append_json(self._grp_path, {"key": list(gk), "family": C.family_of_sel(s.sel), "instance": instance, "step": step})
          rec["learned_group"] = list(gk)
      if not p["rollback_ok"] and why == "D2":
        # the two-level detour through this edge could not be undone by Back: never go deeper through it again
        key = (ui.package, ui.activity, C.family_of_sel(s.sel))
        self._d2_bad.add(key)
        SS.append_json(self._d2_path, {"key": list(key), "instance": instance, "step": step})
      if not p["rollback_ok"] and ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
        # restoration ladder, second rung: a safe path over the joint graph back to the origin page
        ok_j, orgs = JG.restore(self, ex.graph, ui, step)
        rec["joint_restore"] = {"ok": ok_j, "hops_origins": orgs}
        if ok_j:
          p["rollback_ok"] = True
      if not p["rollback_ok"] and ARM in ("FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
        ok_r = full9.relaunch_restore(self, ui, step)
        rec["relaunch_restore"] = ok_r
        if ok_r:
          p["rollback_ok"] = True
        if (ok_r or ARM == "FULL45") and ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and not getattr(self, "_relaunch_content_same", True):
          # the screen is back but its data is not: the probed control changes app data. Remember it in the
          # graph so no later task explores it again
          key = (ui.package, C.family_of_sel(s.sel) if why != "D2" else p.get("family", ""))
          self._learned_eff.add(key)
          SS.append_json(self._eff_path, {"key": list(key), "instance": instance, "step": step})
          rec["learned_effectful"] = list(key)
      if not p["rollback_ok"]:
        rec["stop"] = "rollback_failed"
        if ARM == "FULL45":
          rec["recovery_unresolved"] = True
          ex.disabled_reason = f"rollback_failed@{step}"
        if ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
          # reversibility gating by control family, not by episode: never probe this family again; stop
          # exploring only after a second failure
          self._bad_fams = getattr(self, "_bad_fams", set()) | {Graph_app_key(ui, s)}
          self._rb_fail += 1
          if self._rb_fail >= 2 and ARM != "FULL45":
            ex.disabled_reason = f"rollback_failed_x2@{step}"
        else:
          ex.disabled_reason = f"rollback_failed@{step}"
        break
    if ARM == "FULL27" and not rec.get("stop") and getattr(self, "_infer_thread", None) is not None \
        and self._infer_thread.is_alive() and not full9.transient(ui):
      dr = TG.draft(self._tmat, ui, self._gvec_unit)
      if dr is not None:
        # speculative execution: perform the drafted action while the model reasons; kept only if it agrees
        chk = self.dev.capture(f"s{step}_spec_pre", settle=True)
        if R.same_state(chk["ui"], ui):
          sl, fam, act, mg = dr
          self.dev.tap(*ui.nodes[sl.node_idx].center)
          dst = full7.capture_stable(self.dev, f"s{step}_spec")
          self._spec = {"fam": fam, "dst": dst, "pred_act": act, "margin": mg}
          rec["spec"] = {"fam": fam, "margin": mg, "landed": dst["ui"].activity == act}
    if ARM in ("FULL41", "FULL42", "FULL43") and not rec.get("stop") and not rec.get("learned_edit_effect"):
      self._park(ex, pre, rec, step, instance)
    rec["window_s"] = round(time.time() - t0, 3)
    return rec


def _depth2_target(self, ex, ui, known, reversible, exclude=()):
  """No unknown control left on this page: go one level deeper along a known edge that exploration has seen
  reversed by Back, to a page that still has unknown (non-effectful) controls."""
  raw = self.out.parents[1]
  cands = []
  for sl in U.enumerate_slots(ui):
    if not sl.safe or sl.sel.get("in_list") or full9.effectful(sl.sel):
      continue
    fam = C.family_of_sel(sl.sel)
    if (ui.package, ui.activity, fam) not in reversible or (ui.package, ui.activity, fam) in self._d2_bad or fam in exclude:
      continue
    for o in reversed(getattr(ex.graph, "_by_page", {}).get((ui.package, ui.activity, fam), [])):   # newest first
      dd = o.get("dst")
      # the intermediate page must be on the back stack: an activity transition (popup menus and dialogs stay in
      # the same activity and are not)
      if not dd or dd.get("package") != ui.package or dd.get("activity") == ui.activity or not o.get("dst_cid"):
        continue
      task, var = o["instance"].split("/")
      try:
        dui = R.load_ui(SS.episode_path(raw, task, var), o["dst_cid"])
      except Exception:  # pylint: disable=broad-except
        continue
      if full9.transient(dui):
        continue
      evid = full9.container_evidence(ex.graph)
      fresh = [t for t in U.enumerate_slots(dui) if t.safe and not t.sel.get("in_list") and not full9.effectful(t.sel)
               and full9.p_reversible(dui, t.sel, evid) >= P_REV_MIN
               and (dui.package, dui.activity, C.family_of_sel(t.sel)) not in known
               and (dui.package, C.family_of_sel(t.sel)) not in self._learned_eff
               and full9.container_key(dui.package, dui.activity, t.sel) not in self._bad_groups
               and (ARM != "FULL45" or not _risky_slot(self, dui, t))]
      if fresh and ARM in ("FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
        cands.append((sl, dd, dui, fresh, (o["instance"], o["dst_cid"], dd.get("layout_sig"))))
        break
      if fresh:
        sl.d2 = {"via_dst": dd, "inner": full9.frontier_order(dui, fresh, self._fn_ops, ELEMENT_APPS)[0].sel}
        return sl
  if cands:
    # value backup decides where to go deeper: V(destination page) + novelty of the best control behind it
    gv = EV.GraphValue(ex.graph, raw, self._gvec, ui.package, app_doc_idf=ARM in ("FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"))
    best = None
    for sl, dd, dui, fresh, ent in cands:
      inner = max(fresh, key=lambda t: gv.score(dui, t)[0])
      v = gv.V(ent) + gv.score(dui, inner)[0] + EV.BETA_D2 * gv.novelty(inner.sel)
      if best is None or v > best[0]:
        best = (v, sl, dd, inner)
    _, sl, dd, inner = best
    sl.d2 = {"via_dst": dd, "inner": inner.sel}
    return sl
  return None


def _probe_depth2(self, ex, pre, sl, step):
  """Tap the known edge, probe the unknown control on the page behind it (the probe restores that page), then
  Back to the origin page. The origin check (and relaunch-replay repair) is done by the caller."""
  ui0 = pre["ui"]
  x, y = ui0.nodes[sl.node_idx].center
  self.dev.tap(x, y)
  mid = self.dev.capture(f"s{step}_d2_mid", settle=True, pause=0.8)
  inner_fam = C.family_of_sel(sl.d2["inner"])
  target = next((t for t in U.enumerate_slots(mid["ui"]) if C.family_of_sel(t.sel) == inner_fam and t.safe
                 and (ARM != "FULL45" or not _risky_slot(self, mid["ui"], t))), None)
  if mid["ui"].activity != sl.d2["via_dst"]["activity"] or target is None or full9.transient(mid["ui"]):
    self.dev.back()
    chk = self.dev.capture(f"s{step}_d2_back", settle=True)
    return None if R.same_state(chk["ui"], ui0) else {"rollback_ok": False, "outcome": "d2_not_landed", "t_total_s": 0.0,
                                                       "dst_cid": mid["cid"], "family": inner_fam}
  p = ex.probe(mid, target, 0, step, depth=2)
  self.dev.back()
  chk = self.dev.capture(f"s{step}_d2_back", settle=True)
  ok = p["rollback_ok"] and R.same_state(chk["ui"], ui0)
  return dict(p, rollback_ok=ok, family=inner_fam)


FullEpisode.depth2_target = _depth2_target
FullEpisode.probe_depth2 = _probe_depth2


EDGE_WAIT_DEFAULT, EDGE_WAIT_MAX, POLL = 0.6, 3.0, 0.15


def _adaptive_wait(self, prev, m):
  """FULL40: wait for the screen after the model's action. Change detection against the screen the action was
  taken on, then two identical consecutive dumps; if nothing changes, wait the edge's learned stabilisation time
  (graph metadata: median time-to-stable of this (app, activity, control family) in this run, default 0.6 s)."""
  import time as _t  # noqa: E402
  ui0 = prev["pre"]["ui"]
  sig0 = (ui0.exact_sig, ui0.content_sig)
  key = f"{ui0.package}|{ui0.activity}|{prev.get('fam') or prev.get('slot_id')}"
  waits = None if os.environ.get("MAI_SHARED_STATE") == "1" else getattr(self, "_edge_wait", None)
  if waits is None:
    p = APPDIR / ARM_NAME / "edge_wait.json"
    try:
      waits = SS.read_json(p, {})
    except Exception:  # pylint: disable=broad-except
      waits = {}
    self._edge_wait = waits
  hist = waits.get(key) or []
  no_change_wait = sorted(hist)[len(hist) // 2] if hist else EDGE_WAIT_DEFAULT
  t0 = _t.time()
  last, changed, t_change = None, False, None
  while _t.time() - t0 < EDGE_WAIT_MAX:
    try:
      u = self.dev.quick_ui()
      sig = (u.exact_sig, u.content_sig)
    except Exception:  # pylint: disable=broad-except
      sig = None
    if sig is not None and sig != sig0 and not changed:
      changed, t_change = True, _t.time() - t0
    if changed and sig is not None and sig == last and (ARM not in ("FULL41", "FULL42", "FULL43", "FULL44", "FULL45") or any((n.text or n.desc) for n in u.app_nodes())):
      break                                   # changed and now stable (FULL41: and not a blank transitional frame)
    if not changed and _t.time() - t0 >= no_change_wait:
      break                                   # this edge usually shows no change (or changes later than we wait)
    last = sig
    _t.sleep(POLL)
  t_w = _t.time() - t0
  if changed:
    hist.append(round(t_w, 3))
    waits[key] = hist[-9:]
    try:
      self._edge_wait = SS.append_wait(APPDIR / ARM_NAME / "edge_wait.json", key, round(t_w, 3))
    except Exception:  # pylint: disable=broad-except
      pass
  m["adaptive_wait_changed" if changed else "adaptive_wait_nochange"] += 1
  return t_w


FullEpisode._adaptive_wait = _adaptive_wait


def _pixels_caught_up(self, prev, pre, k, m):
  """FULL43: under on-device load the screen is drawn later than the accessibility tree changes. If the tree says the
  page changed but the screenshot is still (almost) the one the last action was decided on, the screenshot is stale:
  wait and capture again (bounded 2 s), so the model never decides on a frame that predates its own action."""
  import numpy as np  # noqa: E402
  ui0, ui1 = prev["pre"]["ui"], pre["ui"]
  if (ui0.activity, ui0.layout_sig) == (ui1.activity, ui1.layout_sig):
    return pre
  def small(px):
    return np.asarray(px)[::16, ::16].astype(np.int16)
  ref = small(prev["pre"]["pixels"])
  t0 = time.time()
  while time.time() - t0 < 2.0 and float(np.abs(small(pre["pixels"]) - ref).mean()) < 3.0:
    m["stale_pixels"] += 1
    time.sleep(0.25)
    pre = self.dev.capture(f"s{k}_pre_redraw", settle=False, pause=None, full_res=True)
  return pre


FullEpisode._pixels_caught_up = _pixels_caught_up

# ---- FULL44 -----------------------------------------------------------------------------------------------------
# Controls whose effect is a mode or a new flow rather than a page (FULL41-43 runs: Settings, Search, Edit, Add Person,
# Go to and the calendar view switches failed Back or confused the next decision): never probed.
_RISKY_WORDS = ("setting", "search", "edit", "manage", "go to", "goto", "filter", "sort", "share", "delete", "remove",
                "view", "mode", "today", "account", "sign", "login", "log in")


def _risky_probe(sel):
  txt = " ".join(str(sel.get(k) or "") for k in ("label", "desc", "rid", "text")).lower().replace("_", " ")
  return any(w in txt for w in _RISKY_WORDS)


def _risky_slot(ep, ui, sl):
  """FULL45: toggles and checkable controls change state with one tap (FULL44: the Wi-Fi / Bluetooth switch bar was
  flipped back by a probe); the control the model acted on in its last step is never re-probed."""
  n = ui.nodes[sl.node_idx]
  cls = (n.cls or "").lower()
  rid = str(sl.sel.get("rid") or "").lower()
  if RG.unsafe_navigation_probe(sl.sel) or _risky_probe(sl.sel) or n.checkable or any(w in cls for w in ("switch", "checkbox", "toggle", "radiobutton")) \
     or any(w in rid for w in ("switch", "toggle", "checkbox")):
    return True
  return C.family_of_sel(sl.sel) == getattr(ep, "_last_act_fam", None)


_DESTRUCTIVE = ("delete", "remove", "discard", "erase", "clear", "reset", "uninstall")


def _destructive_dialog(ui):
  txt = " ".join((n.text or n.desc or "") for n in ui.app_nodes()).lower()
  return any(w in txt for w in _DESTRUCTIVE)


def _dialog_key(ui):
  lab = next(((n.text or n.desc or "") for n in ui.app_nodes() if (n.rid or "").endswith("button1")), "")
  return (ui.package, ui.activity, lab.lower())


def _app_not_first(goal, alias):
  """The instruction asks for a system action (Wi-Fi, Bluetooth, ...) before the app: "turn on Wifi, then open
  the dialer app". Commas and "and" alone are not evidence (place names, time ranges, file lists)."""
  g = goal.lower()
  i = g.find(alias.lower())
  return i > 0 and bool(re.search(r"\b(turn on|turn off|enable|disable|switch on|switch off)\b|wi-?fi|bluetooth", g[:i]))


def _pick_once(ep, live, graph):
  """A navigation macro hop is taken at most once per (page, control) in an episode (FULL43: the same drawer tap
  was replayed twice in a row)."""
  p = IG.pick(ep, live, graph)
  used = getattr(ep, "_macro_used", set())
  if p is not None and ((JG.page_key(live), p["fam"]) in used or (ARM == "FULL45" and any(f == p["fam"] for _, f in used))):
    return None                        # FULL45: once per control family per episode, on any page
  return p

# ---- simulated on-device inference load ----------------------------------------------------------------------
# A Qwen2-VL-2B llama-server runs inside the emulator (resident, like an on-device agent's model). While the real
# decision is computed (remote GELAB-4B, for comparable accuracy), the same device is kept busy by an equivalent
# on-device request (~1700 prompt tokens = ~700 text + ~1024 image tokens); it is cut off when the decision returns.
ONDEVICE_LOAD = os.environ.get("ONDEVICE_LOAD") == "1"
ONDEVICE_PORT = int(os.environ.get("ONDEVICE_PORT", "18096"))
_LOAD_TEXT = " ".join(["The screen shows a list of items with titles, dates, amounts and action buttons."] * 120)


def _load_start():
  import http.client  # noqa: E402
  try:
    conn = http.client.HTTPConnection("127.0.0.1", ONDEVICE_PORT, timeout=60)
    # a fresh prefix and no prompt cache: every decision pays the on-device prefill, then decodes until cut off
    body = json.dumps({"messages": [{"role": "user", "content": f"[{time.time_ns()}] " + _LOAD_TEXT}],
                       "max_tokens": 1024, "cache_prompt": False})
    conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
    return conn
  except Exception:  # pylint: disable=broad-except
    return None


def _load_stop(conn):
  try:
    if conn is not None:
      conn.sock.close() if conn.sock is not None else None
      conn.close()
  except Exception:  # pylint: disable=broad-except
    pass


_decide_remote = FullEpisode.decide


def _decide_with_load(self, agent, goal, pixels):
  if not ONDEVICE_LOAD:
    return _decide_remote(self, agent, goal, pixels)
  conn = _load_start()
  try:
    return _decide_remote(self, agent, goal, pixels)
  finally:
    _load_stop(conn)


FullEpisode.decide = _decide_with_load

# ---- FULL41: decode-overlapped actuation + parking on the most relevant known edge -----------------------------
# GELab emits explain / action / point / ... / summary; the summary is decoded last (~4 s of a 9-22 s decision on the
# Orin). The response is streamed; once the summary key appears every action field is complete, the action is
# executed and the next screen observed while the summary is still decoding. The summary (history for the next
# decision) is written into its slot before the next decision is built.
# STREAM_IO: every decision is streamed and timed (time to first token, time the action fields complete, end, tokens);
# STREAM: the action is executed as soon as its fields are complete (FULL41 by default; any arm with STREAM_ACT=1)
STREAM = os.environ.get("STREAM_ACT", "1" if ARM in ("FULL41", "FULL42", "FULL43", "FULL44", "FULL45") else "0") == "1"
STREAM_IO = STREAM or os.environ.get("STREAM_IO") == "1"
_T_ACTION = []

# ---- FULL42: restoration equivalence + Back that waits for the screen to change ----------------------------------
# Restored = same app/activity, same dialog and keyboard state, and >= 90% of the app's visible controls (class,
# id, label, coarse position) identical. The status bar and self-dismissing overlays (snackbar, toast) are ignored.
_strict_same = R.same_state


def _equiv_items(u):
  out = []
  for n in u.app_nodes():
    rid = n.rid or ""
    lab = (n.text or n.desc or "").strip()
    if not n.visible or not lab or "snackbar" in rid or "toast" in rid:
      continue
    out.append((lab, n.center, n.cls.endswith("EditText") or bool(getattr(n, "editable", False))))
  return out


def _label_match(x, y):
  (la, ca, ea), (lb, cb, eb) = x, y
  if abs(ca[0] - cb[0]) > 48 or abs(ca[1] - cb[1]) > 48 or ea != eb:
    return False
  if la == lb:
    return True
  # a non-editable control whose accessibility text gained or lost a suffix (path, date, state) is the same control;
  # text in an input field must match exactly (it is data)
  return not ea and min(len(la), len(lb)) >= 4 and (la.startswith(lb) or lb.startswith(la))


def _same_state_relaxed(a, b):
  """Restored = same app/activity, same dialog and keyboard state, and the visible labelled controls match one to one
  (same text at the same place, +-48 px; a non-editable control's text may differ by a suffix). Unlabelled
  containers, the status bar and self-dismissing overlays (snackbar, toast) are ignored."""
  if _strict_same(a, b):
    return True
  if a.package != b.package or (a.activity and b.activity and a.activity != b.activity) or a.ime_visible() != b.ime_visible():
    return False
  import state_abstraction as SA  # noqa: E402
  if SA.mode(a) != SA.mode(b):
    return False
  ia, ib = _equiv_items(a), _equiv_items(b)
  if not ia or len(ia) != len(ib):
    return False
  rest = list(ib)
  for x in ia:
    j = next((j for j, y in enumerate(rest) if _label_match(x, y)), None)
    if j is None:
      return False
    rest.pop(j)
  return True


if ARM in ("FULL42", "FULL43", "FULL44", "FULL45"):
  R.same_state = _same_state_relaxed


def _back_wait(dev, orig_back):
  def back():
    try:
      s0 = dev.quick_ui().exact_sig
    except Exception:  # pylint: disable=broad-except
      s0 = None
    orig_back()
    t0 = time.time()
    while s0 is not None and time.time() - t0 < 2.0:
      time.sleep(0.15)
      try:
        if dev.quick_ui().exact_sig != s0:
          break
      except Exception:  # pylint: disable=broad-except
        break
  return back
_SUMMARY_KEY = re.compile(r"(?:^|[\t\n])\s*summary\s*[:\uff1a]", re.I)


def _stream_call(api_url, model, messages, on_prefix, gen=None):
  import requests  # noqa: E402
  payload = {"messages": messages, "temperature": 0.0, "top_p": 1.0, "max_tokens": 1024, "stream": True,
             "stream_options": {"include_usage": True}}
  t0 = time.time()
  gen = {} if gen is None else gen
  if model:
    payload["model"] = model
  headers = {"Content-Type": "application/json"}
  key = os.environ.get("ANDROID_WORLD_LLM_API_KEY") or os.environ.get("VLLM_API_KEY")
  if key:
    headers["Authorization"] = f"Bearer {key}"
  txt, fired = "", False
  with requests.post(api_url, headers=headers, json=payload, stream=True, timeout=3000) as resp:
    resp.raise_for_status()
    for line in resp.iter_lines():
      if not line or not line.startswith(b"data: "):
        continue
      if line.strip() == b"data: [DONE]":
        break
      js = json.loads(line[6:])
      if js.get("usage"):
        gen["prompt_tokens"] = js["usage"].get("prompt_tokens")
        gen["completion_tokens"] = js["usage"].get("completion_tokens")
      if not js.get("choices"):
        continue
      ch = (js["choices"][0].get("delta") or {}).get("content") or ""
      if ch and "ttft_s" not in gen:
        gen["ttft_s"] = round(time.time() - t0, 3)
      txt += ch
      if fired or not ch:
        continue
      low = txt.lower()
      if "<think>" in low and "</think>" not in low:
        continue
      off = low.rfind("</think>") + 8 if "</think>" in low else 0
      mt = _SUMMARY_KEY.search(txt, off)
      if mt and "action" in low[off:mt.start()]:
        fired = True
        gen["t_action_s"] = round(time.time() - t0, 3)
        gen["action_chars"] = mt.start()
        on_prefix(txt[:mt.start()])
  gen["t_end_s"] = round(time.time() - t0, 3)
  gen["chars"] = len(txt)
  gen["think"] = "<think>" in txt.lower()
  if gen.get("completion_tokens") and "ttft_s" in gen and gen["t_end_s"] > gen["ttft_s"]:
    gen["decode_tok_s"] = round((gen["completion_tokens"] - 1) / (gen["t_end_s"] - gen["ttft_s"]), 2)
  return txt


def _parse_decision(self, response, lat):
  ga = R.gelab_agent
  parse_error = None
  try:
    parsed = ga.parse_gelab_response(response)
    action, tool_call, extras = ga.gelab_action_to_json_action(parsed, self.env.logical_screen_size)
  except R.seeact_utils.ParseActionError as e:
    parse_error = str(e)
    parsed = R.OrderedDict(cot="", action="WAIT", value="1", summary="Parser fallback wait", parse_error=parse_error)
    action = json_action.JSONAction(action_type=json_action.WAIT)
    tool_call = {"name": "mobile_use", "arguments": {"action": "wait", "value": 1}}
    extras = {"wait_seconds": 1, "parse_error": parse_error, "fallback": "parse_error_wait"}
  return {"response": response, "parsed": dict(parsed), "action": action, "tool_call": tool_call,
          "extras": extras, "model_latency_s": round(lat, 3), "parse_error": parse_error}


_decide_plain = FullEpisode.decide


def _decide_stream(self, agent, goal, pixels):
  if not STREAM_IO:
    return _decide_plain(self, agent, goal, pixels)
  from PIL import Image as _Image  # noqa: E402
  early = getattr(self, "_early", None)
  messages = R.gelab_agent.build_gelab_messages(goal, agent._history_text(), _Image.fromarray(pixels))  # pylint: disable=protected-access
  conn = _load_start() if ONDEVICE_LOAD else None
  t0 = time.time()

  gen = {}

  def on_prefix(prefix):
    if early is None or not STREAM:
      return
    try:
      d0 = _parse_decision(self, prefix, time.time() - t0)
    except Exception:  # pylint: disable=broad-except
      return
    if d0.get("parse_error"):
      return
    early["d"] = d0
    early["ev"].set()
  try:
    try:
      response = _stream_call(self.args.api_url, self.args.model, messages, on_prefix, gen)
    except Exception as e:  # pylint: disable=broad-except
      self.log("stream_error", {"err": repr(e)[:300]})
      response, _, _ = agent.vllm.predict_mm("", [], messages=messages)
  finally:
    _load_stop(conn)
  d = _parse_decision(self, response, time.time() - t0)
  d["gen"] = gen
  if gen.get("t_action_s"):
    _T_ACTION.append(gen["t_action_s"])
  self.log("gen", dict(gen, early=STREAM))
  if early is not None:
    early["ev"].set()                    # wake the waiter also when no early action was found
  return d


FullEpisode.decide = _decide_stream


def _placeholder_summary(self, agent):
  slot = {"summary": ""}
  agent._actions.append(slot)  # pylint: disable=protected-access
  agent._summaries.append("")  # pylint: disable=protected-access
  self._pending.update(agent=agent, slot=slot, idx=len(agent._summaries) - 1, t_exec=time.time())  # pylint: disable=protected-access
  return ""


def _commit_stream(self, agent, d, goal):
  p = getattr(self, "_pending", None)
  if not STREAM or p is None or p["early"] is not d:
    return _commit_plain(self, agent, d, goal)
  if d["extras"].get("return_text"):
    self.env.interaction_cache = str(d["extras"]["return_text"])
  t0 = time.time()
  agent._execute_action(d["action"], d["extras"])  # pylint: disable=protected-access
  exec_s = time.time() - t0
  return self._placeholder_summary(agent), exec_s


def _resolve_pending(self, steps=None):
  p = getattr(self, "_pending", None)
  if p is None:
    return
  self._pending = None
  t0 = time.time()
  p["th"].join()
  waited = time.time() - t0
  full = p["box"].get("d")
  ga = R.gelab_agent
  src = full if full is not None else p["early"]
  summary = (ga._normalize_space(src["parsed"].get("summary")) or ga._normalize_space(src["parsed"].get("explain"))  # pylint: disable=protected-access
             or str(src["tool_call"].get("arguments") or src["tool_call"]))
  if "slot" in p:
    p["slot"]["summary"] = summary
    p["agent"]._summaries[p["idx"]] = summary  # pylint: disable=protected-access
  mism = full is not None and {k: v for k, v in full["action"].__dict__.items() if v is not None} != \
      {k: v for k, v in p["early"]["action"].__dict__.items() if v is not None}
  self.log("late_summary", {"step": p["step"], "join_wait_s": round(waited, 3),
                            "early_latency_s": p["early"]["model_latency_s"],
                            "full_latency_s": full["model_latency_s"] if full is not None else None,
                            "hidden_s": round(time.time() - waited - p.get("t_exec", t0), 3) if "t_exec" in p else None,
                            "mismatch": mism, "summary": summary[:200]})
  for r in reversed(steps or []):
    if r.get("kind") == "model" and r.get("step") == p["step"]:
      r["summary"] = summary
      r["model_latency_full_s"] = full["model_latency_s"] if full is not None else None
      break


def _drop_pending(self):
  p = getattr(self, "_pending", None)
  if p is not None:
    p["th"].join()
    self._pending = None


_commit_plain = FullEpisode.commit
FullEpisode.commit = _commit_stream
FullEpisode._placeholder_summary = _placeholder_summary
FullEpisode._resolve_pending = _resolve_pending
FullEpisode._drop_pending = _drop_pending


def _park(self, ex, pre, rec, step, instance):
  """FULL41: the window has nothing left to probe and the decision is pending: take the most goal-relevant edge whose
  Back-undo is known (probed in this window or recorded reversible in the graph) and stay on its destination. If the
  model's action leads there (same control, or the graph's deterministic destination of its edge), it is already
  executed; otherwise Back (overlapped with the summary decode)."""
  ui = pre["ui"]
  early = getattr(self, "_early", None)
  if early is None or early["ev"].is_set() or full9.transient(ui):
    return
  _, reversible = full9.page_family_index(ex.graph)
  probed_ok = {p["family"] for p in rec["probes"] if p["rollback_ok"] and p["why"] != "D2" and p.get("dst_ui") is not None}
  bp = getattr(ex.graph, "_by_page", {})
  cands = []
  for sl in U.enumerate_slots(ui):
    fam = C.family_of_sel(sl.sel)
    key = (ui.package, ui.activity, fam)
    if not sl.safe or full9.effectful(sl.sel) or sl.sel.get("in_list") or is_commit_token(ui, fam) \
        or (ARM == "FULL45" and _risky_slot(self, ui, sl)):
      continue
    if fam not in probed_ok and key not in reversible:
      continue
    if ((ui.package, fam) in self._learned_eff or Graph_app_key(ui, sl) in getattr(self, "_bad_fams", set())
        or full9.container_key(ui.package, ui.activity, sl.sel) in self._bad_groups):
      continue
    cands.append(sl)
  if not cands:
    return
  gv = EV.GraphValue(ex.graph, self.out.parents[1], self._gvec, ui.package, app_doc_idf=True, instance=instance)
  gv.pvec = self._pvec
  scored = []
  for sl in cands:
    key = (ui.package, ui.activity, C.family_of_sel(sl.sel))
    ents = [(o["instance"], o["dst_cid"], o["dst"].get("layout_sig")) for o in bp.get(key, []) if o.get("dst_cid")][-3:]
    scored.append((gv.score(ui, sl, ents)[0], sl))
  scored.sort(key=lambda x: -x[0])
  sc, sl = scored[0]
  rec["park_cands"] = len(cands)
  if sc <= 0 or early["ev"].is_set():
    return
  chk = self.dev.capture(f"s{step}_park_chk", settle=True)
  if not R.same_state(chk["ui"], ui) or early["ev"].is_set():
    return
  self.dev.tap(*ui.nodes[sl.node_idx].center)
  dst = full7.capture_stable(self.dev, f"s{step}_park")
  self._spec = {"fam": C.family_of_sel(sl.sel), "dst": dst, "park": True, "score": sc}
  rec["park"] = {"fam": C.family_of_sel(sl.sel), "score": round(sc, 3), "landed": dst["ui"].activity}


FullEpisode._park = _park


def _restore_for_decision(self, ex, ui, d, k, m):
  """FULL35: before paying a re-inference, bring the screen back to the one the model decided on (joint-graph
  safe path, then relaunch + replay of this episode's actions) and keep the decision if it is valid there."""
  ok_s, _ = JG.restore(self, ex.graph, ui, k)
  if not ok_s:
    ok_s = full9.relaunch_restore(self, ui, k)
  v2 = self.dev.capture(f"s{k}_verify2", settle=True)
  if ok_s and (R.same_state(v2["ui"], ui) or full9.decision_still_valid(ui, v2["ui"], d["action"])
               or full9.relocate_decision(ui, v2["ui"], d["action"])):
    m["restored_then_executed"] += 1
    return True
  m["restore_for_decision_failed"] += 1
  return False


FullEpisode._restore_for_decision = _restore_for_decision


def _try_reuse(self, ui, pre, agent, steps, k, own, m, fn):
  """Graph reuse of the model's own decisions in this episode (no inference):
  repeat - the model chose the same navigation control on this page the last two times, landing alike;
  loop   - in the previous record the model's first action on this page was this control, and the task has moved
           on to the next record (multi-record procedures repeat their navigation).
  Only safe, non-list, non-commit controls found on the live screen; landing is verified (brake on mismatch)."""
  P = JG.page_key(ui)
  hist = self._page_acts.get(P, [])
  cand = why = land = None
  moved = lambda h: (h["land"][1], h["land"][2]) != (P[1], P[2])   # noqa: E731
  # FULL31: a page revisited with the same control twice is a cycle, not a procedure - in FULL23-30 every "repeat"
  # reuse (14-27 per run) happened in an episode that failed; the model decides there
  if (ARM != "FULL31" and len(hist) >= 2 and hist[-1]["fam"] == hist[-2]["fam"] and hist[-1]["land"][1] == hist[-2]["land"][1]
      and (ARM not in ("FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") or (moved(hist[-1]) and moved(hist[-2])))):
    # FULL19: only a repeated transition that leaves this page (an in-page repeat - next month, a digit, a stuck
    # tap - needs the model's judgement of when to stop)
    cand, why, land = hist[-1]["fam"], "repeat", hist[-1]["land"]
  elif fn is not None:
    cur = fn.next_rec
    prior = [h for h in hist if h["rec"] < cur and h["first"]]
    if prior and P not in self._rec_pages.get(cur, set()):
      cand, why, land = prior[-1]["fam"], "loop", prior[-1]["land"]
  if cand is None:
    return 0
  sl = next((t for t in U.enumerate_slots(ui) if C.family_of_sel(t.sel) == cand and t.safe and not t.sel.get("in_list")
             and not full9.effectful(t.sel) and t.unsafe_reason not in ("unsafe_label", "dialog_action_button")), None)
  if sl is None:
    m["reuse_absent"] += 1
    return 0
  node = ui.nodes[sl.node_idx]
  action = json_action.JSONAction(action_type="click", x=node.center[0], y=node.center[1])
  agent._execute_action(action, {})  # pylint: disable=protected-access
  self._exec_log.append({"action": {kk: vv for kk, vv in action.__dict__.items() if vv is not None}, "commit": False})
  lab = U.label_of(ui, node) or sl.sel.get("desc") or sl.sel.get("rid")
  summ = f"[graph] Tapped \"{lab}\"."
  agent._actions.append({"summary": summ})  # pylint: disable=protected-access
  agent._summaries.append(summ)  # pylint: disable=protected-access
  own.append(S.own_entry(ui, cand, cand))
  row = {"step": k, "kind": "graph_reuse", "why": why, "tok": cand, "summary": summ, "pre": R.strip(pre)}
  steps.append(row)
  self.log("steps", row)
  cap = full7.capture_stable(self.dev, f"s{k}_reuse")
  rec_ = fn.next_rec if fn is not None else 0
  self._rec_pages.setdefault(rec_, set()).add(P)
  self._page_acts.setdefault(P, []).append({"fam": cand, "land": JG.page_key(cap["ui"]), "rec": rec_,
                                            "first": why == "loop"})
  m["reuse_skip"] += 1
  m["reuse_" + why] += 1
  if cap["ui"].activity != land[1]:
    self._reuse_brake = True
    m["reuse_brake_landing"] += 1
  self._last_reuse_step = k
  return 1


FullEpisode.try_reuse = _try_reuse


def is_commit_token(ui, fam):
  """The model's clicked slot is a commit (deny-listed label or dialog action)."""
  for sl in U.enumerate_slots(ui):
    if C.family_of_sel(sl.sel) == fam:
      return sl.unsafe_reason in ("unsafe_label", "dialog_action_button")
  return False


P_REV_MIN = 0.7
if ARM in ("FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
  # FULL29 calibration: probes with predicted P_rev < 0.8 failed rollback 33% of the time (>= 0.8: 4%) and every
  # unrepaired failure costs a re-inference; probe only controls Back is expected to undo
  P_REV_MIN = 0.8
LAMBDA_U = KAPPA_ = 0.3 * (1 - 0.7) + 1e-3   # FULL33: uncertainty bonus (see the window), = KAPPA * (1 - P_REV_MIN_RV)
ENTRY_VERBS = ("add", "create", "new", "record", "save", "send", "reply")
ENTRY_WORDS = {"new", "add", "create", "compose"}


def _entry_ok(ep, sel):
  """FULL33: a creation entry (New Event / New Recipe / "+") opens an empty editor that Back leaves untouched; it
  is explored when the task has records to fill (the editor is where the record's fields live)."""
  if not getattr(ep, "_create_task", False) or sel.get("in_list") or sel.get("cls") in ("Switch", "CheckBox", "ToggleButton", "RadioButton"):
    return False
  lab = (sel.get("label") or sel.get("desc") or sel.get("text") or "").strip().lower()
  first = re.split(r"[^a-z]+", lab)[0] if lab else ""
  rid = set(re.split(r"[^a-z]+", (sel.get("rid") or "").lower()))
  return first in ENTRY_WORDS or (not lab and bool(rid & (ENTRY_WORDS | {"fab"})))


P_REV_MIN_RV = 0.7          # FULL32: re-observing a known edge fails Back 5% of the time vs 23% for discovery
P_REV_RISKY = 0.5          # FULL17: accepted only while relaunch + replay can repair the state
LAMBDA_T = 0.15            # FULL26: exploration bonus per log(1 + tasks through the page)
KAPPA = 0.3                # FULL23: expected re-inference cost of a probe Back may fail to undo, in value units
PROBES_PER_STEP = 5        # FULL17: fixed exploration effort per inference
if ARM in ("FULL41", "FULL42", "FULL43", "FULL44", "FULL45"):
  PROBES_PER_STEP = 10      # FULL41: exploration runs until the decision's action is known (edge inference windows)
ELEMENT_APPS = {}
try:
  ELEMENT_APPS = json.load(open(HERE.parent.parent / "runtime_graph_exploration" / "graph" / "element_idf_base.json"))["apps"]
except Exception:  # pylint: disable=broad-except
  pass


def Graph_app_key(ui, slot):
  return (ui.package, C.family_of_sel(slot.sel))


def is_dialog_button(ui, fam):
  for sl in U.enumerate_slots(ui):
    if C.family_of_sel(sl.sel) == fam:
      return sl.unsafe_reason == "dialog_action_button"
  return False


def graph_obs(graph, struct, slot_id):
  return graph.slots.get((struct, slot_id), [])


def _run(self):
  res = S._orig_run(self)  # pylint: disable=protected-access
  if ARM in ("FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30", "FULL31", "FULL32", "FULL33", "FULL34", "FULL35", "FULL36", "FULL37", "FULL38", "FULL39", "FULL40", "FULL41", "FULL42", "FULL43", "FULL44", "FULL45") and getattr(self, "_intent", None) is not None:
    # P0-1: a successful episode validates the edges it traversed for this kind of goal
    self._intent.record_episode(self.out / "steps.jsonl", self._gvec, res.get("is_successful") == 1.0)
  if ARM.startswith("FULL") and res.get("is_successful") == 1.0 and getattr(self, "_own_seq", None):
    S.Store(APPDIR / ARM_NAME / "store.json").append(self._own_seq, self.args.task)
  return res


def _setup(self):
  meta = S._orig_setup(self)  # pylint: disable=protected-access
  meta["full_arm"] = ARM
  if ARM in ("FULL42", "FULL43", "FULL44", "FULL45"):
    self.dev.back = _back_wait(self.dev, self.dev.back)
  self._confirmed_destructive, self._last_act_fam = set(), None   # Back waits until the screen changed (bounded 2 s)
  (self.out / "meta.json").write_text(R.jdump(meta))
  return meta


FullEpisode.run = _run
FullEpisode.setup = _setup
R.Episode = FullEpisode

if __name__ == "__main__":
  R.main()
