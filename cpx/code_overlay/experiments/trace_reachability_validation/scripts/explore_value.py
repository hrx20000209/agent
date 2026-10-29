"""FULL16: graph-based exploration value.

One relevance function, rel(goal, T) = TF-IDF cosine between the goal's operation/object terms and a text T, applied
to whatever text evidence the graph holds about a candidate control:

  L1 known edge      T = content of the destination page captured in the graph       confidence 0.8
  L2 graph neighbours T = destination pages of same-label / same-rid controls taken elsewhere (any app),
                         weighted by their similarity to the candidate                  confidence 0.5
  L3 unknown         T = the control's own label + split resource-id + nearby caption  (always mixed in)

  value(c) = conf * rel(goal, T_evidence) + (1 - conf) * rel(goal, T_L3)

Known pages also back up value along the graph: V(page) = max(rel(goal, page), gamma * max V(child page)).
Selection: score(c) = value(c) + beta * novelty(c), among controls with predicted reversibility >= threshold.

Goal side: terms of the instruction minus app names, dates, numbers and the parsed field values (those are
parameters, not operations), weighted by IDF over the task corpus. UI side: term weights are IDF over a corpus of
screens from all apps, so controls present everywhere ("More", "Settings", "Back") weigh little.
"""
from __future__ import annotations

import glob
import gzip
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from android_world import shared_state as SS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import full9  # noqa: E402

R, U, C, FG = full9.R, full9.U, full9.C, full9.FG
UI_IDF_PATH = HERE.parent.parent / "runtime_graph_exploration" / "graph" / "ui_idf.json"
CONF_L1, CONF_L2, GAMMA = 0.8, 0.5, 0.7
# FX candidates are already unknown (page-level dedup), so novelty adds only noise there (offline: beta 0 ->
# 16 vs 7 controls later clicked by the model; 0.1-0.3 -> 15); novelty is used to choose where to go deeper.
BETA, BETA_D2 = 0.0, 0.3
PARAM_W = 1.0     # FULL17: list rows are entity instances; the goal's parameter values are their relevance

_STOP = {"the", "a", "an", "to", "of", "in", "on", "at", "for", "and", "or", "with", "from", "into", "by", "is", "are",
         "be", "it", "this", "that", "my", "me", "i", "you", "your", "all", "following", "app", "please", "using",
         "use", "as", "do", "not", "if", "then", "there", "their", "its", "only", "which", "what", "how", "any", "should",
         "make", "sure", "named", "called", "titled", "e", "g"}
_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
           "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tomorrow",
           "yesterday", "am", "pm"}


def stem(t):
  if len(t) > 4 and t.endswith("ies"):
    return t[:-3] + "y"
  if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
    return t[:-1]
  return t


def tokens(text):
  text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
  return [stem(t) for t in re.split(r"[^a-z]+", text.lower()) if len(t) >= 2]


# ---- UI-side IDF (screens of all apps are the documents) ----------------------------------------------------
_UI_IDF = None


def build_ui_idf(arms=("FULL11", "FULL12", "FULL13", "FULL14")):
  df, n = Counter(), 0
  for arm in arms:
    for f in glob.glob(str(HERE.parent / "raw" / "full" / arm / "*" / "p0" / "a11y" / "*_pre.json.gz")):
      try:
        with gzip.open(f, "rt") as fh:
          d = json.load(fh)
        ui = U.parse_flat(d["payload"], activity=d["activity"])
      except Exception:  # pylint: disable=broad-except
        continue
      n += 1
      df.update(set(page_tokens(ui)))
  out = {"n": n, "df": dict(df)}
  UI_IDF_PATH.write_text(json.dumps(out))
  return out


def ui_idf():
  global _UI_IDF
  if _UI_IDF is None:
    d = json.loads(UI_IDF_PATH.read_text())
    n = d["n"]
    _UI_IDF = (n, {t: math.log((n + 1) / (c + 1)) + 1 for t, c in d["df"].items()})
  return _UI_IDF


def w_ui(t):
  n, idf = ui_idf()
  return idf.get(t, math.log(n + 1) + 1)


# ---- FULL32: app-as-document global IDF ----------------------------------------------------------------------
APP_TOKENS_PATH = HERE.parent.parent / "runtime_graph_exploration" / "graph" / "app_tokens.json"
_APP_DOC = None


def app_doc():
  """Profiling corpus with one document per app: {package: set(terms)} and the per-term app frequency."""
  global _APP_DOC
  if _APP_DOC is None:
    toks = {k: set(v) for k, v in json.loads(APP_TOKENS_PATH.read_text()).items()}
    df = Counter()
    for v in toks.values():
      df.update(v)
    _APP_DOC = (toks, df)
  return _APP_DOC


def w_app(pkg):
  """Global IDF with apps as documents, the current app left out: a term every other app shows weighs little,
  a term few other apps show weighs much."""
  toks, df = app_doc()
  own = toks.get(pkg, set())
  n = len(toks) - (1 if pkg in toks else 0)
  def w(t):
    return math.log((n + 1) / (df.get(t, 0) - (1 if t in own else 0) + 1)) + 1
  return w


# ---- goal side ---------------------------------------------------------------------------------------------
_APP_PATTERNS = None


def app_patterns():
  global _APP_PATTERNS
  if _APP_PATTERNS is None:
    from android_world.env import adb_utils  # pylint: disable=import-outside-toplevel
    pats = set()
    for k in adb_utils._PATTERN_TO_ACTIVITY:  # pylint: disable=protected-access
      pats |= {p.strip() for p in k.split("|") if p.strip()}
    _APP_PATTERNS = sorted(pats, key=len, reverse=True)
  return _APP_PATTERNS


def goal_vector(goal, fx=None):
  """Operation/object terms of the goal with task-corpus IDF weights."""
  g = goal
  for p in app_patterns():                       # app names name the app, not the operation
    g = re.sub(r"\b" + re.escape(p) + r"\b(\s+app)?", " ", g, flags=re.I)
  if fx is not None:                             # parsed field values are parameters
    for vs in fx.recs.values():
      for v in vs:
        if v:
          g = g.replace(v, " ")
  g = re.sub(r"'[^']*'|\"[^\"]*\"", " ", g)      # quoted literals are parameters
  g = re.sub(r"\S*\d\S*", " ", g)                 # dates, times, numbers, amounts
  idf = full9.goal_idf()
  vec = {}
  for t in tokens(g):
    if t in _STOP or t in _MONTHS or t in vec:
      continue                                    # presence, not count: repeated entity words are not operations
    vec[t] = max(min(idf(t), idf(t + "s"), idf(t[:-1] + "ies") if t.endswith("y") else 99), 0.1)
  return vec


def rel(gvec, toks, wfun=None):
  """TF-IDF cosine between the goal vector and a text (UI-side IDF, or the graph's mixed IDF)."""
  if not gvec or not toks:
    return 0.0
  wfun = wfun or w_ui
  tv = Counter()
  for t in toks:
    tv[t] += wfun(t)
  dot = sum(gvec[t] * tv[t] for t in gvec if t in tv)
  if dot == 0:
    return 0.0
  return dot / (math.sqrt(sum(v * v for v in gvec.values())) * math.sqrt(sum(v * v for v in tv.values())))


# ---- text evidence -------------------------------------------------------------------------------------------
def page_tokens(ui):
  out = []
  for n in ui.app_nodes():
    if not n.visible:
      continue
    out += tokens(n.text) + tokens(n.desc) + tokens(n.short_rid.replace("_", " "))
  return out


def own_tokens(ui, sel):
  """L3: the control's label, split resource-id and the caption right above it."""
  toks = tokens(sel.get("label") or "") + tokens(sel.get("desc") or "") + tokens((sel.get("rid") or "").replace("_", " "))
  b = sel.get("bounds")
  if b:
    best = None
    for m in ui.app_nodes():
      if not m.visible or not m.text or m.clickable:
        continue
      if m.bounds[3] <= b[1] + 5 and b[1] - m.bounds[3] < 200:
        d = b[1] - m.bounds[3]
        if best is None or d < best[0]:
          best = (d, m.text)
    if best:
      toks += tokens(best[1])
  return toks


_CAP = {}


def load_cap(raw, instance, cid):
  key = (str(raw), instance, cid)
  if key not in _CAP:
    task, var = instance.split("/")
    try:
      _CAP[key] = R.load_ui(SS.episode_path(raw, task, var), cid)
    except Exception:  # pylint: disable=broad-except
      _CAP[key] = None
  return _CAP[key]


class GraphValue:
  """Graph-side evidence for one episode window (rebuilt when the graph grows)."""

  def __init__(self, graph, raw, gvec, package=None, app_doc_idf=False, instance=None):
    self.raw, self.gvec, self.package = raw, gvec, package
    self.wg = w_app(package) if app_doc_idf else w_ui        # FULL32: global IDF over apps (profiling corpus)
    self.instance, self.ep_pages = instance, {}
    self.by_label, self.by_rid, self.children, self.page_rel, self.app_pages = {}, {}, {}, {}, {}
    self.seen = Counter()
    for (struct, sid), obs in graph.slots.items():
      for o in obs:
        sel, d = o.get("selector") or {}, o.get("dst")
        lab = FG.norm(sel.get("label") or sel.get("desc") or "")
        rid = sel.get("rid") or ""
        self.seen[lab or rid] += 1
        if not d or not o.get("dst_cid"):
          continue
        ent = (o["instance"], o["dst_cid"], d.get("layout_sig"))
        if lab:
          self.by_label.setdefault(lab, []).append(ent)
        if rid:
          self.by_rid.setdefault(rid, []).append(ent)
        src = o["src"].get("layout_sig")
        if d.get("layout_sig") != src:
          self.children.setdefault(src, set()).add(ent)
        if package and d.get("package") == package:
          self.app_pages.setdefault(d.get("layout_sig"), ent)
          if instance and o.get("instance") == instance:
            self.ep_pages.setdefault(d.get("layout_sig"), ent)
    self._local_idf()

  @staticmethod
  def _df(raw, ents):
    df, n = Counter(), 0
    for ent in ents[:80]:
      ui = load_cap(raw, ent[0], ent[1])
      if ui is None:
        continue
      n += 1
      df.update(set(page_tokens(ui)))
    return n, df

  def _local_idf(self):
    """Graph-local IDF: the documents are the distinct pages of the current app held in this graph. Terms on
    every page of the app (its toolbar, its navigation) weigh little; terms of a single page weigh much. It
    sharpens as the graph grows."""
    df, n = Counter(), 0
    for ent in list(self.app_pages.values())[:80]:
      ui = load_cap(self.raw, ent[0], ent[1])
      if ui is None:
        continue
      n += 1
      df.update(set(page_tokens(ui)))
    self.local_n, self.local_df = n, df
    # FULL32: task-local IDF over the pages of this app reached in the current episode (agent + probes)
    self.ep_n, self.ep_df = self._df(self.raw, list(self.ep_pages.values())) if self.instance else (0, Counter())

  def w(self, t):
    """Term weight: geometric mean of the global (all apps' screens) and the graph-local (this app's pages)
    IDF; global only while the app has fewer than 3 pages in the graph."""
    g = self.wg(t)
    ws = [g]
    if self.local_n >= 3:
      ws.append(math.log((self.local_n + 1) / (self.local_df.get(t, 0) + 1)) + 1)
    if self.instance and self.ep_n >= 3:
      ws.append(math.log((self.ep_n + 1) / (self.ep_df.get(t, 0) + 1)) + 1)
    if len(ws) == 1:
      return g
    if not self.instance and len(ws) == 2:
      return math.sqrt(ws[0] * ws[1])
    # FULL32: geometric mean of global (apps), graph-local (this app's pages) and task-local (this episode)
    out = 1.0
    for x in ws:
      out *= x
    return out ** (1.0 / len(ws))

  def _page_rel(self, ent):
    if ent not in self.page_rel:
      ui = load_cap(self.raw, ent[0], ent[1])
      self.page_rel[ent] = rel(self.gvec, page_tokens(ui), self.w) if ui is not None else 0.0
    return self.page_rel[ent]

  def V(self, ent, depth=2, _seen=None):
    """Value backup: a page is worth its own relevance or a discounted best child."""
    v = self._page_rel(ent)
    if depth <= 0:
      return v
    _seen = (_seen or set()) | {ent[2]}
    for ch in list(self.children.get(ent[2], ()))[:12]:
      if ch[2] in _seen:
        continue
      v = max(v, GAMMA * self.V(ch, depth - 1, _seen))
    return v

  def value(self, ui, sl, known_edge_ents=()):
    sel = sl.sel
    own = rel(self.gvec, own_tokens(ui, sel), self.w)
    if known_edge_ents:                                         # L1
      ev = max(self.V(e) for e in known_edge_ents)
      return CONF_L1 * ev + (1 - CONF_L1) * own, "L1"
    lab = FG.norm(sel.get("label") or sel.get("desc") or "")
    nb = (self.by_label.get(lab, []) if lab else []) + (self.by_rid.get(sel.get("rid") or "", []) if sel.get("rid") else [])
    if nb:                                                      # L2 (same label / rid: similarity 1)
      ev = sum(self.V(e, depth=1) for e in nb[:8]) / min(len(nb), 8)
      return CONF_L2 * ev + (1 - CONF_L2) * own, "L2"
    return own, "L3"

  def novelty(self, sel):
    lab = FG.norm(sel.get("label") or sel.get("desc") or "") or (sel.get("rid") or "")
    return 1.0 / (1.0 + self.seen.get(lab, 0))

  def score(self, ui, sl, known_edge_ents=()):
    v, lvl = self.value(ui, sl, known_edge_ents)
    if sl.sel.get("in_list") and getattr(self, "pvec", None):
      v += PARAM_W * rel(self.pvec, own_tokens(ui, sl.sel), self.w)
    return v + getattr(self, "beta", BETA) * self.novelty(sl.sel), v, lvl


if __name__ == "__main__":
  out = build_ui_idf()
  print("screens", out["n"], "terms", len(out["df"]))


def param_vector(fx):
  """Parameter side of the goal: the parsed field values (entity names, titles, file names)."""
  vec = {}
  if fx is None:
    return vec
  for vs in fx.recs.values():
    for v in vs:
      for t in tokens(v or ""):
        if t not in _STOP and t not in _MONTHS:
          vec[t] = 1.0
  return vec
