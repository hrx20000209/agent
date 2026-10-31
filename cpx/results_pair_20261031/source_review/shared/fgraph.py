"""Function graph: parameterized sub-trajectories over the observation graph.

A FUNCTION is a sub-path of a successful trajectory between commits:
    entry screen -> [NAV click | SLOT_CLICK(field) | SLOT_TYPE(field)]* -> (commit, left to the model)
A SLOT is an action whose clicked label or typed text equals a parameter value of
that episode's goal; the slot remembers WHICH goal field the value came from, so
the function can be re-instantiated with another goal's value of the same field.

Goals are split into PARAMETERS (records of a table / key: value blocks, quoted
strings, enumerations, file names, phone numbers ...) and OPERATION terms (the
rest). Parameters bind slots; operation terms authorize a function (same verb)
and rank exploration targets (TF-IDF similarity), never execute anything.

Sources of functions: other tasks of the same app (history, leave-one-template-
out), the current episode's own completed segments, and exploration edges, which
connect the current screen to a function's entry screen.
"""

from __future__ import annotations

import json
import math
import re
import os as _os
_m = re.match(r"FULL(\d+)", _os.environ.get("FULL_ARM", ""))
_ARM_N = int(_m.group(1)) if _m else 0
from collections import Counter, defaultdict
from pathlib import Path

import memory as M

C, U, R = M.C, M.U, M.R
STOP = M.STOP | set("following named with into from the in on at to of for by app apps".split())
VERBS = {"add": "add", "create": "add", "new": "add", "enter": "add", "delete": "delete", "remove": "delete",
         "edit": "edit", "update": "edit", "change": "edit", "rename": "edit", "move": "move", "copy": "copy",
         "send": "send", "reply": "send", "resend": "send", "forward": "send", "mark": "mark", "turn": "turn",
         "record": "record", "take": "take", "set": "set", "save": "save", "merge": "merge", "share": "send"}


def norm(t):
  return re.sub(r"\s+", " ", (t or "").strip().lower()).strip(" .,;:'\"")


def _key(words):
  ws = [w for w in re.findall(r"[a-z]+", words.lower()) if w not in STOP]
  return " ".join(ws[-2:])


# ------------------------------------------------------------------ goal parsing
def parse_goal(goal):
  """-> {"slots": [{"field", "value", "rec"}], "verb": str|None, "ops": [terms]}"""
  slots, covered = [], []
  lines = goal.split("\n")
  # 1. table: header "a|b|c" followed by rows
  for i, l in enumerate(lines):
    if "|" in l and i + 1 < len(lines) and "|" in lines[i + 1]:
      hdr = [norm(h) for h in l.split("|")]
      rec = 0
      for row in lines[i + 1:]:
        if "|" not in row:
          break
        vals = [v.strip() for v in row.split("|")]
        for h, v in zip(hdr, vals):
          if v:
            slots.append({"field": h, "value": v.lstrip("$"), "rec": rec})
            slots.append({"field": h, "value": v, "rec": rec, "alias": True})
        rec += 1
      covered.append(l)
      break
  # 2. key: value lines / blocks ("Recipe: X" opens a record)
  rec = -1
  for l in lines:
    m = re.match(r"^\s*([A-Za-z][A-Za-z _]{0,30}):\s*(.+)$", l)
    if m and "|" not in l and len(m.group(1).split()) <= 3:
      k, v = norm(m.group(1)), m.group(2).strip()
      if not l.startswith(" ") or rec < 0:
        rec += 1
      slots.append({"field": k, "value": v, "rec": max(rec, 0)})
  # 3. inline "Key: value, Key2: value2" pairs
  for m in re.finditer(r"([A-Z][A-Za-z ]{1,20}):\s*([^,.\n]+?)(?=,\s*[A-Z][A-Za-z ]{1,20}:|\.\s|\.$|$)", goal):
    k, v = norm(m.group(1)), m.group(2).strip()
    if not any(s["value"] == v for s in slots):
      slots.append({"field": k, "value": v, "rec": 0})
  # 4. quoted strings; field = last content words before the quote
  for m in re.finditer(r"'([^']{1,200})'|\"([^\"]{1,200})\"", goal):
    v = m.group(1) or m.group(2)
    slots.append({"field": " ".join(_key(goal[max(0, m.start() - 40):m.start()]).split()[-1:]), "value": v, "rec": 0})
  # 5. enumeration after a colon: "...: A, B and C."
  # FULL33: a colon inside a time ("7:45am") does not start a list, and a question is not a list
  _pat = r"(?<!\d):(?!\d)\s*([^:\n?]+?)\.?\s*$" if _ARM_N >= 33 else r":\s*([^:\n]+?)\.?\s*$"
  m = re.search(_pat, lines[0]) if not any(s["field"] for s in slots if not s["field"].startswith(("file:", "number"))) else None
  if m:
    head = _key(lines[0][:m.start()])
    if _ARM_N >= 33 and "," in m.group(1):
      # FULL33: list items are comma-separated; " and " inside an item is part of a name ("Kale and Quinoa Salad")
      items = [re.sub(r"^and\s+", "", x.strip()) for x in m.group(1).split(",") if x.strip()]
    elif _ARM_N >= 33:
      items = [m.group(1).strip()]
    else:
      items = [x.strip() for x in re.split(r",\s*|\s+and\s+", m.group(1)) if x.strip()]
    if 1 <= len(items) <= 8:
      for j, it in enumerate(items):
        slots.append({"field": "enum:" + head, "value": it, "rec": j})
  # 6. file names, phone numbers, message tails
  for m in re.finditer(r"[\w\-]+\.(?:md|txt|jpg|png|mp3|mp4|m4a|pdf|csv)\b", goal):
    slots.append({"field": "file:" + _key(goal[max(0, m.start() - 30):m.start()]), "value": m.group(0), "rec": 0})
  for m in re.finditer(r"\+?\d[\d\- ]{6,}\d", goal):
    slots.append({"field": "number", "value": m.group(0).strip(), "rec": 0})
  m = re.search(r"(?:message|text):\s*(.+)$", goal, re.I | re.S)
  if m:
    slots.append({"field": "message", "value": m.group(1).strip().rstrip("."), "rec": 0})
  # operation terms: goal words outside parameter values
  rest = goal
  for s in sorted(slots, key=lambda s: -len(s["value"])):
    rest = rest.replace(s["value"], " ")
  ops = [w for w in re.findall(r"[a-z]+", rest.lower()) if w not in STOP and len(w) > 2]
  verb = next((VERBS[w] for w in re.findall(r"[a-z]+", goal.lower()) if w in VERBS), None)
  return {"slots": slots, "verb": verb, "ops": ops}


def records(parse):
  """field -> list of values by record index (first value per (field, rec))."""
  out = defaultdict(dict)
  for s in parse["slots"]:
    if s.get("alias"):
      continue
    out[s["field"]].setdefault(s["rec"], s["value"])
  return {f: [v for _, v in sorted(d.items())] for f, d in out.items()}


def value_index(parse):
  idx = {}
  for s in parse["slots"]:
    for v in {norm(s["value"]), norm(s["value"].lstrip("$"))}:
      if len(v) >= 2:
        idx.setdefault(v, s)
  return idx


# ------------------------------------------------------------------ extraction
def _field_selector(ui, x, y):
  n = U.element_at(ui, int(x), int(y), clickable_only=False)
  edits = sorted([m for m in ui.app_nodes() if m.editable and m.visible], key=lambda m: (m.bounds[1], m.bounds[0]))
  k = next((i for i, m in enumerate(edits) if n is not None and m.idx == n.idx), None)
  if k is None and n is not None:   # tapped label/container: nearest edit below the point
    below = [i for i, m in enumerate(edits) if m.bounds[1] >= n.bounds[1] - 5]
    k = below[0] if below else None
  rid = n.short_rid if n is not None else ""
  return {"rid": rid, "k": k, "n_edits": len(edits)}


def extract(ep_dir, goal, pkg=None):
  """Functions from one successful episode: list of dicts."""
  ep_dir = Path(ep_dir)
  P = parse_goal(goal)
  vidx = value_index(P)
  rows = [json.loads(l) for l in open(ep_dir / "steps.jsonl")]
  rows = [r for r in rows if r.get("kind") == "model" and r.get("pre")]
  funcs, cur, entry = [], [], None
  dropped = False
  for r in rows:
    try:
      ui = R.load_ui(ep_dir, r["pre"]["cid"])
    except Exception:  # pylint: disable=broad-except
      continue
    if pkg is not None and ui.package != pkg and "launcher" not in ui.package:
      pass
    a = r["action"]
    at = a.get("action_type")
    step = None
    if at == "input_text" and a.get("text"):
      s = vidx.get(norm(a["text"]))
      if s is None:
        dropped = True              # typed text that is not a goal value: the function does not reproduce it
      if s is not None:
        step = {"op": "type", "field": s["field"], "rec": s["rec"], "sel": _field_selector(ui, a.get("x") or 0, a.get("y") or 0)
                if a.get("x") is not None else {"rid": "", "k": None}, "act": ui.activity}
    elif at in ("click", "long_press") and a.get("x") is not None:
      n = U.element_at(ui, int(a["x"]), int(a["y"]))
      if n is None:
        continue
      sel = U.selector(ui, n)
      fam = C.family_of_sel(sel)
      texts = [norm(t) for t in M.page_labels(ui) if t]
      lab = norm(U.label_of(ui, n))
      inner = [norm(m.text or m.desc) for m in ui.app_nodes()
               if (m.text or m.desc) and n.bounds[0] <= m.center[0] < n.bounds[2] and n.bounds[1] <= m.center[1] < n.bounds[3]]
      hit = next((vidx[t] for t in [lab] + inner if t in vidx), None)
      slot = next((sl for sl in U.enumerate_slots(ui) if sl.node_idx == n.idx), None)
      commit = slot is not None and slot.unsafe_reason in ("unsafe_label", "dialog_action_button")
      if commit and hit is None:
        if cur and any(s["op"] != "nav" for s in cur):
          funcs.append({"entry_act": entry[0], "entry_key": entry[1], "steps": cur, "commit": lab, "verb": P["verb"],
                        "fields": sorted({s["field"] for s in cur if s["op"] != "nav"}), "pkg": ui.package,
                        "complete": not dropped})
        cur, entry, dropped = [], None, False
        continue
      op = at if at == "long_press" else "click"
      step = ({"op": "slot_" + op, "field": hit["field"], "rec": hit["rec"], "row": f"{n.short_cls}|{n.short_rid}",
               "act": ui.activity} if hit is not None else
              {"op": "nav", "fam": fam, "label": lab, "act": ui.activity, "kind": op,
               "editable": bool(slot is None or slot.unsafe_reason == "text_field")})
    elif at in ("status", "answer", "open_app"):
      cur, entry, dropped = [], None, False
      continue
    if step is not None:
      if not cur:
        entry = (ui.activity, M.state_key(ui))
      cur.append(step)
  return funcs


CACHE = Path(__file__).resolve().parent.parent / "graph" / "function_cache.jsonl"


def build_cache(roots, out=CACHE, extra=False):
  """Extract functions from every successful episode once (immutable cache, tagged by template)."""
  n = 0
  with Path(out).open("a" if extra else "w") as fo:
    for root in roots:
      for res in sorted(Path(root).glob("*/*/p*/result.json")):
        ep = res.parent
        try:
          if json.loads(res.read_text()).get("is_successful") != 1.0:
            continue
          goal = json.loads((ep / "meta.json").read_text())["goal"]
          for f in extract(ep, goal):
            f["src"] = f"{ep.parent.parent.name}/{ep.parent.name}/{ep.name}"
            f["template"] = M.template_key(ep.parent.name)
            fo.write(json.dumps(f) + "\n")
            n += 1
        except Exception:  # pylint: disable=broad-except
          continue
  return n


def self_reported_done(ep):
  """The model itself ended the episode as complete (no benchmark evaluator involved)."""
  last = None
  for line in open(ep / "steps.jsonl"):
    r = json.loads(line)
    if r.get("kind") == "model":
      last = r
  if last is None:
    return False
  a = last.get("action") or {}
  return (a.get("action_type") == "status" and a.get("goal_status") == "task_complete") or a.get("action_type") == "answer"


def build_online(run_dir, out):
  """FULL33 online function library: functions extracted from the episodes of THIS run that the model itself
  completed, in run order; nothing from earlier runs. Appends episodes not yet processed."""
  out = Path(out)
  done_p = out.with_suffix(".done")
  done = set(done_p.read_text().split()) if done_p.exists() else set()
  out.touch()
  new = []
  for ep in sorted(Path(run_dir).glob("*/p0")):
    key = ep.parent.name
    if key in done or not (ep / "result.json").exists() or not (ep / "steps.jsonl").exists():
      continue
    done.add(key)
    try:
      if not self_reported_done(ep):
        continue
      goal = json.loads((ep / "meta.json").read_text())["goal"]
      for f in extract(ep, goal):
        f["src"] = f"online/{key}"
        f["template"] = M.template_key(key)
        new.append(f)
    except Exception:  # pylint: disable=broad-except
      continue
  with out.open("a") as fo:
    for f in new:
      fo.write(json.dumps(f) + "\n")
  done_p.write_text("\n".join(sorted(done)))
  return len(new)


class Library:
  """Leave-one-template-out view of the function cache (+ functions added during the run)."""

  def __init__(self, task, cache=CACHE):
    tk = M.template_key(task)
    self.funcs, seen = [], set()
    for l in open(cache):
      f = json.loads(l)
      if f["template"] == tk and _os.environ.get("SAME_TPL") != "1":   # SAME_TPL: seen-task split, same template allowed
        continue
      sig = (f["entry_act"], tuple((s["op"], s.get("field") or s.get("fam")) for s in f["steps"]))
      if sig in seen:
        continue
      seen.add(sig)
      self.funcs.append(f)

  def add(self, funcs):
    self.funcs.extend(funcs)

  def candidates(self, goal_parse, act):
    recs = records(goal_parse)
    out = []
    for f in self.funcs:
      if f["entry_act"] != act or f["verb"] is None or f["verb"] != goal_parse["verb"]:
        continue
      if not all(fd in recs for fd in f["fields"]):
        continue
      out.append(f)
    out.sort(key=lambda f: (-len(f["fields"]), -len(f["steps"])))
    return out


# ------------------------------------------------------------------ element statistics (TF-IDF)
class ElementStats:
  """Structural element identities per task (documents = tasks). Data text is excluded."""

  def __init__(self, path):
    self.path = Path(path)
    d = json.loads(self.path.read_text()) if self.path.exists() else {"docs": {}, "apps": {}}
    self.docs = {k: set(v) for k, v in d["docs"].items()}
    self.apps = {k: set(v) for k, v in d["apps"].items()}

  @staticmethod
  def ident(sl):
    if sl.sel.get("in_list"):
      return None                                  # rows are data, not functions
    return norm(sl.sel.get("label") or sl.sel.get("desc") or sl.sel.get("rid") or "") or None

  def add_doc(self, task, pkg, idents):
    for t in idents:
      self.docs.setdefault(t, set()).add(task)
      self.apps.setdefault(t, set()).add(pkg)

  def save(self):
    self.path.write_text(json.dumps({"docs": {k: sorted(v) for k, v in self.docs.items()},
                                     "apps": {k: sorted(v) for k, v in self.apps.items()}}))

  def idf(self, t):
    n = max(1, len({x for v in self.docs.values() for x in v}))
    return math.log((1 + n) / (1 + len(self.docs.get(t, ()))))

  def n_apps(self, t):
    return len(self.apps.get(t, ()))


def op_similarity(label, ops, stats):
  """TF-IDF cosine between an element label and the goal's operation terms."""
  lw = [w for w in re.findall(r"[a-z]+", (label or "").lower()) if w not in STOP]
  if not lw or not ops:
    return 0.0
  ow = Counter(ops)
  num = sum(ow[w] * stats.idf(w) ** 2 for w in set(lw) if w in ow)
  den = math.sqrt(sum((c * stats.idf(w)) ** 2 for w, c in ow.items())) * math.sqrt(sum(stats.idf(w) ** 2 for w in set(lw)))
  return num / den if den else 0.0
