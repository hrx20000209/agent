"""Replay the no-op guard over the archived Qwen3 a11y episodes.

Code semantics (full_run.py): at the start of each counted step k the previous
model action is compared with the screen captured now; `prev.step == k-1` is
required, so only consecutive steps count.  Reaching LIMIT consecutive
identical actions with an unchanged screen ends the episode as a failure.

Here the "screen unchanged" test is byte equality of the decompressed a11y
snapshot before the action and the snapshot captured before the next step.
Model calls before the trigger include requery calls (one extra call each).
"""
import gzip, hashlib, json, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 5


class NoopGuard:
    def __init__(self, limit):
        self.limit, self.sig, self.n, self.fired = limit, None, 0, False

    def observe(self, sig, unchanged):
        if not unchanged:
            self.sig, self.n, self.fired = None, 0, False
        elif sig == self.sig:
            self.n += 1
        else:
            self.sig, self.n, self.fired = sig, 1, False
        if self.n >= self.limit and not self.fired:
            self.fired = True
            return True
        return False


def calls_of(rows):
    return sum(1 + (1 if r.get("requery") else 0) for r in rows)


def replay(task_dir):
    res_p = task_dir / "result.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {}
    steps_p = task_dir / "steps.jsonl"
    if not steps_p.exists():
        return {"trigger": False, "calls": res.get("model_calls"), "success": res.get("is_successful")}
    rows = sorted((json.loads(l) for l in steps_p.read_text().splitlines() if l.strip()), key=lambda r: r["step"])
    model = [r for r in rows if r.get("kind") == "model"]
    cache = {}

    def h(cid):
        if cid is None:
            return None
        if cid not in cache:
            f = task_dir / "a11y" / f"{cid}.json.gz"
            cache[cid] = hashlib.sha256(gzip.open(f, "rb").read()).hexdigest() if f.exists() else None
        return cache[cid]

    guard = NoopGuard(LIMIT)
    for i, r in enumerate(model):
        if i == 0:
            continue
        prev = model[i - 1]
        if r["step"] != prev["step"] + 1:          # a graph/bootstrap step intervened
            guard.observe(None, False)
            continue
        pre = (prev.get("pre") or {}).get("cid")
        after = (r.get("pre") or {}).get("cid")
        unchanged = pre is not None and h(pre) is not None and h(pre) == h(after)
        a = prev.get("action") or {}
        sig = (a.get("action_type"), a.get("x"), a.get("y"), a.get("index"), a.get("text"))
        if guard.observe(sig, unchanged):
            return {"trigger": True, "step": r["step"], "calls": calls_of(model[:i]),
                    "orig_calls": res.get("model_calls"), "success": 0.0,
                    "orig_success": res.get("is_successful")}
    return {"trigger": False, "calls": res.get("model_calls"), "success": res.get("is_successful")}


def main():
    out = {}
    for arm in ("BASE", "FULL45"):
        tasks = {}
        for td in sorted(p for p in (ROOT / arm).glob("*/p0") if p.is_dir()):
            tasks[td.parent.name] = replay(td)
        out[arm] = tasks
    (ROOT / f"noop_replay_limit{LIMIT}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for arm in ("BASE", "FULL45"):
        trig = [k for k, v in out[arm].items() if v.get("trigger")]
        print(arm, "tasks", len(out[arm]), "triggered", len(trig))
        for k in trig:
            if out[arm][k].get("orig_success") == 1:
                print("   SUCCESS->FAIL", k, out[arm][k])


if __name__ == "__main__":
    main()
