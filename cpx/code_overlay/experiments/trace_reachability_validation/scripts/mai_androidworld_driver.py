"""Paired AndroidWorld driver for the official MAI client, with isolated outputs.

Runs on the device host/container, not on the model server. Does not create an
emulator or a background load model. Original GELab e2e_driver.py is unchanged.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from android_world import shared_state as SS


def worker_tasks(tasks, workers, worker_id):
  return [(i, task) for i, task in enumerate(tasks) if i % workers == worker_id]


def preflight(api, model, config):
  for relative, expected in config.get("statistics", {}).get("files_sha256", {}).items():
    path = (REPO / relative).resolve()
    if not path.is_relative_to(REPO) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
      raise RuntimeError("Statistics asset missing or hash mismatch: " + relative)
  opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
  url = api.split("/v1/")[0] + "/v1/models"
  models = json.load(opener.open(url, timeout=10))["data"]
  match = next((m for m in models if m["id"] == model), None)
  if not match or match.get("max_model_len", 0) < config["max_model_len"]:
    raise RuntimeError("Model missing, or context smaller than the explicit official-client config")
  subprocess.run(["adb", "-s", "emulator-5554", "get-state"], check=True, capture_output=True)
  subprocess.run(["adb", "-s", "emulator-5554", "forward", "tcp:8765", "localabstract:androidworld_fast_a11y"], check=True, capture_output=True)
  def read_exact(s, n):
    data = b""
    while len(data) < n:
      chunk = s.recv(n - len(data))
      if not chunk:
        raise RuntimeError("fast-a11y closed the connection; install/enable the dumper before running")
      data += chunk
    return data
  with socket.create_connection(("127.0.0.1", 8765), timeout=5) as s:
    s.settimeout(5)
    s.sendall(b"flat 0 10000\n")
    size = struct.unpack(">I", read_exact(s, 4))[0]
    if size > 32 * 1024 * 1024:
      raise RuntimeError("Invalid a11y frame length")
    tree = json.loads(read_exact(s, size))
    if not tree.get("ok") or len(tree.get("nodes", [])) < 3:
      raise RuntimeError("fast-a11y returned an empty/unready UI tree")


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--tag", required=True)
  p.add_argument("--arms", default="BASE,FULL45")
  p.add_argument("--tasks", help="Comma-separated tasks; default is all 116 registry tasks")
  p.add_argument("--task-order", type=Path, help="Original e2e116n_task_order.txt for matching original order")
  p.add_argument("--seed", type=int, default=30)
  p.add_argument("--api-url")
  p.add_argument("--model")
  p.add_argument("--config", type=Path, default=ROOT / "configs/mai_ui_8b_official.json")
  p.add_argument("--preflight-only", action="store_true")
  p.add_argument("--smoke", action="store_true", help="8 model calls per task, neutral IDF and empty offline priors; requires explicit tasks")
  p.add_argument("--workers", type=int, default=1, help="Workers per arm; FULL45 workers share run/graph/FULL45")
  p.add_argument("--worker-id", type=int, default=0, help="Worker index, from 0 to workers-1")
  args = p.parse_args()
  if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.tag):
    p.error("Use a new tag containing only letters, digits, - or _")
  arms = args.arms.split(",")
  if len(set(arms)) != len(arms) or any(a not in {"BASE", "FULL45"} for a in arms):
    p.error("--arms supports BASE,FULL45 only")
  if args.tasks and args.task_order:
    p.error("Choose --tasks or --task-order")
  if args.smoke and not args.tasks:
    p.error("--smoke requires an explicit --tasks list")
  if args.workers < 1 or not 0 <= args.worker_id < args.workers:
    p.error("Require --workers >= 1 and 0 <= --worker-id < --workers")
  if args.workers > 1 and len(arms) != 1:
    p.error("Parallel workers run exactly one --arms BASE or FULL45 per container")
  config = json.loads(args.config.read_text())
  api, model = args.api_url or config["api_url"], args.model or config["model"]
  preflight(api, model, config)
  if args.preflight_only:
    print("PASS: model/context, emulator, and fast-a11y readiness")
    return
  from android_world import registry
  tasks_registry = registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)
  tasks = (args.tasks.split(",") if args.tasks else args.task_order.read_text().strip().split(",")
           if args.task_order else sorted(tasks_registry))
  if len(set(tasks)) != len(tasks) or any(t not in tasks_registry for t in tasks):
    p.error("Unknown or duplicate tasks")
  run = ROOT / "raw" / "mai_official" / args.tag
  run.mkdir(parents=True, exist_ok=True)
  settings = {"arms": ["BASE", "FULL45"] if args.workers > 1 else arms,
              "workers_per_arm": args.workers, "shared_full45_graph": args.workers > 1,
              "tasks": tasks, "task_seed": args.seed, "api_url": api,
              "model": model, "mai_config": config, "background_load": False, "early_action": False,
              "graph_initialization": "empty_on_new_tag_persist_within_run",
              "historical_page_records": False,
              "smoke": args.smoke, "statistics_mode": "neutral_no_offline_priors" if args.smoke else config.get("statistics", {}).get("mode", "original"),
              "source_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                                 for name in ("mai_run.py", "mai_protocol.py", "mai_client.py", "full_run.py", "full7.py",
                                              "mai_androidworld_driver.py", "full9.py", "graph_exec.py", "intent_graph.py", "seq_run.py", "explore_value.py")},
              "shared_source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (REPO / "android_world/shared_state.py", REPO / "experiments/runtime_graph_study/scripts/rgs_run.py",
                                                 REPO / "experiments/runtime_graph_exploration/scripts/fgraph.py")}}
  manifest = run / "run_config.json"
  def establish(current):
    if current is not None and current != settings:
      raise RuntimeError("Existing tag uses different settings; choose a new tag")
    if current is None and (run / "graph").exists() and any((run / "graph").rglob("*.json*")):
      raise RuntimeError("New tag already contains graph data without a manifest")
    return settings
  SS.update_json(manifest, establish, None)
  with ExitStack() as stack:
    for arm in arms:
      stack.enter_context(SS.lock(run / "workers" / f"{arm}-{args.worker_id}.claim", blocking=False))
    run_tasks(args, arms, tasks, run, api, model, config)


def run_tasks(args, arms, tasks, run, api, model, config):
  for i, task in worker_tasks(tasks, args.workers, args.worker_id):
    for arm in arms if i % 2 == 0 else arms[::-1]:
      out = run / arm / task / "p0"
      result_path = out / "result.json"
      if result_path.exists():
        if json.loads(result_path.read_text()).get("error"):
          raise RuntimeError("Prior error retained; inspect it before choosing a new run tag")
        if args.workers > 1 and not (out / ".shared_complete").exists():
          raise RuntimeError("Prior result has no worker completion marker; inspect the interrupted episode")
        continue
      if out.exists():
        raise RuntimeError(f"Partial episode retained at {out}; do not silently rerun with a warm graph")
      preflight(api, model, config)
      env = dict(os.environ, FULL_ARM=arm, SEQ_APPDIR=str(run / "graph"),
                 MAI_CONFIG=str(args.config.resolve()), ONDEVICE_LOAD="0")
      env["MAI_SMOKE_EMPTY_STATS"] = "1" if args.smoke else "0"
      env["MAI_SHARED_STATE"] = "1" if args.workers > 1 and arm == "FULL45" else "0"
      env["MAI_WORKER_ID"] = str(args.worker_id)
      cmd = [sys.executable, str(ROOT / "scripts/mai_run.py"), "--task", task, "--seed", str(args.seed),
             "--variant", f"{args.tag}_{arm}", "--mode", "agent", "--out", str(out),
             "--api_url", api, "--model", model]
      if args.smoke:
        cmd += ["--max_steps", "8"]
      t0 = time.time()
      with (run / f"{arm}_{task}.log").open("w") as log:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=2400).returncode
      result = json.loads(result_path.read_text()) if result_path.exists() else {"error": "no result"}
      attempts = out / "mai_attempts.jsonl"
      row = {"task": task, "arm": arm, "rc": rc, "e2e_s": round(time.time() - t0, 2),
             "success": result.get("is_successful"), "model_calls": result.get("model_calls"),
             "http_attempts": len(attempts.read_text().splitlines()) if attempts.exists() else 0,
             "error": result.get("error")}
      row["worker_id"] = args.worker_id
      SS.append_json(run / "runs.jsonl", row)
      print(json.dumps(row), flush=True)
      if rc != 0 or row["error"]:
        raise RuntimeError("Episode failed with an execution/environment error; run stopped, logs retained")
      SS.atomic_text(out / ".shared_complete", json.dumps(row))


if __name__ == "__main__":
  main()
