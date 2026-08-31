"""Evaluate progressive-graph memory growth under real ADB exploration.

Two questions are measured together:
1. How graph node/edge growth changes Python cache/RSS and device memory.
2. How many verified exploration probes fit in a fixed wall-clock window under
   different enforced graph cache budgets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from PIL import Image

from Explorer.GoalExplorer import A11yTreeOnlineExplorer
from Explorer.progressive_belief_graph import ProgressiveBeliefGraph, describe_ui_state
from Explorer.state_graph_information import MatrixAblations, parse_reasoning_prior
from Explorer.utils import collect_clickable_nodes, ensure_dir
from MobileAgentE.controller import get_a11y_tree, get_screenshot, home
from MobileAgentE.tree import parse_a11y_tree
from main import _default_adb_path, ensure_adb_ready, resolve_adb_prefix


def _run_adb(adb_prefix: str, *args: str, timeout: float = 8.0) -> str:
    proc = subprocess.run(
        [*shlex.split(adb_prefix), *args], capture_output=True, text=True, timeout=timeout
    )
    return (proc.stdout or "") if proc.returncode == 0 else ""


def _host_rss_kb() -> int:
    proc = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    )
    try:
        return int((proc.stdout or "0").strip())
    except ValueError:
        return 0


def _deep_size(value: Any, seen=None) -> int:
    if seen is None:
        seen = set()
    oid = id(value)
    if oid in seen:
        return 0
    seen.add(oid)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_deep_size(k, seen) + _deep_size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_deep_size(item, seen) for item in value)
    elif hasattr(value, "__dict__"):
        size += _deep_size(vars(value), seen)
    return size


def _device_memory(adb_prefix: str) -> Dict[str, Any]:
    text = _run_adb(adb_prefix, "shell", "cat", "/proc/meminfo")
    values = {k: int(v) for k, v in re.findall(r"^(MemTotal|MemFree|MemAvailable|Cached|SwapFree):\s+(\d+)", text, re.M)}
    window = _run_adb(adb_prefix, "shell", "dumpsys", "window", "windows")
    activity = _run_adb(adb_prefix, "shell", "dumpsys", "activity", "activities")
    match = re.search(r"mCurrentFocus=.*?\s([A-Za-z0-9_.]+)/", window)
    if not match:
        match = re.search(
            r"(?:topResumedActivity=|ResumedActivity:|mFocusedApp=).*?\su\d+\s+([A-Za-z0-9_.]+)/",
            activity,
        )
    package = match.group(1) if match else ""
    app_pss_kb = None
    if package:
        meminfo = _run_adb(adb_prefix, "shell", "dumpsys", "meminfo", package)
        patterns = [r"TOTAL PSS:\s*(\d+)", r"^\s*TOTAL\s+(\d+)"]
        for pattern in patterns:
            found = re.search(pattern, meminfo, re.M)
            if found:
                app_pss_kb = int(found.group(1))
                break
    values.update({"foreground_package": package, "foreground_app_pss_kb": app_pss_kb})
    return values


def _device_identity(adb_prefix: str) -> Dict[str, str]:
    serial = _run_adb(adb_prefix, "get-serialno").strip()
    model = _run_adb(adb_prefix, "shell", "getprop", "ro.product.model").strip()
    qemu = _run_adb(adb_prefix, "shell", "getprop", "ro.kernel.qemu").strip()
    hardware = _run_adb(adb_prefix, "shell", "getprop", "ro.hardware").strip()
    return {"serial": serial, "model": model, "qemu": qemu, "hardware": hardware}


def _require_physical_device(adb_prefix: str, allow_emulator: bool = False) -> Dict[str, str]:
    identity = _device_identity(adb_prefix)
    serial = identity.get("serial", "").lower()
    hardware = identity.get("hardware", "").lower()
    is_emulator = bool(
        identity.get("qemu") == "1"
        or serial.startswith("emulator-")
        or "goldfish" in hardware
        or "ranchu" in hardware
    )
    if is_emulator and not allow_emulator:
        raise RuntimeError(
            "Refusing graph-memory evaluation on an emulator. "
            f"Detected serial={identity.get('serial')} model={identity.get('model')} "
            f"hardware={identity.get('hardware')}. Connect a physical Android device "
            "or pass --allow_emulator only for an explicitly non-final smoke test."
        )
    return identity


def _slope(samples: List[Dict[str, Any]], x_key: str, y_key: str) -> Optional[float]:
    pairs = [(float(s[x_key]), float(s[y_key])) for s in samples if s.get(x_key) is not None and s.get(y_key) is not None]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom <= 0:
        return None
    return sum((x - xbar) * (y - ybar) for x, y in pairs) / denom


def run_budget(args, budget_mb: float, run_index: int, output_jsonl: str) -> Dict[str, Any]:
    graph = ProgressiveBeliefGraph()
    budget_bytes = 0 if budget_mb <= 0 else int(budget_mb * 1024 * 1024)
    run_dir = os.path.join(args.output_dir, f"budget_{budget_mb:g}mb")
    ensure_dir(run_dir)
    home(args.adb_path)
    time.sleep(0.35)

    screenshot_path = os.path.join(run_dir, "root.jpg")
    xml_path = os.path.join(run_dir, "root.xml")
    get_screenshot(args, screenshot_path, scale=args.scale)
    get_a11y_tree(args, xml_path)
    width, height = Image.open(screenshot_path).size
    root = parse_a11y_tree(xml_path=xml_path)
    candidates = collect_clickable_nodes(root)
    state = describe_ui_state(root, len(candidates))
    snapshot = graph.snapshot()
    snapshot_node = snapshot.match_state(state)
    live_node = graph.observe_state(state)

    explorer = A11yTreeOnlineExplorer(
        args=args,
        adb_path=args.adb_path,
        xml_path=xml_path,
        explore_vis_dir=os.path.join(run_dir, "explore"),
        embed_model_name="hashing",
        ui_lock=threading.Lock(),
        stop_event=threading.Event(),
        rollback_done_event=threading.Event(),
        width=width,
        height=height,
        explorer_mode="collect_demo",
    )
    args._current_screenshot_path = screenshot_path
    args._current_xml_path = xml_path
    args._current_width = width
    args._current_height = height
    explorer.prepare_graph_iteration(
        graph, snapshot, snapshot_node, live_node, state,
        parse_reasoning_prior("", args.task), 1, f"memory_eval_{run_index}",
        exploration_policy="graph_matrix", matrix_ablations=MatrixAblations(), recent_nodes=[],
    )

    device_start = _device_memory(args.adb_path)
    rss_start = _host_rss_kb()
    start = time.time()
    explorer.start(
        max_steps=args.max_steps,
        max_depth=args.max_depth,
        leaf_width=args.leaf_width,
        time_budget_sec=args.duration_sec,
        trigger_reason="memory_evaluation",
    )
    samples: List[Dict[str, Any]] = []
    last_probe_count = -1
    while time.time() - start < args.duration_sec and explorer.thread and explorer.thread.is_alive():
        probe_count = int(explorer.graph_probe_ingest_count)
        budget_needs_prune = bool(
            budget_bytes > 0
            and not explorer._pending_probe_edge_ids
            and graph.approximate_serialized_bytes() > budget_bytes
        )
        if probe_count != last_probe_count or budget_needs_prune:
            prune = (
                graph.prune_to_budget(budget_bytes, protected_node_ids=[live_node])
                if not explorer._pending_probe_edge_ids
                else {"pruned_edges": 0, "pruned_nodes": 0, "bytes": graph.approximate_serialized_bytes()}
            )
            sample = {
                "record_type": "sample",
                "run_index": run_index,
                "budget_mb": budget_mb,
                "budget_bytes": budget_bytes,
                "elapsed_sec": time.time() - start,
                "probe_count": probe_count,
                "explorer_action_count": explorer.cur_steps,
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "graph_python_bytes": _deep_size({"nodes": graph.nodes, "edges": graph.edges}),
                "graph_serialized_bytes": graph.approximate_serialized_bytes(),
                "host_rss_kb": _host_rss_kb(),
                **prune,
            }
            samples.append(sample)
            with open(output_jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            last_probe_count = probe_count
        time.sleep(0.05)

    exploration_window_elapsed = time.time() - start
    explorer.stop(min_steps=0, min_runtime_sec=0.0, max_wait_sec=2.0)
    elapsed = time.time() - start
    final_prune = graph.prune_to_budget(budget_bytes, protected_node_ids=[live_node])
    device_end = _device_memory(args.adb_path)
    rss_end = _host_rss_kb()
    total_probes = int(explorer.graph_probe_ingest_count)
    rollback_successes = sum(edge.rollback_success_count for edge in graph.edges.values())
    rollback_failures = sum(edge.rollback_failure_count for edge in graph.edges.values())
    summary = {
        "record_type": "budget_summary",
        "run_index": run_index,
        "budget_mb": budget_mb,
        "budget_bytes": budget_bytes,
        "target_duration_sec": args.duration_sec,
        "exploration_window_elapsed_sec": exploration_window_elapsed,
        "actual_duration_sec": elapsed,
        "exploration_probe_count": total_probes,
        "explorer_action_count": explorer.cur_steps,
        "probes_per_second": total_probes / max(1e-9, exploration_window_elapsed),
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "rollback_success_count": rollback_successes,
        "rollback_failure_count": rollback_failures,
        "graph_python_bytes": _deep_size({"nodes": graph.nodes, "edges": graph.edges}),
        "graph_serialized_bytes": graph.approximate_serialized_bytes(),
        "host_rss_start_kb": rss_start,
        "host_rss_end_kb": rss_end,
        "host_rss_delta_kb": rss_end - rss_start,
        "device_mem_start": device_start,
        "device_mem_end": device_end,
        "device_mem_available_delta_kb": (
            device_end.get("MemAvailable", 0) - device_start.get("MemAvailable", 0)
        ),
        "bytes_per_node_slope": _slope(samples, "node_count", "graph_python_bytes"),
        "rss_kb_per_node_slope": _slope(samples, "node_count", "host_rss_kb"),
        **final_prune,
    }
    with open(output_jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Explore non-destructive launcher controls")
    parser.add_argument("--duration_sec", type=float, default=8.0)
    parser.add_argument("--budgets_mb", default="0.01,0.03,0", help="Comma-separated; 0 means unlimited.")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--leaf_width", type=int, default=2)
    parser.add_argument("--output_dir", default="./evaluation_results/graph_memory_adb")
    parser.add_argument("--adb_path", default=_default_adb_path())
    parser.add_argument("--adb_serial", default="")
    parser.add_argument("--adb_port", type=int, default=5037)
    parser.add_argument("--adb_cmd_timeout", type=float, default=8.0)
    parser.add_argument("--adb_retries", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--on_device", action="store_true")
    parser.add_argument(
        "--allow_emulator",
        action="store_true",
        help="Permit emulator smoke tests. Final evaluation rejects emulators by default.",
    )
    args = parser.parse_args()
    args.adb_path = resolve_adb_prefix(args.adb_path, args.adb_serial, args.adb_port)
    ensure_adb_ready(args.adb_path)
    device_identity = _require_physical_device(args.adb_path, allow_emulator=args.allow_emulator)
    ensure_dir(args.output_dir)
    output_jsonl = os.path.join(args.output_dir, "samples.jsonl")
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)
    budgets = [float(x.strip()) for x in args.budgets_mb.split(",") if x.strip()]
    summaries = [run_budget(args, budget, i + 1, output_jsonl) for i, budget in enumerate(budgets)]
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "device": device_identity, "results": summaries}, fh, ensure_ascii=False, indent=2)
    csv_path = os.path.join(args.output_dir, "summary.csv")
    flat_keys = [
        "budget_mb", "target_duration_sec", "actual_duration_sec", "exploration_probe_count",
        "probes_per_second", "node_count", "edge_count", "graph_python_bytes",
        "graph_serialized_bytes", "host_rss_delta_kb", "device_mem_available_delta_kb",
        "rollback_success_count", "rollback_failure_count",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat_keys)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({k: summary.get(k) for k in flat_keys})
    print(json.dumps({"summary_json": summary_path, "summary_csv": csv_path, "results": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
