#!/usr/bin/env python3
"""Profile real-phone ADB exploration without model inference or graph preload.

Each repeat starts with a fresh, empty ProgressiveBeliefGraph.  The graph is
only used to ingest verified exploration probes; it is not preloaded and no
LLM request is issued.  Emulators are always rejected.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import multiprocessing
import os
import queue as queue_module
import re
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

from Explorer.GoalExplorer import A11yTreeOnlineExplorer
from Explorer.progressive_belief_graph import ProgressiveBeliefGraph, describe_ui_state
from Explorer.state_graph_information import MatrixAblations, parse_reasoning_prior
from Explorer.utils import collect_clickable_nodes, ensure_dir
from MobileAgentE.controller import get_a11y_tree, get_screenshot, home
from MobileAgentE.tree import parse_a11y_tree
from evaluate_graph_memory_adb import (
    _deep_size,
    _device_memory,
    _host_memory,
    _host_rss_kb,
    _require_physical_device,
    _run_adb,
)
from main import _default_adb_path, ensure_adb_ready, resolve_adb_prefix


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latest(values: List[float]) -> Optional[float]:
    return float(values[-1]) if values else None


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def require_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def foreground_package(adb_path: str, timeout: float) -> str:
    window = _run_adb(adb_path, "shell", "dumpsys", "window", "windows", timeout=timeout)
    match = re.search(r"mCurrentFocus=.*?\s([A-Za-z0-9_.]+)/", window)
    if match:
        return match.group(1)
    activity = _run_adb(adb_path, "shell", "dumpsys", "activity", "activities", timeout=timeout)
    match = re.search(
        r"(?:topResumedActivity=|ResumedActivity:|mFocusedApp=).*?\su\d+\s+([A-Za-z0-9_.]+)/",
        activity,
    )
    return match.group(1) if match else ""


def start_target(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    if args.package:
        if args.force_stop_between_repeats:
            _run_adb(args.adb_path, "shell", "am", "force-stop", args.package)
        launch_command_started = time.perf_counter()
        _run_adb(
            args.adb_path,
            "shell",
            "monkey",
            "-p",
            args.package,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
            timeout=max(8.0, args.adb_cmd_timeout),
        )
        launch_command_sec = time.perf_counter() - launch_command_started
    else:
        launch_command_started = time.perf_counter()
        home(args.adb_path)
        launch_command_sec = time.perf_counter() - launch_command_started

    reached = not bool(args.package)
    observed_package = ""
    time_to_foreground_sec: Optional[float] = None
    if args.package:
        deadline = time.perf_counter() + max(0.1, float(args.launch_timeout_sec))
        while time.perf_counter() < deadline:
            observed_package = foreground_package(args.adb_path, args.adb_cmd_timeout)
            if observed_package == args.package:
                reached = True
                time_to_foreground_sec = time.perf_counter() - started
                break
            time.sleep(max(0.01, float(args.launch_poll_sec)))
    if args.launch_wait_sec > 0:
        time.sleep(float(args.launch_wait_sec))
    return {
        "launch_started_perf_counter": started,
        "launch_command_sec": launch_command_sec,
        "foreground_reached": reached,
        "foreground_package_after_launch": observed_package,
        "time_to_foreground_sec": time_to_foreground_sec,
    }


def capture_initial_state(
    args: argparse.Namespace,
    run_dir: Path,
    launch_started: float,
) -> Dict[str, Any]:
    screenshot_path = run_dir / "root.png"
    xml_path = run_dir / "root.xml"
    screenshot_started = time.perf_counter()
    get_screenshot(args, str(screenshot_path), scale=args.scale)
    initial_screenshot_sec = time.perf_counter() - screenshot_started
    time_to_first_frame_sec = time.perf_counter() - launch_started
    width, height = Image.open(screenshot_path).size

    required_stable = max(1, int(args.ui_stability_checks))
    stable_count = 0
    last_signature = None
    first_a11y_sec: Optional[float] = None
    tree_capture_count = 0
    root = None
    candidates = []
    state = None
    deadline = time.perf_counter() + max(0.1, float(args.ui_stability_timeout_sec))
    while time.perf_counter() < deadline:
        get_a11y_tree(args, str(xml_path))
        tree_capture_count += 1
        root = parse_a11y_tree(xml_path=str(xml_path))
        candidates = collect_clickable_nodes(root)
        state = describe_ui_state(root, len(candidates))
        if first_a11y_sec is None:
            first_a11y_sec = time.perf_counter() - launch_started
        if state.signature == last_signature:
            stable_count += 1
        else:
            last_signature = state.signature
            stable_count = 1
        if stable_count >= required_stable:
            break
        time.sleep(max(0.01, float(args.ui_stability_poll_sec)))
    if root is None or state is None:
        raise RuntimeError("Could not capture an accessibility tree after launching the target")
    return {
        "screenshot_path": screenshot_path,
        "xml_path": xml_path,
        "width": width,
        "height": height,
        "root": root,
        "candidates": candidates,
        "state": state,
        "initial_screenshot_sec": initial_screenshot_sec,
        "time_to_first_frame_sec": time_to_first_frame_sec,
        "time_to_first_a11y_sec": first_a11y_sec,
        "time_to_stable_ui_sec": time.perf_counter() - launch_started,
        "ui_stable": stable_count >= required_stable,
        "ui_tree_capture_count": tree_capture_count,
    }


def run_once(
    args: argparse.Namespace,
    run_index: int,
    samples_path: Path,
) -> Dict[str, Any]:
    gc.collect()
    run_dir = Path(args.output_dir) / f"run_{run_index:03d}"
    ensure_dir(str(run_dir))
    launch = start_target(args)
    capture = capture_initial_state(
        args, run_dir, float(launch["launch_started_perf_counter"])
    )
    screenshot_path = capture["screenshot_path"]
    xml_path = capture["xml_path"]
    width, height = capture["width"], capture["height"]
    candidates = capture["candidates"]
    state = capture["state"]

    graph = ProgressiveBeliefGraph()
    live_node_id = graph.observe_state(state)
    snapshot_started = time.perf_counter()
    snapshot = graph.snapshot()
    snapshot_build_sec = time.perf_counter() - snapshot_started
    snapshot_python_bytes = _deep_size({"nodes": snapshot.nodes, "edges": snapshot.edges})
    snapshot_node_id = snapshot.match_state(state)

    explorer = A11yTreeOnlineExplorer(
        args=args,
        adb_path=args.adb_path,
        xml_path=str(xml_path),
        explore_vis_dir=str(run_dir / "explore"),
        embed_model_name="hashing",
        ui_lock=threading.Lock(),
        stop_event=threading.Event(),
        rollback_done_event=threading.Event(),
        width=width,
        height=height,
        explorer_mode="collect_demo",
    )
    args._current_screenshot_path = str(screenshot_path)
    args._current_xml_path = str(xml_path)
    args._current_width = width
    args._current_height = height
    explorer.prepare_graph_iteration(
        graph,
        snapshot,
        snapshot_node_id,
        live_node_id,
        state,
        parse_reasoning_prior("", args.task),
        1,
        f"exploration_profile_{run_index}",
        exploration_policy="graph_matrix",
        matrix_ablations=MatrixAblations(),
        recent_nodes=[],
    )

    device_start = _device_memory(args.adb_path)
    host_start = _host_memory()
    rss_start_kb = _host_rss_kb()
    started = time.time()
    last_probe_count = -1
    last_probe_elapsed = 0.0
    samples: List[Dict[str, Any]] = []

    explorer.start(
        max_steps=args.max_steps,
        max_depth=args.max_depth,
        leaf_width=args.leaf_width,
        time_budget_sec=args.duration_sec,
        trigger_reason="standalone_exploration_profile",
    )
    while time.time() - started < args.duration_sec and explorer.thread and explorer.thread.is_alive():
        probe_count = int(explorer.graph_probe_ingest_count)
        if probe_count != last_probe_count:
            elapsed = time.time() - started
            row = {
                "record_type": "sample",
                "run_index": run_index,
                "timestamp": time.time(),
                "elapsed_sec": elapsed,
                "probe_interval_sec": (
                    elapsed - last_probe_elapsed
                    if probe_count > last_probe_count and probe_count > 0
                    else None
                ),
                "verified_probe_count": probe_count,
                "explorer_action_count": int(explorer.cur_steps),
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "graph_python_bytes": graph.approximate_python_bytes(),
                "graph_serialized_bytes": graph.approximate_serialized_bytes(),
                "host_rss_kb": _host_rss_kb(),
                "probe_total_sec": latest(explorer.total_latency),
                "selection_sec": latest(explorer.selection_latency),
                "action_sec": latest(explorer.action_latency),
                "screenshot_sec": latest(explorer.screenshot_latency),
                "a11y_tree_sec": latest(explorer.adb_tree_latency),
                "parse_tree_sec": latest(explorer.tree_latency),
            }
            samples.append(row)
            append_jsonl(samples_path, row)
            if probe_count > last_probe_count and probe_count > 0:
                last_probe_elapsed = elapsed
            last_probe_count = probe_count
        time.sleep(0.05)

    window_elapsed_sec = time.time() - started
    explorer.stop(min_steps=0, min_runtime_sec=0.0, max_wait_sec=2.0)
    total_elapsed_sec = time.time() - started
    device_end = _device_memory(args.adb_path)
    host_end = _host_memory()
    rss_end_kb = _host_rss_kb()
    graph.save(str(run_dir / "belief_graph.json"))

    total_latencies = list(explorer.total_latency)
    summary = {
        "record_type": "run_summary",
        "run_index": run_index,
        "exploration_started_timestamp": started,
        "exploration_ended_timestamp": time.time(),
        "launch_command_sec": launch["launch_command_sec"],
        "foreground_reached": launch["foreground_reached"],
        "foreground_package_after_launch": launch["foreground_package_after_launch"],
        "time_to_foreground_sec": launch["time_to_foreground_sec"],
        "initial_screenshot_sec": capture["initial_screenshot_sec"],
        "time_to_first_frame_sec": capture["time_to_first_frame_sec"],
        "time_to_first_a11y_sec": capture["time_to_first_a11y_sec"],
        "time_to_stable_ui_sec": capture["time_to_stable_ui_sec"],
        "ui_stable": capture["ui_stable"],
        "ui_tree_capture_count": capture["ui_tree_capture_count"],
        "target_duration_sec": args.duration_sec,
        "exploration_window_elapsed_sec": window_elapsed_sec,
        "total_elapsed_sec": total_elapsed_sec,
        "verified_probe_count": int(explorer.graph_probe_ingest_count),
        "explorer_action_count": int(explorer.cur_steps),
        "verified_probes_per_sec": int(explorer.graph_probe_ingest_count) / max(1e-9, window_elapsed_sec),
        "actions_per_sec": int(explorer.cur_steps) / max(1e-9, window_elapsed_sec),
        "probe_latency_mean_sec": statistics.mean(total_latencies) if total_latencies else None,
        "probe_latency_p50_sec": percentile(total_latencies, 0.50),
        "probe_latency_p95_sec": percentile(total_latencies, 0.95),
        "steady_probe_latency_mean_sec": statistics.mean(total_latencies[1:]) if len(total_latencies) > 1 else None,
        "selection_latency_mean_sec": statistics.mean(explorer.selection_latency) if explorer.selection_latency else None,
        "action_latency_mean_sec": statistics.mean(explorer.action_latency) if explorer.action_latency else None,
        "screenshot_latency_mean_sec": statistics.mean(explorer.screenshot_latency) if explorer.screenshot_latency else None,
        "a11y_tree_latency_mean_sec": statistics.mean(explorer.adb_tree_latency) if explorer.adb_tree_latency else None,
        "parse_tree_latency_mean_sec": statistics.mean(explorer.tree_latency) if explorer.tree_latency else None,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "graph_python_bytes": graph.approximate_python_bytes(),
        "graph_serialized_bytes": graph.approximate_serialized_bytes(),
        "initial_snapshot_python_bytes": snapshot_python_bytes,
        "initial_snapshot_build_sec": snapshot_build_sec,
        "host_rss_start_kb": rss_start_kb,
        "host_rss_end_kb": rss_end_kb,
        "host_rss_delta_kb": rss_end_kb - rss_start_kb,
        "host_memory_start": host_start,
        "host_memory_end": host_end,
        "device_memory_start": device_start,
        "device_memory_end": device_end,
        "device_mem_available_delta_kb": (
            device_end.get("MemAvailable", 0) - device_start.get("MemAvailable", 0)
        ),
        "foreground_app_pss_delta_kb": (
            (device_end.get("foreground_app_pss_kb") or 0)
            - (device_start.get("foreground_app_pss_kb") or 0)
        ),
    }
    append_jsonl(samples_path, summary)
    return summary


def worker(args_dict: Dict[str, Any], run_index: int, samples_path: str, result_queue) -> None:
    try:
        result = run_once(argparse.Namespace(**args_dict), run_index, Path(samples_path))
        result_queue.put({"ok": True, "result": result})
    except Exception as exc:
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def aggregate(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [
        "launch_command_sec", "time_to_foreground_sec", "initial_screenshot_sec",
        "time_to_first_frame_sec", "time_to_first_a11y_sec", "time_to_stable_ui_sec",
        "verified_probe_count", "explorer_action_count", "verified_probes_per_sec",
        "actions_per_sec", "probe_latency_mean_sec", "steady_probe_latency_mean_sec",
        "selection_latency_mean_sec", "action_latency_mean_sec", "screenshot_latency_mean_sec",
        "a11y_tree_latency_mean_sec", "host_rss_delta_kb", "device_mem_available_delta_kb",
        "foreground_app_pss_delta_kb", "graph_python_bytes",
    ]
    output: Dict[str, Any] = {"repeat_count": len(summaries)}
    for metric in metrics:
        values = [float(row[metric]) for row in summaries if row.get(metric) is not None]
        output[f"{metric}_mean"] = statistics.mean(values) if values else None
        output[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else None
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb_serial", required=True, help="Explicit physical-device serial; emulator serials are rejected.")
    parser.add_argument("--adb_path", default=_default_adb_path())
    parser.add_argument("--adb_port", type=int, default=5037)
    parser.add_argument("--adb_cmd_timeout", type=float, default=12.0)
    parser.add_argument("--adb_retries", type=int, default=3)
    parser.add_argument("--package", default="", help="Optional app package; empty profiles launcher exploration.")
    parser.add_argument("--task", default="Explore safe, non-destructive controls relevant to the current screen")
    parser.add_argument("--duration_sec", type=float, default=60.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--leaf_width", type=int, default=2)
    parser.add_argument(
        "--launch_wait_sec",
        type=float,
        default=0.0,
        help="Optional extra fixed delay after foreground detection; normally leave at 0.",
    )
    parser.add_argument("--launch_poll_sec", type=float, default=0.10)
    parser.add_argument("--launch_timeout_sec", type=float, default=15.0)
    parser.add_argument("--ui_stability_poll_sec", type=float, default=0.25)
    parser.add_argument("--ui_stability_checks", type=int, default=2)
    parser.add_argument("--ui_stability_timeout_sec", type=float, default=20.0)
    parser.add_argument("--force_stop_between_repeats", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--on_device", action="store_true")
    parser.add_argument("--no_isolate_runs", dest="isolate_runs", action="store_false")
    parser.set_defaults(isolate_runs=True)
    parser.add_argument("--output_dir", default="evaluation_results/profile_adb_exploration")
    args = parser.parse_args()

    if args.adb_serial.lower().startswith("emulator-"):
        raise RuntimeError(f"Refusing emulator serial: {args.adb_serial}")
    args.adb_path = resolve_adb_prefix(args.adb_path, args.adb_serial, args.adb_port)
    ensure_adb_ready(args.adb_path)
    device = _require_physical_device(args.adb_path, allow_emulator=False)
    output_dir = Path(args.output_dir).resolve()
    require_empty_output_dir(output_dir)
    args.output_dir = str(output_dir)
    samples_path = output_dir / "samples.jsonl"
    config = {**vars(args), "device": device}
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summaries: List[Dict[str, Any]] = []
    for run_index in range(1, max(1, int(args.repeats)) + 1):
        if args.isolate_runs:
            context = multiprocessing.get_context("spawn")
            result_queue = context.Queue()
            process = context.Process(
                target=worker,
                args=(vars(args), run_index, str(samples_path), result_queue),
                name=f"adb-exploration-profile-{run_index}",
            )
            process.start()
            try:
                message = result_queue.get(timeout=max(300.0, args.duration_sec + 180.0))
            except queue_module.Empty as exc:
                process.terminate()
                process.join(timeout=5.0)
                raise RuntimeError(f"Exploration repeat {run_index} timed out") from exc
            process.join(timeout=5.0)
            if not message.get("ok"):
                raise RuntimeError(message.get("error", "isolated exploration run failed"))
            summaries.append(message["result"])
        else:
            summaries.append(run_once(args, run_index, samples_path))

    flat_fields = [
        "run_index", "launch_command_sec", "foreground_reached",
        "foreground_package_after_launch", "time_to_foreground_sec",
        "initial_screenshot_sec", "time_to_first_frame_sec", "time_to_first_a11y_sec",
        "time_to_stable_ui_sec", "ui_stable", "ui_tree_capture_count",
        "target_duration_sec", "exploration_window_elapsed_sec",
        "total_elapsed_sec", "verified_probe_count", "explorer_action_count",
        "verified_probes_per_sec", "actions_per_sec", "probe_latency_mean_sec",
        "probe_latency_p50_sec", "probe_latency_p95_sec", "steady_probe_latency_mean_sec",
        "selection_latency_mean_sec", "action_latency_mean_sec", "screenshot_latency_mean_sec",
        "a11y_tree_latency_mean_sec", "parse_tree_latency_mean_sec", "node_count", "edge_count",
        "graph_python_bytes", "graph_serialized_bytes", "initial_snapshot_python_bytes",
        "initial_snapshot_build_sec", "host_rss_start_kb", "host_rss_end_kb",
        "host_rss_delta_kb", "device_mem_available_delta_kb", "foreground_app_pss_delta_kb",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in flat_fields} for row in summaries)

    report = {"config": config, "runs": summaries, "aggregate": aggregate(summaries)}
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
