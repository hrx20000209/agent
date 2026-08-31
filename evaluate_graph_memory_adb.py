"""Evaluate progressive-graph memory growth under real ADB exploration.

Two questions are measured together:
1. How graph node/edge growth changes Python cache/RSS and device memory.
2. How many verified exploration probes fit in a fixed wall-clock window under
   different enforced graph cache budgets.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import multiprocessing
import os
import queue as queue_module
import random
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from PIL import Image

from Explorer.GoalExplorer import A11yTreeOnlineExplorer
from Explorer.progressive_belief_graph import (
    ProgressiveBeliefGraph,
    UIStateDescriptor,
    describe_ui_state,
)
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


def _host_memory() -> Dict[str, Any]:
    try:
        import psutil  # type: ignore
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "total_kb": int(memory.total / 1024),
            "available_kb": int(memory.available / 1024),
            "used_percent": float(memory.percent),
            "swap_used_kb": int(swap.used / 1024),
        }
    except Exception:
        pass
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo", encoding="utf-8") as stream:
            text = stream.read()
        parsed = {
            key: int(value)
            for key, value in re.findall(
                r"^(MemTotal|MemAvailable|SwapTotal|SwapFree):\s+(\d+)", text, re.M
            )
        }
        return {
            "total_kb": parsed.get("MemTotal"),
            "available_kb": parsed.get("MemAvailable"),
            "swap_used_kb": (
                parsed.get("SwapTotal", 0) - parsed.get("SwapFree", 0)
            ),
        }
    return {}


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


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * max(0.0, min(1.0, percentile))
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _latest(values: List[float]) -> Optional[float]:
    return float(values[-1]) if values else None


class _InferenceLoad:
    """Repeatedly run an explicit model command during an exploration window."""

    def __init__(self, command: str, timeout_sec: float):
        self.command = command.strip()
        self.timeout_sec = max(0.1, float(timeout_sec))
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.latencies_sec: List[float] = []
        self.failures: List[str] = []

    def _run_once(self) -> None:
        started = time.perf_counter()
        try:
            subprocess.run(
                shlex.split(self.command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_sec,
                check=True,
            )
            self.latencies_sec.append(time.perf_counter() - started)
        except Exception as exc:  # The failure is part of the experiment output.
            self.failures.append(f"{type(exc).__name__}: {exc}")

    def warmup(self, count: int) -> None:
        for _ in range(max(0, int(count))):
            self._run_once()

    def start(self) -> None:
        if not self.command:
            return

        def loop() -> None:
            while not self.stop_event.is_set():
                self._run_once()

        self.thread = threading.Thread(target=loop, name="inference-load", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.timeout_sec + 1.0)

    def summary(self) -> Dict[str, Any]:
        values = list(self.latencies_sec)
        return {
            "inference_command_enabled": bool(self.command),
            "inference_call_count": len(values),
            "inference_failure_count": len(self.failures),
            "inference_latency_mean_sec": statistics.mean(values) if values else None,
            "inference_latency_p50_sec": _percentile(values, 0.50),
            "inference_latency_p95_sec": _percentile(values, 0.95),
            "inference_failures": self.failures[:10],
        }


def _device_memory(adb_prefix: str) -> Dict[str, Any]:
    text = _run_adb(adb_prefix, "shell", "cat", "/proc/meminfo")
    memory_keys = (
        "MemTotal|MemFree|MemAvailable|Cached|SwapCached|SwapTotal|SwapFree|"
        "Active|Inactive|Dirty|Writeback"
    )
    values = {
        k: int(v)
        for k, v in re.findall(rf"^({memory_keys}):\s+(\d+)", text, re.M)
    }
    pressure = _run_adb(adb_prefix, "shell", "cat", "/proc/pressure/memory")
    for scope, avg10, avg60, avg300, total in re.findall(
        r"^(some|full)\s+avg10=([0-9.]+)\s+avg60=([0-9.]+)\s+avg300=([0-9.]+)\s+total=(\d+)",
        pressure,
        re.M,
    ):
        values.update({
            f"psi_{scope}_avg10": float(avg10),
            f"psi_{scope}_avg60": float(avg60),
            f"psi_{scope}_avg300": float(avg300),
            f"psi_{scope}_total_us": int(total),
        })
    vmstat = _run_adb(adb_prefix, "shell", "cat", "/proc/vmstat")
    for key, raw in re.findall(r"^(pgmajfault|pswpin|pswpout|oom_kill)\s+(\d+)", vmstat, re.M):
        values[key] = int(raw)
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


def _counter_delta(start: Dict[str, Any], end: Dict[str, Any], key: str) -> Optional[float]:
    if key not in start or key not in end:
        return None
    return float(end[key]) - float(start[key])


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


def _graph_bytes(graph: ProgressiveBeliefGraph, metric: str) -> int:
    return (
        graph.approximate_python_bytes()
        if metric == "python"
        else graph.approximate_serialized_bytes()
    )


def _preload_graph(
    graph: ProgressiveBeliefGraph,
    target_bytes: int,
    max_nodes: int,
) -> Dict[str, Any]:
    """Populate a realistic chain so a run starts at a controlled graph size."""
    target = max(0, int(target_bytes))
    if target <= 0:
        return {"preloaded_nodes": 0, "preloaded_edges": 0, "preload_python_bytes": 0}
    started = time.perf_counter()
    source = graph.observe_state(UIStateDescriptor(
        signature="eval_preload_00000000",
        labels=("Synthetic launcher", "Entry 0"),
        package="evaluation.synthetic",
        coarse_context="button:8,text:16",
        candidate_element_count=8,
    ))
    created = 1
    # Measuring the recursive live size is deliberately batched; doing it on
    # every node turns preloading into an O(n^2) benchmark of the evaluator.
    while created < max(1, int(max_nodes)):
        if created == 1 or created % 16 == 0:
            if graph.approximate_python_bytes() >= target:
                break
        index = created
        destination = graph.observe_state(UIStateDescriptor(
            signature=f"eval_preload_{index:08d}",
            labels=(f"Synthetic screen {index}", f"Item {index}", "Back", "More options"),
            package="evaluation.synthetic",
            coarse_context="button:8,text:16",
            candidate_element_count=8,
        ))
        edge_id = graph.record_probe(
            source,
            f"evaluation.synthetic:id/item_{index}",
            "click",
            "button",
            "click",
            "button:8,text:16",
            "generic",
            {
                "action_type": "click",
                "coord_space": "norm1000",
                "action_inputs": {"coordinate": [100 + index % 700, 200 + index % 600]},
            },
            destination,
            (f"Synthetic destination {index}", "Details", "Back"),
            0.5,
            0.25,
            (10, 10, 200, 100),
            0.0,
        )
        graph.record_rollback_result(edge_id, True)
        source = destination
        created += 1
    return {
        "preloaded_nodes": created,
        "preloaded_edges": max(0, created - 1),
        "preload_python_bytes": graph.approximate_python_bytes(),
        "preload_serialized_bytes": graph.approximate_serialized_bytes(),
        "preload_elapsed_sec": time.perf_counter() - started,
        "preload_target_reached": graph.approximate_python_bytes() >= target,
    }


def run_budget(
    args,
    budget_mb: float,
    run_index: int,
    output_jsonl: str,
    repeat_index: int = 1,
) -> Dict[str, Any]:
    gc.collect()
    graph = ProgressiveBeliefGraph()
    budget_bytes = 0 if budget_mb <= 0 else int(budget_mb * 1024 * 1024)
    run_dir = os.path.join(
        args.output_dir,
        f"run_{run_index:03d}_repeat_{repeat_index:02d}_budget_{budget_mb:g}mb",
    )
    ensure_dir(run_dir)
    preload_target = (
        int(budget_bytes * args.preload_fraction)
        if args.preload_to_budget and budget_bytes > 0
        else int(max(0.0, args.preload_graph_mb) * 1024 * 1024)
    )
    preload = _preload_graph(graph, preload_target, args.preload_max_nodes)
    preload_prune = graph.prune_to_budget(
        budget_bytes,
        size_metric=args.budget_size_metric,
    )
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
    snapshot_started = time.perf_counter()
    snapshot = graph.snapshot()
    snapshot_build_sec = time.perf_counter() - snapshot_started
    snapshot_python_bytes = _deep_size({"nodes": snapshot.nodes, "edges": snapshot.edges})
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

    inference_load = _InferenceLoad(args.inference_command, args.inference_timeout_sec)
    inference_load.warmup(args.inference_warmup_runs)
    inference_load.latencies_sec.clear()
    inference_load.failures.clear()
    device_start = _device_memory(args.adb_path)
    host_memory_start = _host_memory()
    rss_start = _host_rss_kb()
    start = time.time()
    inference_load.start()
    explorer.start(
        max_steps=args.max_steps,
        max_depth=args.max_depth,
        leaf_width=args.leaf_width,
        time_budget_sec=args.duration_sec,
        trigger_reason="memory_evaluation",
    )
    samples: List[Dict[str, Any]] = []
    last_probe_count = -1
    last_probe_elapsed = 0.0
    total_pruned_edges = int(preload_prune.get("pruned_edges", 0))
    total_pruned_nodes = int(preload_prune.get("pruned_nodes", 0))
    while time.time() - start < args.duration_sec and explorer.thread and explorer.thread.is_alive():
        probe_count = int(explorer.graph_probe_ingest_count)
        budget_needs_prune = bool(
            budget_bytes > 0
            and not explorer._pending_probe_edge_ids
            and _graph_bytes(graph, args.budget_size_metric) > budget_bytes
        )
        if probe_count != last_probe_count or budget_needs_prune:
            prune_started = time.perf_counter()
            prune = (
                graph.prune_to_budget(
                    budget_bytes,
                    protected_node_ids=[live_node],
                    size_metric=args.budget_size_metric,
                )
                if not explorer._pending_probe_edge_ids
                else {
                    "pruned_edges": 0,
                    "pruned_nodes": 0,
                    "bytes": _graph_bytes(graph, args.budget_size_metric),
                    "size_metric": args.budget_size_metric,
                }
            )
            prune_elapsed_sec = time.perf_counter() - prune_started
            total_pruned_edges += int(prune.get("pruned_edges", 0))
            total_pruned_nodes += int(prune.get("pruned_nodes", 0))
            now_elapsed = time.time() - start
            sample = {
                "record_type": "sample",
                "run_index": run_index,
                "repeat_index": repeat_index,
                "budget_mb": budget_mb,
                "budget_bytes": budget_bytes,
                "budget_size_metric": args.budget_size_metric,
                "elapsed_sec": now_elapsed,
                "probe_interval_sec": (
                    now_elapsed - last_probe_elapsed if probe_count > last_probe_count and probe_count > 0 else None
                ),
                "probe_count": probe_count,
                "explorer_action_count": explorer.cur_steps,
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "graph_python_bytes": graph.approximate_python_bytes(),
                "graph_snapshot_python_bytes": snapshot_python_bytes,
                "graph_total_python_bytes": (
                    graph.approximate_python_bytes() + snapshot_python_bytes
                ),
                "graph_serialized_bytes": graph.approximate_serialized_bytes(),
                "host_rss_kb": _host_rss_kb(),
                "probe_total_sec": _latest(explorer.total_latency),
                "selection_sec": _latest(explorer.selection_latency),
                "action_sec": _latest(explorer.action_latency),
                "screenshot_sec": _latest(explorer.screenshot_latency),
                "a11y_tree_sec": _latest(explorer.adb_tree_latency),
                "parse_tree_sec": _latest(explorer.tree_latency),
                "prune_elapsed_sec": prune_elapsed_sec,
                **prune,
            }
            samples.append(sample)
            with open(output_jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if probe_count > last_probe_count and probe_count > 0:
                last_probe_elapsed = now_elapsed
            last_probe_count = probe_count
        time.sleep(0.05)

    exploration_window_elapsed = time.time() - start
    explorer.stop(min_steps=0, min_runtime_sec=0.0, max_wait_sec=2.0)
    inference_load.stop()
    elapsed = time.time() - start
    final_prune = graph.prune_to_budget(
        budget_bytes,
        protected_node_ids=[live_node],
        size_metric=args.budget_size_metric,
    )
    total_pruned_edges += int(final_prune.get("pruned_edges", 0))
    total_pruned_nodes += int(final_prune.get("pruned_nodes", 0))
    device_end = _device_memory(args.adb_path)
    host_memory_end = _host_memory()
    rss_end = _host_rss_kb()
    total_probes = int(explorer.graph_probe_ingest_count)
    rollback_successes = sum(edge.rollback_success_count for edge in graph.edges.values())
    rollback_failures = sum(edge.rollback_failure_count for edge in graph.edges.values())
    summary = {
        "record_type": "budget_summary",
        "run_index": run_index,
        "repeat_index": repeat_index,
        "budget_mb": budget_mb,
        "budget_bytes": budget_bytes,
        "budget_size_metric": args.budget_size_metric,
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
        "graph_python_bytes": graph.approximate_python_bytes(),
        "graph_snapshot_python_bytes": snapshot_python_bytes,
        "graph_total_python_bytes": graph.approximate_python_bytes() + snapshot_python_bytes,
        "graph_serialized_bytes": graph.approximate_serialized_bytes(),
        "snapshot_build_sec": snapshot_build_sec,
        "host_rss_start_kb": rss_start,
        "host_rss_end_kb": rss_end,
        "host_rss_delta_kb": rss_end - rss_start,
        "host_memory_start": host_memory_start,
        "host_memory_end": host_memory_end,
        "host_mem_available_delta_kb": (
            host_memory_end.get("available_kb", 0)
            - host_memory_start.get("available_kb", 0)
        ),
        "host_swap_used_delta_kb": (
            host_memory_end.get("swap_used_kb", 0)
            - host_memory_start.get("swap_used_kb", 0)
        ),
        "device_mem_start": device_start,
        "device_mem_end": device_end,
        "device_mem_available_delta_kb": (
            device_end.get("MemAvailable", 0) - device_start.get("MemAvailable", 0)
        ),
        "device_psi_some_total_delta_us": _counter_delta(
            device_start, device_end, "psi_some_total_us"
        ),
        "device_psi_full_total_delta_us": _counter_delta(
            device_start, device_end, "psi_full_total_us"
        ),
        "device_pgmajfault_delta": _counter_delta(device_start, device_end, "pgmajfault"),
        "device_pswpin_delta": _counter_delta(device_start, device_end, "pswpin"),
        "device_pswpout_delta": _counter_delta(device_start, device_end, "pswpout"),
        "bytes_per_node_slope": _slope(samples, "node_count", "graph_python_bytes"),
        "rss_kb_per_node_slope": _slope(samples, "node_count", "host_rss_kb"),
        "probe_latency_sec_per_python_mb_slope": _slope(
            [
                {**sample, "graph_python_mb": sample["graph_python_bytes"] / 1024.0 / 1024.0}
                for sample in samples
                if sample.get("probe_total_sec") is not None
            ],
            "graph_python_mb",
            "probe_total_sec",
        ),
        "probe_latency_mean_sec": (
            statistics.mean(explorer.total_latency) if explorer.total_latency else None
        ),
        "probe_latency_p50_sec": _percentile(explorer.total_latency, 0.50),
        "probe_latency_p95_sec": _percentile(explorer.total_latency, 0.95),
        "steady_probe_latency_mean_sec": (
            statistics.mean(explorer.total_latency[1:])
            if len(explorer.total_latency) > 1 else None
        ),
        "selection_latency_mean_sec": (
            statistics.mean(explorer.selection_latency) if explorer.selection_latency else None
        ),
        "action_latency_mean_sec": (
            statistics.mean(explorer.action_latency) if explorer.action_latency else None
        ),
        "screenshot_latency_mean_sec": (
            statistics.mean(explorer.screenshot_latency) if explorer.screenshot_latency else None
        ),
        "a11y_tree_latency_mean_sec": (
            statistics.mean(explorer.adb_tree_latency) if explorer.adb_tree_latency else None
        ),
        "parse_tree_latency_mean_sec": (
            statistics.mean(explorer.tree_latency) if explorer.tree_latency else None
        ),
        "pruned_edges": total_pruned_edges,
        "pruned_nodes": total_pruned_nodes,
        "budget_bytes_after_prune": final_prune.get("bytes"),
        **preload,
        **inference_load.summary(),
    }
    with open(output_jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return summary


def _run_budget_worker(
    args_dict: Dict[str, Any],
    budget_mb: float,
    run_index: int,
    repeat_index: int,
    output_jsonl: str,
    queue,
) -> None:
    try:
        result = run_budget(
            argparse.Namespace(**args_dict),
            budget_mb,
            run_index,
            output_jsonl,
            repeat_index,
        )
        queue.put({"ok": True, "result": result})
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _aggregate_summaries(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[float, List[Dict[str, Any]]] = {}
    for row in summaries:
        groups.setdefault(float(row["budget_mb"]), []).append(row)
    metrics = [
        "exploration_probe_count",
        "probes_per_second",
        "probe_latency_mean_sec",
        "steady_probe_latency_mean_sec",
        "selection_latency_mean_sec",
        "action_latency_mean_sec",
        "screenshot_latency_mean_sec",
        "a11y_tree_latency_mean_sec",
        "graph_python_bytes",
        "graph_snapshot_python_bytes",
        "graph_total_python_bytes",
        "snapshot_build_sec",
        "host_rss_delta_kb",
        "host_mem_available_delta_kb",
        "host_swap_used_delta_kb",
        "device_mem_available_delta_kb",
        "device_psi_some_total_delta_us",
        "device_psi_full_total_delta_us",
        "device_pgmajfault_delta",
        "inference_call_count",
        "inference_latency_mean_sec",
    ]
    output = []
    for budget, rows in sorted(groups.items()):
        item: Dict[str, Any] = {
            "budget_mb": budget,
            "repeat_count": len(rows),
            "budget_size_metric": rows[0].get("budget_size_metric"),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            item[f"{metric}_mean"] = statistics.mean(values) if values else None
            item[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else None
        output.append(item)
    return output


def _system_effects(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [
        {
            **row,
            "graph_total_python_mb": float(row.get("graph_total_python_bytes", 0)) / 1024.0 / 1024.0,
        }
        for row in summaries
    ]
    return {
        "run_count": len(rows),
        "probe_latency_sec_per_graph_mb_slope": _slope(
            rows, "graph_total_python_mb", "probe_latency_mean_sec"
        ),
        "probes_per_second_per_graph_mb_slope": _slope(
            rows, "graph_total_python_mb", "probes_per_second"
        ),
        "inference_latency_sec_per_graph_mb_slope": _slope(
            rows, "graph_total_python_mb", "inference_latency_mean_sec"
        ),
        "snapshot_build_sec_per_graph_mb_slope": _slope(
            rows, "graph_total_python_mb", "snapshot_build_sec"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Explore non-destructive launcher controls")
    parser.add_argument("--duration_sec", type=float, default=8.0)
    parser.add_argument("--budgets_mb", default="0.01,0.03,0", help="Comma-separated; 0 means unlimited.")
    parser.add_argument(
        "--budget_size_metric",
        choices=["python", "serialized"],
        default="python",
        help="Enforce the budget against live Python objects (system test) or JSON bytes.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--randomize_order",
        dest="randomize_order",
        action="store_true",
        help="Randomize budget order within every repeat.",
    )
    parser.add_argument("--no_randomize_order", dest="randomize_order", action="store_false")
    parser.set_defaults(randomize_order=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--isolate_runs",
        dest="isolate_runs",
        action="store_true",
        help="Run every budget/repeat in a fresh Python process to isolate RSS warm-up.",
    )
    parser.add_argument("--no_isolate_runs", dest="isolate_runs", action="store_false")
    parser.set_defaults(isolate_runs=True)
    parser.add_argument(
        "--preload_to_budget",
        action="store_true",
        help="Pre-populate a realistic graph chain to a fraction of each positive budget.",
    )
    parser.add_argument("--preload_fraction", type=float, default=0.90)
    parser.add_argument("--preload_graph_mb", type=float, default=0.0)
    parser.add_argument("--preload_max_nodes", type=int, default=100000)
    parser.add_argument(
        "--inference_command",
        default="",
        help="Optional command repeatedly executed concurrently with exploration.",
    )
    parser.add_argument("--inference_timeout_sec", type=float, default=120.0)
    parser.add_argument("--inference_warmup_runs", type=int, default=0)
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
    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use a new directory "
            "so repeated experiments cannot mix artifacts."
        )
    ensure_dir(args.output_dir)
    output_jsonl = os.path.join(args.output_dir, "samples.jsonl")
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)
    budgets = [float(x.strip()) for x in args.budgets_mb.split(",") if x.strip()]
    run_specs = []
    rng = random.Random(args.seed)
    run_index = 0
    for repeat_index in range(1, max(1, args.repeats) + 1):
        ordered = list(budgets)
        if args.randomize_order:
            rng.shuffle(ordered)
        for budget in ordered:
            run_index += 1
            run_specs.append((budget, run_index, repeat_index))

    summaries = []
    for budget, current_run, repeat_index in run_specs:
        if args.isolate_runs:
            context = multiprocessing.get_context("spawn")
            queue = context.Queue()
            process = context.Process(
                target=_run_budget_worker,
                args=(vars(args), budget, current_run, repeat_index, output_jsonl, queue),
                name=f"graph-memory-budget-{budget:g}-repeat-{repeat_index}",
            )
            process.start()
            try:
                message = queue.get(timeout=max(
                    300.0,
                    args.duration_sec + args.inference_timeout_sec + 120.0,
                ))
            except queue_module.Empty as exc:
                process.terminate()
                process.join(timeout=5.0)
                raise RuntimeError(
                    f"Budget run timed out: budget={budget:g} repeat={repeat_index}"
                ) from exc
            process.join()
            if not message.get("ok"):
                raise RuntimeError(message.get("error", "isolated budget run failed"))
            summaries.append(message["result"])
        else:
            summaries.append(run_budget(
                args, budget, current_run, output_jsonl, repeat_index
            ))
    aggregates = _aggregate_summaries(summaries)
    effects = _system_effects(summaries)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": vars(args),
                "device": device_identity,
                "results": summaries,
                "aggregates": aggregates,
                "system_effects": effects,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    csv_path = os.path.join(args.output_dir, "summary.csv")
    flat_keys = [
        "run_index", "repeat_index", "budget_mb", "budget_size_metric",
        "target_duration_sec", "actual_duration_sec", "exploration_probe_count",
        "probes_per_second", "node_count", "edge_count", "graph_python_bytes",
        "graph_snapshot_python_bytes", "graph_total_python_bytes", "graph_serialized_bytes",
        "snapshot_build_sec",
        "host_rss_delta_kb", "host_mem_available_delta_kb", "host_swap_used_delta_kb",
        "device_mem_available_delta_kb",
        "rollback_success_count", "rollback_failure_count", "probe_latency_mean_sec",
        "steady_probe_latency_mean_sec", "selection_latency_mean_sec", "action_latency_mean_sec",
        "screenshot_latency_mean_sec", "a11y_tree_latency_mean_sec",
        "device_psi_some_total_delta_us", "device_psi_full_total_delta_us",
        "device_pgmajfault_delta", "inference_call_count", "inference_latency_mean_sec",
        "pruned_edges", "pruned_nodes", "preloaded_nodes", "preload_python_bytes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat_keys)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({k: summary.get(k) for k in flat_keys})
    aggregate_path = os.path.join(args.output_dir, "aggregate.csv")
    aggregate_keys = sorted({key for row in aggregates for key in row})
    with open(aggregate_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=aggregate_keys)
        writer.writeheader()
        writer.writerows(aggregates)
    print(json.dumps({
        "summary_json": summary_path,
        "summary_csv": csv_path,
        "aggregate_csv": aggregate_path,
        "results": summaries,
        "aggregates": aggregates,
        "system_effects": effects,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
