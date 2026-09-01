#!/usr/bin/env python3
"""Profile ProgressiveBeliefGraph growth inside Android Termux.

This script intentionally has no psutil, ADB, Pillow, or model dependency.  It
reads its own Android process memory from /proc and records pure insertion time
separately from expensive graph-size and snapshot measurements.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from Explorer.progressive_belief_graph import (
    ProgressiveBeliefGraph,
    UIStateDescriptor,
    _deep_size,
)


def read_key_values(path: str, keys: Iterable[str]) -> Dict[str, int]:
    wanted = set(keys)
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    result: Dict[str, int] = {}
    for key, value in re.findall(r"^([A-Za-z_()]+):\s+(\d+)", text, re.M):
        if key in wanted:
            result[key] = int(value)
    return result


def process_memory() -> Dict[str, Optional[int]]:
    status = read_key_values(
        "/proc/self/status", ("VmRSS", "VmHWM", "VmSize", "VmSwap")
    )
    rollup = read_key_values(
        "/proc/self/smaps_rollup",
        ("Pss", "Private_Clean", "Private_Dirty", "SwapPss"),
    )
    private_kb = None
    if "Private_Clean" in rollup or "Private_Dirty" in rollup:
        private_kb = rollup.get("Private_Clean", 0) + rollup.get("Private_Dirty", 0)
    return {
        "rss_bytes": status.get("VmRSS", 0) * 1024,
        "rss_hwm_bytes": status.get("VmHWM", 0) * 1024,
        "vms_bytes": status.get("VmSize", 0) * 1024,
        "swap_bytes": status.get("VmSwap", 0) * 1024,
        "pss_bytes": rollup.get("Pss", 0) * 1024 if "Pss" in rollup else None,
        "uss_bytes": private_kb * 1024 if private_kb is not None else None,
        "swap_pss_bytes": rollup.get("SwapPss", 0) * 1024 if "SwapPss" in rollup else None,
    }


def system_memory() -> Dict[str, Any]:
    values = read_key_values(
        "/proc/meminfo",
        ("MemTotal", "MemFree", "MemAvailable", "Cached", "SwapTotal", "SwapFree"),
    )
    output: Dict[str, Any] = {f"{key.lower()}_kb": value for key, value in values.items()}
    try:
        pressure = Path("/proc/pressure/memory").read_text(encoding="utf-8")
    except OSError:
        pressure = ""
    for scope, avg10, avg60, avg300, total in re.findall(
        r"^(some|full)\s+avg10=([0-9.]+)\s+avg60=([0-9.]+)\s+avg300=([0-9.]+)\s+total=(\d+)",
        pressure,
        re.M,
    ):
        output.update({
            f"psi_{scope}_avg10": float(avg10),
            f"psi_{scope}_avg60": float(avg60),
            f"psi_{scope}_avg300": float(avg300),
            f"psi_{scope}_total_us": int(total),
        })
    return output


def descriptor(index: int, labels_per_node: int, evidence_chars: int) -> UIStateDescriptor:
    payload = "x" * max(0, evidence_chars)
    return UIStateDescriptor(
        signature=f"termux_profile_{index:012d}",
        labels=tuple(
            f"Synthetic screen {index} control {label_index} {payload}"
            for label_index in range(max(1, labels_per_node))
        ),
        package="profiling.termux",
        coarse_context=f"button:{labels_per_node},text:{labels_per_node * 2}",
        candidate_element_count=max(1, labels_per_node),
    )


def add_transition(
    graph: ProgressiveBeliefGraph,
    source_node_id: str,
    destination_node_id: str,
    index: int,
    evidence_chars: int,
) -> None:
    payload = "e" * max(0, evidence_chars)
    edge_id = graph.record_probe(
        source_node_id,
        f"profiling.termux:id/control_{index}",
        "click",
        "button",
        "click",
        "button:8,text:16",
        "generic",
        {
            "action_type": "click",
            "coord_space": "norm1000",
            "action_inputs": {
                "coordinate": [100 + index % 800, 100 + (index * 7) % 800],
                "label": f"Synthetic control {index} {payload}",
            },
        },
        destination_node_id,
        (f"Destination {index} {payload}", "Back", "More options"),
        0.5,
        0.25,
        (10, 10, 200, 100),
        0.0,
    )
    graph.record_rollback_result(edge_id, True)


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def require_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_nodes", type=int, default=20000)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--snapshot_every", type=int, default=1000)
    parser.add_argument(
        "--expensive_measure_every",
        type=int,
        default=-1,
        help=(
            "Measure deep/serialized graph size every N nodes. -1 follows "
            "--sample_every (legacy behavior); 0 measures only the final graph."
        ),
    )
    parser.add_argument("--labels_per_node", type=int, default=4)
    parser.add_argument("--evidence_chars", type=int, default=32)
    parser.add_argument("--gc_at_sample", action="store_true")
    parser.add_argument("--save_graph", action="store_true")
    parser.add_argument(
        "--baseline_sec",
        type=float,
        default=0.0,
        help="Sample the imported but graph-free Python process before construction.",
    )
    parser.add_argument("--baseline_sample_sec", type=float, default=1.0)
    parser.add_argument(
        "--hold_sec",
        type=float,
        default=0.0,
        help="Keep the completed live graph and snapshot resident for concurrent experiments.",
    )
    parser.add_argument("--hold_sample_sec", type=float, default=1.0)
    parser.add_argument("--output_dir", default="evaluation_results/termux_graph_growth")
    args = parser.parse_args()

    target_nodes = max(1, args.target_nodes)
    sample_every = max(1, args.sample_every)
    snapshot_every = max(0, args.snapshot_every)
    expensive_measure_every = (
        sample_every if args.expensive_measure_every < 0
        else max(0, int(args.expensive_measure_every))
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    require_empty_output_dir(output_dir)
    config = {**vars(args), "output_dir": str(output_dir), "pid": os.getpid()}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    baseline_rows: List[Dict[str, Any]] = []
    baseline_started = time.perf_counter()
    baseline_deadline = baseline_started + max(0.0, float(args.baseline_sec))
    while time.perf_counter() < baseline_deadline:
        baseline_rows.append({
            "timestamp": time.time(),
            "baseline_elapsed_sec": time.perf_counter() - baseline_started,
            **{f"process_{key}": value for key, value in process_memory().items()},
            **system_memory(),
        })
        time.sleep(min(
            max(0.05, float(args.baseline_sample_sec)),
            max(0.0, baseline_deadline - time.perf_counter()),
        ))
    if baseline_rows:
        baseline_fields = sorted({key for row in baseline_rows for key in row})
        with (output_dir / "baseline_memory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=baseline_fields)
            writer.writeheader()
            writer.writerows(baseline_rows)

    graph = ProgressiveBeliefGraph()
    source = graph.observe_state(descriptor(0, args.labels_per_node, args.evidence_chars))
    retained_snapshot = None
    retained_snapshot_bytes = 0
    insertion_latencies: List[float] = []
    insertion_total_sec = 0.0
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    started_timestamp = time.time()

    def sample(snapshot_due: bool, expensive_due: bool) -> None:
        nonlocal retained_snapshot, retained_snapshot_bytes
        gc_sec = 0.0
        if args.gc_at_sample:
            gc_started = time.perf_counter()
            gc.collect()
            gc_sec = time.perf_counter() - gc_started

        snapshot_build_sec = None
        snapshot_size_measure_sec = None
        overlap_memory: Dict[str, Optional[int]] = {}
        if snapshot_due:
            snapshot_started = time.perf_counter()
            new_snapshot = graph.snapshot()
            snapshot_build_sec = time.perf_counter() - snapshot_started
            overlap_memory = process_memory()
            retained_snapshot = new_snapshot
            if expensive_due:
                snapshot_size_started = time.perf_counter()
                retained_snapshot_bytes = _deep_size(
                    {"nodes": new_snapshot.nodes, "edges": new_snapshot.edges}
                )
                snapshot_size_measure_sec = time.perf_counter() - snapshot_size_started

        live_python_bytes = None
        live_size_measure_sec = None
        serialized_bytes = None
        serialized_size_measure_sec = None
        if expensive_due:
            live_size_started = time.perf_counter()
            live_python_bytes = graph.approximate_python_bytes()
            live_size_measure_sec = time.perf_counter() - live_size_started
            serialized_started = time.perf_counter()
            serialized_bytes = graph.approximate_serialized_bytes()
            serialized_size_measure_sec = time.perf_counter() - serialized_started
        memory = process_memory()
        row: Dict[str, Any] = {
            "timestamp": time.time(),
            "elapsed_sec": time.perf_counter() - started,
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "insertion_total_sec": insertion_total_sec,
            "insertion_mean_ms": statistics.mean(insertion_latencies) * 1000.0 if insertion_latencies else None,
            "insertion_p95_ms": percentile(insertion_latencies, 0.95) * 1000.0 if insertion_latencies else None,
            "live_graph_python_bytes": live_python_bytes,
            "immutable_snapshot_python_bytes": retained_snapshot_bytes,
            "total_graph_python_bytes": (
                live_python_bytes + retained_snapshot_bytes
                if live_python_bytes is not None else None
            ),
            "serialized_graph_bytes": serialized_bytes,
            "snapshot_refreshed": snapshot_due,
            "expensive_size_measured": expensive_due,
            "snapshot_build_sec": snapshot_build_sec,
            "snapshot_size_measure_sec": snapshot_size_measure_sec,
            "live_size_measure_sec": live_size_measure_sec,
            "serialized_size_measure_sec": serialized_size_measure_sec,
            "gc_sec": gc_sec,
            **{f"process_{key}": value for key, value in memory.items()},
            **{f"snapshot_overlap_{key}": value for key, value in overlap_memory.items()},
            **system_memory(),
        }
        rows.append(row)

    sample(snapshot_due=False, expensive_due=target_nodes == 1)
    for index in range(1, target_nodes):
        insertion_started = time.perf_counter()
        destination = graph.observe_state(
            descriptor(index, args.labels_per_node, args.evidence_chars)
        )
        add_transition(graph, source, destination, index, args.evidence_chars)
        insertion_sec = time.perf_counter() - insertion_started
        insertion_total_sec += insertion_sec
        insertion_latencies.append(insertion_sec)
        source = destination
        node_count = len(graph.nodes)
        if node_count % sample_every == 0 or node_count == target_nodes:
            is_final = node_count == target_nodes
            sample(
                snapshot_due=bool(
                    snapshot_every and (node_count % snapshot_every == 0 or is_final)
                ),
                expensive_due=bool(
                    is_final
                    or (expensive_measure_every and node_count % expensive_measure_every == 0)
                ),
            )

    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "growth.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    growth_elapsed_sec = time.perf_counter() - started
    ready = {
        "ready": True,
        "timestamp": time.time(),
        "pid": os.getpid(),
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "live_graph_python_bytes": rows[-1]["live_graph_python_bytes"],
        "snapshot_python_bytes": rows[-1]["immutable_snapshot_python_bytes"],
        "process_memory": process_memory(),
    }
    (output_dir / "ready.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
    print("GRAPH_READY " + json.dumps(ready), flush=True)

    hold_rows: List[Dict[str, Any]] = []
    hold_deadline = time.perf_counter() + max(0.0, float(args.hold_sec))
    while time.perf_counter() < hold_deadline:
        hold_rows.append({
            "timestamp": time.time(),
            "hold_elapsed_sec": max(0.0, float(args.hold_sec) - (hold_deadline - time.perf_counter())),
            **{f"process_{key}": value for key, value in process_memory().items()},
            **system_memory(),
        })
        time.sleep(min(
            max(0.05, float(args.hold_sample_sec)),
            max(0.0, hold_deadline - time.perf_counter()),
        ))
    if hold_rows:
        hold_fields = sorted({key for row in hold_rows for key in row})
        with (output_dir / "hold_memory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=hold_fields)
            writer.writeheader()
            writer.writerows(hold_rows)

    snapshot_latencies = [float(row["snapshot_build_sec"]) for row in rows if row.get("snapshot_build_sec") is not None]
    final = rows[-1]
    summary = {
        "config": config,
        "started_timestamp": started_timestamp,
        "ended_timestamp": time.time(),
        "metrics": {
            # Backward-compatible name now means graph construction only; hold
            # duration is reported separately instead of inflating build time.
            "elapsed_sec": growth_elapsed_sec,
            "total_elapsed_sec": time.perf_counter() - started,
            "growth_elapsed_sec": growth_elapsed_sec,
            "configured_hold_sec": max(0.0, float(args.hold_sec)),
            "final_node_count": len(graph.nodes),
            "final_edge_count": len(graph.edges),
            "insertion_total_sec": insertion_total_sec,
            "insertion_mean_ms": statistics.mean(insertion_latencies) * 1000.0 if insertion_latencies else None,
            "insertion_p50_ms": percentile(insertion_latencies, 0.50) * 1000.0 if insertion_latencies else None,
            "insertion_p95_ms": percentile(insertion_latencies, 0.95) * 1000.0 if insertion_latencies else None,
            "final_live_graph_python_bytes": final["live_graph_python_bytes"],
            "final_snapshot_python_bytes": final["immutable_snapshot_python_bytes"],
            "final_process_rss_bytes": final["process_rss_bytes"],
            "final_process_pss_bytes": final["process_pss_bytes"],
            "final_process_swap_bytes": final["process_swap_bytes"],
            "baseline_sample_count": len(baseline_rows),
            "baseline_rss_mean_bytes": statistics.mean(
                row["process_rss_bytes"] for row in baseline_rows
            ) if baseline_rows else None,
            "baseline_pss_mean_bytes": statistics.mean(
                row["process_pss_bytes"] for row in baseline_rows
                if row["process_pss_bytes"] is not None
            ) if any(row["process_pss_bytes"] is not None for row in baseline_rows) else None,
            "graph_rss_over_baseline_bytes": (
                final["process_rss_bytes"]
                - statistics.mean(row["process_rss_bytes"] for row in baseline_rows)
            ) if baseline_rows else None,
            "snapshot_build_mean_sec": statistics.mean(snapshot_latencies) if snapshot_latencies else None,
            "snapshot_build_p95_sec": percentile(snapshot_latencies, 0.95),
            "hold_sample_count": len(hold_rows),
            "hold_rss_mean_bytes": statistics.mean(
                row["process_rss_bytes"] for row in hold_rows
            ) if hold_rows else None,
            "hold_rss_peak_bytes": max(
                (row["process_rss_bytes"] for row in hold_rows), default=None
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_graph:
        graph.save(str(output_dir / "belief_graph.json"))
    print(json.dumps(summary, indent=2))
    _ = retained_snapshot
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
