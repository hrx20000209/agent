#!/usr/bin/env python3
"""Profile simulated ProgressiveBeliefGraph growth without ADB or a model."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import psutil

from Explorer.progressive_belief_graph import ProgressiveBeliefGraph, UIStateDescriptor
from evaluate_graph_memory_adb import _deep_size


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def process_memory() -> Dict[str, int]:
    process = psutil.Process(os.getpid())
    basic = process.memory_info()
    full = process.memory_full_info()
    return {
        "rss_bytes": int(basic.rss),
        "uss_bytes": int(getattr(full, "uss", 0) or 0),
    }


def require_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def descriptor(index: int, labels_per_node: int, evidence_chars: int) -> UIStateDescriptor:
    suffix = ("x" * max(0, evidence_chars))[: max(0, evidence_chars)]
    labels = tuple(
        f"Synthetic screen {index} control {label_index} {suffix}"
        for label_index in range(max(1, labels_per_node))
    )
    return UIStateDescriptor(
        signature=f"profile_{index:012d}",
        labels=labels,
        package="profiling.synthetic",
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
    payload = ("e" * max(0, evidence_chars))[: max(0, evidence_chars)]
    edge_id = graph.record_probe(
        source_node_id=source_node_id,
        element_identity=f"profiling.synthetic:id/control_{index}",
        action_type="click",
        role="button",
        probe_type="click",
        coarse_context="button:8,text:16",
        information_need_type="generic",
        action={
            "action_type": "click",
            "coord_space": "norm1000",
            "action_inputs": {
                "coordinate": [100 + index % 800, 100 + (index * 7) % 800],
                "label": f"Synthetic control {index} {payload}",
            },
        },
        destination_node_id=destination_node_id,
        discovered_labels=(f"Destination {index} {payload}", "Back", "More options"),
        exploration_cost=0.5,
        realized_information_gain=0.25,
        bounds=(10, 10, 200, 100),
        risk_level=0.0,
    )
    graph.record_rollback_result(edge_id, True)


def linear_slope(rows: List[Dict[str, Any]], x_key: str, y_key: str) -> Optional[float]:
    pairs = [
        (float(row[x_key]), float(row[y_key]))
        for row in rows
        if row.get(x_key) is not None and row.get(y_key) is not None
    ]
    if len(pairs) < 2:
        return None
    x_mean = statistics.mean(x for x, _ in pairs)
    y_mean = statistics.mean(y for _, y in pairs)
    denominator = sum((x - x_mean) ** 2 for x, _ in pairs)
    if denominator <= 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_nodes", type=int, default=10000)
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument(
        "--snapshot_every",
        type=int,
        default=1000,
        help="Refresh and retain an immutable snapshot every N nodes; 0 disables snapshots.",
    )
    parser.add_argument("--labels_per_node", type=int, default=4)
    parser.add_argument(
        "--evidence_chars",
        type=int,
        default=32,
        help="Synthetic label/action payload per node and edge.",
    )
    parser.add_argument("--gc_at_sample", action="store_true")
    parser.add_argument("--output_dir", default="evaluation_results/profile_graph_growth")
    args = parser.parse_args()

    target_nodes = max(1, int(args.target_nodes))
    sample_every = max(1, int(args.sample_every))
    snapshot_every = max(0, int(args.snapshot_every))
    output_dir = Path(args.output_dir).resolve()
    require_empty_output_dir(output_dir)
    (output_dir / "config.json").write_text(
        json.dumps({**vars(args), "output_dir": str(output_dir)}, indent=2), encoding="utf-8"
    )

    graph = ProgressiveBeliefGraph()
    source_node_id = graph.observe_state(descriptor(0, args.labels_per_node, args.evidence_chars))
    retained_snapshot = None
    retained_snapshot_bytes = 0
    last_sample_node_count = 1
    last_sample_time = time.perf_counter()
    started = last_sample_time
    rows: List[Dict[str, Any]] = []

    def sample(snapshot_due: bool) -> None:
        nonlocal retained_snapshot, retained_snapshot_bytes, last_sample_node_count, last_sample_time
        if args.gc_at_sample:
            gc.collect()
        snapshot_build_sec: Optional[float] = None
        snapshot_overlap_rss_bytes: Optional[int] = None
        if snapshot_due:
            snapshot_started = time.perf_counter()
            new_snapshot = graph.snapshot()
            snapshot_build_sec = time.perf_counter() - snapshot_started
            retained_snapshot_bytes = _deep_size({"nodes": new_snapshot.nodes, "edges": new_snapshot.edges})
            # retained_snapshot is intentionally still alive here.  This RSS is
            # the replacement peak when old and new snapshots overlap.
            snapshot_overlap_rss_bytes = process_memory()["rss_bytes"]
            retained_snapshot = new_snapshot
            gc.collect()
        size_started = time.perf_counter()
        live_python_bytes = graph.approximate_python_bytes()
        python_size_measure_sec = time.perf_counter() - size_started
        serialized_started = time.perf_counter()
        serialized_bytes = graph.approximate_serialized_bytes()
        serialized_size_measure_sec = time.perf_counter() - serialized_started
        now = time.perf_counter()
        node_count = len(graph.nodes)
        memory = process_memory()
        rows.append({
            "elapsed_sec": now - started,
            "node_count": node_count,
            "edge_count": len(graph.edges),
            "nodes_added_since_sample": node_count - last_sample_node_count,
            "build_sec_since_sample": now - last_sample_time,
            "build_nodes_per_sec_since_sample": (
                (node_count - last_sample_node_count) / max(1e-9, now - last_sample_time)
            ),
            "live_graph_python_bytes": live_python_bytes,
            "immutable_snapshot_python_bytes": retained_snapshot_bytes,
            "total_graph_python_bytes": live_python_bytes + retained_snapshot_bytes,
            "serialized_graph_bytes": serialized_bytes,
            "process_rss_bytes": memory["rss_bytes"],
            "process_uss_bytes": memory["uss_bytes"],
            "snapshot_refreshed": snapshot_due,
            "snapshot_build_sec": snapshot_build_sec,
            "snapshot_overlap_rss_bytes": snapshot_overlap_rss_bytes,
            "python_size_measure_sec": python_size_measure_sec,
            "serialized_size_measure_sec": serialized_size_measure_sec,
        })
        last_sample_node_count = node_count
        last_sample_time = now

    sample(snapshot_due=bool(snapshot_every and 1 % snapshot_every == 0))
    for index in range(1, target_nodes):
        destination_node_id = graph.observe_state(
            descriptor(index, args.labels_per_node, args.evidence_chars)
        )
        add_transition(graph, source_node_id, destination_node_id, index, args.evidence_chars)
        source_node_id = destination_node_id
        node_count = len(graph.nodes)
        is_final = node_count == target_nodes
        is_sample = node_count % sample_every == 0 or is_final
        if is_sample:
            snapshot_due = bool(
                snapshot_every
                and (node_count % snapshot_every == 0 or is_final)
            )
            sample(snapshot_due=snapshot_due)

    fieldnames = list(rows[0].keys())
    with (output_dir / "growth.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    snapshot_latencies = [
        float(row["snapshot_build_sec"])
        for row in rows
        if row.get("snapshot_build_sec") is not None
    ]
    summary = {
        "config": {**vars(args), "output_dir": str(output_dir)},
        "metrics": {
            "elapsed_sec": time.perf_counter() - started,
            "final_node_count": len(graph.nodes),
            "final_edge_count": len(graph.edges),
            "final_live_graph_python_bytes": rows[-1]["live_graph_python_bytes"],
            "final_snapshot_python_bytes": rows[-1]["immutable_snapshot_python_bytes"],
            "final_total_graph_python_bytes": rows[-1]["total_graph_python_bytes"],
            "final_serialized_graph_bytes": rows[-1]["serialized_graph_bytes"],
            "final_process_rss_bytes": rows[-1]["process_rss_bytes"],
            "peak_process_rss_bytes": max(int(row["process_rss_bytes"]) for row in rows),
            "peak_snapshot_overlap_rss_bytes": max(
                [int(row["snapshot_overlap_rss_bytes"]) for row in rows if row.get("snapshot_overlap_rss_bytes")]
                or [0]
            ),
            "live_python_bytes_per_node_slope": linear_slope(rows, "node_count", "live_graph_python_bytes"),
            "rss_bytes_per_node_slope": linear_slope(rows, "node_count", "process_rss_bytes"),
            "snapshot_build_mean_sec": statistics.mean(snapshot_latencies) if snapshot_latencies else None,
            "snapshot_build_p95_sec": percentile(snapshot_latencies, 0.95),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    graph.save(str(output_dir / "belief_graph.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # Keep a strong reference until after all final measurements are written.
    _ = retained_snapshot
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
