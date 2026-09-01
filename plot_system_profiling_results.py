#!/usr/bin/env python3
"""Plot and summarize the current isolated MobileExplorer system profiles."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


MIB = 1024.0 * 1024.0


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows: List[Dict[str, Any]] = list(csv.DictReader(stream))
    for row in rows:
        for key, raw in list(row.items()):
            if raw in (None, ""):
                row[key] = None
                continue
            try:
                row[key] = float(raw)
            except (TypeError, ValueError):
                pass
    return rows


def values(rows: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    found = values(rows, key)
    return statistics.mean(found) if found else None


def stdev(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    found = values(rows, key)
    return statistics.stdev(found) if len(found) > 1 else None


def bootstrap_mean_difference(a: List[float], b: List[float], seed: int = 42) -> List[float]:
    rng = random.Random(seed)
    differences = []
    for _ in range(20000):
        sample_a = [rng.choice(a) for _ in a]
        sample_b = [rng.choice(b) for _ in b]
        differences.append(statistics.mean(sample_b) - statistics.mean(sample_a))
    differences.sort()
    return [differences[int(0.025 * len(differences))], differences[int(0.975 * len(differences))]]


def annotate(ax, bars, fmt: str = "{:.1f}", offset: int = 4) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="evaluation_results/profilng/system_profiling")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    phone: Dict[str, List[Dict[str, Any]]] = {}
    for trial in (
        "baseline_r01", "model_idle_r01", "model_active_r01",
        "model_active_llamacpp_r01", "graph_10000_r01", "exploration_cold_r01",
    ):
        phone[trial] = load_csv(root / trial / "phone_system" / "samples.csv")

    graph_summary = load_json(root / "graph_10000_r01" / "graph" / "summary.json")
    graph_growth = load_csv(root / "graph_10000_r01" / "graph" / "growth.csv")
    graph_baseline = load_csv(root / "graph_10000_r01" / "graph" / "baseline_memory.csv")
    exploration = load_json(root / "exploration_cold_r01" / "exploration" / "summary.json")
    model_remote = load_json(root / "model_active_r01" / "model" / "summary.json")
    model_phone_load = load_json(root / "model_active_llamacpp_r01" / "model" / "summary.json")
    requests_remote = load_csv(root / "model_active_r01" / "model" / "requests.csv")
    requests_phone_load = load_csv(root / "model_active_llamacpp_r01" / "model" / "requests.csv")

    graph_metrics = graph_summary["metrics"]
    exploration_agg = exploration["aggregate"]
    llama_idle_mib = float(mean(phone["model_idle_r01"], "process_llama_pss_kb") or 0) / 1024.0
    llama_phone_active_mib = float(mean(phone["model_active_llamacpp_r01"], "process_llama_pss_kb") or 0) / 1024.0
    graph_pss_increment_mib = (
        graph_metrics["final_process_pss_bytes"] - graph_metrics["baseline_pss_mean_bytes"]
    ) / MIB
    graph_logical_mib = (
        graph_metrics["final_live_graph_python_bytes"] + graph_metrics["final_snapshot_python_bytes"]
    ) / MIB
    exploration_app_delta_mib = exploration_agg["foreground_app_pss_delta_kb_mean"] / 1024.0
    exploration_host_delta_mib = exploration_agg["host_rss_delta_kb_mean"] / 1024.0

    remote_latencies = values(requests_remote, "latency_sec")
    phone_load_latencies = values(requests_phone_load, "latency_sec")
    latency_difference = statistics.mean(phone_load_latencies) - statistics.mean(remote_latencies)
    latency_difference_ci = bootstrap_mean_difference(remote_latencies, phone_load_latencies)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle("Physical-phone isolated profiling — memory and latency", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    labels = ["llama loaded\nPSS", "10k graph\nPSS increment", "10k graph\nlogical bytes", "exploration\napp PSS delta", "host explorer\nRSS delta"]
    component_values = [llama_idle_mib, graph_pss_increment_mib, graph_logical_mib, exploration_app_delta_mib, exploration_host_delta_mib]
    bars = ax.bar(labels, component_values, color=["#7e57c2", "#26a69a", "#80cbc4", "#ef5350", "#5c6bc0"])
    ax.set_yscale("log")
    ax.set_ylabel("Memory (MiB, log scale)")
    ax.set_title("A. Measured component memory (definitions differ; see labels)")
    annotate(ax, bars)
    ax.tick_params(axis="x", labelsize=9)

    ax = axes[0, 1]
    model_labels = ["Remote 8084\nphone llama idle", "Remote 8084\nphone llama active"]
    latency_means = [model_remote["metrics"]["latency_mean_sec"], model_phone_load["metrics"]["latency_mean_sec"]]
    latency_stdevs = [statistics.stdev(remote_latencies), statistics.stdev(phone_load_latencies)]
    ttft_means = [model_remote["metrics"]["ttft_mean_sec"], model_phone_load["metrics"]["ttft_mean_sec"]]
    x = np.arange(2)
    width = 0.36
    b1 = ax.bar(x - width / 2, latency_means, width, yerr=latency_stdevs, capsize=4, label="Total latency", color="#ffb300")
    b2 = ax.bar(x + width / 2, ttft_means, width, label="TTFT", color="#42a5f5")
    annotate(ax, b1, "{:.3f}")
    annotate(ax, b2, "{:.3f}")
    ax.set_xticks(x, model_labels)
    ax.set_ylabel("Seconds")
    ax.set_title(f"B. Remote-model latency: +{latency_difference * 1000:.1f} ms (+{latency_difference / latency_means[0] * 100:.1f}%)")
    ax.legend(frameon=True)

    ax = axes[1, 0]
    nodes = np.array(values(graph_growth, "node_count"))
    baseline_pss = statistics.mean(values(graph_baseline, "process_pss_bytes")) / MIB
    ax.plot(nodes, np.array(values(graph_growth, "process_pss_bytes")) / MIB - baseline_pss, marker="o", markersize=3, label="Process PSS over baseline")
    ax.plot(nodes, np.array(values(graph_growth, "live_graph_python_bytes")) / MIB, label="Live graph deep size")
    ax.plot(nodes, np.array(values(graph_growth, "immutable_snapshot_python_bytes")) / MIB, label="Retained snapshot deep size")
    ax.plot(nodes, np.array(values(graph_growth, "total_graph_python_bytes")) / MIB, linestyle="--", label="Live + snapshot logical bytes")
    ax.set_xlabel("Graph nodes")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title("C. 10k graph growth in Termux")
    ax.legend(frameon=True, fontsize=8)

    ax = axes[1, 1]
    total = exploration_agg["probe_latency_mean_sec_mean"]
    pieces = [
        exploration_agg["selection_latency_mean_sec_mean"],
        exploration_agg["action_latency_mean_sec_mean"],
        exploration_agg["screenshot_latency_mean_sec_mean"],
        exploration_agg["a11y_tree_latency_mean_sec_mean"],
    ]
    recovery = max(0.0, total - sum(pieces))
    names = ["Selection", "ADB action", "Screenshot", "A11y tree", "Recovery/other"]
    colors = ["#7e57c2", "#3f7fe8", "#11acc3", "#16ad9c", "#b8c4d2"]
    left = 0.0
    for name, part, color in zip(names, pieces + [recovery], colors):
        ax.barh([0], [part], left=left, label=f"{name}: {part:.2f}s", color=color)
        left += part
    ax.axvline(total, color="#172033", marker="D", linewidth=2, label=f"Measured total: {total:.2f}s")
    ax.set_yticks([0], ["Mean probe"])
    ax.set_xlabel("Seconds")
    ax.set_title(f"D. Exploration: {exploration_agg['verified_probe_count_mean']:.0f} probes / 60s")
    ax.legend(frameon=True, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))

    overview_path = output_dir / "isolated_profiling_overview.png"
    fig.savefig(overview_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle("Physical-phone workload traces and confounders", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    condition_names = ["llama idle", "remote 8084 requests", "phone llama workload"]
    condition_trials = ["model_idle_r01", "model_active_r01", "model_active_llamacpp_r01"]
    llama_pss = [float(mean(phone[t], "process_llama_pss_kb") or 0) / 1024.0 for t in condition_trials]
    bars = ax.bar(condition_names, llama_pss, color=["#9575cd", "#7e57c2", "#5e35b1"])
    ax.set_ylim(min(llama_pss) - 3.0, max(llama_pss) + 3.0)
    ax.set_ylabel("llama process-group PSS (MiB)")
    ax.set_title("A. Model memory is dominated by loaded weights")
    annotate(ax, bars)

    ax = axes[0, 1]
    elapsed = np.array(values(graph_growth, "timestamp")) - values(graph_growth, "timestamp")[0]
    ax.plot(elapsed, np.array(values(graph_growth, "process_pss_bytes")) / MIB, marker="o", markersize=3, label="Termux graph PSS")
    ax.set_xlabel("Graph-growth elapsed time (s)")
    ax.set_ylabel("Process PSS (MiB)")
    ax.set_title("B. PSS rises monotonically during graph growth")
    ax.legend(frameon=True)

    ax = axes[1, 0]
    system_exploration = phone["exploration_cold_r01"]
    run_indices = []
    run_latency = []
    run_temperature = []
    for run in exploration["runs"]:
        start = run["exploration_started_timestamp"]
        end = run["exploration_ended_timestamp"]
        overlap = [row for row in system_exploration if start <= float(row["timestamp"]) <= end]
        run_indices.append(run["run_index"])
        run_latency.append(run["probe_latency_mean_sec"])
        run_temperature.append(float(mean(overlap, "battery_temperature_c") or np.nan))
    line1 = ax.plot(run_indices, run_latency, marker="o", color="#ef5350", label="Probe latency")[0]
    ax.set_xlabel("Exploration repeat")
    ax.set_ylabel("Mean probe latency (s)", color="#ef5350")
    twin = ax.twinx()
    line2 = twin.plot(run_indices, run_temperature, marker="s", color="#fb8c00", label="Battery temperature")[0]
    twin.set_ylabel("Temperature (°C)", color="#fb8c00")
    ax.set_title("C. Phone warmed up, but probe latency did not rise monotonically")
    ax.legend([line1, line2], ["Probe latency", "Temperature"], frameon=True, loc="best")

    ax = axes[1, 1]
    graph_wall = float(graph_growth[-1]["timestamp"]) - float(graph_growth[0]["timestamp"])
    graph_costs = {
        "Pure inserts": graph_metrics["insertion_total_sec"],
        "Deep-size scans": sum(values(graph_growth, "live_size_measure_sec")),
        "Serialization scans": sum(values(graph_growth, "serialized_size_measure_sec")),
        "Snapshot build": sum(values(graph_growth, "snapshot_build_sec")),
        "Snapshot deep-size": sum(values(graph_growth, "snapshot_size_measure_sec")),
        "GC": sum(values(graph_growth, "gc_sec")),
    }
    bars = ax.bar(graph_costs.keys(), graph_costs.values(), color=["#26a69a", "#42a5f5", "#5c6bc0", "#ffb300", "#fb8c00", "#90a4ae"])
    ax.set_ylabel("Accumulated seconds")
    ax.set_title(f"D. 10k growth wall time {graph_wall:.1f}s; profiler scans dominate")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    annotate(ax, bars, "{:.2f}")

    traces_path = output_dir / "workload_traces_and_confounders.png"
    fig.savefig(traces_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = {
        "paths": {"overview": str(overview_path), "traces": str(traces_path)},
        "component_memory_mib": {
            "llama_loaded_idle_pss": llama_idle_mib,
            "llama_phone_active_pss": llama_phone_active_mib,
            "llama_active_minus_idle_pss": llama_phone_active_mib - llama_idle_mib,
            "graph_10k_process_pss_over_baseline": graph_pss_increment_mib,
            "graph_10k_live_plus_snapshot_logical": graph_logical_mib,
            "exploration_target_app_pss_delta_mean": exploration_app_delta_mib,
            "exploration_host_controller_rss_delta_mean": exploration_host_delta_mib,
        },
        "remote_model_comparison": {
            "no_phone_workload_latency_mean_sec": statistics.mean(remote_latencies),
            "phone_llama_workload_latency_mean_sec": statistics.mean(phone_load_latencies),
            "difference_sec": latency_difference,
            "difference_percent": latency_difference / statistics.mean(remote_latencies) * 100.0,
            "bootstrap_95_percent_ci_difference_sec": latency_difference_ci,
            "warning": "8084 is a remote model; this comparison is not phone-model inference latency.",
        },
        "graph": {
            "reported_elapsed_including_hold_sec": graph_metrics["elapsed_sec"],
            "growth_wall_sec": graph_wall,
            "hold_sec": graph_summary["config"]["hold_sec"],
            "pure_insertion_total_sec": graph_metrics["insertion_total_sec"],
            "snapshot_build_total_sec": sum(values(graph_growth, "snapshot_build_sec")),
            "profiler_scan_total_sec": (
                sum(values(graph_growth, "live_size_measure_sec"))
                + sum(values(graph_growth, "serialized_size_measure_sec"))
                + sum(values(graph_growth, "snapshot_size_measure_sec"))
            ),
        },
        "exploration": {
            "repeat_count": exploration_agg["repeat_count"],
            "verified_probes_mean": exploration_agg["verified_probe_count_mean"],
            "probe_latency_mean_sec": exploration_agg["probe_latency_mean_sec_mean"],
            "a11y_latency_mean_sec": exploration_agg["a11y_tree_latency_mean_sec_mean"],
            "time_to_foreground_mean_sec": exploration_agg["time_to_foreground_sec_mean"],
            "time_to_stable_ui_mean_sec": exploration_agg["time_to_stable_ui_sec_mean"],
            "target_app_pss_delta_mean_mib": exploration_app_delta_mib,
        },
    }
    report_path = output_dir / "analysis_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
