#!/usr/bin/env python3
"""Plot physical-device graph-memory system evaluation results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    0.0: "#64748B",
    1.0: "#22C55E",
    8.0: "#F59E0B",
    32.0: "#EF4444",
}


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else math.nan


def stdev(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def regression(xs, ys):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    r2 = 1.0 - np.sum((y - predicted) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return float(slope), float(intercept), float(r2)


def style_axis(ax):
    ax.grid(axis="y", color="#CBD5E1", alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_dashboard(root, results, groups, budgets, labels, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(len(budgets))
    colors = [COLORS.get(b, "#2563EB") for b in budgets]

    live = [mean(groups[b], "graph_python_bytes") / 2**20 for b in budgets]
    snapshot = [mean(groups[b], "graph_snapshot_python_bytes") / 2**20 for b in budgets]
    axes[0, 0].bar(x, live, color=colors, label="Live graph")
    axes[0, 0].bar(x, snapshot, bottom=live, color=colors, alpha=0.42, label="Immutable snapshot")
    for i, total in enumerate(np.asarray(live) + np.asarray(snapshot)):
        axes[0, 0].text(i, total + max(total * 0.025, 0.25), f"{total:.1f} MB", ha="center", fontsize=9)
    axes[0, 0].set_title("Actual graph memory at exploration start")
    axes[0, 0].set_ylabel("Python heap (MiB)")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].legend(frameon=False)
    style_axis(axes[0, 0])

    counts = [mean(groups[b], "exploration_probe_count") for b in budgets]
    count_err = [stdev(groups[b], "exploration_probe_count") for b in budgets]
    rates = [mean(groups[b], "probes_per_second") for b in budgets]
    axes[0, 1].bar(x, counts, yerr=count_err, capsize=5, color=colors, alpha=0.88)
    for i, value in enumerate(counts):
        axes[0, 1].text(i, value + 0.22, f"{value:.1f}", ha="center", fontsize=9)
    axes[0, 1].set_title("Exploration throughput in a fixed 60 s window")
    axes[0, 1].set_ylabel("Completed probes / 60 s")
    axes[0, 1].set_xticks(x, labels)
    style_axis(axes[0, 1])
    rate_axis = axes[0, 1].twinx()
    rate_axis.plot(x, rates, color="#0F172A", marker="o", linewidth=2, label="Probes/s")
    rate_axis.set_ylabel("Probes/s")
    rate_axis.set_ylim(0, max(rates) * 1.35)
    rate_axis.spines["top"].set_visible(False)

    width = 0.36
    total_latency = [mean(groups[b], "probe_latency_mean_sec") for b in budgets]
    total_err = [stdev(groups[b], "probe_latency_mean_sec") for b in budgets]
    steady = [mean(groups[b], "steady_probe_latency_mean_sec") for b in budgets]
    steady_err = [stdev(groups[b], "steady_probe_latency_mean_sec") for b in budgets]
    axes[1, 0].bar(x - width / 2, total_latency, width, yerr=total_err, capsize=4,
                   color=colors, label="All probes")
    axes[1, 0].bar(x + width / 2, steady, width, yerr=steady_err, capsize=4,
                   color=colors, alpha=0.45, label="Excluding first probe")
    axes[1, 0].set_title("Mean latency of one exploration probe")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].legend(frameon=False)
    style_axis(axes[1, 0])

    components = [
        ("Selection", "selection_latency_mean_sec", "#8B5CF6"),
        ("ADB action", "action_latency_mean_sec", "#3B82F6"),
        ("Screenshot", "screenshot_latency_mean_sec", "#06B6D4"),
        ("UI tree", "a11y_tree_latency_mean_sec", "#14B8A6"),
    ]
    bottom = np.zeros(len(budgets))
    for name, key, color in components:
        values = np.array([mean(groups[b], key) for b in budgets])
        axes[1, 1].bar(x, values, bottom=bottom, label=name, color=color)
        bottom += values
    residual = np.maximum(0, np.asarray(total_latency) - bottom)
    axes[1, 1].bar(x, residual, bottom=bottom, label="Recovery / other", color="#CBD5E1")
    axes[1, 1].plot(x, total_latency, color="#0F172A", marker="D", linewidth=1.8, label="Measured total")
    axes[1, 1].set_title("Probe latency decomposition")
    axes[1, 1].set_ylabel("Seconds")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].legend(frameon=False, ncol=2, fontsize=8)
    style_axis(axes[1, 1])

    device = root.get("device", {})
    fig.suptitle(
        f"Physical-phone graph-memory evaluation — {device.get('model', 'device')} "
        f"({len(results)} runs, {len(groups[budgets[0]])} repeats per budget)",
        fontsize=16,
        fontweight="bold",
    )
    path = out_dir / "system_evaluation_dashboard.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_scatter(results, budgets, out_dir):
    graph_mb = [row["graph_total_python_bytes"] / 2**20 for row in results]
    latency = [row["probe_latency_mean_sec"] for row in results]
    throughput = [row["probes_per_second"] for row in results]
    snapshot = [row["snapshot_build_sec"] for row in results]
    point_colors = [COLORS.get(float(row["budget_mb"]), "#2563EB") for row in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    panels = [
        (latency, "Probe latency (s)", "Latency", "s/MB"),
        (throughput, "Probe throughput (probes/s)", "Throughput", "probes/s/MB"),
        (snapshot, "Snapshot build time (s)", "Snapshot construction", "s/MB"),
    ]
    for ax, (ys, ylabel, title, slope_unit) in zip(axes, panels):
        ax.scatter(graph_mb, ys, c=point_colors, s=58, alpha=0.82, edgecolor="white", linewidth=0.7)
        slope, intercept, r2 = regression(graph_mb, ys)
        line_x = np.linspace(0, max(graph_mb) * 1.03, 200)
        ax.plot(line_x, intercept + slope * line_x, color="#0F172A", linewidth=2)
        ax.text(
            0.04, 0.94, f"slope = {slope:+.4f} {slope_unit}\n$R^2$ = {r2:.3f}",
            transform=ax.transAxes, va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD5E1"},
        )
        ax.set_title(title)
        ax.set_xlabel("Actual live + snapshot graph memory (MiB)")
        ax.set_ylabel(ylabel)
        style_axis(ax)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[b], markersize=8,
                   label="Empty" if b == 0 else f"{b:g} MB budget")
        for b in budgets
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    path = out_dir / "system_effect_scatter.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_pressure(root, groups, budgets, labels, out_dir):
    results = root["results"]
    x = np.arange(len(budgets))
    colors = [COLORS.get(b, "#2563EB") for b in budgets]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    rss_start = [mean(groups[b], "host_rss_start_kb") / 1024 for b in budgets]
    rss_end = [mean(groups[b], "host_rss_end_kb") / 1024 for b in budgets]
    axes[0].bar(x - 0.18, rss_start, 0.36, color=colors, alpha=0.5, label="Start")
    axes[0].bar(x + 0.18, rss_end, 0.36, color=colors, label="End")
    axes[0].set_title("Host evaluator RSS")
    axes[0].set_ylabel("MiB")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    available = [mean(groups[b], "device_mem_available_delta_kb") / 1024 for b in budgets]
    available_err = [stdev(groups[b], "device_mem_available_delta_kb") / 1024 for b in budgets]
    axes[1].bar(x, available, yerr=available_err, capsize=5, color=colors)
    axes[1].axhline(0, color="#0F172A", linewidth=1)
    axes[1].set_title("Phone MemAvailable change")
    axes[1].set_ylabel("MiB (end − start)")
    axes[1].set_xticks(x, labels)
    style_axis(axes[1])

    psi = [mean(groups[b], "device_psi_full_total_delta_us") / 1000 for b in budgets]
    psi_err = [stdev(groups[b], "device_psi_full_total_delta_us") / 1000 for b in budgets]
    axes[2].bar(x, psi, yerr=psi_err, capsize=5, color=colors)
    axes[2].set_title("Phone full memory-stall pressure")
    axes[2].set_ylabel("PSI full stall increase (ms)")
    axes[2].set_xticks(x, labels)
    style_axis(axes[2])

    fig.suptitle("Host memory footprint and phone-side pressure signals", fontsize=15, fontweight="bold")
    path = out_dir / "memory_pressure_signals.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    source = args.result_dir / "summary.json"
    root = json.loads(source.read_text(encoding="utf-8"))
    results = root["results"]
    budgets = sorted({float(row["budget_mb"]) for row in results})
    groups = {budget: [row for row in results if float(row["budget_mb"]) == budget] for budget in budgets}
    labels = ["Empty" if budget == 0 else f"{budget:g} MB" for budget in budgets]
    out_dir = args.result_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        save_dashboard(root, results, groups, budgets, labels, out_dir),
        save_scatter(results, budgets, out_dir),
        save_pressure(root, groups, budgets, labels, out_dir),
    ]
    print(json.dumps({"plots": [str(path.resolve()) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
