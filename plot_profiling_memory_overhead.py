#!/usr/bin/env python3
"""Plot measured exploration and inference memory overhead.

The figure deliberately separates the Windows HTTP client from the Termux
llama.cpp server: the latter was not captured by the current profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MIB = 1024.0 * 1024.0


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def annotate_bars(ax, bars, fmt="{:.1f}"):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiling_dir", default="evaluation_results/profilng")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.profiling_dir).resolve()
    exploration = load_json(root / "ctrip_exploration_profile_60s_v1" / "summary.json")
    model_summary = load_json(root / "model_inference_profile_v1" / "summary.json")
    model_memory = load_jsonl(root / "model_inference_profile_v1" / "memory.jsonl")
    runs = exploration["runs"]

    app_delta = np.array([run["foreground_app_pss_delta_kb"] / 1024.0 for run in runs])
    available_consumed = np.array([-run["device_mem_available_delta_kb"] / 1024.0 for run in runs])
    host_delta = np.array([run["host_rss_delta_kb"] / 1024.0 for run in runs])
    app_start = np.array([
        run["device_memory_start"]["foreground_app_pss_kb"] / 1024.0 for run in runs
    ])
    app_end = np.array([
        run["device_memory_end"]["foreground_app_pss_kb"] / 1024.0 for run in runs
    ])

    first_ts = model_memory[0]["timestamp"]
    model_time = np.array([row["timestamp"] - first_ts for row in model_memory])
    client_rss = np.array([row["client"]["rss_bytes"] / MIB for row in model_memory])
    client_uss = np.array([row["client"]["uss_bytes"] / MIB for row in model_memory])
    client_rss_delta = client_rss[-1] - client_rss[0]
    client_uss_delta = client_uss[-1] - client_uss[0]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        "Measured memory overhead — physical-phone exploration vs. model-inference client",
        fontsize=19,
        fontweight="bold",
    )

    # A. Per-repeat exploration deltas.
    ax = axes[0, 0]
    x = np.arange(1, len(runs) + 1)
    width = 0.25
    b1 = ax.bar(x - width, app_delta, width, label="Ctrip app PSS increase", color="#ef5350")
    b2 = ax.bar(x, available_consumed, width, label="Device MemAvailable consumed", color="#ffb300")
    b3 = ax.bar(x + width, host_delta, width, label="Host controller RSS increase", color="#5c6bc0")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    annotate_bars(ax, b3)
    ax.set_title("A. Exploration overhead in each 60 s repeat")
    ax.set_xlabel("Repeat")
    ax.set_ylabel("Memory change (MiB; positive = more memory used)")
    ax.set_xticks(x)
    ax.legend(frameon=True, fontsize=9)

    # B. Paired app PSS changes show consistency across repeats.
    ax = axes[0, 1]
    for index, (start, end) in enumerate(zip(app_start, app_end), start=1):
        ax.plot([0, 1], [start, end], color="#ef5350", alpha=0.55, linewidth=2)
        ax.scatter([0, 1], [start, end], color=["#78909c", "#ef5350"], s=45, zorder=3)
        ax.annotate(f"R{index}", (1, end), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
    mean_start = float(app_start.mean())
    mean_end = float(app_end.mean())
    ax.plot([0, 1], [mean_start, mean_end], color="#b71c1c", linewidth=4, label="Mean")
    ax.set_xticks([0, 1], ["Start", "End"])
    ax.set_xlim(-0.2, 1.25)
    ax.set_ylabel("Ctrip foreground PSS (MiB)")
    ax.set_title(f"B. Ctrip PSS consistently grows: +{app_delta.mean():.1f} ± {app_delta.std(ddof=1):.1f} MiB")
    ax.legend(frameon=True)

    # C. What the model profiler actually captured: the Windows client only.
    ax = axes[1, 0]
    ax.plot(model_time, client_rss, color="#1565c0", linewidth=2, label="HTTP client RSS")
    ax.plot(model_time, client_uss, color="#00897b", linewidth=2, label="HTTP client USS")
    ax.scatter([model_time[0], model_time[-1]], [client_rss[0], client_rss[-1]], color="#1565c0", s=35)
    ax.set_title(
        "C. Model-inference profile: Windows HTTP client only\n"
        f"30 requests, client RSS Δ {client_rss_delta:+.2f} MiB; USS Δ {client_uss_delta:+.2f} MiB"
    )
    ax.set_xlabel("Elapsed inference-profile time (s)")
    ax.set_ylabel("Process memory (MiB)")
    ax.legend(frameon=True)

    # D. Compact comparison, with a visible N/A for the missing model server.
    ax = axes[1, 1]
    labels = ["Ctrip app\n(exploration)", "Host controller\n(exploration)", "HTTP client\n(inference)"]
    values = [float(app_delta.mean()), float(host_delta.mean()), float(client_rss_delta)]
    errors = [float(app_delta.std(ddof=1)), float(host_delta.std(ddof=1)), 0.0]
    colors = ["#ef5350", "#5c6bc0", "#1565c0"]
    y = np.arange(len(labels))
    bars = ax.barh(y, values, xerr=errors, color=colors, alpha=0.9, capsize=4)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Measured memory increase (MiB, log scale)")
    ax.set_title("D. Measured components only — not a full model-vs-exploration comparison")
    for bar, value in zip(bars, values):
        ax.text(value * 1.08, bar.get_y() + bar.get_height() / 2, f"{value:.2f} MiB", va="center", fontsize=10)
    ax.text(
        0.03,
        -0.28,
        "Termux llama.cpp server: N/A",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="#c62828",
        bbox={"facecolor": "#ffebee", "edgecolor": "#ef9a9a", "boxstyle": "round,pad=0.4"},
    )
    candidate = ((model_summary.get("config") or {}).get("server_candidates") or [{}])[0]
    detected = candidate.get("description", "unknown process")
    ax.text(
        0.03,
        -0.43,
        f"Auto-detected process was '{detected}', not the Android/Termux model server.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )

    output = Path(args.output).resolve() if args.output else root / "plots" / "memory_overhead_dashboard.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({
        "output": str(output),
        "exploration_app_pss_delta_mean_mib": float(app_delta.mean()),
        "exploration_device_available_consumed_mean_mib": float(available_consumed.mean()),
        "exploration_host_rss_delta_mean_mib": float(host_delta.mean()),
        "model_client_rss_delta_mib": float(client_rss_delta),
        "model_server_memory_available": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
