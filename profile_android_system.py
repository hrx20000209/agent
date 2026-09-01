#!/usr/bin/env python3
"""Continuously sample physical-Android memory and selected process memory.

Run this script on the host computer.  It only reads the connected phone via
ADB and refuses emulators.  Workload scripts may run on the host, in Termux,
or in the foreground Android app; timestamps are epoch seconds so their output
can be aligned after the experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evaluate_graph_memory_adb import _require_physical_device, _run_adb
from main import _default_adb_path, ensure_adb_ready, resolve_adb_prefix


MEMINFO_KEYS = (
    "MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapCached",
    "Active", "Inactive", "Dirty", "Writeback", "SwapTotal", "SwapFree",
)
VMSTAT_KEYS = ("pgfault", "pgmajfault", "pswpin", "pswpout", "oom_kill")


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_labeled_regex(raw: str) -> Tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected LABEL=REGEX")
    label, pattern = raw.split("=", 1)
    if not label.strip() or not pattern.strip():
        raise argparse.ArgumentTypeError("LABEL and REGEX must both be non-empty")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"invalid regex: {exc}") from exc
    return label.strip(), pattern.strip()


def parse_labeled_package(raw: str) -> Tuple[str, str]:
    if "=" in raw:
        label, package = raw.split("=", 1)
    else:
        label = package = raw
    if not label.strip() or not package.strip():
        raise argparse.ArgumentTypeError("expected PACKAGE or LABEL=PACKAGE")
    return label.strip(), package.strip()


def parse_key_values(text: str, keys: Iterable[str]) -> Dict[str, int]:
    wanted = set(keys)
    return {
        key: int(value)
        for key, value in re.findall(r"^([A-Za-z_()]+):?\s+(\d+)", text, re.M)
        if key in wanted
    }


def parse_pressure(text: str, prefix: str) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for scope, avg10, avg60, avg300, total in re.findall(
        r"^(some|full)\s+avg10=([0-9.]+)\s+avg60=([0-9.]+)\s+avg300=([0-9.]+)\s+total=(\d+)",
        text,
        re.M,
    ):
        values.update({
            f"{prefix}_{scope}_avg10": float(avg10),
            f"{prefix}_{scope}_avg60": float(avg60),
            f"{prefix}_{scope}_avg300": float(avg300),
            f"{prefix}_{scope}_total_us": int(total),
        })
    return values


def list_processes(adb_path: str, timeout: float) -> List[Tuple[int, str]]:
    text = _run_adb(adb_path, "shell", "ps", "-A", "-o", "PID,NAME,ARGS", timeout=timeout)
    if not text.strip():
        text = _run_adb(adb_path, "shell", "ps", "-A", timeout=timeout)
    processes: List[Tuple[int, str]] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)\s+(.+)$", line)
        if match:
            processes.append((int(match.group(1)), match.group(2).strip()))
            continue
        columns = line.split()
        if len(columns) >= 2:
            pid_index = next((i for i, value in enumerate(columns) if value.isdigit()), None)
            if pid_index is not None and pid_index + 1 < len(columns):
                processes.append((int(columns[pid_index]), " ".join(columns[pid_index + 1:])))
    return processes


def pidof(adb_path: str, package: str, timeout: float) -> List[int]:
    text = _run_adb(adb_path, "shell", "pidof", package, timeout=timeout)
    return [int(value) for value in re.findall(r"\b\d+\b", text)]


def parse_process_meminfo(text: str) -> Dict[str, Optional[int]]:
    def first(patterns: Iterable[str]) -> Optional[int]:
        for pattern in patterns:
            match = re.search(pattern, text, re.M | re.I)
            if match:
                return int(match.group(1).replace(",", ""))
        return None

    return {
        "pss_kb": first((r"TOTAL PSS:\s*([\d,]+)", r"^\s*TOTAL\s+([\d,]+)")),
        "rss_kb": first((r"TOTAL RSS:\s*([\d,]+)", r"^\s*TOTAL\s+[\d,]+(?:\s+[\d,]+){5}\s+([\d,]+)")),
        "swap_pss_kb": first((r"TOTAL SWAP PSS:\s*([\d,]+)",)),
    }


def process_group_memory(adb_path: str, pids: Iterable[int], timeout: float) -> Dict[str, Any]:
    unique_pids = sorted(set(int(pid) for pid in pids))
    members = []
    for pid in unique_pids:
        text = _run_adb(adb_path, "shell", "dumpsys", "meminfo", str(pid), timeout=timeout)
        memory = parse_process_meminfo(text)
        if any(value is not None for value in memory.values()):
            members.append({"pid": pid, **memory})
    return {
        "pids": unique_pids,
        "member_count": len(members),
        "pss_kb": sum(member["pss_kb"] or 0 for member in members) if members else None,
        "rss_kb": sum(member["rss_kb"] or 0 for member in members) if members else None,
        "swap_pss_kb": sum(member["swap_pss_kb"] or 0 for member in members) if members else None,
        "members": members,
    }


def battery_temperature_c(adb_path: str, timeout: float) -> Optional[float]:
    text = _run_adb(adb_path, "shell", "dumpsys", "battery", timeout=timeout)
    match = re.search(r"^\s*temperature:\s*(-?\d+)", text, re.M)
    return int(match.group(1)) / 10.0 if match else None


def foreground_package(adb_path: str, timeout: float) -> str:
    text = _run_adb(adb_path, "shell", "dumpsys", "window", "windows", timeout=timeout)
    match = re.search(r"mCurrentFocus=.*?\s([A-Za-z0-9_.]+)/", text)
    return match.group(1) if match else ""


def sample_phone(
    adb_path: str,
    timeout: float,
    packages: List[Tuple[str, str]],
    process_patterns: List[Tuple[str, str]],
) -> Dict[str, Any]:
    started = time.perf_counter()
    meminfo = _run_adb(adb_path, "shell", "cat", "/proc/meminfo", timeout=timeout)
    vmstat = _run_adb(adb_path, "shell", "cat", "/proc/vmstat", timeout=timeout)
    memory_pressure = _run_adb(adb_path, "shell", "cat", "/proc/pressure/memory", timeout=timeout)
    cpu_pressure = _run_adb(adb_path, "shell", "cat", "/proc/pressure/cpu", timeout=timeout)
    processes = list_processes(adb_path, timeout) if process_patterns else []
    groups: Dict[str, Any] = {}
    for label, package in packages:
        groups[label] = process_group_memory(adb_path, pidof(adb_path, package, timeout), timeout)
        groups[label]["selector"] = {"type": "package", "value": package}
    for label, pattern in process_patterns:
        regex = re.compile(pattern, re.I)
        matched = [pid for pid, description in processes if regex.search(description)]
        groups[label] = process_group_memory(adb_path, matched, timeout)
        groups[label]["selector"] = {"type": "process_regex", "value": pattern}
    values: Dict[str, Any] = {
        "timestamp": time.time(),
        "sample_duration_sec": time.perf_counter() - started,
        "memory_kb": parse_key_values(meminfo, MEMINFO_KEYS),
        "vmstat": {
            key: value
            for key, value in parse_key_values(vmstat, VMSTAT_KEYS).items()
        },
        "pressure": {
            **parse_pressure(memory_pressure, "memory"),
            **parse_pressure(cpu_pressure, "cpu"),
        },
        "battery_temperature_c": battery_temperature_c(adb_path, timeout),
        "foreground_package": foreground_package(adb_path, timeout),
        "process_groups": groups,
    }
    values["sample_duration_sec"] = time.perf_counter() - started
    return values


def flatten(sample: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "timestamp": sample["timestamp"],
        "elapsed_sec": sample.get("elapsed_sec"),
        "sample_duration_sec": sample["sample_duration_sec"],
        "battery_temperature_c": sample.get("battery_temperature_c"),
        "foreground_package": sample.get("foreground_package"),
    }
    row.update({f"mem_{key}_kb": value for key, value in sample["memory_kb"].items()})
    row.update({f"vmstat_{key}": value for key, value in sample["vmstat"].items()})
    row.update(sample["pressure"])
    for label, group in sample["process_groups"].items():
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_").lower()
        row[f"process_{safe}_pid_count"] = len(group["pids"])
        row[f"process_{safe}_pss_kb"] = group["pss_kb"]
        row[f"process_{safe}_rss_kb"] = group["rss_kb"]
        row[f"process_{safe}_swap_pss_kb"] = group["swap_pss_kb"]
    return row


def numeric_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"sample_count": len(rows)}
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        summary[key] = {
            "start": values[0],
            "end": values[-1],
            "delta": values[-1] - values[0],
            "mean": statistics.mean(values),
            "peak": max(values),
            "p95": percentile(values, 0.95),
        }
    return summary


def require_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb_serial", required=True)
    parser.add_argument("--adb_path", default=_default_adb_path())
    parser.add_argument("--adb_port", type=int, default=5037)
    parser.add_argument("--adb_timeout_sec", type=float, default=12.0)
    parser.add_argument("--duration_sec", type=float, default=60.0, help="0 means run until Ctrl-C.")
    parser.add_argument("--interval_sec", type=float, default=1.0)
    parser.add_argument("--package", action="append", default=[], type=parse_labeled_package,
                        help="PACKAGE or LABEL=PACKAGE; repeat for multiple apps.")
    parser.add_argument("--process", action="append", default=[], type=parse_labeled_regex,
                        help="LABEL=REGEX matched against Android ps output; repeatable.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if args.adb_serial.lower().startswith("emulator-"):
        raise RuntimeError(f"Refusing emulator serial: {args.adb_serial}")
    args.adb_path = resolve_adb_prefix(args.adb_path, args.adb_serial, args.adb_port)
    ensure_adb_ready(args.adb_path)
    identity = _require_physical_device(args.adb_path, allow_emulator=False)
    output_dir = Path(args.output_dir).expanduser().resolve()
    require_empty_output_dir(output_dir)
    config = {**vars(args), "device": identity, "output_dir": str(output_dir)}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    jsonl_path = output_dir / "samples.jsonl"
    samples: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    started_wall = time.time()
    deadline = None if args.duration_sec <= 0 else time.perf_counter() + args.duration_sec
    try:
        with jsonl_path.open("w", encoding="utf-8") as stream:
            while deadline is None or time.perf_counter() < deadline:
                sample = sample_phone(
                    args.adb_path,
                    max(1.0, args.adb_timeout_sec),
                    list(args.package),
                    list(args.process),
                )
                sample["elapsed_sec"] = sample["timestamp"] - started_wall
                row = flatten(sample)
                samples.append(sample)
                flat_rows.append(row)
                stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stream.flush()
                remaining = max(0.0, args.interval_sec - sample["sample_duration_sec"])
                if remaining:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass

    if flat_rows:
        fields = sorted({key for row in flat_rows for key in row})
        with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat_rows)
    summary = {
        "config": config,
        "metrics": numeric_summary(flat_rows),
        "started_timestamp": started_wall,
        "ended_timestamp": time.time(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "sample_count": len(samples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
