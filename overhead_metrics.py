import os
import re
import subprocess
import time
from typing import Any, Dict, Optional


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _run_adb_cat_int(adb_path: str, file_path: str) -> Optional[int]:
    try:
        proc = subprocess.run(
            [adb_path, "shell", "cat", file_path],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if proc.returncode != 0:
            return None
        txt = (proc.stdout or "").strip()
        m = re.search(r"-?\d+", txt)
        if not m:
            return None
        return int(m.group(0))
    except Exception:
        return None


def _run_adb_text(adb_path: str, shell_cmd: str) -> str:
    try:
        proc = subprocess.run(
            [adb_path, "shell", shell_cmd],
            capture_output=True,
            text=True,
            timeout=2.5,
        )
        if proc.returncode != 0:
            return ""
        return (proc.stdout or "").strip()
    except Exception:
        return ""


def _parse_int_by_patterns(text: str, patterns: list[str]) -> Optional[int]:
    if not text:
        return None
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None


def read_android_power(adb_path: str = "adb") -> Dict[str, Any]:
    current_candidates = [
        "/sys/class/power_supply/battery/current_now",
        "/sys/class/power_supply/Battery/current_now",
        "/sys/class/power_supply/bms/current_now",
    ]
    voltage_candidates = [
        "/sys/class/power_supply/battery/voltage_now",
        "/sys/class/power_supply/Battery/voltage_now",
        "/sys/class/power_supply/bms/voltage_now",
    ]

    current_ua = None
    current_src = None
    for p in current_candidates:
        v = _run_adb_cat_int(adb_path, p)
        if v is not None:
            current_ua = v
            current_src = p
            break

    voltage_uv = None
    voltage_src = None
    for p in voltage_candidates:
        v = _run_adb_cat_int(adb_path, p)
        if v is not None:
            voltage_uv = v
            voltage_src = p
            break

    # Fallback: parse dumpsys when sysfs nodes are inaccessible.
    if current_ua is None or voltage_uv is None:
        txt_props = _run_adb_text(adb_path, "dumpsys batteryproperties")
        txt_batt = _run_adb_text(adb_path, "dumpsys battery")
        merged = (txt_props + "\n" + txt_batt).strip()
        if current_ua is None:
            current_ua = _parse_int_by_patterns(
                merged,
                [
                    r"batteryCurrentNow\s*[:=]\s*(-?\d+)",
                    r"current[_\s]?now\s*[:=]\s*(-?\d+)",
                    r"current\s*[:=]\s*(-?\d+)",
                ],
            )
            if current_ua is not None:
                current_src = "dumpsys"

        if voltage_uv is None:
            voltage_raw = _parse_int_by_patterns(
                merged,
                [
                    r"batteryVoltage\s*[:=]\s*(-?\d+)",
                    r"voltage[_\s]?now\s*[:=]\s*(-?\d+)",
                    r"voltage\s*[:=]\s*(-?\d+)",
                ],
            )
            if voltage_raw is not None:
                # dumpsys battery often reports mV; sysfs is usually uV.
                voltage_uv = int(voltage_raw * 1000) if abs(voltage_raw) < 100000 else int(voltage_raw)
                voltage_src = "dumpsys"

    # Some devices report sentinel values (e.g., -99 / 0) when unsupported.
    if current_ua is not None and abs(int(current_ua)) <= 100:
        current_ua = None
    if voltage_uv is not None and int(voltage_uv) <= 0:
        voltage_uv = None

    power_w = None
    if current_ua is not None and voltage_uv is not None:
        power_w = abs(float(current_ua) * float(voltage_uv)) / 1e12

    return {
        "current_ua": current_ua,
        "voltage_uv": voltage_uv,
        "power_w": power_w,
        "current_source": current_src,
        "voltage_source": voltage_src,
    }


def begin_overhead_sample(adb_path: str = "adb", collect_power: bool = True) -> Dict[str, Any]:
    token = {
        "wall_start": time.time(),
        "cpu_start": time.process_time(),
        "cpu_count": max(1, int(os.cpu_count() or 1)),
        "power_start": read_android_power(adb_path) if collect_power else None,
    }
    return token


def end_overhead_sample(token: Dict[str, Any], adb_path: str = "adb", collect_power: bool = True) -> Dict[str, Any]:
    wall_end = time.time()
    cpu_end = time.process_time()

    wall_s = max(1e-8, wall_end - float(token["wall_start"]))
    cpu_s = max(0.0, cpu_end - float(token["cpu_start"]))
    cpu_pct_single_core = 100.0 * (cpu_s / wall_s)
    cpu_pct_system = cpu_pct_single_core / float(token["cpu_count"])

    power_start = token.get("power_start")
    power_end = read_android_power(adb_path) if collect_power else None

    p0 = _safe_float((power_start or {}).get("power_w")) if power_start else None
    p1 = _safe_float((power_end or {}).get("power_w")) if power_end else None
    power_avg_w = None
    if p0 is not None and p1 is not None:
        power_avg_w = 0.5 * (p0 + p1)
    elif p0 is not None:
        power_avg_w = p0
    elif p1 is not None:
        power_avg_w = p1

    return {
        "wall_ms": wall_s * 1000.0,
        "process_cpu_time_ms": cpu_s * 1000.0,
        "process_cpu_pct_single_core": cpu_pct_single_core,
        "process_cpu_pct_system": cpu_pct_system,
        "power_start": power_start,
        "power_end": power_end,
        "power_avg_w": power_avg_w,
    }
