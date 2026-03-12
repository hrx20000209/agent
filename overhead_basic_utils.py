import time
from typing import Dict, List


def begin_cpu_sample() -> Dict[str, float]:
    return {
        "wall_start": time.time(),
        "cpu_start": time.process_time(),
    }


def end_cpu_sample(token: Dict[str, float]) -> Dict[str, float]:
    wall_s = max(1e-9, time.time() - float(token["wall_start"]))
    cpu_s = max(0.0, time.process_time() - float(token["cpu_start"]))
    return {
        "wall_ms": wall_s * 1000.0,
        "cpu_ms": cpu_s * 1000.0,
        "cpu_pct_single_core": (cpu_s / wall_s) * 100.0,
    }


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
