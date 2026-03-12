import argparse
import json
from typing import List

from overhead_metrics import begin_overhead_sample, end_overhead_sample
from sim_no_adb_utils import compute_phash, verify_with_anchor_phash


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def run_module2(args):
    anchor_hash = compute_phash(args.screenshot_path)

    verify_ms_list: List[float] = []
    total_ms_list: List[float] = []
    cpu_pct_single_list: List[float] = []
    cpu_pct_system_list: List[float] = []
    power_w_list: List[float] = []
    ok_count = 0

    for itr in range(1, args.rounds + 1):
        collect_power_now = bool(args.collect_power) and ((itr - 1) % max(1, int(args.power_sample_every)) == 0)
        token = begin_overhead_sample(adb_path=args.adb_path, collect_power=collect_power_now)
        result = verify_with_anchor_phash(
            screenshot_path=args.screenshot_path,
            anchor_hash=anchor_hash,
            threshold=args.phash_threshold,
        )
        overhead = end_overhead_sample(token=token, adb_path=args.adb_path, collect_power=collect_power_now)

        verify_ms_list.append(float(result.get("verification_latency_ms", 0.0)))
        total_ms_list.append(float(overhead["wall_ms"]))
        cpu_pct_single_list.append(float(overhead["process_cpu_pct_single_core"]))
        cpu_pct_system_list.append(float(overhead["process_cpu_pct_system"]))
        if overhead.get("power_avg_w") is not None:
            power_w_list.append(float(overhead["power_avg_w"]))
        if bool(result.get("ok", False)):
            ok_count += 1

    summary = {
        "module": "rollback_verification_same_screenshot",
        "screenshot_path": args.screenshot_path,
        "rounds": int(args.rounds),
        "phash_threshold": int(args.phash_threshold),
        "success_rate": float(ok_count / max(1, args.rounds)),
        "avg_verification_ms": _safe_mean(verify_ms_list),
        "avg_total_ms": _safe_mean(total_ms_list),
        "avg_process_cpu_pct_single_core": _safe_mean(cpu_pct_single_list),
        "avg_process_cpu_pct_system": _safe_mean(cpu_pct_system_list),
        "avg_power_w": _safe_mean(power_w_list),
        "power_samples": len(power_w_list),
        "power_sample_every": int(max(1, int(args.power_sample_every))),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot_path", type=str, default="./screenshot/screenshot.png")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--phash_threshold", type=int, default=8)
    parser.add_argument("--adb_path", type=str, default="adb")
    parser.add_argument(
        "--disable_power",
        dest="collect_power",
        action="store_false",
        help="Disable adb battery current/voltage power sampling.",
    )
    parser.add_argument(
        "--power_sample_every",
        type=int,
        default=5,
        help="Sample power once every N rounds to reduce measurement perturbation.",
    )
    parser.set_defaults(collect_power=True)
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    run_module2(args)
