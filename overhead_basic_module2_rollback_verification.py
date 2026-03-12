import argparse
import json
from typing import List

from overhead_basic_utils import begin_cpu_sample, end_cpu_sample, mean
from sim_no_adb_utils import compute_phash, verify_with_anchor_phash


def run(args):
    anchor_hash = compute_phash(args.screenshot_path)

    verify_ms_list: List[float] = []
    total_ms_list: List[float] = []
    cpu_pct_list: List[float] = []
    ok_count = 0

    for _ in range(args.rounds):
        token = begin_cpu_sample()
        result = verify_with_anchor_phash(
            screenshot_path=args.screenshot_path,
            anchor_hash=anchor_hash,
            threshold=args.phash_threshold,
        )
        cpu = end_cpu_sample(token)

        verify_ms_list.append(float(result.get("verification_latency_ms", 0.0)))
        total_ms_list.append(float(cpu["wall_ms"]))
        cpu_pct_list.append(float(cpu["cpu_pct_single_core"]))
        if bool(result.get("ok", False)):
            ok_count += 1

    summary = {
        "module": "rollback_verification_same_screenshot",
        "screenshot_path": args.screenshot_path,
        "rounds": int(args.rounds),
        "phash_threshold": int(args.phash_threshold),
        "success_rate": float(ok_count / max(1, args.rounds)),
        "avg_verification_ms": mean(verify_ms_list),
        "avg_total_ms": mean(total_ms_list),
        "avg_cpu_pct_single_core": mean(cpu_pct_list),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot_path", type=str, default="./screenshot/screenshot.png")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--phash_threshold", type=int, default=8)
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()
    run(args)
