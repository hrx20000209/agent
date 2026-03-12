import argparse
import json
import time
from typing import Dict, List

from overhead_basic_utils import begin_cpu_sample, end_cpu_sample, mean
from sim_no_adb_utils import text_similarity, verify_with_anchor_phash, compute_phash


TASK_TEXT = "Search attractions in Los Angeles in Trip app and open the first attraction."
DISCOVERED_TEXTS = [
    "Featured attractions",
    "Attractions",
    "Save attraction",
    "Trip Moments",
    "Open today at 10:00–18:00",
    "Universal Studios Hollywood",
    "Hollywood Walk of Fame",
    "The Getty Center",
    "Santa Monica Pier",
    "Los Angeles County Museum of Art",
    "Sort by relevance",
    "Filter by distance",
    "Map view",
]


def run(args):
    task = args.task or TASK_TEXT
    anchor_hash = compute_phash(args.screenshot_path)

    verify_ms_list: List[float] = []
    align_ms_list: List[float] = []
    total_ms_list: List[float] = []
    cpu_pct_list: List[float] = []
    ok_count = 0
    aligned_items = 0
    last_topk: List[Dict] = []

    for _ in range(args.rounds):
        token = begin_cpu_sample()

        verify = verify_with_anchor_phash(
            screenshot_path=args.screenshot_path,
            anchor_hash=anchor_hash,
            threshold=args.phash_threshold,
        )
        verify_ok = bool(verify.get("ok", False))
        verify_ms_list.append(float(verify.get("verification_latency_ms", 0.0)))
        if verify_ok:
            ok_count += 1

        t0 = time.time()
        topk = []
        if verify_ok:
            scored = []
            for txt in DISCOVERED_TEXTS:
                score, jaccard, seq_ratio = text_similarity(task, txt)
                scored.append(
                    {
                        "text": txt,
                        "score": float(score),
                        "jaccard": float(jaccard),
                        "seq_ratio": float(seq_ratio),
                    }
                )
            scored.sort(key=lambda x: x["score"], reverse=True)
            topk = scored[: args.top_k]
            aligned_items += len(topk)
        align_ms = (time.time() - t0) * 1000.0
        align_ms_list.append(align_ms)
        last_topk = topk

        cpu = end_cpu_sample(token)
        total_ms_list.append(float(cpu["wall_ms"]))
        cpu_pct_list.append(float(cpu["cpu_pct_single_core"]))

    summary = {
        "module": "alignment_with_same_rollback_verification",
        "task": task,
        "rounds": int(args.rounds),
        "verification_success_rate": float(ok_count / max(1, args.rounds)),
        "avg_verification_ms": mean(verify_ms_list),
        "avg_alignment_ms": mean(align_ms_list),
        "avg_total_ms": mean(total_ms_list),
        "avg_aligned_items_per_round": float(aligned_items / max(1, args.rounds)),
        "avg_cpu_pct_single_core": mean(cpu_pct_list),
        "last_round_topk": last_topk,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=TASK_TEXT)
    parser.add_argument("--screenshot_path", type=str, default="./screenshot/screenshot.png")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--phash_threshold", type=int, default=8)
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()
    run(args)
