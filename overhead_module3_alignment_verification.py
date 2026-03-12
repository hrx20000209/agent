import argparse
import json
import time
from typing import List

from Explorer.utils import collect_clickable_nodes, node_to_text
from MobileAgentE.tree import parse_a11y_tree
from overhead_metrics import begin_overhead_sample, end_overhead_sample
from sim_no_adb_utils import compute_phash, text_similarity, verify_with_anchor_phash


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_candidate_texts(xml_path: str) -> List[str]:
    root = parse_a11y_tree(xml_path)
    texts = []
    for n in collect_clickable_nodes(root):
        t = (node_to_text(n) or "").strip()
        if t:
            texts.append(t)
    if not texts:
        texts = ["empty_page"]
    return texts


def run_module3(args):
    anchor_hash = compute_phash(args.screenshot_path)
    candidate_texts = _load_candidate_texts(args.xml_path)

    verify_ms_list: List[float] = []
    alignment_ms_list: List[float] = []
    total_ms_list: List[float] = []
    cpu_pct_single_list: List[float] = []
    cpu_pct_system_list: List[float] = []
    power_w_list: List[float] = []
    ok_count = 0
    aligned_count = 0
    last_topk = []

    for itr in range(1, args.rounds + 1):
        collect_power_now = bool(args.collect_power) and ((itr - 1) % max(1, int(args.power_sample_every)) == 0)
        token = begin_overhead_sample(adb_path=args.adb_path, collect_power=collect_power_now)

        verify = verify_with_anchor_phash(
            screenshot_path=args.screenshot_path,
            anchor_hash=anchor_hash,
            threshold=args.phash_threshold,
        )
        verify_ms_list.append(float(verify.get("verification_latency_ms", 0.0)))
        verify_ok = bool(verify.get("ok", False))
        if verify_ok:
            ok_count += 1

        align_start = time.time()
        topk = []
        if verify_ok:
            scored = []
            for t in candidate_texts:
                score, jaccard, seq_ratio = text_similarity(args.task, t)
                scored.append(
                    {
                        "text": t,
                        "score": float(score),
                        "jaccard": float(jaccard),
                        "seq_ratio": float(seq_ratio),
                    }
                )
            scored.sort(key=lambda x: x["score"], reverse=True)
            topk = scored[: max(1, int(args.top_k))]
            aligned_count += len(topk)
        alignment_ms = (time.time() - align_start) * 1000
        alignment_ms_list.append(alignment_ms)
        last_topk = topk

        overhead = end_overhead_sample(token=token, adb_path=args.adb_path, collect_power=collect_power_now)
        total_ms_list.append(float(overhead["wall_ms"]))
        cpu_pct_single_list.append(float(overhead["process_cpu_pct_single_core"]))
        cpu_pct_system_list.append(float(overhead["process_cpu_pct_system"]))
        if overhead.get("power_avg_w") is not None:
            power_w_list.append(float(overhead["power_avg_w"]))

    summary = {
        "module": "alignment_with_same_rollback_verification",
        "task": args.task,
        "screenshot_path": args.screenshot_path,
        "xml_path": args.xml_path,
        "rounds": int(args.rounds),
        "candidate_count": len(candidate_texts),
        "phash_threshold": int(args.phash_threshold),
        "verification_success_rate": float(ok_count / max(1, args.rounds)),
        "avg_verification_ms": _safe_mean(verify_ms_list),
        "avg_alignment_ms": _safe_mean(alignment_ms_list),
        "avg_total_ms": _safe_mean(total_ms_list),
        "avg_aligned_items_per_round": float(aligned_count / max(1, args.rounds)),
        "avg_process_cpu_pct_single_core": _safe_mean(cpu_pct_single_list),
        "avg_process_cpu_pct_system": _safe_mean(cpu_pct_system_list),
        "avg_power_w": _safe_mean(power_w_list),
        "power_samples": len(power_w_list),
        "power_sample_every": int(max(1, int(args.power_sample_every))),
        "last_round_topk": last_topk,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="Search attractions in Los Angeles in Trip app and open the first attraction.",
    )
    parser.add_argument("--screenshot_path", type=str, default="./screenshot/screenshot.png")
    parser.add_argument("--xml_path", type=str, default="./screenshot/a11y.xml")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=5)
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

    run_module3(args)
