import argparse
import json
import os
import time
from typing import Dict, List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sentence_transformers import SentenceTransformer

from Explorer.utils import collect_clickable_nodes, extract_task_queries, node_to_text
from MobileAgentE.tree import parse_a11y_tree
from overhead_metrics import begin_overhead_sample, end_overhead_sample


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return x / denom


def run_module1(args):
    model_load_start = time.time()
    model = SentenceTransformer(args.model_name)
    model_load_ms = (time.time() - model_load_start) * 1000

    task_queries = extract_task_queries(args.task)
    if not task_queries:
        task_queries = [args.task.strip() or "task"]

    q_vec = model.encode(task_queries, convert_to_numpy=True)
    q_vec = _normalize_rows(np.asarray(q_vec, dtype=np.float32))

    emb_cache: Dict[str, np.ndarray] = {}
    parse_ms_list: List[float] = []
    embed_ms_list: List[float] = []
    sim_ms_list: List[float] = []
    total_ms_list: List[float] = []
    candidate_count_list: List[int] = []
    cpu_pct_single_list: List[float] = []
    cpu_pct_system_list: List[float] = []
    power_w_list: List[float] = []
    top_debug = []

    for itr in range(1, args.rounds + 1):
        collect_power_now = bool(args.collect_power) and ((itr - 1) % max(1, int(args.power_sample_every)) == 0)
        overhead_token = begin_overhead_sample(adb_path=args.adb_path, collect_power=collect_power_now)
        step_start = time.time()

        parse_start = time.time()
        root = parse_a11y_tree(args.xml_path)
        nodes = collect_clickable_nodes(root)
        parse_ms = (time.time() - parse_start) * 1000
        parse_ms_list.append(parse_ms)

        texts = []
        for n in nodes:
            t = (node_to_text(n) or "").strip()
            if t:
                texts.append(t)
        if not texts:
            texts = ["empty_page"]

        candidate_count_list.append(len(texts))

        missing = [t for t in texts if t not in emb_cache]
        embed_start = time.time()
        if missing:
            vecs = model.encode(missing, convert_to_numpy=True)
            vecs = _normalize_rows(np.asarray(vecs, dtype=np.float32))
            for t, v in zip(missing, vecs):
                emb_cache[t] = v
        embed_ms = (time.time() - embed_start) * 1000
        embed_ms_list.append(embed_ms)

        sim_start = time.time()
        cand_mat = np.stack([emb_cache[t] for t in texts], axis=0)
        score_mat = cand_mat @ q_vec.T
        scores = score_mat.max(axis=1)
        order = np.argsort(-scores)
        topk = []
        for idx in order[: max(1, int(args.top_k))]:
            topk.append({"text": texts[int(idx)], "score": float(scores[int(idx)])})
        sim_ms = (time.time() - sim_start) * 1000
        sim_ms_list.append(sim_ms)

        total_ms = (time.time() - step_start) * 1000
        total_ms_list.append(total_ms)

        overhead = end_overhead_sample(
            adb_path=args.adb_path,
            token=overhead_token,
            collect_power=collect_power_now,
        )
        cpu_pct_single_list.append(float(overhead["process_cpu_pct_single_core"]))
        cpu_pct_system_list.append(float(overhead["process_cpu_pct_system"]))
        if overhead.get("power_avg_w") is not None:
            power_w_list.append(float(overhead["power_avg_w"]))

        top_debug.append({"round": itr, "top": topk})

    summary = {
        "module": "exploration+embedding+similarity",
        "task": args.task,
        "xml_path": args.xml_path,
        "rounds": int(args.rounds),
        "model_name": args.model_name,
        "model_load_ms": model_load_ms,
        "avg_parse_tree_ms": _safe_mean(parse_ms_list),
        "avg_embedding_ms": _safe_mean(embed_ms_list),
        "avg_similarity_ms": _safe_mean(sim_ms_list),
        "avg_total_ms": _safe_mean(total_ms_list),
        "avg_candidate_count": _safe_mean(candidate_count_list),
        "avg_process_cpu_pct_single_core": _safe_mean(cpu_pct_single_list),
        "avg_process_cpu_pct_system": _safe_mean(cpu_pct_system_list),
        "avg_power_w": _safe_mean(power_w_list),
        "power_samples": len(power_w_list),
        "power_sample_every": int(max(1, int(args.power_sample_every))),
        "cache_size": len(emb_cache),
        "last_round_topk": top_debug[-1]["top"] if top_debug else [],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "top_debug": top_debug}, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="Search attractions in Los Angeles in Trip app and open the first attraction.",
    )
    parser.add_argument("--xml_path", type=str, default="./screenshot/a11y.xml")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/paraphrase-MiniLM-L6-v2",
    )
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

    run_module1(args)
