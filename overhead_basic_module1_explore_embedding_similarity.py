import argparse
import hashlib
import json
import math
import re
import time
from typing import Dict, List, Tuple

from overhead_basic_utils import begin_cpu_sample, end_cpu_sample, mean


TASK_TEXT = "Search attractions in Los Angeles in Trip app and open the first attraction."
PAGE_ELEMENTS = [
    "Search box",
    "Flights tab",
    "Hotels tab",
    "Attractions tab",
    "Los Angeles city chip",
    "Universal Studios Hollywood card",
    "Hollywood Walk of Fame card",
    "Getty Center card",
    "Santa Monica Pier card",
    "Sort button",
    "Filter button",
    "Map view button",
    "Price filter",
    "Rating filter",
    "Open now filter",
    "Distance filter",
    "Back button",
    "Home button",
    "Account button",
    "Saved attraction button",
]


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def text_to_embedding(text: str, dim: int = 128) -> List[float]:
    vec = [0.0] * dim
    toks = _tokens(text)
    if not toks:
        return vec
    for tok in toks:
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=16).digest()
        idx = int.from_bytes(h[:4], byteorder="little", signed=False) % dim
        sign = 1.0 if (h[4] & 1) == 0 else -1.0
        mag = 1.0 + (h[5] % 7) * 0.1
        vec[idx] += sign * mag
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def run(args):
    task = args.task or TASK_TEXT
    task_emb = text_to_embedding(task, dim=args.embedding_dim)

    explore_ms_list: List[float] = []
    emb_ms_list: List[float] = []
    sim_ms_list: List[float] = []
    total_ms_list: List[float] = []
    cpu_pct_list: List[float] = []
    top_debug: List[Dict] = []

    n = len(PAGE_ELEMENTS)
    width = max(1, min(args.explore_width, n))

    for i in range(args.rounds):
        sample = begin_cpu_sample()
        total_start = time.time()

        # 1) exploration (simulated candidate picking from current "page")
        t0 = time.time()
        start_idx = (i * args.explore_stride) % n
        candidates = [PAGE_ELEMENTS[(start_idx + j) % n] for j in range(width)]
        if args.explore_sleep_ms > 0:
            time.sleep(args.explore_sleep_ms / 1000.0)
        explore_ms = (time.time() - t0) * 1000.0
        explore_ms_list.append(explore_ms)

        # 2) embedding
        t1 = time.time()
        cand_embs = [text_to_embedding(c, dim=args.embedding_dim) for c in candidates]
        emb_ms = (time.time() - t1) * 1000.0
        emb_ms_list.append(emb_ms)

        # 3) similarity
        t2 = time.time()
        scored: List[Tuple[str, float]] = []
        for c, emb in zip(candidates, cand_embs):
            scored.append((c, cosine(task_emb, emb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        sim_ms = (time.time() - t2) * 1000.0
        sim_ms_list.append(sim_ms)

        total_ms = (time.time() - total_start) * 1000.0
        total_ms_list.append(total_ms)
        cpu = end_cpu_sample(sample)
        cpu_pct_list.append(float(cpu["cpu_pct_single_core"]))

        top_debug.append(
            {
                "round": i + 1,
                "top_candidates": [{"text": t, "score": s} for t, s in scored[: args.top_k]],
            }
        )

    summary = {
        "module": "exploration+embedding+similarity",
        "task": task,
        "rounds": int(args.rounds),
        "embedding_dim": int(args.embedding_dim),
        "explore_width": int(width),
        "explore_stride": int(args.explore_stride),
        "avg_exploration_ms": mean(explore_ms_list),
        "avg_embedding_ms": mean(emb_ms_list),
        "avg_similarity_ms": mean(sim_ms_list),
        "avg_total_ms": mean(total_ms_list),
        "avg_cpu_pct_single_core": mean(cpu_pct_list),
        "last_round_topk": top_debug[-1]["top_candidates"] if top_debug else [],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rounds": top_debug}, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=TASK_TEXT)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--explore_width", type=int, default=12)
    parser.add_argument("--explore_stride", type=int, default=3)
    parser.add_argument("--explore_sleep_ms", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()
    run(args)
