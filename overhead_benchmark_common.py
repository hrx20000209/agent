import json
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from agents.mai_prompt import MAI_MOBILE_SYS_PROMPT, build_user_prompt as build_mai_user_prompt
from sim_no_adb_utils import (
    COMMON_ELEMENT_TEXTS,
    call_llama_cpp_with_image,
    compute_phash,
    parse_action_from_llm_text,
    text_similarity,
)


SYSTEM_PROMPT = MAI_MOBILE_SYS_PROMPT
DEFAULT_HISTORY = [
    "open(Trip.com)",
    "click(search box)",
    "type(Los Angeles attractions)",
]
SYNTHETIC_TRIP_ELEMENTS = [
    "Attractions",
    "Things to do",
    "Top sights",
    "Los Angeles",
    "Hollywood Walk of Fame",
    "Universal Studios Hollywood",
    "Santa Monica Pier",
    "Griffith Observatory",
    "Search destinations",
    "Recent searches",
    "Popular nearby attractions",
    "Flights",
    "Hotels",
    "Trains",
    "Travel guide",
    "Maps",
    "Book now",
    "See details",
]


def _safe_mean(values: List[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _history_to_prompt_text(history: List[str], max_items: int = 8) -> str:
    if not history:
        return "None"
    return "\n".join(history[-max_items:])


def _build_user_prompt(task: str, history: List[str]) -> str:
    return build_mai_user_prompt(task, _history_to_prompt_text(history))


def _stable_hash(text: str) -> int:
    value = 2166136261
    for ch in text.encode("utf-8", errors="ignore"):
        value ^= ch
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def _hashed_embedding(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(int(dim), dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    if not tokens:
        tokens = ["<empty>"]
    for token in tokens:
        ngrams = [token[i : i + 3] for i in range(max(1, len(token) - 2))] or [token]
        for ngram in ngrams:
            idx = _stable_hash(ngram) % int(dim)
            sign = -1.0 if (_stable_hash(ngram + "#") & 1) else 1.0
            vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec


def _generate_synthetic_elements(task: str, count: int) -> List[str]:
    base = list(dict.fromkeys(COMMON_ELEMENT_TEXTS + SYNTHETIC_TRIP_ELEMENTS))
    tokens = re.findall(r"[a-z0-9]+", (task or "").lower())
    token_text = " ".join(tokens[:6]) or "trip attraction"
    idx = 0
    while len(base) < int(count):
        label = idx % 7
        base.append(f"{token_text} option {idx}")
        base.append(f"{token_text} result card {idx}")
        base.append(f"Nearby attraction suggestion {idx} label {label}")
        idx += 1
    return base[: int(count)]


def _extract_power_w_from_text(text: str, key_hint: str = "") -> Optional[float]:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    mw_match = re.search(r"(-?\d+(?:\.\d+)?)\s*mW", raw, flags=re.IGNORECASE)
    if mw_match:
        return float(mw_match.group(1)) / 1000.0

    w_match = re.search(r"(-?\d+(?:\.\d+)?)\s*W", raw, flags=re.IGNORECASE)
    if w_match:
        return float(w_match.group(1))

    num_match = re.search(r"(-?\d+(?:\.\d+)?)", raw)
    if not num_match:
        return None

    value = float(num_match.group(1))
    key_low = key_hint.lower()
    if "mw" in raw.lower() or "mw" in key_low:
        return value / 1000.0
    if "power" in key_low or "vdd" in key_low or "pom" in key_low:
        return value / 1000.0 if abs(value) > 100.0 else value
    return None


def _flatten_mapping(prefix: str, value: Any, out: List[Tuple[str, Any]]):
    if isinstance(value, dict):
        for k, v in value.items():
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            _flatten_mapping(child_prefix, v, out)
        return
    out.append((prefix, value))


def _extract_jtop_stats(stats: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    flat: List[Tuple[str, Any]] = []
    _flatten_mapping("", stats or {}, flat)
    power_w = None
    ram_used_mb = None

    preferred_power_keys = [
        "vdd_in",
        "pom_5v_in",
        "power.tot",
        "power_tot",
        "power total",
        "power in",
    ]
    for key, value in flat:
        key_low = key.lower()
        if any(token in key_low for token in preferred_power_keys) or ("power" in key_low and "tot" in key_low):
            power_w = _extract_power_w_from_text(value, key)
            if power_w is not None:
                break

    if power_w is None:
        for key, value in flat:
            key_low = key.lower()
            if "power" in key_low or "vdd" in key_low or "pom" in key_low:
                power_w = _extract_power_w_from_text(value, key)
                if power_w is not None:
                    break

    for key, value in flat:
        key_low = key.lower()
        if "ram" not in key_low:
            continue
        raw = str(value)
        match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*MB", raw, flags=re.IGNORECASE)
        if match:
            ram_used_mb = float(match.group(1))
            break
        match = re.search(r"(\d+(?:\.\d+)?)\s*MB", raw, flags=re.IGNORECASE)
        if match:
            ram_used_mb = float(match.group(1))
            break

    return power_w, ram_used_mb


def _extract_jtop_power_w(power_info: Any) -> Optional[float]:
    if not isinstance(power_info, dict):
        return None

    for total_key in ("all", "ALL", "tot", "TOT"):
        total = power_info.get(total_key)
        if isinstance(total, dict):
            for key in ("power", "avg", "instant"):
                value_mw = _safe_float(total.get(key))
                if value_mw is not None:
                    return float(value_mw) / 1000.0
        elif total is not None:
            value_w = _extract_power_w_from_text(total, key_hint=total_key)
            if value_w is not None:
                return value_w

    rails = power_info.get("rail")
    if isinstance(rails, dict):
        rail_all = rails.get("all") or rails.get("ALL")
        if isinstance(rail_all, dict):
            for key in ("power", "avg", "instant"):
                value_mw = _safe_float(rail_all.get(key))
                if value_mw is not None:
                    return float(value_mw) / 1000.0

        for rail_name in ("VDD_IN", "POM_5V_IN"):
            rail = rails.get(rail_name)
            if not isinstance(rail, dict):
                continue
            for key in ("power", "avg", "instant"):
                value_mw = _safe_float(rail.get(key))
                if value_mw is not None:
                    return float(value_mw) / 1000.0

        rail_sum_mw = 0.0
        rail_count = 0
        for rail in rails.values():
            if not isinstance(rail, dict):
                continue
            value_mw = _safe_float(rail.get("power"))
            if value_mw is None:
                value_mw = _safe_float(rail.get("avg"))
            if value_mw is None:
                continue
            rail_sum_mw += float(value_mw)
            rail_count += 1
        if rail_count > 0:
            return rail_sum_mw / 1000.0

    return None


def _extract_jtop_memory_mb(memory_info: Any) -> Optional[float]:
    if not isinstance(memory_info, dict):
        return None
    ram = memory_info.get("RAM")
    if not isinstance(ram, dict):
        return None

    used_value = _safe_float(ram.get("used"))
    if used_value is not None:
        # jtop.memory["RAM"]["used"] is reported in KB in the official API.
        return float(used_value) / 1024.0

    for key in ("used", "use", "shared", "cached"):
        text_value = ram.get(key)
        if text_value is None:
            continue
        raw = str(text_value)
        match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*MB", raw, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"(\d+(?:\.\d+)?)\s*MB", raw, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _extract_jtop_reader_metrics(reader: Any) -> Tuple[Optional[float], Optional[float]]:
    power_w = None
    system_used_mb = None

    try:
        power_w = _extract_jtop_power_w(getattr(reader, "power", None))
    except Exception:
        power_w = None

    try:
        system_used_mb = _extract_jtop_memory_mb(getattr(reader, "memory", None))
    except Exception:
        system_used_mb = None

    if power_w is None or system_used_mb is None:
        try:
            stats = dict(getattr(reader, "stats", {}) or {})
            stats_power_w, stats_ram_mb = _extract_jtop_stats(stats)
            if power_w is None:
                power_w = stats_power_w
            if system_used_mb is None:
                system_used_mb = stats_ram_mb
        except Exception:
            pass

    return power_w, system_used_mb


def _parse_tegrastats_line(line: str) -> Tuple[Optional[float], Optional[float]]:
    power_w = None
    ram_used_mb = None

    ram_match = re.search(r"RAM\s+(\d+)(?:\/(\d+))?MB", line, flags=re.IGNORECASE)
    if ram_match:
        ram_used_mb = float(ram_match.group(1))

    power_match = re.search(
        r"\b(?:VDD_IN|POM_5V_IN)\s+(\d+(?:\.\d+)?)(?:mW)?(?:\/(\d+(?:\.\d+)?)(?:mW)?)?",
        line,
        flags=re.IGNORECASE,
    )
    if power_match:
        power_w = float(power_match.group(1)) / 1000.0

    return power_w, ram_used_mb


class ResourceSampler:
    def __init__(self, use_jtop: bool = True, sample_interval_sec: float = 0.1):
        self.use_jtop = bool(use_jtop)
        self.sample_interval_sec = max(0.02, float(sample_interval_sec))
        self.process = psutil.Process(os.getpid())
        self.backend = "psutil"
        self.backend_note = ""
        self._samples: List[Dict[str, Optional[float]]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tegrastats_proc: Optional[subprocess.Popen] = None
        self._tegrastats_thread: Optional[threading.Thread] = None
        self._latest_power_w: Optional[float] = None
        self._latest_system_used_mb: Optional[float] = None

    def start(self):
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._stop_tegrastats()

    def mark(self) -> int:
        with self._lock:
            return len(self._samples)

    def summarize_window(self, start_idx: int) -> Dict[str, Any]:
        with self._lock:
            window = list(self._samples[int(start_idx) :])

        power_vals = [float(s["power_w"]) for s in window if s.get("power_w") is not None]
        process_vals = [float(s["process_rss_mb"]) for s in window if s.get("process_rss_mb") is not None]
        system_vals = [float(s["system_used_mb"]) for s in window if s.get("system_used_mb") is not None]

        return {
            "resource_backend": self.backend,
            "resource_backend_note": self.backend_note,
            "resource_samples": len(window),
            "avg_power_w": _safe_mean(power_vals),
            "peak_power_w": max(power_vals) if power_vals else None,
            "peak_process_rss_mb": max(process_vals) if process_vals else None,
            "peak_system_used_mb": max(system_vals) if system_vals else None,
        }

    def _start_tegrastats(self):
        path = shutil.which("tegrastats")
        if not path:
            return
        interval_ms = max(20, int(self.sample_interval_sec * 1000))
        try:
            self._tegrastats_proc = subprocess.Popen(
                [path, "--interval", str(interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            self.backend = "tegrastats"
            self._tegrastats_thread = threading.Thread(target=self._read_tegrastats_loop, daemon=True)
            self._tegrastats_thread.start()
        except Exception as exc:
            self.backend_note = f"tegrastats_unavailable: {exc}"
            self._tegrastats_proc = None

    def _stop_tegrastats(self):
        if self._tegrastats_proc is not None:
            try:
                self._tegrastats_proc.terminate()
                self._tegrastats_proc.wait(timeout=1.0)
            except Exception:
                try:
                    self._tegrastats_proc.kill()
                except Exception:
                    pass
            self._tegrastats_proc = None
        if self._tegrastats_thread is not None:
            self._tegrastats_thread.join(timeout=1.0)
            self._tegrastats_thread = None

    def _read_tegrastats_loop(self):
        proc = self._tegrastats_proc
        if proc is None or proc.stdout is None:
            return
        while not self._stop_event.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            power_w, ram_used_mb = _parse_tegrastats_line(line)
            if power_w is not None:
                self._latest_power_w = power_w
            if ram_used_mb is not None:
                self._latest_system_used_mb = ram_used_mb

    def _append_sample(self, power_w: Optional[float], system_used_mb: Optional[float]):
        if power_w is None:
            power_w = self._latest_power_w
        else:
            self._latest_power_w = power_w

        if system_used_mb is None:
            system_used_mb = self._latest_system_used_mb
        else:
            self._latest_system_used_mb = system_used_mb

        if system_used_mb is None:
            system_used_mb = float(psutil.virtual_memory().used) / (1024.0 * 1024.0)

        process_rss_mb = float(self.process.memory_info().rss) / (1024.0 * 1024.0)
        sample = {
            "ts": time.time(),
            "power_w": power_w,
            "process_rss_mb": process_rss_mb,
            "system_used_mb": system_used_mb,
        }
        with self._lock:
            self._samples.append(sample)

    def _sample_loop(self):
        if self.use_jtop:
            try:
                from jtop import jtop

                self.backend = "jtop"
                jtop_sample_count = 0
                with jtop() as jetson_reader:
                    while not self._stop_event.is_set():
                        if not jetson_reader.ok():
                            break
                        try:
                            power_w, system_used_mb = _extract_jtop_reader_metrics(jetson_reader)
                        except Exception as exc:
                            power_w = None
                            system_used_mb = None
                            if not self.backend_note:
                                self.backend_note = f"jtop_read_failed: {exc}"
                        self._append_sample(power_w=power_w, system_used_mb=system_used_mb)
                        jtop_sample_count += 1
                    if self._stop_event.is_set() or jtop_sample_count > 0:
                        return
                    if not self.backend_note:
                        self.backend_note = "jtop_no_samples"
            except Exception as exc:
                self.backend_note = f"jtop_unavailable: {exc}"

        self._start_tegrastats()
        if self.backend != "tegrastats":
            self.backend = "psutil"
            if not self.backend_note:
                self.backend_note = "power_metrics_unavailable"

        while not self._stop_event.is_set():
            self._append_sample(power_w=self._latest_power_w, system_used_mb=self._latest_system_used_mb)
            time.sleep(self.sample_interval_sec)


def _run_threaded(target, kwargs: Dict[str, Any]) -> Tuple[threading.Thread, Dict[str, Any]]:
    box: Dict[str, Any] = {}

    def _runner():
        try:
            box["result"] = target(**kwargs)
        except Exception as exc:
            box["error"] = str(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    return thread, box


def _run_explore_worker(
    task: str,
    corpus: List[str],
    worker_rounds: int,
    embedding_dim: int,
    top_k: int,
    per_round_sleep_sec: float,
) -> Dict[str, Any]:
    traces = []
    start_ts = time.time()
    for itr in range(1, int(worker_rounds) + 1):
        round_start = time.time()
        task_vec = _hashed_embedding(task, dim=embedding_dim)
        cand_mat = np.stack([_hashed_embedding(text, dim=embedding_dim) for text in corpus], axis=0)
        cosine_scores = cand_mat @ task_vec

        ranked = []
        for idx, text in enumerate(corpus):
            lexical_score, jaccard, seq_ratio = text_similarity(task, text)
            final_score = 0.7 * float(cosine_scores[idx]) + 0.3 * float(lexical_score)
            ranked.append(
                {
                    "text": text,
                    "score": final_score,
                    "cosine": float(cosine_scores[idx]),
                    "lexical": float(lexical_score),
                    "jaccard": float(jaccard),
                    "seq_ratio": float(seq_ratio),
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        traces.append(
            {
                "round": itr,
                "latency_ms": (time.time() - round_start) * 1000.0,
                "top_k": ranked[: max(1, int(top_k))],
            }
        )
        if per_round_sleep_sec > 0:
            time.sleep(per_round_sleep_sec)

    return {
        "mode": "explore",
        "worker_rounds": int(worker_rounds),
        "worker_latency_ms": (time.time() - start_ts) * 1000.0,
        "last_top_k": traces[-1]["top_k"] if traces else [],
        "traces": traces,
    }


def _run_threaded_rollback(
    screenshot_path: str,
    repeats: int,
    threshold: int,
    per_round_sleep_sec: float,
) -> Dict[str, Any]:
    anchor_hash = compute_phash(screenshot_path)
    traces = []
    start_ts = time.time()
    for itr in range(1, int(repeats) + 1):
        verify_start = time.time()
        curr_hash = compute_phash(screenshot_path)
        diff = int(curr_hash - anchor_hash)
        traces.append(
            {
                "round": itr,
                "diff": diff,
                "ok": diff <= int(threshold),
                "latency_ms": (time.time() - verify_start) * 1000.0,
            }
        )
        if per_round_sleep_sec > 0:
            time.sleep(per_round_sleep_sec)
    return {
        "mode": "rollback",
        "worker_rounds": int(repeats),
        "worker_latency_ms": (time.time() - start_ts) * 1000.0,
        "traces": traces,
    }


def _run_noop_worker(idle_sleep_sec: float) -> Dict[str, Any]:
    start_ts = time.time()
    if idle_sleep_sec > 0:
        time.sleep(idle_sleep_sec)
    return {
        "mode": "noop",
        "worker_rounds": 0,
        "worker_latency_ms": (time.time() - start_ts) * 1000.0,
        "traces": [],
    }


def _run_selection_verification(
    screenshot_path: str,
    repeats: int,
    threshold: int,
    per_round_sleep_sec: float,
) -> Dict[str, Any]:
    anchor_hash = compute_phash(screenshot_path)
    traces = []
    start_ts = time.time()
    for itr in range(1, int(repeats) + 1):
        verify_start = time.time()
        curr_hash = compute_phash(screenshot_path)
        diff = int(curr_hash - anchor_hash)
        traces.append(
            {
                "round": itr,
                "diff": diff,
                "ok": diff <= int(threshold),
                "latency_ms": (time.time() - verify_start) * 1000.0,
            }
        )
        if per_round_sleep_sec > 0:
            time.sleep(per_round_sleep_sec)
    return {
        "mode": "selection",
        "selection_rounds": int(repeats),
        "selection_latency_ms": (time.time() - start_ts) * 1000.0,
        "traces": traces,
    }


def _build_static_history(args) -> List[str]:
    if args.history_json:
        try:
            data = json.loads(args.history_json)
            if isinstance(data, list):
                return [str(item) for item in data]
        except Exception:
            pass
    return list(DEFAULT_HISTORY)


def add_common_args(parser):
    parser.add_argument(
        "--task",
        type=str,
        default="Search attractions in Los Angeles in Trip app and open the first attraction.",
    )
    parser.add_argument("--screenshot_path", type=str, default="./resized_screenshot.png")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--llama_api_url", type=str, default="http://localhost:8100/v1/chat/completions")
    parser.add_argument("--history_json", type=str, default="")
    parser.add_argument("--sample_interval_sec", type=float, default=0.1)
    parser.add_argument(
        "--disable_jtop",
        dest="use_jtop",
        action="store_false",
        help="Disable jtop sampling and fall back to tegrastats/psutil.",
    )
    parser.set_defaults(use_jtop=True)
    parser.add_argument("--output_json", type=str, default="")
    return parser


def add_explore_args(parser):
    parser.add_argument("--worker_rounds", type=int, default=4)
    parser.add_argument("--candidate_pool_size", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=192)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--worker_sleep_sec", type=float, default=0.01)
    return parser


def add_phash_args(parser):
    parser.add_argument("--phash_repeats", type=int, default=3)
    parser.add_argument("--phash_threshold", type=int, default=8)
    parser.add_argument("--worker_sleep_sec", type=float, default=0.01)
    return parser


def run_overhead_case(args, case_name: str):
    history = _build_static_history(args)
    sampler = ResourceSampler(use_jtop=args.use_jtop, sample_interval_sec=args.sample_interval_sec)
    sampler.start()

    synthetic_corpus = None
    if case_name == "explore":
        synthetic_corpus = _generate_synthetic_elements(args.task, count=args.candidate_pool_size)

    steps: List[Dict[str, Any]] = []

    try:
        for itr in range(1, int(args.rounds) + 1):
            iter_start = time.time()
            resource_mark = sampler.mark()

            worker_thread = None
            worker_box: Dict[str, Any] = {}
            if case_name == "explore":
                worker_thread, worker_box = _run_threaded(
                    _run_explore_worker,
                    {
                        "task": args.task,
                        "corpus": synthetic_corpus or [],
                        "worker_rounds": args.worker_rounds,
                        "embedding_dim": args.embedding_dim,
                        "top_k": args.top_k,
                        "per_round_sleep_sec": args.worker_sleep_sec,
                    },
                )
            elif case_name == "rollback":
                worker_thread, worker_box = _run_threaded(
                    _run_threaded_rollback,
                    {
                        "screenshot_path": args.screenshot_path,
                        "repeats": args.phash_repeats,
                        "threshold": args.phash_threshold,
                        "per_round_sleep_sec": args.worker_sleep_sec,
                    },
                )
            elif case_name == "selection":
                worker_thread, worker_box = _run_threaded(
                    _run_noop_worker,
                    {"idle_sleep_sec": 0.0},
                )

            query_start = time.time()
            llm_output = call_llama_cpp_with_image(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=_build_user_prompt(args.task, history),
                screenshot_path=args.screenshot_path,
                api_url=args.llama_api_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            query_latency_ms = (time.time() - query_start) * 1000.0

            parse_error = None
            parsed_action = None
            try:
                parsed_action = parse_action_from_llm_text(llm_output)
            except Exception as exc:
                parse_error = str(exc)

            if worker_thread is not None:
                worker_thread.join()
            worker_result = worker_box.get("result")
            worker_error = worker_box.get("error")

            selection_result = None
            if case_name == "selection":
                selection_result = _run_selection_verification(
                    screenshot_path=args.screenshot_path,
                    repeats=args.phash_repeats,
                    threshold=args.phash_threshold,
                    per_round_sleep_sec=args.worker_sleep_sec,
                )

            iteration_latency_ms = (time.time() - iter_start) * 1000.0
            resource_stats = sampler.summarize_window(resource_mark)

            step = {
                "round": itr,
                "query_latency_ms": query_latency_ms,
                "iteration_latency_ms": iteration_latency_ms,
                "llm_output": llm_output,
                "parsed_action": parsed_action,
                "parse_error": parse_error,
                "worker_result": worker_result,
                "worker_error": worker_error,
                "selection_result": selection_result,
                "resources": resource_stats,
            }
            steps.append(step)

            print(
                f"[Round {itr}] query={query_latency_ms:.1f} ms, total={iteration_latency_ms:.1f} ms, "
                f"power={resource_stats.get('avg_power_w')}, peak_mem={resource_stats.get('peak_system_used_mb')}"
            )
    finally:
        sampler.stop()

    query_latencies = [float(step["query_latency_ms"]) for step in steps]
    total_latencies = [float(step["iteration_latency_ms"]) for step in steps]
    avg_power_vals = [
        float(step["resources"]["avg_power_w"])
        for step in steps
        if step.get("resources", {}).get("avg_power_w") is not None
    ]
    peak_power_vals = [
        float(step["resources"]["peak_power_w"])
        for step in steps
        if step.get("resources", {}).get("peak_power_w") is not None
    ]
    peak_process_vals = [
        float(step["resources"]["peak_process_rss_mb"])
        for step in steps
        if step.get("resources", {}).get("peak_process_rss_mb") is not None
    ]
    peak_system_vals = [
        float(step["resources"]["peak_system_used_mb"])
        for step in steps
        if step.get("resources", {}).get("peak_system_used_mb") is not None
    ]

    summary = {
        "case": case_name,
        "task": args.task,
        "screenshot_path": args.screenshot_path,
        "rounds": int(args.rounds),
        "llama_api_url": args.llama_api_url,
        "avg_query_latency_ms": _safe_mean(query_latencies),
        "avg_iteration_latency_ms": _safe_mean(total_latencies),
        "avg_power_w": _safe_mean(avg_power_vals),
        "peak_power_w": max(peak_power_vals) if peak_power_vals else None,
        "avg_peak_process_rss_mb": _safe_mean(peak_process_vals),
        "peak_process_rss_mb": max(peak_process_vals) if peak_process_vals else None,
        "avg_peak_system_used_mb": _safe_mean(peak_system_vals),
        "peak_system_used_mb": max(peak_system_vals) if peak_system_vals else None,
        "resource_backend": steps[-1]["resources"]["resource_backend"] if steps else sampler.backend,
        "resource_backend_note": steps[-1]["resources"]["resource_backend_note"] if steps else sampler.backend_note,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "steps": steps}, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {args.output_json}")

    return {"summary": summary, "steps": steps}
