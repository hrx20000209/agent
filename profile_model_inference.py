#!/usr/bin/env python3
"""Profile one OpenAI-compatible model endpoint in isolation.

The script does not touch ADB or the belief graph.  It records request latency,
optional streaming time-to-first-token, token usage, and memory for both this
client and a local llama.cpp server process.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - reported as a runtime configuration error
    psutil = None


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def process_memory(pid: Optional[int]) -> Dict[str, Optional[int]]:
    if psutil is None or not pid:
        return {"rss_bytes": None, "uss_bytes": None, "vms_bytes": None}
    try:
        process = psutil.Process(int(pid))
        basic = process.memory_info()
        full = process.memory_full_info()
        return {
            "rss_bytes": int(basic.rss),
            "uss_bytes": int(getattr(full, "uss", 0) or 0),
            "vms_bytes": int(basic.vms),
        }
    except (psutil.Error, OSError):
        return {"rss_bytes": None, "uss_bytes": None, "vms_bytes": None}


def discover_server_pid(explicit_pid: int, process_pattern: str) -> Tuple[Optional[int], List[Dict[str, Any]]]:
    if explicit_pid > 0:
        if psutil is None or not psutil.pid_exists(explicit_pid):
            raise RuntimeError(f"--server_pid does not exist: {explicit_pid}")
        return explicit_pid, []
    if psutil is None or not process_pattern:
        return None, []
    pattern = re.compile(process_pattern, re.I)
    candidates: List[Dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            cmdline = " ".join(process.info.get("cmdline") or [])
            description = f"{process.info.get('name') or ''} {cmdline}".strip()
            if process.pid != os.getpid() and pattern.search(description):
                memory_info = process.info.get("memory_info")
                candidates.append({
                    "pid": int(process.pid),
                    "rss_bytes": int(memory_info.rss) if memory_info else 0,
                    "description": description[:500],
                })
        except (psutil.Error, OSError):
            continue
    candidates.sort(key=lambda item: item["rss_bytes"], reverse=True)
    return (int(candidates[0]["pid"]) if candidates else None), candidates


class MemorySampler:
    def __init__(self, path: Path, server_pid: Optional[int], interval_sec: float):
        self.path = path
        self.server_pid = server_pid
        self.interval_sec = max(0.02, float(interval_sec))
        self.current_request: Optional[int] = None
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="model-memory-profiler", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            while not self.stop_event.is_set():
                sample = {
                    "timestamp": time.time(),
                    "request_index": self.current_request,
                    "client_pid": os.getpid(),
                    "client": process_memory(os.getpid()),
                    "server_pid": self.server_pid,
                    "server": process_memory(self.server_pid),
                }
                stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stream.flush()
                self.stop_event.wait(self.interval_sec)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return str(args.prompt)


def build_messages(prompt: str, image_path: str) -> Tuple[List[Dict[str, Any]], int]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    image_bytes = 0
    if image_path:
        path = Path(image_path)
        raw = path.read_bytes()
        image_bytes = len(raw)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"},
        })
    return [{"role": "user", "content": content}], image_bytes


def request_once(
    api_url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_sec: float,
    stream_response: bool,
) -> Dict[str, Any]:
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    started = time.perf_counter()
    first_token_sec: Optional[float] = None
    usage: Dict[str, Any] = {}
    response_bytes = 0
    generated_text_parts: List[str] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            if stream_response:
                for raw_line in response:
                    response_bytes += len(raw_line)
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        if first_token_sec is None:
                            first_token_sec = time.perf_counter() - started
                        generated_text_parts.append(str(text))
            else:
                raw = response.read()
                response_bytes = len(raw)
                body = json.loads(raw.decode("utf-8"))
                if not body.get("choices"):
                    raise RuntimeError(f"Inference returned no choices: {str(body)[:1000]}")
                usage = body.get("usage") or {}
                message = ((body.get("choices") or [{}])[0].get("message") or {})
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        str(item.get("text", "")) for item in content if isinstance(item, dict)
                    )
                generated_text_parts.append(str(content))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {details[:1000]}") from exc
    return {
        "latency_sec": time.perf_counter() - started,
        "time_to_first_token_sec": first_token_sec,
        "response_bytes": response_bytes,
        "response_chars": len("".join(generated_text_parts)),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def summarize(rows: List[Dict[str, Any]], memory_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [row for row in rows if row.get("ok")]
    latencies = [float(row["latency_sec"]) for row in successful]
    ttft = [float(row["time_to_first_token_sec"]) for row in successful if row.get("time_to_first_token_sec") is not None]
    completion_tokens = [
        float(row["completion_tokens"]) for row in successful if row.get("completion_tokens") is not None
    ]
    server_rss = [
        int(row["server"]["rss_bytes"])
        for row in memory_rows
        if (row.get("server") or {}).get("rss_bytes") is not None
    ]
    server_uss = [
        int(row["server"]["uss_bytes"])
        for row in memory_rows
        if (row.get("server") or {}).get("uss_bytes") is not None
    ]
    output: Dict[str, Any] = {
        "request_count": len(rows),
        "success_count": len(successful),
        "failure_count": len(rows) - len(successful),
        "latency_mean_sec": statistics.mean(latencies) if latencies else None,
        "latency_p50_sec": percentile(latencies, 0.50),
        "latency_p95_sec": percentile(latencies, 0.95),
        "ttft_mean_sec": statistics.mean(ttft) if ttft else None,
        "ttft_p50_sec": percentile(ttft, 0.50),
        "completion_tokens_per_sec": (
            sum(completion_tokens) / sum(latencies) if completion_tokens and sum(latencies) > 0 else None
        ),
        "server_rss_start_bytes": server_rss[0] if server_rss else None,
        "server_rss_end_bytes": server_rss[-1] if server_rss else None,
        "server_rss_peak_bytes": max(server_rss) if server_rss else None,
        "server_uss_start_bytes": server_uss[0] if server_uss else None,
        "server_uss_end_bytes": server_uss[-1] if server_uss else None,
        "server_uss_peak_bytes": max(server_uss) if server_uss else None,
    }
    return output


def require_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api_url", default="http://127.0.0.1:8084/v1/chat/completions")
    parser.add_argument("--model", default="GELAB-ZERO-4B")
    parser.add_argument("--prompt", default="Describe the safest useful next action on this mobile UI.")
    parser.add_argument("--prompt_file", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup_runs", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--timeout_sec", type=float, default=180.0)
    parser.add_argument("--stream", action="store_true", help="Use SSE streaming and report TTFT.")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--server_pid", type=int, default=0)
    parser.add_argument(
        "--server_process_pattern",
        default=r"llama-server|llama\.cpp|server\.exe",
        help="Regex used to find a local server when --server_pid is omitted.",
    )
    parser.add_argument("--memory_interval_sec", type=float, default=0.05)
    parser.add_argument("--cooldown_sec", type=float, default=0.0)
    parser.add_argument("--output_dir", default="evaluation_results/profile_model_inference")
    args = parser.parse_args()

    if psutil is None:
        raise RuntimeError("psutil is required; install the repository requirements first.")
    output_dir = Path(args.output_dir).resolve()
    require_empty_output_dir(output_dir)
    prompt = load_prompt(args)
    messages, image_bytes = build_messages(prompt, args.image)
    server_pid, candidates = discover_server_pid(args.server_pid, args.server_process_pattern)
    api_key = os.environ.get(args.api_key_env, "")
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max(1, int(args.max_tokens)),
        "stream": bool(args.stream),
    }
    if args.stream:
        payload["stream_options"] = {"include_usage": True}

    config = vars(args).copy()
    config.update({
        "output_dir": str(output_dir),
        "prompt_chars": len(prompt),
        "image_bytes": image_bytes,
        "server_pid_resolved": server_pid,
        "server_candidates": candidates,
    })
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for _ in range(max(0, int(args.warmup_runs))):
        request_once(args.api_url, api_key, payload, args.timeout_sec, args.stream)

    memory_path = output_dir / "memory.jsonl"
    sampler = MemorySampler(memory_path, server_pid, args.memory_interval_sec)
    sampler.start()
    rows: List[Dict[str, Any]] = []
    try:
        for index in range(1, max(1, int(args.runs)) + 1):
            sampler.current_request = index
            before = process_memory(server_pid)
            try:
                result = request_once(args.api_url, api_key, payload, args.timeout_sec, args.stream)
                row = {"request_index": index, "ok": True, **result}
            except Exception as exc:
                row = {
                    "request_index": index,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            after = process_memory(server_pid)
            row.update({
                "server_rss_before_bytes": before.get("rss_bytes"),
                "server_rss_after_bytes": after.get("rss_bytes"),
                "server_uss_before_bytes": before.get("uss_bytes"),
                "server_uss_after_bytes": after.get("uss_bytes"),
            })
            rows.append(row)
            if args.cooldown_sec > 0:
                time.sleep(args.cooldown_sec)
    finally:
        sampler.current_request = None
        sampler.stop()

    memory_rows = [json.loads(line) for line in memory_path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        request_samples = [
            sample for sample in memory_rows
            if sample.get("request_index") == row["request_index"]
        ]
        for owner in ("client", "server"):
            for metric in ("rss_bytes", "uss_bytes"):
                values = [
                    int(sample[owner][metric])
                    for sample in request_samples
                    if (sample.get(owner) or {}).get(metric) is not None
                ]
                row[f"{owner}_{metric.removesuffix('_bytes')}_peak_bytes"] = max(values) if values else None

    fieldnames = [
        "request_index", "ok", "latency_sec", "time_to_first_token_sec",
        "prompt_tokens", "completion_tokens", "total_tokens", "response_bytes",
        "response_chars", "server_rss_before_bytes", "server_rss_after_bytes",
        "server_rss_peak_bytes", "server_uss_before_bytes", "server_uss_after_bytes",
        "server_uss_peak_bytes", "client_rss_peak_bytes", "client_uss_peak_bytes", "error",
    ]
    with (output_dir / "requests.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)

    summary = {"config": config, "metrics": summarize(rows, memory_rows), "failures": [r for r in rows if not r.get("ok")]}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["metrics"]["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
