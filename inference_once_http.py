#!/usr/bin/env python3
"""Issue one OpenAI-compatible inference request for concurrency evaluation."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_url",
        default="http://127.0.0.1:8100/v1/chat/completions",
    )
    parser.add_argument("--model", default="qwen2.5vl")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--timeout_sec", type=float, default=120.0)
    parser.add_argument(
        "--prompt",
        default="Inspect the mobile UI and describe the safest useful next action.",
    )
    parser.add_argument("--image", default="")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    args = parser.parse_args()

    content = [{"type": "text", "text": args.prompt}]
    if args.image:
        mime = mimetypes.guess_type(args.image)[0] or "image/jpeg"
        with open(args.image, "rb") as stream:
            encoded = base64.b64encode(stream.read()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
        })
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(args.api_key_env, "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        args.api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.timeout_sec) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not body.get("choices"):
        raise RuntimeError(f"Inference returned no choices: {body}")
    print(json.dumps({
        "latency_sec": time.perf_counter() - started,
        "usage": body.get("usage"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
