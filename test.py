import time
from PIL import Image
import io
import base64
import os
import requests


########################################
#        llama-server API CALL
########################################
def inference_chat_llama_cpp(
        chat,
        api_url="http://localhost:8080/v1/chat/completions",
        temperature=0.0,
        max_tokens=200
):
    headers = {"Content-Type": "application/json"}
    messages = []

    for role, content_items in chat:
        content_list = []

        for item in content_items:
            if item["type"] == "text":
                content_list.append({
                    "type": "text",
                    "text": item["text"]
                })

            elif item["type"] == "image_url":
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": item["image_url"]["url"]
                    }
                })

        messages.append({
            "role": role,
            "content": content_list
        })

    data = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    res = requests.post(api_url, headers=headers, json=data)

    if res.status_code != 200:
        print(f"\n[Error] Server Response: {res.text}\n")

    res.raise_for_status()
    js = res.json()

    return {
        "content": js["choices"][0]["message"]["content"],
        "timings": js.get("timings", {}),
        "usage": js.get("usage", {})
    }


########################################
#        IMAGE LOADING + RESIZE
########################################
def load_image_as_data_uri(path, resize_divisor=1):
    """
    resize_divisor = 1  -> original
    resize_divisor = 2  -> width/height / 2
    resize_divisor = 4  -> width/height / 4
    """
    img = Image.open(os.path.expanduser(path)).convert("RGB")

    if resize_divisor > 1:
        w, h = img.size
        img = img.resize(
            (max(1, w // resize_divisor), max(1, h // resize_divisor)),
            Image.LANCZOS
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{b64}"


########################################
#                MAIN
########################################
def main():

    # ---------------- CONFIG ----------------
    IMAGE_PATHS = [
        "./data/episode_000003/screenshot_000.png",
        "./data/episode_000003/screenshot_001.png",
        "./data/episode_000003/screenshot_002.png",
    ]

    RESIZE_DIVISOR = 2   # change this
    N = 5                # number of runs

    # ---------------- PRELOAD IMAGES ----------------
    image_uris = [
        load_image_as_data_uri(p, resize_divisor=RESIZE_DIVISOR)
        for p in IMAGE_PATHS
    ]

    # ---------------- BUILD CHAT ----------------
    chat = [
        (
            "user",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": image_uris[0]}
                },
                {
                    "type": "text",
                    "text": "Describe what is shown on the phone screen."
                }
            ]
        )
    ]

    # ---------------- WARM-UP ----------------
    inference_chat_llama_cpp(chat)

    # ---------------- STATS ----------------
    prefill_times = []
    decode_times = []
    e2e_times = []

    prefill_tps_list = []
    decode_tps_list = []

    # ---------------- BENCHMARK ----------------
    for i in range(N):
        img_uri = image_uris[i % len(image_uris)]
        chat[0][1][0]["image_url"]["url"] = img_uri

        t0 = time.time()
        result = inference_chat_llama_cpp(chat)
        t1 = time.time()

        timings = result["timings"]
        usage = result["usage"]

        # latency (seconds)
        prefill_s = timings.get("prompt_ms", 0.0) / 1000.0
        decode_s = timings.get("predicted_ms", 0.0) / 1000.0
        e2e_s = t1 - t0

        # tokens
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # token/s
        prefill_tps = (
            prompt_tokens / prefill_s if prefill_s > 0 else 0.0
        )
        decode_tps = (
            completion_tokens / decode_s if decode_s > 0 else 0.0
        )

        # record
        prefill_times.append(prefill_s)
        decode_times.append(decode_s)
        e2e_times.append(e2e_s)

        prefill_tps_list.append(prefill_tps)
        decode_tps_list.append(decode_tps)

        print(
            f"[Run {i+1}] Image={i % len(image_uris)} | "
            f"Prefill={prefill_s:.3f}s ({prefill_tps:.1f} tok/s) | "
            f"Decode={decode_s:.3f}s ({decode_tps:.1f} tok/s) | "
            f"E2E={e2e_s:.3f}s"
        )

    # ---------------- SUMMARY ----------------
    print("\n=== Averages ===")
    print(f"Resize divisor: {RESIZE_DIVISOR}")

    print(
        f"Prefill avg: {sum(prefill_times)/N:.3f}s | "
        f"{sum(prefill_tps_list)/N:.1f} tok/s"
    )
    print(
        f"Decode  avg: {sum(decode_times)/N:.3f}s | "
        f"{sum(decode_tps_list)/N:.1f} tok/s"
    )
    print(
        f"E2E     avg: {sum(e2e_times)/N:.3f}s"
    )


if __name__ == "__main__":
    main()
