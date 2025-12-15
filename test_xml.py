import time
import os
import requests

from xml_tree import convert_forest_any_to_xml


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
#                MAIN
########################################
def main():

    # ---------------- CONFIG ----------------
    XML_PATHS = [
        "./data/episode_000003/accessibility_tree_000.txt",
        "./data/episode_000003/accessibility_tree_001.txt",
        "./data/episode_000003/accessibility_tree_002.txt",
    ]

    TASK = "Describe the current UI and decide the next action."
    N = 5

    # ---------------- PRELOAD XML ----------------
    xml_strings = []

    for p in XML_PATHS:
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read()

        xml = convert_forest_any_to_xml(raw)
        xml_strings.append(xml)

    # ---------------- BUILD CHAT TEMPLATE ----------------
    def build_chat(xml_str):
        return [
            (
                "user",
                [
                    {
                        "type": "text",
                        "text":
                            "You are a GUI agent.\n\n"
                            f"Task:\n{TASK}\n\n"
                            f"Accessibility Tree (XML):\n{xml_str}\n\n"
                            "Output the next action."
                    }
                ]
            )
        ]

    # ---------------- WARM-UP ----------------
    inference_chat_llama_cpp(build_chat(xml_strings[0]))

    # ---------------- STATS ----------------
    prefill_times = []
    decode_times = []
    e2e_times = []

    prefill_tps_list = []
    decode_tps_list = []

    # ---------------- BENCHMARK ----------------
    for i in range(N):
        xml = xml_strings[i % len(xml_strings)]
        chat = build_chat(xml)

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
            f"[Run {i+1}] XML={i % len(xml_strings)} | "
            f"Prefill={prefill_s:.3f}s ({prefill_tps:.1f} tok/s) | "
            f"Decode={decode_s:.3f}s ({decode_tps:.1f} tok/s) | "
            f"E2E={e2e_s:.3f}s"
        )

    # ---------------- SUMMARY ----------------
    print("\n=== Averages (XML-only) ===")

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
