import os
import base64
import requests
from time import sleep
import json


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def track_usage(res_json, api_key):
    """
    {'id': 'chatcmpl-AbJIS3o0HMEW9CWtRjU43bu2Ccrdu', 'object': 'chat.completion', 'created': 1733455676, 'model': 'gpt-4o-2024-11-20', 'choices': [...], 'usage': {'prompt_tokens': 2731, 'completion_tokens': 235, 'total_tokens': 2966, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'system_fingerprint': 'fp_28935134ad'}
    """
    model = res_json['model']
    usage = res_json['usage']
    if "prompt_tokens" in usage and "completion_tokens" in usage:
        prompt_tokens, completion_tokens = usage['prompt_tokens'], usage['completion_tokens']
    elif "promptTokens" in usage and "completionTokens" in usage:
        prompt_tokens, completion_tokens = usage['promptTokens'], usage['completionTokens']
    elif "input_tokens" in usage and "output_tokens" in usage:
        prompt_tokens, completion_tokens = usage['input_tokens'], usage['output_tokens']
    else:
        prompt_tokens, completion_tokens = None, None

    prompt_token_price = None
    completion_token_price = None
    if prompt_tokens is not None and completion_tokens is not None:
        if "gpt-4o" in model:
            prompt_token_price = (2.5 / 1000000) * prompt_tokens
            completion_token_price = (10 / 1000000) * completion_tokens
        elif "gemini" in model:
            prompt_token_price = (1.25 / 1000000) * prompt_tokens
            completion_token_price = (5 / 1000000) * completion_tokens
        elif "claude" in model:
            prompt_token_price = (3 / 1000000) * prompt_tokens
            completion_token_price = (15 / 1000000) * completion_tokens
    return {
        # "api_key": api_key, # remove for better safety
        "id": res_json['id'] if "id" in res_json else None,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_token_price": prompt_token_price,
        "completion_token_price": completion_token_price
    }


def inference_chat(chat, model, api_url, token, usage_tracking_jsonl=None, max_tokens=2048, temperature=0.0):
    if token is None:
        raise ValueError("API key is required")

    if 'gpt' in model:
        headers = {
            "Content-Type": "application/json",
            "api-key": token
        }
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": max_tokens,
        'temperature': temperature
    }

    if "claude" in model:
        if "47.88.8.18:8088" not in api_url:
            # using official api url
            headers = {
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        for role, content in chat:
            if role == "system":
                assert content[0]['type'] == "text" and len(content) == 1
                data['system'] = content[0]['text']
            else:
                converted_content = []
                for item in content:
                    if item['type'] == "text":
                        converted_content.append({"type": "text", "text": item['text']})
                    elif item['type'] == "image_url":
                        converted_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": item['image_url']['url'].replace("data:image/jpeg;base64,", "")
                            }
                        })
                    else:
                        raise ValueError(f"Invalid content type: {item['type']}")
                data["messages"].append({"role": role, "content": converted_content})
    else:
        for role, content in chat:
            data["messages"].append({"role": role, "content": content})

    max_retry = 5
    sleep_sec = 20

    while True:
        try:
            if "claude" in model:
                res = requests.post(api_url, headers=headers, data=json.dumps(data))
                res_json = res.json()
                # print(res_json)
                res_content = res_json['content'][0]['text']
            else:
                res = requests.post(api_url, headers=headers, json=data)
                res_json = res.json()
                # print(res_json)
                res_content = res_json['choices'][0]['message']['content']
            if usage_tracking_jsonl:
                usage = track_usage(res_json, api_key=token)
                with open(usage_tracking_jsonl, "a") as f:
                    f.write(json.dumps(usage) + "\n")
        except:
            print("Network Error:")
            try:
                print(res.json())
            except:
                print("Request Failed")
        else:
            break
        print(f"Sleep {sleep_sec} before retry...")
        sleep(sleep_sec)
        max_retry -= 1
        if max_retry < 0:
            print(f"Failed after {max_retry} retries...")
            return None

    return res_content


def inference_chat_ollama(
        chat,
        model="0000/ui-tars-1.5-7b-q8_0:7b",
        api_url="http://localhost:11434/api/chat",
        temperature=0.0,
        stream=False,
        num_predict=100
):
    headers = {"Content-Type": "application/json"}
    messages = []

    for role, content in chat:
        text_parts = []
        image_list = []

        for item in content:
            if item["type"] == "text":
                text_parts.append(item["text"])

            elif item["type"] == "image_url":
                url = item["image_url"]["url"]
                if os.path.exists(url):
                    image_list.append(url)  # 本地文件路径
                elif url.startswith("data:image"):
                    image_list.append(url.split(",")[1])  # 去掉 data:image/...;base64, 前缀
                else:
                    raise ValueError(f"Ollama only supports local path or base64, got: {url}")

            elif item["type"] == "image":
                # 如果是 PIL.Image 类型，保存成临时文件
                temp_path = "./temp_image.png"
                item["image"].save(temp_path)
                image_list.append(temp_path)

        msg = {
            "role": role,
            "content": "\n".join(text_parts) if text_parts else " "
        }
        if image_list:
            msg["images"] = image_list

        messages.append(msg)

    # === 构造请求体 ===
    data = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict
        }
    }

    # 调试用
    # print("👉 请求数据:", json.dumps(data, indent=2, ensure_ascii=False)[:500])

    res = requests.post(api_url, headers=headers, json=data)
    res.raise_for_status()
    res_json = res.json()

    # Ollama /api/chat 返回格式: {"message": {"role": "...", "content": "..."}}
    return res_json["message"]["content"]


def inference_chat_llama_cpp(
        chat,
        api_url="http://localhost:8080/v1/chat/completions",
        temperature=0.0,
        max_tokens=200
):
    import requests

    headers = {"Content-Type": "application/json"}
    messages = []

    # 遍历 chat 历史
    for role, content_items in chat:
        content_list = []

        for item in content_items:
            # 1. 处理文本
            if item["type"] == "text":
                content_list.append({
                    "type": "text",
                    "text": item["text"]
                })

            # 2. 处理图像 (关键修改部分)
            elif item["type"] == "image_url":
                url = item["image_url"]["url"]

                # 确保格式是 data:image 开头 (add_chat 已经保证了这点，这里做个保险)
                if not url.startswith("data:image"):
                    # 如果不是标准格式，可能需要这里做额外处理，或者抛错
                    pass

                    # 【重要】llama.cpp server 需要标准的 OpenAI 格式
                # 不要改成 "type": "image"
                # 不要去掉 "data:image...base64," 前缀
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": url  # 直接传完整的 data uri
                    }
                })

        messages.append({
            "role": role,
            "content": content_list
        })

    # 构建请求体
    data = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    # 发送请求

    res = requests.post(api_url, headers=headers, json=data)

    # 如果出错，打印服务器返回的具体信息，而不是笼统的 500
    if res.status_code != 200:
        print(f"\n[Error] Server Response: {res.text}\n")

    res.raise_for_status()
    js = res.json()

    # print(js["choices"][0])  # debug

    return js["choices"][0]["message"]["content"]

