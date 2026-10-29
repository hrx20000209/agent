"""OpenAI request parameters from the pinned official MAI navigation agent."""
import os


class OfficialClient:
  def __init__(self, api_url, model_name, config):
    self.api_url, self.model_name, self.config = api_url, model_name, config

  def payload(self, messages):
    c = self.config
    return {"model": self.model_name, "messages": messages,
            "max_tokens": c["max_tokens"], "temperature": c["temperature"],
            "top_p": c["top_p"], "top_k": c["top_k"],
            "frequency_penalty": c["frequency_penalty"],
            "presence_penalty": c["presence_penalty"],
            "repetition_penalty": c["repetition_penalty"], "seed": c["request_seed"], "stream": False}

  def predict_mm(self, text_prompt, images, messages=None):
    import requests
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("ANDROID_WORLD_LLM_API_KEY") or os.environ.get("VLLM_API_KEY")
    if token:
      headers["Authorization"] = "Bearer " + token
    response = requests.post(self.api_url, json=self.payload(messages), headers=headers,
                             timeout=self.config["request_timeout_s"])
    if response.status_code != 200:
      raise RuntimeError(f"MAI HTTP {response.status_code}: {response.text[:1200]}")
    result = response.json()
    if result["choices"][0].get("finish_reason") == "length":
      raise RuntimeError("MAI response reached max_tokens=2048; no partial action executed")
    return result["choices"][0]["message"]["content"], True, result
