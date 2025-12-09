import time
import torch
import base64
from transformers import AutoProcessor, AutoModelForVision2Seq

# ================================================================
# Load model
# ================================================================
model_path = "../models/ui_tars_2B"

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
    device_map="auto",
    dtype=torch.float16
)
model.eval()

# ================================================================
# Base64 image
# ================================================================
image_path = "./screenshot/screenshot.png"
with open(image_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

data_url = f"data:image/png;base64,{b64}"

# ================================================================
# Build chat messages
# ================================================================
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": data_url},
            {"type": "text", "text": "Describe what is in the screenshot?"}
        ]
    }
]

# ================================================================
# Convert to model inputs
# ================================================================
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

input_ids = inputs["input_ids"]
num_prefill_tokens = input_ids.shape[-1]     # ← prefill 的 token 数

# ================================================================
# Prefill latency: 单次 forward
# ================================================================
torch.cuda.synchronize()
t0 = time.time()

with torch.no_grad():
    _ = model(
        **inputs,
        use_cache=True
    )

torch.cuda.synchronize()
prefill_latency = time.time() - t0
prefill_tokens_per_sec = num_prefill_tokens / prefill_latency

print(f"\n=== Prefill ===")
print(f"Prefill latency: {prefill_latency:.4f} s")
print(f"Prefill tokens:  {num_prefill_tokens}")
print(f"Prefill speed:   {prefill_tokens_per_sec:.2f} tokens/s")

# ================================================================
# Generation
# ================================================================
torch.cuda.synchronize()
t1 = time.time()

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True
    )

torch.cuda.synchronize()
gen_total_latency = time.time() - t1

# 计算 decode token 数（真实生成量）
generated_tokens = output_ids.shape[-1] - num_prefill_tokens
generated_tokens = max(generated_tokens, 0)

decode_latency = max(gen_total_latency - prefill_latency, 0)
avg_decode_latency = decode_latency / generated_tokens if generated_tokens > 0 else float("nan")
decode_tokens_per_sec = generated_tokens / decode_latency if decode_latency > 0 else float("inf")

print(f"\n=== Decode ===")
print(f"Generated tokens: {generated_tokens}")
print(f"Decode total latency: {decode_latency:.4f} s")
print(f"Decode avg latency/token: {avg_decode_latency:.4f} s")
print(f"Decode speed: {decode_tokens_per_sec:.2f} tokens/s")

# ================================================================
# Decode text
# ================================================================
generated_text = processor.decode(
    output_ids[0][num_prefill_tokens:],
    skip_special_tokens=True
)

print("\n=== Model Output ===")
print(generated_text)

# ================================================================
# Summary
# ================================================================
print("\n=== Summary ===")
print(f"Prefill latency: {prefill_latency:.4f} s")
print(f"Prefill tokens: {num_prefill_tokens}")
print(f"Prefill speed: {prefill_tokens_per_sec:.2f} tokens/s")

print(f"\nGeneration latency (total): {gen_total_latency:.4f} s")
print(f"Generated tokens: {generated_tokens}")
print(f"Decode speed: {decode_tokens_per_sec:.2f} tokens/s")
