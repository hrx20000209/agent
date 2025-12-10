import time
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# ================================================================
# Load model (按照你给的方式)
# ================================================================
processor = AutoProcessor.from_pretrained(
    "ByteDance-Seed/UI-TARS-2B-SFT",
    trust_remote_code=True,
    use_fast=False,
    local_files_only=True,
    ignore_mismatched_sizes=True,
)

model = AutoModelForVision2Seq.from_pretrained(
    "ByteDance-Seed/UI-TARS-2B-SFT",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ================================================================
# Build messages
# ================================================================
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]

# ================================================================
# Convert to inputs
# ================================================================
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)
prefill_tokens = inputs["input_ids"].numel()

# ================================================================
# 1. Prefill latency (generate 只做 prefill + first token)
# ================================================================
torch.cuda.synchronize()
t0 = time.time()

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=1,
        use_cache=True,
        return_dict_in_generate=True,
        output_logits=True
    )

torch.cuda.synchronize()
prefill_latency = time.time() - t0

prefill_speed = prefill_tokens / prefill_latency

print("\n=== PREFILL ===")
print(f"Prefill tokens:  {prefill_tokens}")
print(f"Prefill latency: {prefill_latency:.4f}s")
print(f"Prefill speed:   {prefill_speed:.2f} tokens/s")


# ================================================================
# 2. Decode latency（多生成一些 token）
# ================================================================
decode_steps = 20

torch.cuda.synchronize()
t1 = time.time()

with torch.no_grad():
    out2 = model.generate(
        **inputs,
        max_new_tokens=decode_steps,
        use_cache=True,
    )

torch.cuda.synchronize()
decode_latency = time.time() - t1

decode_speed = decode_steps / decode_latency

print("\n=== DECODE ===")
print(f"Decode tokens:   {decode_steps}")
print(f"Decode latency:  {decode_latency:.4f}s")
print(f"Decode speed:    {decode_speed:.2f} tokens/s")


# ================================================================
# Decode text
# ================================================================
decoded = processor.decode(
    out2[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print("\n=== MODEL OUTPUT ===")
print(decoded)
