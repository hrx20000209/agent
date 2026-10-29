#!/usr/bin/env bash
set -euo pipefail
service_root=/home/pxchen/workspace/services/mai_ui_8b_gpu4_20261024
runtime_root=/home/pxchen/workspace/services/minicpmo45_20260929
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="*" no_proxy="*"
export CUDA_VISIBLE_DEVICES=GPU-d152c6a7-1dcf-0ff2-1802-80647045e446
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$runtime_root/venv312/bin:$PATH"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_CACHE_ROOT="$service_root/vllm-cache"
cd "$service_root"
exec "$runtime_root/venv312/bin/python" -m vllm.entrypoints.openai.api_server \
 --model /home/pxchen/workspace/models/MAI-UI-8B \
 --served-model-name MAI-UI-8B --host 0.0.0.0 --port 8097 \
 --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 24576 \
 --gpu-memory-utilization 0.97 --max-num-seqs 8 --max-num-batched-tokens 1024 \
 --limit-mm-per-prompt "{\"image\":3,\"video\":0}" --seed 0 \
 --enforce-eager --attention-backend TRITON_ATTN
