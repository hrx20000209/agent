# 启动脚本与复现

## 代码部署

先 checkout 独立 AndroidWorld 工作目录至 `643303c9fb8c010edeb0dec3a63887055e39ca95`，安装该仓库依赖，再复制本分支覆盖层：

```sh
cp -R cpx/code_overlay/. /path/to/android_world/
```

覆盖层包括修改源码、新增 MAI runner、配置、官方系统提示词和词频先验。`patches/` 仅用于审阅，不要在复制覆盖层之后再次 apply patch。实验使用 Python3.11 容器环境，模型服务 vLLM0.30.0；依赖并非锁版本环境，不保证逐 token 重现。

## 原始启动顺序（历史脚本）

这些是当时实际使用的 host-side Python 脚本，保留硬编码路径和 tag；不是在本仓库里直接运行的通用安装器。脚本执行会启动实验，查看请只读。

- 修复前 FULL45：`historical/mai_full50_launch.py`。
- 修复后 FULL45：`historical/mai_recovery_full50_launch.py`，依赖上一轮完成审计及修复源码 SHA 清单。
- BASE：`historical/mai_base12_create.py` 准备新增容器，`mai_base12_launch8.py` 启动旧8个，`mai_base12_ready_start.py` 加入新4个。
- 当时 worker11 预检失败，未产生任务记录；`mai_base12_replace11.py` 替换，`mai_base12_start11.py` 单独启动。此文件是实际现场补救记录，不是常规流程。

## worker 命令模板

假定代码挂载 `/androidworld`，容器已通过 a11y/短信/model 预检，模型 API 从容器可达；每个 worker 必须使用独立模拟器。每轮使用全新 tag。`<SOURCE_TAG>` 指已恢复的原始 `mai8b_full232_shared_20261025`，worker 从该 run_config 读取116题顺序、task seed和 endpoint。

BASE 每容器一个 worker，ID0..11：

```sh
python -u experiments/trace_reachability_validation/scripts/mai_base_full50_worker.py \
 --tag <NEW_BASE_TAG> --worker-id <ID> --workers 12 --max-calls 50 \
 --source-run mai8b_full232_shared_20261025 \
 --config experiments/trace_reachability_validation/configs/mai_ui_8b_base_full50_20261026.json
```

FULL45 每容器一个 worker，ID0..7（完整参数以 `--help` 与历史脚本为准）：

```sh
python -u experiments/trace_reachability_validation/scripts/mai_full45_full50_worker.py \
 --tag <NEW_FULL_TAG> --worker-id <ID> --workers 8 --max-calls 50 \
 --config experiments/trace_reachability_validation/configs/mai_ui_8b_full50_20261025.json
```

历史脚本使用 `docker exec -d ... sh -c 'exec nohup ... > log 2>&1 < /dev/null'`，并启动 `mai_completion_watch.py --tag TAG`，无需 SSH 会话存活。共享图目录由 worker 初始化；不要把上轮共享图复制成新轮初始图。worker 的已完成目录保护不允许静默覆盖/择优重跑。

服务参数：MAI-UI-8B BF16、max_model_len24576、max_num_seqs8、max_num_batched_tokens1024、max_images3、TRITON_ATTN、enforce_eager、gpu_memory_utilization0.97。12个容器并不代表模型服务有12路完整上下文并发。
