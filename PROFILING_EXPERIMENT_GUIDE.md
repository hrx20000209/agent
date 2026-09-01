# MobileExplorer 系统 Profiling 实验指南

## 1. 实验目标与测量边界

这套实验有两个目标：

1. 单独测量各部分的内存开销：模型推理、Progressive Belief Graph 建图、ADB exploration。
2. 测量各部分之间的相互影响：一个 workload 是否会改变另一个 workload 的内存或延迟。

实验中必须区分下面两个概念：

- **逻辑图大小**：`live_graph_python_bytes`、snapshot bytes、节点数和边数。在建图参数完全相同的情况下，理论上不应该仅仅因为 llama.cpp 正在运行就发生变化。
- **物理内存占用**：进程 RSS/PSS/USS/Swap，以及设备的 `MemAvailable`。这些指标可能受到 Python allocator、Android 页面回收、内存压缩和交换的影响。

当前实际 MobileExplorer 的图位于电脑端 Python 进程中。因此，Termux 建图实验是一个有意设计的“图与模型、App 共置”系统压力实验。它回答的是：如果图内存也放在手机上，会不会影响手机上的模型和 exploration；它不代表当前电脑端图内存会直接计入 Android 内存。

## 2. 各程序的运行平台

| 程序 | 运行平台 | 作用 |
|---|---|---|
| `profile_android_system.py` | 连接 ADB 的 macOS/Linux/Windows 电脑 | 采集整机内存、PSI、Swap、温度，以及目标 App、Termux、llama.cpp 的 PSS/RSS |
| `profile_model_inference.py` | 电脑端 | 向手机上的 llama.cpp endpoint 发送固定请求，记录 TTFT 和总延迟 |
| `profile_adb_exploration.py` | 电脑端 | 通过 ADB 控制**真机**，记录 App 启动和 exploration 延迟 |
| `profile_graph_growth_termux.py` | Android Termux | 在手机中构建真实的 `ProgressiveBeliefGraph` 数据结构，并通过 `/proc` 读取自身内存 |

正式实验不要使用模拟器。两个 ADB profiling 脚本都会拒绝 emulator serial。

## 3. 一次性环境准备

### 3.1 电脑端

在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
adb devices -l
```

后续每条命令都应使用 `adb devices -l` 显示的真机 serial。不要依赖 ADB 自动选择设备，避免误连模拟器或其他手机。

如果 Termux 中的 llama.cpp 监听 8081 端口，将它映射到电脑上的空闲端口：

```bash
adb -s PHYSICAL_SERIAL forward tcp:8084 tcp:8081
```

因此，后面的例子使用：

```text
http://127.0.0.1:8084/v1/chat/completions
```

模型实验开始前，保存一张有代表性的手机截图到：

```text
evaluation_inputs/fixed_screen.png
```

仓库已经提供固定 prompt：

```text
evaluation_inputs/fixed_prompt.txt
```

所有实验条件必须复用完全相同的 prompt 和截图。

### 3.2 Android Termux

安装 Python、Git，并允许 Termux 访问共享存储：

```bash
pkg update
pkg install python git
termux-setup-storage
```

将本仓库复制或 clone 到 `~/agent`。手机端至少需要：

```text
profile_graph_growth_termux.py
Explorer/__init__.py
Explorer/progressive_belief_graph.py
```

验证脚本：

```bash
cd ~/agent
python profile_graph_growth_termux.py --help
```

## 4. 输出目录规范

不要复用非空输出目录。建议使用下面的结构：

```text
evaluation_results/system_profiling/
  baseline_r01/
    phone_system/
  model_active_r01/
    phone_system/
    model/
  graph_10000_r01/
    phone_system/
    graph/
  graph_10000_exploration_r01/
    phone_system/
    graph/
    exploration/
```

内存实验每个条件至少重复 5 次；延迟实验建议重复 10 次。使用 `r01`、`r02` 等区分重复实验。

不要总是按照从小 workload 到大 workload 的顺序运行。应该随机化实验顺序，避免最后几个大 workload 总是在手机已经升温时运行。

所有新输出都包含 Unix timestamp，因此可以在实验结束后对齐：

- 手机系统 `samples.jsonl`
- Termux graph CSV
- 模型 inference requests
- exploration probes

## 5. 通用手机系统采样器

每次实验开始前，先在电脑端启动下面的采样器。根据实际目标 App package 和进程名称调整参数：

```bash
python profile_android_system.py \
  --adb_serial PHYSICAL_SERIAL \
  --duration_sec 600 \
  --interval_sec 1 \
  --package target_app=ctrip.english \
  --package termux_app=com.termux \
  --process 'llama=llama-server|llama-cli|llama\.cpp' \
  --process 'termux_python=python.*profile_graph_growth_termux' \
  --output_dir evaluation_results/system_profiling/TRIAL_ID/phone_system
```

设置 `--duration_sec 0` 可以一直采样到按下 Ctrl-C。

`dumpsys meminfo` 本身可能比较慢。如果一次采样超过 1 秒，脚本不会并发发送新的 ADB 请求，而是等待本次采样完成。因此，`--interval_sec 1` 是目标间隔，不保证实际严格每秒得到一个样本。

正式实验前先做一次 pilot run，并检查 `samples.jsonl` 前几行。预期的 process group 应满足：

- `pids` 非空；
- `pss_kb` 不为 null；
- llama group 匹配的确实是手机上的 llama.cpp，而不是电脑端 HTTP client。

如果 llama group 为空，先检查：

```bash
adb -s PHYSICAL_SERIAL shell ps -A
```

然后根据实际进程名称修改 `--process` 正则表达式。

## 6. 单独 Profiling

### 6.1 手机空闲基线

条件：

- 目标 App 停留在同一个初始页面；
- 不运行模型推理；
- 不在 Termux 中建图。

运行手机系统采样器 120 秒。这组结果作为 `MemAvailable`、memory PSI、Swap、major page fault 和手机温度的整机基线。

### 6.2 模型内存与推理延迟

模型需要分成两个条件测试。

#### 条件 A：模型已经加载，但处于 idle

1. 启动 Termux 中的 llama.cpp server。
2. 发送两次 warm-up 请求。
3. 停止发送请求，只运行手机系统采样器 120 秒。

这组结果给出模型权重加载后的常驻内存。

#### 条件 B：模型持续进行推理

同时运行手机系统采样器和下面的电脑端命令：

```bash
python profile_model_inference.py \
  --api_url http://127.0.0.1:8084/v1/chat/completions \
  --model GELAB-ZERO-4B \
  --prompt_file evaluation_inputs/fixed_prompt.txt \
  --image evaluation_inputs/fixed_screen.png \
  --runs 30 \
  --warmup_runs 2 \
  --max_tokens 256 \
  --stream \
  --output_dir evaluation_results/system_profiling/model_active_r01/model
```

所有模型实验必须固定 prompt、screenshot、`max_tokens`、模型文件、量化方式、llama.cpp context size 和启动参数。

手机模型内存应使用系统采样器中的 `process_llama_pss_kb`，不能使用电脑端 HTTP client 的 RSS。

模型单独 profiling 最终报告：

- loaded-idle llama PSS/RSS；
- active inference 时的 llama PSS/RSS mean 和 peak；
- active PSS 减去 idle PSS；
- TTFT、总延迟、completion tokens/s；
- memory PSI、Swap、major faults 和温度。

### 6.3 Termux 建图 Profiling

每次实验只测试一个 graph size。在 Termux 中运行：

```bash
cd ~/agent
python profile_graph_growth_termux.py \
  --target_nodes 10000 \
  --sample_every 250 \
  --snapshot_every 1000 \
  --labels_per_node 4 \
  --evidence_chars 32 \
  --gc_at_sample \
  --baseline_sec 10 \
  --baseline_sample_sec 1 \
  --hold_sec 180 \
  --hold_sample_sec 1 \
  --output_dir ~/profiling/graph_10000_r01
```

建议测试：

```text
1,000 / 5,000 / 10,000 / 20,000 nodes
```

还应增加 snapshot-disabled 对照组：

```bash
--snapshot_every 0
```

启动 Termux 命令前，应先在电脑端启动手机系统采样器。

Graph profiling 的主要输出：

- `baseline_memory.csv`：导入 Python 模块后、真正创建 graph 前的进程内存；
- `growth.csv`：内存、插入延迟、snapshot 延迟随节点数的变化；
- `ready.json`：目标图已经构建完成；
- `hold_memory.csv`：图和最新 snapshot 保持存活期间的内存；
- `summary.json`：graph RSS over baseline 和延迟汇总。

实验结束后，把 Termux 私有目录结果复制到共享存储，再由电脑拉取：

```bash
cp -r ~/profiling/graph_10000_r01 ~/storage/downloads/
adb -s PHYSICAL_SERIAL pull /sdcard/Download/graph_10000_r01 \
  evaluation_results/system_profiling/graph_10000_r01/graph
```

### 6.4 ADB Exploration Profiling

冷启动实验：

```bash
python profile_adb_exploration.py \
  --adb_serial PHYSICAL_SERIAL \
  --package ctrip.english \
  --duration_sec 60 \
  --repeats 10 \
  --force_stop_between_repeats \
  --launch_wait_sec 0 \
  --launch_poll_sec 0.10 \
  --ui_stability_poll_sec 0.25 \
  --ui_stability_checks 2 \
  --output_dir evaluation_results/system_profiling/exploration_cold_r01/exploration
```

温启动实验去掉：

```bash
--force_stop_between_repeats
```

在整个 exploration 命令运行期间，同时运行手机系统采样器。

最终报告目标 App PSS、设备 `MemAvailable`、每 60 秒 verified probes 数、probe latency、time to foreground、first screenshot proxy、first A11y tree 和 stable UI time。

`time_to_first_frame_sec` 是第一次截图成功完成的 proxy，不是 Android framework 提供的精确 first-frame 指标，因此结果中不要把它表述为精确 first-frame time。

## 7. 相互影响实验

相互影响实验复用前面的单独 profiling 命令，只是让 workload 在时间上重叠。每个方向回答的问题不同，不能只做一个组合实验后同时推断两个方向。

### 7.1 图大小是否影响模型推理

对 `0 / 1k / 5k / 10k / 20k nodes` 分别测试：

1. 启动手机系统采样器。
2. 在 Termux 中建图，并设置 `--hold_sec 300`。
3. 等待 Termux 输出已经 flush 的 `GRAPH_READY`。
4. 图保持在内存中时，运行 30 次 model inference profiling。
5. 模型 profiling 结束后停止本组实验。

分析模型 TTFT、总延迟、tokens/s 与 graph node count、`live_graph_python_bytes`、Termux Python retained PSS 的关系。

这组实验测试：**graph → model inference**。

### 7.2 持续模型推理是否影响建图

1. 启动手机系统采样器。
2. 在电脑端启动较长的 active model run，例如 `--runs 500`。
3. 在模型持续处理请求时，在 Termux 中开始建图。
4. Graph 输出 `GRAPH_READY` 后停止持续模型请求。

将结果与“不运行模型时建图”的结果比较：

- insertion latency p50/p95；
- graph 总构建时间；
- snapshot latency；
- Termux Python RSS/PSS；
- logical deep size。

如果 logical deep size 不变，但是 RSS 增大，说明模型改变了物理内存环境，而不是改变了图的内容。

这组实验测试：**model inference → graph**。

### 7.3 图大小是否影响 Exploration

对 `0 / 1k / 5k / 10k / 20k nodes` 分别测试：

1. 启动手机系统采样器。
2. 在 Termux 中构建并 hold 对应大小的图。
3. 等待 `GRAPH_READY`。
4. 图仍在内存中时，运行 60 秒 exploration profiling。

比较 time to foreground、stable UI time、probes per 60 seconds、probe latency、A11y latency、screenshot latency 和目标 App PSS。

这组实验测试：**graph → exploration**。

### 7.4 持续模型推理是否影响 Exploration

1. 不在 Termux 中建图。
2. 启动手机系统采样器。
3. 启动持续 model inference。
4. 模型仍持续推理时，运行 exploration profiling。

与单独 exploration 对照组比较。这组实验测试：**model inference → exploration**，也可以帮助区分 CPU contention、thermal throttling 和 memory pressure。

### 7.5 三部分同时运行

1. 启动手机系统采样器。
2. 在 Termux 中建图并保持图存活。
3. 启动持续 model inference。
4. 运行 60 秒 exploration profiling。

至少测试 `0 / 10k / 20k nodes`。这组实验测量三部分共同运行的整体影响，但必须先完成两两实验，再解释三者组合结果。

## 8. 最小完整实验矩阵

| 实验组 | Graph nodes | 模型状态 | 主要被测 workload |
|---|---:|---|---|
| baseline | 0 | off | phone idle |
| model isolated | 0 | loaded-idle / active | model |
| graph isolated | 1k/5k/10k/20k | off | graph |
| exploration isolated | 0 | off | exploration |
| graph → model | 1k/5k/10k/20k | active | model |
| model → graph | 1k/5k/10k/20k | active | graph |
| graph → exploration | 1k/5k/10k/20k | off | exploration |
| model → exploration | 0 | active | exploration |
| combined | 0/10k/20k | active | exploration + model |

如果时间有限，每个 cell 至少重复 5 次。如果需要得出较强的 latency 结论，建议重复 10 次，并报告置信区间，而不仅是 mean。

## 9. 保证实验有效性的控制变量

- 固定手机型号、Android 版本、电池/充电状态、屏幕亮度、网络路径和目标 App 版本。
- Cold launch 和 warm launch 必须分开报告。
- 模型对比必须固定 prompt、image 和 token 数。
- 始终记录温度。如果 latency 随温度升高而恶化，但 PSI/Swap 没有变化，更可能是热降频，而不是内存争用。
- 随机化实验顺序，并在长时间 active model 实验之间让手机降温。
- 除非研究目标明确是“clear data 后第一次启动”，否则不要在 repeats 之间清除 App data。
- Pilot run 必须验证 process matching。缺少 llama/Termux PSS 时，不能得出可靠的组合内存结论。
- `MemAvailable` 是噪声较大的系统级指标。内存归属优先看 process PSS；是否产生真实压力需要结合 PSI、major fault 和 Swap 变化。

## 10. 如何回答研究问题

### 10.1 单个部分占用多少内存

使用该部分自己的基线计算：

- 模型：loaded-idle PSS 与 active inference PSS；
- 图：graph-free Termux Python PSS 与建图后 PSS；
- exploration：相同 App 状态下 exploration 前后的 App PSS，以及整机指标变化。

不要只使用整机 before/after RAM 差值作为进程内存。

### 10.2 三部分共同占用多少内存

报告各 process group PSS、总 PSS、`MemAvailable`、Swap 和 PSI。

不能直接假设三个 isolated memory cost 可以线性相加，因为 Android 可能回收 page cache，进程之间也可能共享页面。

### 10.3 图大小是否影响 Exploration 或模型

将 retained graph PSS/node count 作为自变量，对 latency 或 throughput 做回归或趋势分析。

最有说服力的证据是：

1. 随 graph size 增大，latency 呈 dose-response 上升；
2. probes/s 或 tokens/s 同步下降；
3. 同一阈值附近 PSI、Swap 或 major faults 开始上升；
4. 温度变化不能解释相同趋势。

### 10.4 模型是否影响建图

保持 graph 参数完全相同，比较 no-model 和 active-model 条件下的 insertion latency、snapshot latency、total construction time、logical graph bytes 和 Termux process PSS/RSS。

### 10.5 因果结论注意事项

仅观察到 latency 增加，不能证明存在 memory issue。

更强的证据应同时包含：

- graph size 或 model memory 的剂量变化；
- process PSS 增长；
- PSI、Swap 或 major faults 变化；
- latency/throughput 同步变化；
- 没有明显的温度或 CPU workload 混杂因素。
