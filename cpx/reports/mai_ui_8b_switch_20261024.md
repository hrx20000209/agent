# MAI-UI-8B / AndroidWorld 接入核验（2026-10-24）

基于 android_world main 643303c，工作分支 codex/mai-ui-8b。代码保存在独立 checkout，不覆盖原目录的未提交改动。

## 官方来源与配置

官方仓库固定至 3deabf1d1fcf31964511929938b7c9485589b29c：
- MAI-UI/src/prompt.py 的 MAI_MOBILE_SYS_PROMPT（原始 AndroidWorld 导航版，无 MCP）。
- MAI-UI/src/mai_naivigation_agent.py 的默认参数、坐标转换、历史序列化及消息拼装。
- 官方来源：https://github.com/Tongyi-MAI/MAI-UI/tree/3deabf1d1fcf31964511929938b7c9485589b29c/MAI-UI

configs/mai_ui_8b_official.json 明确记录：history_n=3，temperature=0，top_k=-1，top_p=1，max_tokens=2048，frequency_penalty=0，presence_penalty=0，repetition_penalty=1，请求 seed=42，失败最多尝试3次。AndroidWorld 任务 seed 另外设置为30，两者不同。

configs/mai_ui_official_system.txt 是官方原始提示词，SHA256 写入配置并在启动时核对。客户端不发 GELab 提示词，也不附加 GELab 的 stuck hint。官方文本版权为 Copyright (c) 2025, Alibaba Cloud and its affiliates，Apache License 2.0；来源链接如上。

## 已实现

1. 每次发送官方 system、独立的 user task、全部历史 assistant 动作，以及最近两张历史截图和当前截图。全历史文本保留，旧图片不再发送。图像不额外缩放。结构化动作按999坐标制保存，执行前转换为像素。
2. menu 保留在官方动作空间中，通过 Android ADB keyevent 82 执行。AndroidWorld JSONAction 没有单独 menu 类型，适配层以 unknown + KEYCODE_MENU 标识并专门执行；恢复重放也识别该标记，不会把它当普通 unknown 忽略。
3. 执行成功后才记入历史；探索、回滚和任务初始化不写成模型主历史。FULL45 自动执行的图动作需要模型知道，故将真实已执行动作转换成相同原生结构，同时标注日志 origin=graph。其未填字段反馈通过 user tool-result 消息保留。这是 FULL45 在官方基础消息格式上的必要扩展，官方单模型 agent 本身没有图执行模块。
4. 按官方代码处理2维点、4维框、thinking/think兼容。官方拖动历史序列化对 start/end 字段保持0..1值的现有行为也保留；差分检查覆盖拖动、bbox、Unicode。没有自行修正该上游行为。
5. 模型请求采用官方非流式模式。FULL45 的决策就绪事件在完整响应后发出，异常路径也保证唤醒；探索截止时间使用近期完整响应耗时，保留2秒恢复余量。未实现 GELab 的 summary 尾部解码重叠。
6. mai_androidworld_driver.py 支持 BASE/FULL45 配对、seed30、交错顺序、独立空图目录、结果续跑和配置/源码哈希检查。记录HTTP尝试数与决策调用数，遇执行错误即停止，失败数据保留。原 e2e_driver.py 不改。

## 已验证

- 8组协议回归和8组执行边界回归通过，覆盖menu/恢复重放、异常事件通知、探测不入主历史、失败动作不记历史、重定位后的坐标。
- 直接编译官方消息拼装与历史序列化方法，与适配输出逐项比较：0/1/2/5步历史通过，包含点击、bbox、拖动和Unicode。
- 真实 MAI-UI-8B 三张1080×2400图片请求通过，按钮定位正确，4.849秒，无OOM。
- 独立 AndroidWorld 模拟器实测原生menu解析到KEYCODE_MENU执行成功，随后Back；两个动作均写入官方结构化历史。
- AndroidWorld BASE 的 RecipeAddSingleRecipe，seed30，完成20次有效模型调用和执行，未在预算内完成任务，评测0，wall 174.42秒；不作为效果结论。第一次因新容器缺采集服务失败的记录另行保留。
- 首次FULL45小试暴露了遗留历史页面加载依赖；已按在线设计修订，见下节。该次失败记录保留，不作为任务效果结论。

## 在线空初始化修订

`observations.jsonl` 是旧版 `MEM.Snapshot` 的历史转移记录。`full7.StuckExplorer` 仅从中提取 src/dst 页面键用于页面新颖性判断，它不是本轮在线图。先前将缺文件判断成“必须补齐历史资产”不符合 FULL33+ 的在线设计，已撤回该结论。

按 `mobileexplorer_design_v34_zh.md` 的“本轮运行中逐步追加，跨任务持久。不读 profiling 图和历史轨迹”，FULL33–FULL45 的 StuckExplorer 现使用 online_only：新运行的 known 集合为空，只加载当前运行目录内在线产生的 known_states.txt；旧版实验分支保留其原有历史加载行为。批量驱动移除了历史文件前置检查，并在 manifest 中记录空初始化策略及 full7.py 哈希。没有创建占位 observations 文件。

每个新 tag 使用独立图目录；同一轮内跨任务累积，不是每个任务都清空。已有 tag 的续跑保留本轮在线数据，正式新实验须用新 tag。

## 必须汇报的未满足条件

### app_tokens.json 统计文件缺失

全新 tag `full45_empty_graph_20261024` 已通过空图启动并发出1次模型请求，不再读 observations.jsonl；随后 `explore_value.app_doc()` 因缺少 `experiments/runtime_graph_exploration/graph/app_tokens.json` 停止。这是按 app 汇总的词表，用于全局 IDF 权重，不是在线节点/边；设计明确保留此类离线统计。该次原始失败结果保留在该 tag 下。下方显式空统计 smoke 随后通过；正式原统计配置仍有此缺失项。

### 运行环境差异

- 当前推理设备为UST RTX4090 D，而原e2e116n在Orin；耗时不能直接横比。
- 官方README固定vLLM0.11.0；当前已验证服务是0.30.0、BF16、eager，不宣称复现官方运行时版本。
- 单卡24GB配置为16K上下文、最多3图、并发上限8（启动smoke时为1）。保留全部历史时，长任务可能超上下文；会记录错误停止，不静默删图或截断历史。
- 官方非流式请求与GELab原FULL45的summary尾部重叠不同。
- 独立容器没有原实验的Qwen背景负载，驱动明确记录background_load=false。

## 位置与后续运行

服务器代码：/home/pxchen/workspace/android_world_mai8b_20261024
模型服务：GPU4，pxchen，8097，MAI-UI-8B；服务日志位于 /home/pxchen/workspace/services/mai_ui_8b_gpu4_20261024/server_official_3img.log。
独立测试容器：pxchen-mai8b-official-20261024；未操作学长原容器的GUI。测试镜像原先缺openai、matplotlib、fast-a11y APK，已在独立容器补齐。fast-a11y APK来自已有实验安装包，仅读取原安装包；SHA256为c150a0ed0d3c8f656da7d216cfa284e8cfe8bf2b5008134354a5a28f24571bf6。

在独立AndroidWorld容器中，用全新 tag 从空图开始：

```bash
cd /androidworld
python experiments/trace_reachability_validation/scripts/mai_androidworld_driver.py \
  --tag mai8b_official_run1 --arms BASE,FULL45 --seed 30 \
  --tasks RecipeAddSingleRecipe \
  --api-url http://172.17.0.1:8097/v1/chat/completions
```

正式116题可用 --task-order 指向原e2e116n_task_order.txt，并用全新tag。尚未启动全量实验。

## 空初始化 smoke 模式（2026-10-24）

`mai_androidworld_driver.py --smoke --tasks RecipeAddSingleRecipe` 显式限制每题8次模型推理，使用等权 IDF 和空离线可回滚先验；没有读取或创建伪造的 app_tokens/observations 文件。该模式保留算法中的在线证据更新和内置回滚先验均值0.75，属于接口与运行稳定性检查，不是原FULL45效果复现。MAI系统提示、采样参数和3图历史保持官方适配配置。默认不加 --smoke 的正式模式仍要求原统计文件。

运行 tag：full45_cold_smoke_20261024。仅指定一题，执行结束后自动退出，不启动全量实验。

smoke 结果：PASS。RecipeAddSingleRecipe，seed30，任务判定 is_successful=1.0、done=true，5次模型调用/5次HTTP请求，无重试，无运行错误；达到任务完成条件提前退出（上限8次）。wall111.62秒，driver端到端115.72秒。实际执行11个动作，表单模块执行6步。本题未触发探索探针（n_probes=0），因此不证明探索/回滚分支稳定。docker top 已确认没有残留 mai_run.py / mai_androidworld_driver.py 进程；未启动116题实验，保留模型服务和模拟器。原始输出、在线图与smoke_summary.json均保留在该tag目录。

## vLLM 并发8压测（2026-10-24）

服务空闲后将 `--max-num-seqs 1` 改为8，其他服务参数保持不变。当前API PID1173986，日志 `server_concurrency8.log`；旧启动脚本备份为 `launch_seq1_before_concurrency8_20261024.sh`。单实例仍为GPU4/8097，未启动新的AndroidWorld批量任务。

使用 `mai_concurrency_probe.py` 发送同一批8个并发回放请求：取已完成smoke的不同1080x2400截图，使用官方系统提示、采样配置，构造两步wait历史加当前图；不实际执行设备动作。三图两组payload哈希逐项一致，completion token数也逐项一致。未重置prefix缓存，且每组只有一批，以下为容量/吞吐探测，不能当正式性能基准。

| 服务上限 | 每请求图片/输入tokens | 成功请求 | 批次耗时 | 每分钟请求 | 采样峰值运行 | 峰值KV占用 | 抢占增量 |
|---|---|---|---|---|---|---|---|
| 1 | 3 / 8260 | 8/8 | 28.90s | 16.61 | 1 | 43.5% | 0 |
| 8 | 3 / 8260 | 8/8 | 20.21s | 23.75 | 2 | 84.2% | 0 |
| 8 | 1 / 3072 | 8/8 | 10.71s | 44.81 | 7 | 99.7% | 2 |

结论：并发上限8可以启动并接住8个请求，但三图输入下当前19,200-token KV容量只能让约2个这类请求同时运行；其余排队。三图单批吞吐提升约1.43倍。短的一图请求采样到7路同时运行，但发生2次抢占，不代表稳定8路常驻解码。保留上限8，增加容器前仍需按实际请求长度压测；正式时延对照应排除其他组的请求竞争。

原始数据：服务目录下 `probe_limit1_3img.json`、`probe_limit8_3img.json`、`probe_limit8_1img.json`；本地 `mai_official_validation_20261024/` 同名副本。

## 跨容器共享图

已补充4 baseline workers + 4 FULL45 workers的任务分片与FULL45全局在线图共享入口，详见 `mai_shared_graph_parallel_20261024.md`。不是每worker单独一张图。24项回归和双FULL45容器真实任务smoke通过；正式全量尚未启动，原统计资产缺失仍需处理。
