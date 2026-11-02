# Qwen3-8B：纯 a11y 输入的 BASE / FULL45 全116题实验

这是唯一计分的 `qwen3_8b_a11y_pair_20261101_full_v2` 成对顺序运行，两组各116题。11月2日00:02（北京时间）开始；FULL45于05:09结束，BASE于05:41结束。**本轮含未修复的恢复回放适配缺陷，不能作为干净的最终对比。** 原始失败和异常完整保留，没有删除任务、补跑择优或合并启动测试成绩。

## 结果

| 口径 | 方法 | 成功率 | 平均步数* | 成功题步数 | 共同成功步数（vs BASE） |
|---|---|---:|---:|---:|---|
| 全部116 · Qwen3-8B | BASE | **13.79%（16/116）** | 14.71 | 6.50 | — |
| 全部116 · Qwen3-8B | FULL45 | **18.10%（21/116）** | 12.89 | 8.33 | **13道：8.85 vs 6.77（+30.7%）** |

- 步数为 `model_calls`：LLM生成请求数，不含图动作、探索探针、Android操作或本地tokenize请求。没有额外LLM摘要请求。
- 成功严格按 `success == 1` 且无执行/退出/teardown异常；分母固定116，异常计失败。所有题的实际调用数均已记录，没有把缺失数填0。
- **平均步数包含异常发生前的实际调用数。** 初始化就失败的题为0，执行中异常为部分调用，提前终止会拉低平均值，不能据14.71→12.89宣称调用加速。
- 仅正常结束的题：BASE 113题，平均15.10；FULL45 106题，平均13.91。这两个集合不同。两组均正常结束的同106题，BASE15.81、FULL45 13.91；同样混有成功/失败及不同终止行为。
- FULL45净多成功5题（+4.31个百分点）；共同成功13题的调用数反而增加30.7%。FULL45独有成功8题、BASE独有成功3题，完整列表见 `computed_summary.json`。
- 两容器无重启。墙钟耗时：BASE约5小时39分，FULL45约5小时07分；异常中断及终止行为不同，不能直接当作方法的端到端加速结论。

## 未修复异常：3题共同初始化失败，FULL45另7题恢复回放越界

两组共同的初始化失败：

- `SimpleSmsReplyMostRecent`：初始化读取接收短信列表第0项时列表为空，`IndexError`；0次模型调用。
- `VlcCreatePlaylist`、`VlcCreateTwoPlaylists`：初始化找不到 `/data/data/org.videolan.vlc/app_db`；0次模型调用。此前的24应用快照检查通过不等于这两题所需的VLC数据库已就绪。

FULL45另有7题在 `full9.relaunch_restore` 中回放索引滚动动作，调用 `_execute_action(a,{})`，未带入对应的UI元素列表；官方actuation访问 `screen_elements[action.index]` 后越界。这是本次适配缺陷，不应归因于模型能力，也不是可忽略的正常失败：

`SportsTrackerActivitiesOnDate`、`SportsTrackerTotalDurationForCategoryThisWeek`、`TasksCompletedTasksForDate`、`TasksDueNextWeek`、`TasksDueOnDate`、`TasksHighPriorityTasks`、`TasksHighPriorityTasksDueOnDate`。

这些异常未在本次归档时修复或重跑。对应 `result.json` 原始traceback可直接在 `records/episodes/` 查看，最后界面的关键截图在 `key_screenshots/`。修复后的实验应另开版本，保留顺序图更新影响，不能用事后修改的源码解释本轮结果。

## 实际配置和代码边界

- 模型 `Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`；BF16、GPU4/5 TP2、CPU offload0、上下文32768、并发上限2、max_tokens2048。
- 每次请求显式 `enable_thinking=false, temperature=0, top_p=1, top_k=-1, seed=42`。贪心采样是本实验设置，不能说是模型卡推荐采样。
- 模型只收任务目标、可见a11y树（层级、文本、ID、状态、边界、索引）、最近24条动作文字历史；超过上下文时先删最旧历史，树本身超限则报错，未悄悄截节点。**没有给模型发送图片**；截图仍可留作内部证据。
- 两组各一个独立Android容器（12GiB/4CPU，模拟器4GiB/4核，API33 Pixel6，emulator `-gpu off`），同一预装镜像，分别完成官方24应用初始化和快照恢复验证。图片模型此前的成绩不是本轮BASE。
- 官方字母顺序116题；suite seed30，实例seed为 `SHA256(30_task_0) mod 2**32`；预算 `int(10*complexity)`，无固定20/30/50封顶。FULL45保留最多2倍预算的计数动作保护、探索窗口上限30秒/3探针。
- BASE关闭图执行与探索。FULL45在线图从空开始，跨题顺序积累；保留app_tokens、function cache等相同历史先验，空在线图不等于没有先验。
- AndroidWorld核心来自 MobileAgent commit `81fa28b564ef7a695521500bc15a0e1324b2b394` 的 `Mobile-Agent-v3.5/android_world_v3.5`；FULL45方法来自学长commit `81a1531758da249c2acb74baa761d2da65e01b1c`。
- **这是CPX控制循环＋官方AndroidWorld核心＋T3A文字动作选择提示词及索引协议，不是未经修改的官方T3A runner。** 未使用T3A额外LLM摘要；每题独立初始化RNG也与此前连续GUI-Owl官方runner不同，不能跨模型直接作严格归因比较。
- 共用适配修复包括：兼容同行`Action:`；保证FULL45决策完成事件被通知；原生a11y通过adb reverse连接当前gRPC服务并在ADB重启后恢复；每次检查实际root/快照状态；关闭视觉帧等待；精确树边界验证索引动作。
- 模型动作执行错误按照官方T3A方式返回文字反馈并消耗预算，继续当前任务；不猜测未知app别名，也不把失败动作学习为成功图边。图恢复执行的原异常路径仍保留，因而暴露了上面的7题缺陷。

实际配置、逐题meta、源码哈希及原始状态以 `records/` 和 `actual_source.tar.gz` 为准。启动时曾遇到多项适配问题，早期测试和中断的初始全量都在独立的 `startup_history_logs.tar.gz` 中，**未合并入本表**。root状态中的部分人工记录时间是检查时间；正式起止以 `records/BASE.status.json` / `FULL45.status.json` 为准。

## 原始数据与复现入口

- `records/BASE.runs.jsonl`、`FULL45.runs.jsonl`：全部232条原始逐题记录；`records/episodes/{BASE,FULL45}/{task}/` 可直接读每题meta/result和异常traceback。
- `formal_logs.tar.gz`：本轮**全部非截图**文件，包含完整模型请求/响应、动作、steps/windows、原生a11y树gzip、恢复日志、最终在线图和审计文件。
- `actual_source.tar.gz`：服务器实际源码、静态先验、配置和实际host启动辅助脚本；`code/` 是关键适配器、FULL45及官方T3A/actuation源码的便于阅读副本。
- `startup_history_logs.tar.gz`：所有早期smoke、中断的初始全量、初始化过程、回归验证、模型下载校验和vLLM服务日志/参数，仅供审计。
- `formal_files.json`、`source_files.json`、`startup_history_files.json`：每个原始文件的服务器路径、大小、SHA256和是否入压缩包。正式截图全集约523MB仍在服务器；Git仅保存关键异常截图和全部截图校验清单，**没有声称截图全集已上传**。
- `environment_versions.json`：Android Python及vLLM环境包版本、容器镜像和资源设置；不包含模型权重。权重的下载revision/逐文件校验在startup归档的service目录。

服务器正式运行目录：`/home/pxchen/workspace/qwen3_a11y_20261101/experiments/trace_reachability_validation/raw/a11y_text/qwen3_8b_a11y_pair_20261101_full_v2`。

无需SSH、Android或模型即可复算/校验：

```bash
python3 summarize.py
python3 verify_archive.py
```

分支根目录可执行 `python3 cpx/tools/verify_artifacts.py`。所有数据是原始记录的快照；归档不会启动或修改实验。
