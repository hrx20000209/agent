# GUI-Owl-1.5-8B-Instruct：完整实验记录与 open_app 修复

本目录归档10月31日原始 BASE/FULL45 全量、11月1日 BASE 七题修复重跑，以及修复后的 FULL45 从空图全量重跑。仅修复 `open_app` 动作别名；模型、提示词、采样、预算、种子、图算法与探索配置保持不变。

## 最新结果

| 口径 | 方法 | 成功率 | 平均步数 | 成功题步数 | 共同成功步数（vs BASE） |
|---|---|---|---|---|---|
| 全部116 · GUI-Owl 8B | BASE：原109题＋修复后7题 | **49.14%（57/116）** | 14.41 | 9.14 | — |
| 全部116 · GUI-Owl 8B | FULL45：修复后从空图全量 | **48.28%（56/116）** | 9.94*（−31.1%） | 7.14 | **48道：7.38 vs 9.00（−18.1%）** |

- 步数指 `model_calls`，不包含图执行、探索探针或额外 Android 操作。成功严格为 `success == 1` 且无执行错误/非零退出码；部分得分不算完整成功。
- BASE 是固定选定的7道接口异常题替换原结果，其他109题保留，不是重新完整运行116题，也不是对所有失败题反复重试取最好结果。
- FULL45四题异常的 `model_calls` 缺失，因此均值按112题计算；BASE均值按116题。−31.1%不是相同题集上的严格配对比较。共同成功48题的−18.1%才是配对调用数比较。
- 成功率分母均为116，异常题计失败，不删题。缺失调用数不填0。
- BASE七题于11月1日01:08（北京时间）结束。FULL45从00:49至04:53，约4小时04分钟；全量最后一题发生容器OOM，仍保留为异常失败记录。
- BASE合并结果不能作为一次连续全量运行的总墙钟时间；本表不据此声称端到端加速比例。

## 原始成绩与修复效果

| 阶段 | BASE成功率 | FULL45成功率 | 说明 |
|---|---|---|---|
| 原始全量 | 46.55%（54/116） | 45.69%（53/116） | BASE七题、FULL45三题因不支持open_app退出；FULL45另有三题坐标解析异常 |
| 本次修复后 | 49.14%（57/116） | 48.28%（56/116） | BASE补测固定七题；FULL45重新从空图顺序跑116题 |

FULL45是新的顺序运行，图和探索会随轨迹变化，净增三题不能全部归因到某三条原始异常。两轮原始数据均保留，`computed_summary.json` 提供增益/退化任务列表。

### BASE固定七题

| 任务 | 重跑结果 | 模型调用数 | 终止原因 |
|---|---|---:|---|
| MarkorCreateNote | 成功 | 15 | 模型结束 |
| MarkorChangeNoteContent | 失败 | 12 | 预算耗尽 |
| MarkorMergeNotes | 失败 | 78 | 预算耗尽 |
| NotesMeetingAttendeeCount | 成功 | 5 | 模型结束 |
| SimpleCalendarDeleteEventsOnRelativeDay | 成功 | 9 | 模型结束 |
| SportsTrackerLongestDistanceActivity | 失败 | 2 | 模型结束但评测未通过 |
| TasksDueNextWeek | 失败 | 10 | 预算耗尽 |

七题均没有再发生open_app异常。启动时曾发生无障碍与短信数据库就绪探测失败，监控自动重启两次后才开始正式任务，没有将启动失败计成任务失败。正式七题无执行异常。

### FULL45仍存在的异常

- ExpenseDeleteSingle、SystemBluetoothTurnOn、SystemBluetoothTurnOnVerify：模型给出 `[1015, 993]`，当前解析器拒绝超出其接受范围的坐标。此轮未修改坐标逻辑，也未补跑这三题。
- VlcCreateTwoPlaylists：最后一题执行到41条有效决策、42次HTTP尝试时容器退出。Docker记录 `OOMKilled=true`、退出码137，容器内存上限8GiB。最终model_calls缺失，不能把已观察41条当作最终调用总数。记录在[最终审计](records/full45_fixed/final_audit.json)及原始任务日志中。

本次修复只消除了动作别名不兼容，不代表全部官方执行语义或69%成绩条件已经复现。

## 启动设置

- 权重：`mPLUG/GUI-Owl-1.5-8B-Instruct`，revision `06d5faecff74840bab2be2425e9c42667a5d04fc`。
- 同一个纯GPU vLLM服务：GPU4/5、TP2、BF16、CPU offload0，32K上下文，最多5图，两路并发。
- 每次请求：temperature0、top_p1、top_k−1、seed42、max_tokens2048。
- 官方GUIOwl系统提示词与最近五图历史处理；更早动作转文字。执行桥仍是本项目适配器。
- 每题预算 `int(10 × complexity)`，不封顶；同一直接任务seed30，原学长116题序。
- 题序SHA256：`595290a05c7c4a3012f04086dc50ff6631126b62c66700ea527e4ed275ff8074`。
- FULL45独立单容器，空在线图启动，跨116题顺序积累；保留同一历史统计先验；explore开启，窗口30秒、max_probes3。
- Android容器8GiB/4CPU，模拟器4GiB/4核，原镜像与命令不变。host-resident nohup监控保留超时、容器崩溃和下一题恢复策略。
- BASE和FULL45未改成官方create_suite的哈希任务种子；显式贪心采样也不能宣称就是官方69%报告的采样设置。

实际配置见[原始run_config](records/original/run_config.json)、[BASE七题run_config](records/base7/run_config.json)、[FULL45修复后run_config](records/full45_fixed/run_config.json)。部署脚本会核对配置、题序、镜像、源码哈希及允许的代码差异，不自动启动另一组。

## 代码与数据入口

- [BASE七题修复启动包](launch/base7/)、[FULL45修复全量启动包](launch/full45_fixed/)：部署脚本、完整package.tar.gz、可读适配器、测试和host supervisor。
- [唯一算法修复补丁](launch/base7/adapter_fix.patch)：官方转换器支持open/open_app两种写法，本地增加别名并保留原始输出历史。
- [模型服务与原始成对启动设置](../guiowl_run_20261031/)：原始模型下载/服务参数及两容器启动包。
- [逐题记录](records/)：不解压即可查看runs.jsonl、配置、审计、图评分文本和BASE修复合并记录。
- `original_logs.tar.gz`、`base7_logs.tar.gz`、`full45_fixed_logs.tar.gz`：完整非截图文件，包括HTTP尝试/响应、决策、动作、meta/result、steps/windows、a11y界面树、最终在线图、异常日志。`actual_launch/`额外保存服务器实际源码和配置。
- `*_files.json`：每个运行的所有文件大小、SHA256、归档包含标记和服务器路径；包含完整截图清单。
- [关键异常截图](key_screenshots/)：各异常任务最后两张已保存截图，原始BASE部分任务只有一张。

全部截图约1.00GB，完整保存在服务器，不放入Git中的日志压缩包。没有声称截图全集已上传；关键截图已入库，其余可按清单定位及校验。日志、动作、界面树与图文件均完整归档。服务器父目录为 `/home/pxchen/workspace/android_world_mai8b_20261024`，每轮完整路径见各文件清单的 `server_root`。

## 无模型、无Android的复算与校验

```bash
python3 summarize.py
python3 verify_run_archives.py
```

`summarize.py`从原始逐题行重新构造BASE固定七题替换、均值分母和共同成功交集，并与存储统计/合并行比对。`verify_run_archives.py`逐一校验全部非截图文件、服务器实际源码、便于阅读的记录副本和关键截图。分支根目录还可执行 `python3 cpx/tools/verify_artifacts.py`。

验证包括14项协议/host运行器测试、42条原始open_app失败输出经实际运行时动作转换的回放，FULL45启动前容器内协议测试及budget/import校验。实际输出分别见日志包及launch测试记录。

官方动作转换参考：[mobile_agent_utils_new.py](https://github.com/X-PLUG/MobileAgent/blob/main/Mobile-Agent-v3.5/android_world_v3.5/android_world/agents/mobile_agent_utils_new.py)。
