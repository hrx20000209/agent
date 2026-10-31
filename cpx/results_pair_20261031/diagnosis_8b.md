# 8B FULL45 成功率下降和加速有限：逐题诊断

## 结论与证据边界

这轮FULL45并非在BASE之上只增加可选提示：它会自动启动应用、执行函数/填表、复用导航，并在模型推理期间探索和恢复。因此会改变真实设备状态、模型看到的截图和后续历史。已有几处具体实现问题，不能假定是无损增量。

BASE 61/116，FULL45 56/116，下降 **4.31个百分点**。FULL45新成功12题，同时丢掉17道BASE成功题。不是所有题都变差，也不是单纯预算改动造成：同轮两组调用预算一致、顺序一致、seed一致、同一8B服务、各自单独模拟器，FULL45从空图在线积累。

下面是一次seed30配对实验的日志诊断；未做禁用单模块的消融，不把“同时出现”当作所有失败的因果证明。普通FULL45使用学长最新版实现，未启用此前CPX的恢复保护；这点已在启动配置记录。

## 1. 已确认的错误参数填入：短信和Markor

### SimpleSmsReply：确切的正文污染

BASE 10次成功；FULL45 3次失败，且这题没有探索探针，不能归因于探索恢复。

目标：`Reply to +18490934066 with message: The library book is due back on the 15th. in Simple SMS Messenger`。

FULL45 `form_function`直接填入 `The library book is due back on the 15th. in Simple SMS Messenger`，最终无障碍树的`thread_message_body`也显示这整段文本；模型随后宣称发送完成，评测为0。应用名后缀进入了真实短信正文。

原因对应[source_review/shared/fgraph.py](source_review/shared/fgraph.py)的`parse_goal`：`(?:message|text):\s*(.+)$`配合`re.S`把尾部应用说明一起捕获。图函数绕过了模型对正文边界的理解。这是可直接由步骤、最终UI及代码交叉验证的问题。

### MarkorCreateNote / MarkorCreateNoteFromClipboard：图字段值未按控件语义规范化

两题BASE均成功（28/13次），FULL45均失败（8/6次）。图填表把完整`copy_cool_snake.txt`写入文件名控件；最终正文分别与要求文本/剪贴板内容一致，但BASE最终标题为`copy_cool_snake`，FULL45为`copy_cool_snake.txt`。

证据指向文件名控件与扩展名处理，而非正文错误。完整文件名直接填入带独立扩展名处理的控件存在重复后缀风险；本次保存的最终UI不等于磁盘文件列表，**不把“.txt.txt实际路径”写成已证实事实**。下一步应以文件名控件schema和实际文件路径校验修复。MarkorCreateNote没有探针，说明仅恢复保护不能修复这一类问题。

原始完整步骤与最终UI在[cases.json](cases.json)，还保留在8B压缩包对应题目目录。

## 2. 图重复复用放大循环，并提前耗尽总动作上限

[source_review/shared/full_run.py](source_review/shared/full_run.py) `_try_reuse`的`repeat`路径，仅依赖同页最近两次相同导航、相同落地activity等条件；它能复用导航循环。FULL31曾有禁用repeat的特例，FULL45并不继承这个禁用。

| 任务 | BASE调用/结果 | FULL45调用/结果 | FULL45 graph_reuse记录 |
|---|---|---|---:|
| VlcCreatePlaylist | 16 / 成功 | 11 / 失败 | 49 |
| TasksHighPriorityTasks | 2 / 成功 | 20 / 失败 | 11 |
| SimpleCalendarDeleteEventsOnRelativeDay | 10 / 成功 | 24 / 失败 | 18 |
| RetroPlaylistDuration | 失败 | 12 / 失败 | 47 |
| RetroSavePlaylist | 失败 | 12 / 失败 | 47 |

VLC、Retro两题均达到`steps=60`而`done=false`。代码的循环还有第二个限制：`k < 2 * max_steps()`。本轮预算30，故图动作加模型动作达到60时也会停，哪怕模型只调用11/12次。

这三题共143次graph_reuse，占全轮209次的68.4%；“有大量复用”主要不是有效完成任务。低模型调用数会因此产生看似加速的假象。

此外发现**监督统计分类缺陷**：host_supervisor.harvest把“done=false、模型调用未到上限、没有异常”的记录标成`model_done`，遗漏总动作上限。三题原始result的`done=false, steps=60`保留，不能按`termination=model_done`误判为主动完成。它不改变success分数，但会误导停止原因分析。本次不改写历史数据；后续runner需单列`action_step_cap`与未知停止。

## 3. 探索并非无副作用，恢复保护仍有缺口

本轮364个探针，312个非空探索窗口；33个探针`rollback_ok=false`，32个窗口joint_restore失败，17个窗口relaunch_restore失败。26题出现探针回退失败，其中20题未完整成功、6题成功；这些是关联统计，不等于20题都由恢复失败造成。

17道BASE成功→FULL45失败的题中，**4题**出现探针回退失败：MarkorCreateNoteFromClipboard、RetroCreatePlaylist、SimpleCalendarAddOneEventRelativeDay、TurnOffWifiAndTurnOnBluetooth。其余13题没有这类记录，不能全部归到恢复。

`full9.relaunch_restore`只用activity+layout判断恢复成功，另行记录content是否一致却未把它纳入返回值。它既不能证明数据库副作用已撤销，也不能证明开关/文本恢复。相关函数见[source_review/shared/full9.py](source_review/shared/full9.py)。本轮启动按上游普通FULL45运行，没有装旧CPX recovery_guard，旧问题仍可能暴露。

例如WiFi/Bluetooth题2次回退失败，最终得0.5分而BASE为1；日志足以证明部分目标未满足并存在探索干扰风险，不足以单独锁定是哪次探针导致哪个系统设置变化。

## 4. 为什么省调用但墙钟加速不明显

| 口径 | BASE | FULL45 | 变化 |
|---|---:|---:|---:|
| 全116题episode耗时合计 | 15399.81秒 | 14420.42秒 | -6.36% |
| 共同成功44题调用总数 | 435 | 394 | -9.43% |
| 共同成功44题episode耗时合计 | 3861.88秒 | 3472.96秒 | -10.07% |
| 成功解析决策的推理耗时合计 | 8415.49秒 | 6926.96秒 | -17.69% |

两组独立容器并行，墙钟比较按各组逐题episode耗时求和；最后结束时间差约16分钟，与此一致。解析失败请求未进入成功决策推理耗时，所以最后一行不是所有HTTP请求精确成本。

FULL45还有：209次图复用、69条form_function、23条function、75条bootstrap及其他系统步骤，364探针和恢复操作；并有367次`m_stale_pixels`、13次恢复重问。探索窗口累计1123.63秒（18.73分钟），但窗口可能与模型推理重叠，**不能直接把这些时间加到推理时间上，也不能当作纯额外耗时**。日志表明控制/探索成本会抵消推理节省，但未提供每段互斥计时，不能声称已完成精确成本归因。

另外，模型已换成纯GPU，平均模型请求只需数秒；固定截图、adb、控件操作、恢复开销占比更高。曾经CPU offload服务很慢时的推理微基准倍数，不能当作图算法的加速倍数。

整体平均模型调用的6.9%下降还包含成功/失败构成改变、一次缺失调用计数、提前动作上限停机，不能单独代表同质量的效率提升。共同成功44题的9.4%调用/10.1%耗时下降更可比，但仍只描述这个交集。

## 5. 后续修复顺序（本次仅归档和诊断，未启动新实验）

1. 修复目标参数schema：短信正文边界、文件名stem/extension、日期/时间与枚举值按实际字段验证。对SMS和Markor上述边界案例做独立回归。
2. 禁止将相同页面往返当作可无限复用的repeat；记录页→动作→页闭环、连续复用上限，命中后停止图复用并交还模型。先验证VLC、Tasks和Calendar上述循环。
3. 恢复需校验内容和任务相关状态；不可逆写操作不做试探，恢复失败后停探索并基于新截图重决策。把旧CPX保护与上游实现差异明确列出，不混用历史成绩。
4. 修正停止原因统计，区分模型预算、2倍总动作上限、模型主动结束和解析错误；保留原始结果，不补造缺失model_calls。
5. 同题序同预算做禁用repeat、禁用自动填表、禁用探索的消融，先看这些退化题，再决定全量。当前只有一次seed30，不能声称修复后必然追回所有17题。

## 逐题差异与复算

[comparison.json](comparison.json)列出17道丢失题和12道新增成功题；[diagnosis_data.json](diagnosis_data.json)包含全部116题结果、复用/探针/恢复计数与响应尾部。[summarize.py](summarize.py)从不改写的runs.jsonl复算成功率和均值。

## 附录：17道退化题

回退失败仅表示日志同时出现，不自动认定因果。

| 任务 | BASE调用 | FULL45调用 | FULL45分数 | 探针数 | 探针回退失败数 | graph_reuse数 |
|---|---:|---:|---:|---:|---:|---:|
| BrowserMaze | 18 | 30 | 0.0 | 2 | 0 | 0 |
| BrowserMultiply | 20 | 30 | 0.0 | 8 | 0 | 0 |
| MarkorCreateNote | 28 | 8 | 0.0 | 0 | 0 | 0 |
| MarkorCreateNoteFromClipboard | 13 | 6 | 0.0 | 1 | 1 | 0 |
| RecipeDeleteMultipleRecipesWithNoise | 20 | 15 | 0.0 | 3 | 0 | 0 |
| RetroCreatePlaylist | 25 | 30 | 0.0 | 9 | 3 | 0 |
| SimpleCalendarEventsInTimeRange | 3 | 16 | 0.0 | 16 | 0 | 0 |
| SimpleCalendarDeleteEventsOnRelativeDay | 10 | 24 | 0.0 | 4 | 0 | 18 |
| SimpleCalendarEventOnDateAtTime | 9 | 20 | 0.0 | 5 | 0 | 0 |
| SimpleCalendarAddOneEventRelativeDay | 19 | 30 | 0.0 | 1 | 1 | 0 |
| SimpleCalendarAddOneEventTomorrow | 19 | 12 | 0.0 | 5 | 0 | 0 |
| SimpleSmsReply | 10 | 3 | 0.0 | 0 | 0 | 0 |
| SportsTrackerLongestDistanceActivity | 3 | 4 | 0.0 | 3 | 0 | 0 |
| SystemCopyToClipboard | 9 | 20 | 0.0 | 0 | 0 | 0 |
| TasksHighPriorityTasks | 2 | 20 | 0.0 | 0 | 0 | 11 |
| TurnOffWifiAndTurnOnBluetooth | 12 | 11 | 0.5 | 3 | 2 | 0 |
| VlcCreatePlaylist | 16 | 11 | 0.0 | 0 | 0 | 49 |
