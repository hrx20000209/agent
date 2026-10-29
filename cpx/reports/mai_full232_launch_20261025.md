# MAI-UI-8B 全量并行实验（2026-10-25）

已于北京时间2026-10-25 01:09启动，tag：`mai8b_full232_shared_20261025`。BASE/FULL45各116题，合计232个episode。每组4个容器，按同一任务列表索引对4取模分配，每worker29题，task seed=30。列表由当前AndroidWorld registry排序生成并保存在本轮task_order.txt。未使用smoke参数；采用原任务复杂度预算，单题最多20次模型调用。

FULL45四个容器共同使用本轮 `graph/FULL45/`，整轮从空初始化；BASE不读取该图。没有载入历史observations或离线流程库。

## 运行位置

服务器代码：`/home/pxchen/workspace/android_world_mai8b_20261024`。
输出：`experiments/trace_reachability_validation/raw/mai_official/mai8b_full232_shared_20261025/`。
配置：`experiments/trace_reachability_validation/configs/mai_ui_8b_full_20261025.json`。
每worker由mai_worker_service.py脱离SSH运行，workers/下有独立log和status.json；任务结果汇总在runs.jsonl。单个worker出错会停止并保留日志，不自动用已有热图重跑。

| 组 | worker | 容器 |
|---|---|---|
| BASE | 0 | pxchen-mai8b-official-20261024 |
| BASE | 1 | pxchen-mai8b-base-w1-20261025 |
| BASE | 2 | pxchen-mai8b-base-w2-20261025 |
| BASE | 3 | pxchen-mai8b-base-w3-20261025 |
| FULL45 | 0 | pxchen-mai8b-shared-w1-20261024 |
| FULL45 | 1 | pxchen-mai8b-full-w1-20261025 |
| FULL45 | 2 | pxchen-mai8b-full-w2-20261025 |
| FULL45 | 3 | pxchen-mai8b-full-w3-20261025 |

所有容器挂载同一工作区到/androidworld，共用GPU4上的MAI-UI-8B 8097服务；并发上限8，16K上下文，最多3图，BF16/eager。三图负载通常仅约2路同时推理，剩余请求排队；混合组wall time不是独占服务的加速比。

## 统计资产恢复

按用户确认，采用rxhuang服务器10月23日重跑使用的重建词表，不宣称是原FULL11–14统计的精确副本。

来源：`/data/rxhuang/android_world_repo/experiments/trace_reachability_validation/graph_snapshots/main643303c_base_full44_sleep10_no_local_model_20261023/historical_app_tokens.json`。

复制到本工作区的 `experiments/runtime_graph_exploration/graph/app_tokens.json`，文件属于pxchen（UID1026/GID1028），逐字节SHA256一致：`4d367968e98b985c0f18ab0d7a4225cd3fba5358d7f0ce9f0ced578e42d8f325`。

26个App，4319个App-token条目；来源manifest记录2357个历史页面，来自此前BASE/FULL44的全量输出。原始来源说明明确该重建数据不等同于原始历史语料。

本工作区任务侧IDF语料原为空，现从同一此前全量的BASE元数据恢复116题指令（106条不同文本）到raw/full/BASE/<task>/p0/meta.json，仅用于goal_idf统计。没有复制动作、截图或结果到此统计目录。既有rev_prior/ui_idf/element_idf继续使用；可回滚先验有25个统计项。资产与语料说明保存在configs/statistics_rxhuang_20261023/，SHA写入本轮配置，每次preflight校验。

## 启动验证与时间预估

8个容器均通过模型/统计哈希/模拟器/fast-a11y预检，且都已实际发送模型请求。首对AudioRecorderRecordAudio均成功：FULL45 7次调用、149.32秒端到端；BASE 8次调用、156.03秒端到端，无重试与运行错误。此为启动观察，不作为总体效果结论。

初步预计全量总耗时4–8小时，即北京时间10月25日05:00–09:00左右。该估计基于单vLLM共享吞吐与任务调用预算；首批样本有限，长任务、排队、错误停止会影响完成时间。
