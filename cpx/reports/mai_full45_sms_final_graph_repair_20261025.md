# FULL45短信环境修复补跑：加载最终图

按用户最终澄清的范围，只补跑FULL45受短信环境故障影响的7题。BASE沿用已完成结果：原始全量的109个非短信任务加已有短信补测的7题，共116题；完整成功数固定为48。其他任务的普通模型失败保持原结果。

补跑tag：`mai8b_full45_sms_warm7_20261025`，4个原FULL45容器通过nohup运行，没有启动全量重跑。任务为MarkorCreateNoteAndSms、SimpleSmsReply、SimpleSmsReplyMostRecent、SimpleSmsResend、SimpleSmsSend、SimpleSmsSendClipboardContent、SimpleSmsSendReceivedAddress。

图来源是原始全量`mai8b_full232_shared_20261025`结束时的FULL45图，不合并独立短信补测的图。冻结快照`graph_snapshots/mai8b_final_full232_20261025`，包含419个节点、1657条观测、13个函数及1863份历史页面数据；manifest SHA256为`cd3046c35f2c745b75fa2426510f6cc69bf2bff4a1aae9ab71ede76c8aae6f36`。

每题从同一冻结快照复制私有图，保留题内学习，不向快照回写，也不把补测新学的图传给下一题。历史图引用通过variant映射读取快照中的历史页面，当前题的页面写入独立输出，避免同任务/capture名称碰撞。保留原代码中的函数同模板排除规则。模型配置、task seed30、预算与原全量一致；短信IPv6修复和overlay安全引用补丁保留。

验证：本地和服务端4项快照隔离/路径测试通过；实际loader抽查54份历史页面成功；加载验证题FULL45成功，2次模型调用，102.32秒端到端，快照哈希未变。此前运行的BASE单题加载对照仅是验证数据，不用于替换用户要求复用的BASE结果。

收尾进程`mai_completion_watch.py`也由nohup运行。完成7题后，自动生成`reconciled_runs.jsonl`和`reconciled_summary.json`：BASE按上述固定来源复用，FULL45保留原109题并仅用本次7题替换原短信任务。不比较多次尝试后取高分，每一行标记source_run和environment_repair。原始结果全部保留。这是用户指定的环境修复补跑汇总，不称为无中断的冷启动全量。

输出目录：`/home/pxchen/workspace/android_world_mai8b_20261024/experiments/trace_reachability_validation/raw/mai_official/mai8b_full45_sms_warm7_20261025`。进度见workers/*.status.json，最终状态见completion_audit.json。

## 最终汇总

北京时间2026-10-25 12:11:34收尾确认为FINISHED。7/7补跑任务结束，完整成功4题，无缺项、无执行错误，4个FULL45 worker全部FINISHED。

按用户指定的环境修复补跑合并口径，232条结果均有明确来源：

| 组别 | 完整成功 | 完整成功率 | 含部分得分的累计分 |
|---|---:|---:|---:|
| BASE | 48/116 | 41.38% | 48.5 |
| FULL45 | 53/116 | 45.69% | 54.0 |

FULL45多成功5题，提高4.31个百分点。BASE复用原109题及已有7题短信补测；FULL45复用原109题，仅替换本次7题最终图补跑。不是从头重新跑完116题的结果。

本次FULL45补跑成功：SimpleSmsReplyMostRecent、SimpleSmsSendClipboardContent、SimpleSmsSendReceivedAddress、SimpleSmsSend。其余MarkorCreateNoteAndSms、SimpleSmsReply、SimpleSmsResend正常执行但任务判定失败。没有按多次尝试取高分，保留全部原始结果及每条最终结果的source_run。
