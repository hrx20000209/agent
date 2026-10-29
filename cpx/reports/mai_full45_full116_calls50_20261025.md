# FULL45 全量116题：空图共享，50次模型调用

用户确认从头重跑实验组。tag：mai8b_full45_full116_calls50_emptyshared_20261025。

2026-10-25 20:43:41（北京时间）启动8个pxchen已有容器，任务分配15、15、15、15、14、14、14、14。任务顺序及seed=30沿用原full232配置；116题均产生新结果，不筛选失败题，不合并既有成功结果。

每题显式--max_steps 50，取消复杂度缩减上限。模型提前宣布结束可提前退出。共享图仅在新目录初始化一次，初始节点、观测、函数均为0；MAI_SHARED_STATE=1，所有8容器共享同一graph/FULL45，跨任务更新；无历史快照、无历史episode映射。保留原来配置的词频与IDF等评分统计文件，并由preflight核验哈希；这些统计不是历史图。

模型与BASE50诊断一致：MAI-UI-8B，BF16，GPU4现有vLLM服务，context24576，max_num_seqs=8；官方提示词、完整文字历史与最近三张图，temperature0/top_p1/request seed42。客户端配置configs/mai_ui_8b_full50_20261025.json。服务无重启。版本与精度均未更改。50次调用是本实验设置，尚未核实为官方70.7%成绩的测试预算。

启动前确认BASE67重跑FINISHED、8个设备claim全部空闲、8容器eth0 IPv6均启用，保留此前短信、a11y恢复与消息转义修复。worker启动真实短信探针且每题前检查a11y。错误记录保留，不自动重试取高分。每题5400秒超时，完成审计观察8小时。

8个worker与收尾监督器均由docker exec -d + exec nohup启动，stdin=/dev/null，日志写入共享磁盘，不依赖SSH。收尾审计每30秒写completion_audit.json；完整无错误结束后自动写summary.json，统计116题完整成功率、分数和、模型调用总数及均值。

输出目录：/home/pxchen/workspace/android_world_mai8b_20261024/experiments/trace_reachability_validation/raw/mai_official/mai8b_full45_full116_calls50_emptyshared_20261025

预计2–3小时，保守预留4小时。BASE67失败题50预算实际用时96.62分钟，可作吞吐参考；新任务轨迹、共享图开销与单服务KV排队会影响耗时。

企稳检查：20:45后8个worker均RUNNING，已完成题无执行错误；各容器共享graph.jsonl的inode均为211949396。nohup进程stdin均为/dev/null且忽略SIGHUP，收尾监督器PID53631。实际episode进程命令含--max_steps 50。
