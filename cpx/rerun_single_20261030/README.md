# 2026-10-30 全量重新运行：每组一个新容器

tag：mai8b_pair_single_c30x20_hostwatch_20261030。北京时间10:46:09启动宿主机监督，两个新模拟器约10:47进入首题。

BASE：pxchen-mai8b-base-single-20261030。FULL45：pxchen-mai8b-full45-single-20261030。相同镜像ID、8GiB/4CPU、模拟器4GiB/4vCPU；从相同预配置镜像新建，不重用昨夜崩溃设备的可写层。只启动这两台；不启动其他已停止设备。

两组全116题从头运行，不使用上轮成功结果；学长e2e116n题序、task seed30，预算min(30,int(20*complexity))。MAI-UI-8B、官方非流式接口。FULL45是上游81a1531普通算法，只有恢复重问预算边界补丁，不包含CPX严格恢复/重复保护或SEENR；图空初始化，跨本轮任务在线积累。历史统计先验沿用、observations为空。

模型服务没有重启或修改。部署时实际进程为gpu_memory_utilization0.9、cpu_offload_gb2、BF16、ctx24576；将实际参数记录到本轮配置，不再把旧launch脚本的0.97当成实际设置。

## 宿主机监督与异常口径

host_supervisor.py在宿主机nohup运行，PID3553278；每5秒检查任务和容器，每15秒更新arm心跳，每10秒写completion_audit。因此模拟器作为容器PID1崩溃后，监督进程仍在。

- 容器退出时保存docker状态与容器日志，将当前题记录为infrastructure_error；不自动重试当前题，不择优。
- 恢复设备后按顺序继续下一题；每次环境恢复最多3次，连续3题基础设施故障停止该组，明确标BLOCKED_ENV。
- 单题无进展600秒或总时长2700秒，终止本题并恢复设备；不让孤儿动作留在容器内。
- 缺失/损坏result.json保留为异常，缺失model_calls不填0。任务零分和预算用尽正常记录并继续。
- 宿主机进程重启时跳过已有持久记录；已有完整result可收集，未完成证据标异常，不静默重跑。

原始模型错误与基础设施错误分别保留；成功率分母固定116，均值需注明有效计数分母。

## 启动验证

5项隔离单元测试通过：结果损坏、预算越界、已记录任务不重试、容器139退出检测、环境恢复有界。Python语法检查通过；上游文件哈希核验，唯一变动full_run.py预算边界补丁；其余接口/监督为新增工具。

10:48:49检查两组RUNNING，errors空、重启次数0，各2次有效模型决策，首题分配30次；模型延迟约22–26秒。此为启动推进检查，不声称已完成首题或保证不再发生宿主机故障。

数据：/home/pxchen/workspace/android_world_mai8b_20261024/cpx_pair_single_rerun_20261030/experiments/trace_reachability_validation/raw/mai_official/mai8b_pair_single_c30x20_hostwatch_20261030

查看completion_audit.json、BASE.runs.jsonl、FULL45.runs.jsonl及各arm.infra.jsonl。完整原始输出保留服务器，最终结果尚未生成。预计12–18小时，随实际进度校准。
