# 各单容器全116题：启动归档

2026-10-29北京时间19:58:36启动，tag=mai8b_pair_single_c30x20_20261029。

BASE使用pxchen-mai8b-base-w9-20261026；FULL45使用pxchen-mai8b-base-w10-20261026。每组一个模拟器，共享MAI-UI-8B服务，学长e2e116n题序、task seed30；每题预算min(30,int(20*complexity))。

FULL45为空图起跑、跨116题积累。上游81a1531普通FULL45，不带SEENR预热及CPX严格恢复/重复保护。唯一上游算法文件补丁budget_boundary.patch防止恢复重问超预算；MAI官方非流式接口另行适配。历史统计先验保留，历史observations为空。均全量新跑，不复用旧成功、不择优补测。

使用既有空闲设备，执行benchmark逐题initialize/reset/teardown，并非新建干净镜像。共享模型服务负载会影响探索窗口，不将墙钟视为隔离服务对照。

根目录：/home/pxchen/workspace/android_world_mai8b_20261024/cpx_pair_single_20261029

结果位于其experiments/trace_reachability_validation/raw/mai_official/mai8b_pair_single_c30x20_20261029。每臂写独立runs/status，watcher每30秒写completion_audit；结束写final_audit。worker/watch均docker exec -d + exec nohup，stdin=/dev/null，不依赖SSH存活。

启动校验：设备claim空闲、源哈希与语法通过；复杂度1/1.2/1.6/1.8/3.4预算分别20/24/30/30/30。

19:59检查：两组RUNNING、errors为空，各完成2次合法模型决策；首题均AudioRecorderRecordAudioWithFileName、预算30。尚未完成首题，不能视为全任务验证通过。本目录不包含尚未产生的最终成绩。
