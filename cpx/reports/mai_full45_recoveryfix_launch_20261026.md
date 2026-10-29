# 修复后FULL45：8容器全量116题，50调用上限

2026-10-26 11:17:47（北京时间）按用户授权nohup启动。

tag：mai8b_full45_full116_calls50_recoveryfix_20261026。

全量116题均重新运行，task seed30、原有顺序，8个容器分配15/15/15/15/14/14/14/14题。图在新目录空初始化一次，8容器跨任务共享更新；不加载前一轮图或既有成功结果。每题最多50次模型调用，模型服务、BF16精度、24K上下文、官方提示词、采样参数、词频/IDF统计与上一轮保持一致。使用20261026恢复修复：危险导航过滤、严格恢复判定、首次最终恢复失败关闭探索并反馈、重复无效指针动作保护。运行时提醒会改变失败后输入，故这是修复后算法实验，不宣称官方原始评测复现。

启动前检查8设备claim均空闲，8容器eth0 IPv6启用，上一轮全量FINISHED，模型API正常。部署源码哈希与修复验证清单匹配。run/source_snapshot保存本轮7个源码与测试文件及sha256，便于复现。已有运行记录不被覆盖。

worker及监督器均docker exec -d + exec nohup，stdin /dev/null，日志落盘，无需维持SSH。completion_audit.json每30秒更新，全部正常结束后写summary.json。每题超时5400秒，监督器观察8小时；错误保留，不自动重试择优。恢复保护触发的停止通过runtime_stop_reason和termination=recovery_guard单独记录。

输出目录：
/home/pxchen/workspace/android_world_mai8b_20261024/experiments/trace_reachability_validation/raw/mai_official/mai8b_full45_full116_calls50_recoveryfix_20261026

预计2–3小时，约13:20–14:20完成，保守预留到15:20。参考上一轮相同规模实际耗时2小时11分钟；修复可能改变轨迹和总调用量，尚不能提前保证提速或成功率。
