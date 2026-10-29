# BASE失败任务50次调用预算诊断

按用户最终指示，只对既有BASE得分为0的67个任务重跑。48个完整成功和1个半分结果保留；FULL45原53/116结果保留，不启动新的FULL45实验。

tag：`mai8b_base50_failed67_20261025`。8个已有pxchen MAI容器全部用于BASE，按失败任务原顺序索引对8取模分配，前三个worker各9题，其余各8题。task seed仍为30。使用nohup启动，不依赖SSH。

每题显式`--max_steps 50`，取消原来`min(20, int(10 * complexity))`的上限。已通过真实FullEpisode.max_steps方法验证显式参数返回50；MAI的决定/提交路径直接调用，不经过GELABAgent.step中的20步退出逻辑。任务由模型自行结束或到50次调用结束，不强制执行到50次。BASE不加载历史图，MAX_AGENT_STEPS全局常数保持20，避免改变原实验的默认设置。

## 模型服务与容量

长轨迹保留官方全部文本历史和最近三张图，旧16K上下文可能提前限制50次调用，因此在服务空闲且无排队请求时仅重启pxchen自己的GPU4服务：上下文24576、显存利用率0.97，权重仍BF16，推理后端/提示词/采样参数/截图格式不变。最大并发仍8；服务启动容量检查为KV缓存3.82 GiB、27776 tokens，可容纳24576上下文。长上下文实际同时执行路数受KV容量限制，其余请求排队，不宣称8倍吞吐。

旧启动配置备份：`/home/pxchen/workspace/services/mai_ui_8b_gpu4_20261024/launch_before_base50_ctx24k_20261025.sh`。新服务日志为server_base50_ctx24k.log。新客户端配置configs/mai_ui_8b_base50_20261025.json。无需量化或更改模型权重。24K仍是有限上下文；如50次调用前出现长度或执行错误，保留并在审计中报告，不静默裁剪官方历史。

## 时间估计与收尾

预计1.5–3小时。原BASE116题合计1354次调用、27664.71秒episode时间之和（包含原共享服务排队）；失败任务通常较长，减少题数不能按比例减少总耗时。估计不是完成承诺。

后台mai_completion_watch.py每30秒保存状态，最长观察6小时；任务本身不因SSH断开停止。每题timeout5400秒，错误结果保留并继续该worker的其余任务，不自动重试取高分。worker完成但有错误时标FINISHED_WITH_ERRORS，收尾标FAILED，避免将环境故障算成可信的完成结果。

全部67题无执行错误结束后，自动生成budget_comparison.json和reconciled_base_runs.jsonl：只替换原67个零分结果，原48成功和1半分保持；统计救回的完整成功题数、合并116题的成功率、达到50次预算的题数及总调用数。替换规则固定，不按多次尝试择优。仅为失败任务扩大预算的诊断，不是重新从头跑116题的官方成绩复现。

代码验证：显式50次预算覆盖、8worker分区、固定67零分选择，以及116题合并保留48成功和1半分的检查通过。

输出目录：`/home/pxchen/workspace/android_world_mai8b_20261024/experiments/trace_reachability_validation/raw/mai_official/mai8b_base50_failed67_20261025`。
