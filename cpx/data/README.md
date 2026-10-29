# 原始数据

`summary/` 保存三轮全量原始 `runs.jsonl`，是统计表直接输入，未剔除异常/未选择最好结果。

`archives/` 为服务器导出的逐步原始日志压缩包；包含模型请求/响应记录、任务结果、worker日志、run_config、图和函数库、可用的当轮source_snapshot、a11y capture 的 `.gz` 等。解压后保持原 tag/arm/task/p0 结构。`manifests/` 给出每个原始文件的路径、大小、SHA256 和是否包含在压缩包中；未随分支上传的图片、锁文件和pid文件也列入索引。

截图原件保留在：
`/home/pxchen/workspace/android_world_mai8b_20261024/experiments/trace_reachability_validation/raw/mai_official/<TAG>/`

截图体积较大，未放入 Git；如果需要检查实际视觉输入，按 manifest 路径读取服务器原件。模型权重、Docker镜像未包含。分支内不把缺失截图描述成完整视觉复现包。

历史合集 `mai8b_full232_shared_20261025` 含 BASE/FULL45 旧20预算运行；失败题补测、短信修复和热图7题分别保留独立tag，不能混合当作一次全新全量实验。当前主表只用三轮 fresh full116。
