# MAI AndroidWorld：4 baseline workers + 4 FULL45 workers

一轮任务按列表索引对4取模分配；两组分片与任务seed一致。所有FULL45 workers共享一张本轮在线图；不按worker隔离图。BASE不读取该图。每个容器有独立Android模拟器，8个worker共同请求一个MAI vLLM。

## 共享范围

相同tag下的 `raw/mai_official/<tag>/graph/FULL45/` 保存图边、成功轨迹、已见页面、不可回滚/有副作用的控件证据、图执行与误用记录、边等待时间和在线流程库。`raw/mai_official/<tag>/FULL45/<task>/p0/` 保存该任务的截图、a11y及执行记录，其他FULL45容器也能读取。

`android_world/shared_state.py` 使用同一宿主机本地文件系统的 `flock`。追加写有互斥锁，JSON读改写在锁内合并最新内容并原子替换。图、执行验证与intent增量刷新；轨迹索引和在线流程库在每次决策前刷新。流程提取受锁保护，仅扫描带 `.shared_complete` 的已结束任务；保留leave-one-template-out筛选和当前episode本地学习。推理与设备操作不持有全局状态锁，worker占用锁仅防止同一个worker被重复启动。

必须给所有容器挂载**同一个宿主机仓库目录到相同的 `/androidworld` 路径**。跨主机、NFS和每个容器自行复制的图目录不在此实现支持范围。每个模拟器使用容器内的5554/8765端口，不使用host网络。

## 启动入口

每个容器运行以下命令，8个容器使用相同tag、任务列表/顺序、seed、API地址、配置与源码；每组worker-id分别取0、1、2、3。BASE容器将 `--arms` 设为BASE，FULL45容器设为FULL45：

```bash
cd /androidworld
python experiments/trace_reachability_validation/scripts/mai_androidworld_driver.py \
  --tag mai8b_shared_run1 --arms FULL45 --workers 4 --worker-id 0 \
  --seed 30 --api-url http://172.17.0.1:8097/v1/chat/completions
```

不指定 `--tasks` 时任务来自AndroidWorld registry（当前116题），每worker约29题。要沿用原实验顺序，所有worker传同一个 `--task-order` 文件。正式新轮使用全新tag，图只在整轮开始时为空；任何worker都不会在启动时清空共享图。manifest建立、runs.jsonl写入均加锁；worker参数或源码不一致会拒绝同tag混跑。

smoke示例：添加 `--tasks RecipeAddSingleRecipe,RecipeDeleteSingleRecipe --smoke`。smoke最多每题8次调用，使用等权IDF与空离线可回滚统计；正式运行不加该参数，仍需补齐原 `app_tokens.json` 等统计资产。该共享机制不会通过创建假历史文件绕过缺失资产。

## 退出与异常

每个worker完成自己分配的任务后退出，结果汇总到同一 `runs.jsonl`。同一 `(arm, worker-id)` 重复启动会被锁拒绝。失败任务、部分输出、缺少完成标记的结果会保留并停止该worker，不自动用已有热图重跑；其他worker不会因此被隐式重启。中途修改worker数量、分片或源码必须换tag。检测到共享日志截断/损坏时停止并人工审查，不丢弃记录继续运行。

并行到达顺序决定图的学习顺序，应保留原始日志；它不是原单worker实验的固定学习顺序。两组共同争用vLLM时的wall time不等于独占服务的加速比。

## 验证

- 协议回归8项、执行回归8项通过。
- 共享状态回归8项通过：4进程累计100条图观察与100条成功轨迹，无丢失；在线等待时间更新合并；已有读者刷新不重复计数；误用记录与跨任务intent共享；4进程提取流程仅一次且跳过未结束任务；保留当前任务本地流程；半条尾记录不被消费；116题分片恰好覆盖且无重叠；重复worker占用被拒绝。
- 双容器真实smoke通过，tag为 `shared_graph_smoke_v2_20261024`。两个容器同时使用同一个空图目录：添加食谱任务成功（5次调用，wall112.42s），删除食谱任务成功（7次调用，wall69.70s）；HTTP请求数分别为5/7，无重试、无运行错误。
- 共享图最终13条观察（添加任务6条、删除任务7条），成功轨迹2条、已提取在线流程1条。按每次刷新读取的全局日志前缀核对：添加worker最多读到7条peer观察，删除worker最多读到3条peer观察。添加worker在运行中还读到了peer成功轨迹和在线流程。两边均实测通过共享路径读取对方截图对应的a11y页面，结构签名与图记录一致。
- 原始验证：`raw/mai_official/shared_graph_smoke_v2_20261024/shared_smoke_summary.json`。本地副本位于 `latest_code_review_20261023/mai_official_validation_20261024/shared_smoke_summary.json`。
- 第一次集成因镜像已安装旧包优先于脚本目录而在模型调用前退出；已修正脚本入口仓库搜索路径，失败tag `shared_graph_smoke_20261024` 保留。新入口的 `mai_run.py --help` 和上述真实调用均通过。
- 尚未跑满8个容器或116题；正式4+4入口已支持，当前验证为4进程并发压力检查加双容器真实任务。smoke结束后没有残留worker，额外测试容器已停止；保留模型服务和准备好的测试镜像。
