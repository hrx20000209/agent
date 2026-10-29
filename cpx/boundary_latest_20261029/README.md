# 最新上游FULL45的危险边界检查

固定上游commit81a1531758da249c2acb74baa761d2da65e01b1c，485个已下载上游文件逐一校验；不带CPX并发图/严格恢复/重复保护修复，仅增加MAI官方消息与动作接口和运行工具。

预算为 min(20,int(10*task.complexity))，不传显式max_steps，task seed30；学长题序中过滤6题，单容器串行，图空初始化，仅在这6题之间积累。普通FULL45，不加载SEENR预热图/同模板示例，也不启用SEENR专用旗标。

继承统计先验沿用此前rxhuang复制的app_tokens/ui_idf/rev_prior/element_idf；历史目标语料只提供goal IDF，observations.jsonl为空。不是完全无统计先验。

MAI接口完整响应后执行，与学长GELab流式提前执行不同；这影响探索窗口，不宣称官方端到端配置完全复现。

新建容器在任何任务开始前开机崩溃（exit139，QEMU CPU3 hang），保留该环境失败记录。真实任务改在确认空闲的pxchen-mai8b-base-w8-20261026运行，代码独立目录，各题执行benchmark初始化。它不是全新镜像的正式对照，不应据6题计算全量性能结论。

静态隔离检查使用实际上游函数、模拟设备I/O：匿名toolbar ImageButton未被过滤；布局相同但内容变化时relaunch_restore仍返回True；message尾部解析仍包含app说明。真实任务结果完成后另附。

本轮只运行边界子集，尚未启动BASE或FULL45全116重跑。plan.json记录之后单容器使用学长题序和复杂度预算的选择。

隔离复现（不访问设备/模型）：

```sh
mkdir upstream
tar -xzf package.tar.gz -C upstream
python3 check_boundaries_isolated.py upstream
```

critical_function_comparison.json 记录与学长10月23日基线的AST对比：parse_goal/relaunch_restore/_risky_probe/_risky_slot 均未变化。
