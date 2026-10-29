# CPX：MAI-UI-8B AndroidWorld 实验归档

本分支用于学长审阅 CPX 的代码修改、启动方式和原始实验结果。实验实际运行在 `hrx20000209/android_world` 的派生工作目录；因目前仅有 `hrx20000209/agent` 的写权限，在此仓库独立 `cpx/` 目录归档，不改动原项目入口。

## 从这里看

- [结果对比与统计限制](analysis/comparison.md)
- [机器可读结果](analysis/comparison.json)、[统计代码](analysis/compare_mai_runs.py)
- [三轮原始逐题 runs.jsonl](data/summary/)
- [与学长代码的区别](DIFFERENCES.md)
- [启动与复现说明](launch/README.md)
- [历史启动脚本](launch/historical/)
- [修改后的源码覆盖层](code_overlay/)、[相对基线的 tracked diff](patches/android_world_tracked_changes.patch)
- [实验日志及诊断报告](reports/)
- [原始数据归档说明](data/README.md)

## 版本和口径

实验代码基于 AndroidWorld 提交 `643303c9fb8c010edeb0dec3a63887055e39ca95`（2026-10-23）。这不是学长 10 月 29 日新增的 SEENR 实现。`code_overlay/` 保存 CPX 修复后代码，历史运行有 source_snapshot 时以归档中的当轮快照为准。不可用修复后代码声称精确复现修复前轨迹。

三轮均 task seed=30、MAI-UI-8B BF16、50 次模型调用上限；BASE 12 容器，FULL45 两轮各 8 容器；FULL45 每轮独立空图初始化，跨任务、跨容器共享在线更新。词频统计先验从学长服务器复制，不能将“空图”说成“无历史统计先验”。

成功数严格按 `success == 1`，0.5 分不计完整成功。修复后 FULL45 一条解析错误保留原样，未补跑、未择优；成功率按 116 题，调用均值按有有效计数的 115 题。共同成功集分别与本轮 BASE 取交集。

## 不需要 Android 或模型的结果复算

在本目录执行：

```sh
python3 analysis/compare_mai_runs.py --root data/summary --out /tmp/cpx_comparison.json
python3 tools/verify_artifacts.py
```

这只读取数据，不启动任何实验。硬件实验环境、原始 tag、服务器路径见各 reports 和 run_config。历史代码中保留原服务器路径用于追溯，迁移机器时需调整路径和 endpoint。
