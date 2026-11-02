# CPX：AndroidWorld 实验归档

本分支用于学长审阅 CPX 的代码修改、启动方式和原始实验结果。实验实际运行在 `hrx20000209/android_world` 的派生工作目录；因目前仅有 `hrx20000209/agent` 的写权限，在此仓库独立 `cpx/` 目录归档，不改动原项目入口。

## 最新纯文本LLM结果（11月2日归档）

- [Qwen3-8B纯a11y：BASE/FULL45完整116题、原始日志、最终图、实际源码及启动配置](qwen3_results_20261102/)
- BASE 16/116（13.79%），FULL45 21/116（18.10%）；共同成功13题调用数8.85 vs6.77（+30.7%），未体现共同成功题上的调用加速。
- 本轮仍有共同3题初始化失败、FULL45额外7题恢复回放适配越界，均保留为失败；平均调用数含异常提前终止，不应据此宣称干净的最终提升。没有混入启动测试或中断的早期全量。
- 非思考、temperature0、int(10×complexity)预算；仅模型输入改为文字树/历史，FULL45在线图从空开始并保留历史先验。不是未经修改的官方T3A runner，不能直接与GUI-Owl视觉成绩作严格归因比较。

## 最新GUI-Owl结果（11月1日归档）

- [GUI-Owl原始全量、BASE七题修复、FULL45修复后全量：结果、日志、图及源码](guiowl_results_20261101/)
- 修复后BASE采用原109题＋固定七题重跑，57/116（49.14%）；FULL45从空图重新顺序跑116题，56/116（48.28%）。共同成功48题调用数7.38 vs9.00（−18.1%）。
- 唯一功能修复为支持官方open_app别名。FULL45仍有三题坐标异常和最后一题容器OOM；缺失调用数及均值分母在结果页明确说明。
- [官方格式GUI-Owl 8B：两独立容器BASE/FULL45、启动脚本和实际设置](guiowl_run_20261031/)
- temperature0，5图history，预算int(10×complexity)，不同于下方历史MAI实验，不能混合模型与预算比较。

## MAI-UI纯GPU实验（10月31日归档）

- [纯GPU 8B/2B四组全116题：结果、原始日志、启动配置](results_pair_20261031/)
- [8B成功率下降与加速有限的逐题诊断](results_pair_20261031/diagnosis_8b.md)
- 最新四组基于upstream81a1531；下方10月29日三轮50调用实验属于历史归档，不能混用预算和成绩。

## 从这里看

- [两张表差距的逐题核查（10月29日）](reports/difference_diagnosis_20261029.md)

- [结果对比与统计限制](analysis/comparison.md)
- [机器可读结果](analysis/comparison.json)、[统计代码](analysis/compare_mai_runs.py)
- [三轮原始逐题 runs.jsonl](data/summary/)
- [与学长代码的区别](DIFFERENCES.md)
- [启动与复现说明](launch/README.md)
- [历史启动脚本](launch/historical/)
- [服务器部署源码参考](reference/)
- [修改后的源码覆盖层](code_overlay/)、[相对基线的 tracked diff](patches/android_world_tracked_changes.patch)
- [实验日志及诊断报告](reports/)
- [原始数据归档说明](data/README.md)
- [本次归档验证](VALIDATION.md)

## 历史三轮的版本和口径（非最新四组）

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
