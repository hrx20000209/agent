# 归档验证（2026-10-29）

- Python 源码 AST 解析通过。
- `python3 -m unittest recovery_guard_test mai_protocol_test`：11项通过，均为纯逻辑测试，不调用模型/设备。
- 使用归档三轮 runs.jsonl 重新运行 compare_mai_runs.py，生成 JSON 与保存的 comparison.json 完全一致。
- 原始三轮任务各116条、任务名唯一、任务集合相同，由统计脚本断言。
- 最终 tracked patch 对基线索引执行 `git apply --cached --check` 通过；仅校验，不修改源文件。
- ARTIFACTS.json 对归档文件记录大小及 SHA256，可用 tools/verify_artifacts.py 复核。
- 本次归档没有启动新实验，也没有宣称做了新的 AndroidWorld 端到端验证。恢复修复时的容器内36项测试记录保存在 reports/mai_recovery_fix_20261026.md，属于此前验证。

- 原始日志归档不包含截图及压缩页面采集文件；这些文件的原始路径、SHA256 和排除标记在 data/manifests 中。

- 8轮原始文本归档：逐文件核对原始 manifest，验证 8124 个文件的大小和SHA256；归档无额外/缺失 included 文件。
- 覆盖层中与服务器部署源码重合的文件逐字节一致。
