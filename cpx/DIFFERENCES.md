# CPX 与学长实验的区别

对照学长 10 月 23 日 e2e116n 的 BASE 53/116、FULL45 54/116。

| 项目 | 学长 e2e116n | CPX 三轮全量 |
|---|---|---|
| 模型 | GELab-Zero-4B-preview | MAI-UI-8B BF16 |
| 预算 | min(20, int(10*complexity)) | 显式 --max_steps 50 |
| 执行方式 | 单模拟器逐题交错两臂 | BASE 12 / FULL45 8 容器 |
| 图经验顺序 | 随逐题顺序积累 | 共享目录加文件锁，并发更新 |
| 设备 | Orin；模拟器内 Qwen2-VL-2B 背景负载 | RTX4090D 模型服务；ONDEVICE_LOAD=0 |
| 模型接口 | GELab 调用/解析 | 官方 MAI system、mobile_use、0..999 坐标、最近3图+全文历史 |
| FULL45 修复 | 当时版本 | 危险探测过滤、严格内容恢复、失败反馈及重复动作保护 |

以上多个变量同时改变，不能将两表差值单独归因于模型或图机制。

## 实际代码变动

- `mai_client.py`, `mai_protocol.py`, `mai_run.py`：官方 MAI 提示词、协议与历史，API 重试、解析和动作接地。
- `android_world/shared_state.py` 及 fgraph/full7/graph_exec/intent_graph/seq_run：跨进程文件锁、原子落盘和共享经验读取。
- `android_world/env/interface.py`：shell 参数引用修复。
- worker/recover：模拟器 a11y 恢复、实际短信探针；容器需启用 IPv6。
- `recovery_guard.py`, `full_run.py`, `full9.py`：危险导航过滤；恢复要求内容匹配；恢复失败关闭探索并反馈；同页重复无效动作提醒/停止；避免恢复重问超预算。
- BASE 不使用 FULL45 的恢复反馈和重复动作保护。

完整 tracked 修改见 patch；新增文件见 code_overlay。未把学长 10 月 29 日 SEENR 新代码合并进 CPX 运行代码。本分支是历史实验归档，不宣称已与最新 SEENR 设置对齐。

## 结果解释

修复后 FULL45 62/116、BASE64/116；共同成功51题调用均值10.61 vs10.78，仅减少1.6%。全体均值降低可能包括提前停止失败任务，不能全部视为成功轨迹加速。修复前57/116；修复后仍有1条解析异常未补跑。
