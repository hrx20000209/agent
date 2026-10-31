# GUI-Owl-1.5-8B-Instruct：BASE / FULL45 配对全116题

用户要求：参考官方提示词与配置、temperature=0，两个独立AndroidWorld容器分别跑BASE和FULL45。

## 固定设置

- 模型权重：`mPLUG/GUI-Owl-1.5-8B-Instruct`，revision `06d5faecff74840bab2be2425e9c42667a5d04fc`，全部16文件及四分片官方SHA256校验通过。
- 服务：GPU4+5 / TP2 / BF16 / CPU offload0 / context32768 / image limit5 / max_num_seqs2 / port18197。实际命令见[model_service/launch.json](model_service/launch.json)。
- 显式固定：temperature0、top_p1、top_k-1、seed42、max_tokens2048。没有继承权重默认temperature0.6/top_p0.95/top_k40。
- 官方移动端system prompt逐字提取于`X-PLUG/MobileAgent/Mobile-Agent-v3.5/android_world_v3.5/android_world/agents/gui_owl.py`；prompt文件和哈希在配置中固定。
- 原生history：最近5图含当前图；更早动作压成Action文字历史。`cut_current_messages` / `convert_format`从官方源码AST提取原函数执行；测试在12个连续决策位置逐一对比官方消息构建结果。模型近期原始响应不重写。图实际执行动作也编码为GUI-Owl原生工具调用，Action文字包含动作参数。
- 坐标：遵循官方`qwen-vl`的999缩放约定，保留swipe的`coordinate`和`coordinate2`两个端点，不转成MAI方向式滑动。图像预处理沿用官方coordinate_resize，服务器再按公开模型卡size参数处理。
- 调用预算：**int(10*complexity)**，来自公开AndroidWorld suite_utils，无统一20/30封顶。FULL45保留上游2倍总计数动作上限，日志区分`action_step_cap`、`budget`、`model_done`和`unknown_stop`。
- 题序：学长e2e116n；每组单容器顺序运行116题。任务仍使用CPX既有直接seed30，便于同框架实验；这不同于官方create_suite把seed/name/index哈希为instance seed，不能声称精确复现官方69.0%。
- 两组独立Android设备：`pxchen-guiowl8b-base-20261031`、`pxchen-guiowl8b-full45-20261031`。此前四个MAI实验已完成，旧容器停止保留，旧结果不复用。
- BASE无图执行、无探索。FULL45空图启动，跨本组116题在线积累，图执行+探索开启，窗口30秒/max3探针。词频/IDF历史先验沿用前轮，空图不等于无历史统计。
- 算法仍为学长普通FULL45 main81a1531，加已有recovery-requery预算边界补丁。未启用SEENR热启动及旧CPX recovery_guard；此次未修改前轮诊断的填表、重复复用、恢复判定逻辑。

## 官方实现与执行桥接边界

官方模型决策提示词和history不换成GELab或MAI模板；借用现有AndroidWorld动作执行器作为桥接。对key/Menu、显式等待时长和长按时长提供ADB落地；严格拒绝不完整或未支持输出，重试次数与原始响应留档，不猜动作。`answer`直接写入interaction_cache并结束当前episode；官方adapter还会额外调用一次模型后强制terminate，当前框架不额外消耗这一调用。这些差异记录在run_config，不作为官方69%完全复现声明。

## 启动和文件

- [deploy.py](deploy.py)：校验空目录/源码哈希/模型服务，创建两容器，先做真实镜像运行时导入及复杂度4.8→48预算检查，再启动nohup宿主监督。
- [package.tar.gz](package.tar.gz)：完整部署源码；包含owl_protocol、owl_upstream_probe、官方prompt及其提取源码、原算法、统计资产、题序和配置。
- [host_supervisor.py](host_supervisor.py)：每组串行，设备退出139/600秒无进展/2700秒硬超时会保留当前题错误、恢复设备后跑下一题；连续3题基础设施错误则停止，避免无限崩溃。
- [readiness.py](readiness.py)：boot、模型、无障碍接口、SMS检查；无障碍短暂未就绪先在同一设备上重试，避免无谓反复重启。
- [model_service](model_service/)：模型服务启动脚本、配置、推理smoke和下载校验记录。
- `run_config.json`为实际服务器部署生成配置，启动后回收；不能用示例README替代实际配置。

部署目录：`/home/pxchen/workspace/android_world_mai8b_20261024/cpx_pair_guiowl_20261031`。
Run tag：`guiowl8b_pair_single_c10_official_20261031`。日志保留既有兼容路径`experiments/trace_reachability_validation/raw/mai_official/<tag>`和`mai_*`文件名，但meta.decision_backend和model均明确是GUI-Owl。

## 验证

两组都开始实际出步、FULL45图/探索标志和图记录校验后才报告运行。纯协议6测试+宿主监督6测试通过；这些是针对性检查，不代表本机已运行全部AndroidWorld测试。模型服务此前通过官方prompt单图及两路5图推理smoke。原始任务评测分数保持环境判定，不由模型自报成功代替。

实际启动：2026-10-31 11:12:32 Asia/Shanghai，宿主监督PID2805116。运行状态需读取服务器最新completion_audit.json，不以本文静态记录代替。
