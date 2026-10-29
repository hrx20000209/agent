# BASE顺序核对与单容器建议

实际run：mai8b_base_full116_calls50_12w_20261026。

- manifest为按任务名排序的116题；与学长同一任务集合，86个列表位置不同。
- worker i按 tasks[i::12] 执行。对各题meta.started排序，12个worker的实际内部顺序全部匹配分片：前8个各10题、后4个各9题。
- 所有worker合起来的实际开始顺序与学长有111个位置不同，这是并发启动与分片调度的正常结果，不是漏题或任务打乱错误。
- 116题唯一，无执行异常、teardown异常、决策计数>50；未发现图步骤。BASE无图，不能把题序差异解释为BASE图经验变化。
- SimpleCalendarEventOnDateAtTime发生两次max_tokens=2048响应截断，随后重试；它们不算成功解析的model_calls，但属于HTTP模型请求。应同时报告两个计数。
- BASE/FULL meta目标与参数核对中，MarkorCreateFolder使用datetime.now生成文件夹名（继承实现），seed不能固定该名字；两组都成功。SaveCopyOfReceiptTaskEval仅PIL.Image对象repr内存地址不同，不能据此声称图片内容不同。

## 单容器方案

建议先做MAI BASE单容器，全116题、使用学长保存的e2e116n_task_order.txt、task seed30，保持50次决策上限及模型/提示词/采样/环境版本不变。这样先检验调度/设备残留影响。模型服务也应确认没有其他实验流量，避免把一个Android容器误当成整机独占。

学长复杂度预算是另一个变量，后续再单独对照。若同时把50改为复杂度上限，只能称整体设置复现，无法单独判断单容器的影响。

使用相同镜像准备新的干净模拟器；明确每题reset/initialize/teardown，记录日期、时区、应用偏好与剪贴板等未必由任务初始化完全复位的状态。原worker仅接受max_calls=50、读取旧source-run的tasks；不能只传workers=1就声称按学长顺序运行。其cohort_note还硬编码12workers，需要新配置/启动入口如实记录。

## 仍未修复

1. FULL45：message尾部解析把应用说明拼进短信正文（确定，源于继承解析函数）。
2. FULL45：重复保护只识别连续完全相同动作，漏掉A/B循环与坐标轻微变化（确定）。不要未经定义把FULL45保护也加到BASE，否则改变baseline。
3. FULL45：恢复到相同页面不能保证任务数据恢复；上一轮仍18题recovery_unresolved，不等价于18题因此失败。
4. FULL45：Markor Name与extension分开时，图仍可能把含后缀的完整文件名填入Name，已有两臂UI差异与失败证据；缺当时文件系统清单，双扩展名路径仍应进一步验证。
5. 日志：拒绝的解析结果/截断响应目前主要只保留错误原因，mai_decisions通常只保存成功解析响应，需补原始失败响应日志以便定位协议异常。
6. 对照可重复性：MarkorCreateFolder的墙钟参数不受task seed控制；需明确是否固定实例参数，不能暗称同seed就逐字相同目标。

Linux adb路径问题已经被mai_run.py末尾的shutil.which覆盖，不应再次列成未修复bug。

本次只核对和记录，未启动新实验、未修改算法。完整机器核对结果在analysis/diagnosis_20261029/baseline_order_audit.json。
