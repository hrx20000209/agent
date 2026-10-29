# 两张表差距的逐题核查（2026-10-29）

不能把 FULL45 收益变化只归因于步数上限。原始记录显示：任务排序/调度不一致、少数成功但未结束的长循环、参数自动填错，以及恢复保护缺口。这里区分日志事实和因果推断；未启动新实验、未修改运行算法。

## 1. 任务顺序没有与学长对齐

学长 e2e116n_task_order.txt 与 CPX run_config.tasks 集合完全相同，但116个位置有86个不同。CPX沿用的是自己首轮按名称排序的任务列表，不是学长保存的题序。报告中“原有顺序”仅指CPX自己的首轮，不能用于声称与学长题序相同。

例如学长短信任务先 SimpleSmsResend，再 ReplyMostRecent，再 Send；CPX列表先 Reply，再 ReplyMostRecent，再 Resend。此外学长逐题两臂交错；CPX FULL45 8容器并发，按下标模8分配。因此每题能读到的在线图/函数库由完成顺序决定，与学长不相同。已证明输入经验不同，尚未以对照实验量化这一因素的独立贡献。

模型也从 GELab-Zero-4B-preview 改成 MAI-UI-8B，不能称“只改了上限”。

## 2. 平均调用翻倍，大部分是原预算之后的长尾

BASE50实际2396调用，其中1073调用发生在该题原复杂度预算 min(20,int(10*complexity)) 之后。把已有轨迹的计数截断到该预算，均值20.66会变成11.41；修复后FULL45对应19.05→10.17（有效115题）。

**这是已有轨迹的计数分解，不是20预算重新评测，也不能推算20预算真实成功率。** 后续图学习和其他任务输入也会随预算变化。它说明全体平均步数的巨大变化确实主要在长尾，不能说明方法本身慢了一倍。

## 3. 共同成功1.6%被少数长循环明显抵消

|共同成功任务|BASE调用|FULL45调用|额外调用|
|---|---:|---:|---:|
|ExpenseDeleteDuplicates|12|50|38|
|ExpenseDeleteDuplicates2|15|50|35|
|MarkorChangeNoteContent|17|50|33|

三题合计额外106调用；而51道共同成功全体 FULL45 541调用、BASE550调用，只净省9次。

三题 result 均 is_successful=1，但 done=false：最终任务状态被评估为成功，模型没有主动结束，在日期选择、菜单或返回/打开之间继续循环到50。不能断言任务在第几步就已成功，因为当时没有逐步调用评估器。

ExpenseDeleteDuplicates最后若干次交替“点月份/年份→关闭”；Duplicates2交替“点From月份→确认”；Markor最后交替“返回→打开文件”。现有 RepetitionGuard 只比较连续相同 action + 页面签名，A→B→A→B每次重置为1。隔离运行部署代码：50次交替点击，最大计数仍1，无法停止。恢复修复没有覆盖该模式。

### 成功任务集合改变也影响横比

学长两臂都成功47题，我们两臂都成功51题，不是相同子集。

只在四组都成功的同一37题上：

- 学长 FULL45 6.243 vs BASE7.486，减少16.61%。
- CPX FULL45 9.459 vs BASE11.081，减少14.63%。

这是诊断用的事后交集，**不替代主表、不应筛掉难题后宣称新成绩**。但足以说明不能直接把两表 −21% vs −1.6% 理解为同一批题的图收益整体消失。

## 4. 确认的参数绑定错误：短信正文包含应用名

SimpleSmsReply：目标正文是 `The library book is due back on the 15th.`。

原始 steps 和 mai_executed 显示图自动填入：
`The library book is due back on the 15th. in Simple SMS Messenger`

读取服务器 final a11y，已发送气泡也确实包含该错误后缀。随后MAI输出answer“task finished”，AndroidWorld评估0；同题BASE9调用成功、FULL45仅3调用失败。

原因是 fgraph.py parse_goal 第110行附近的 message/text 尾部正则贪婪抓取到字符串末尾，把任务末尾应用说明当成正文。AST核对：该函数与学长10月23日基线完全相同，属于继承的参数解析缺陷；但在线库可用性和模型纠错不同，导致当前运行中实际触发。不能说这是CPX新写出的解析错误，也不能因为它是继承代码就忽略当前错误。

Markor另有字段语义证据：图把完整 `copy_cool_snake.txt` 填进Name，同时独立扩展名字段仍为`.txt`；BASE填名称不含后缀。最终FULL标题`copy_cool_snake.txt`，BASE标题`copy_cool_snake`，两者正文字段相同（均带末尾换行）；FULL失败、BASE成功。这高度指向名称与扩展名绑定错误，但未保存当时文件系统清单，不将实际双扩展名路径说成已经直接查证。

## 5. 自报完成不等于经验正确

本轮函数缓存19个函数，来自12个任务；其中7个函数来自4个实际得0分、但模型自报完成的任务：ExpenseAddMultipleFromMarkor(1)、MarkorCreateNote(1)、RecipeDeleteMultipleRecipesWithNoise(4)、SimpleSmsReply(1)。代码确实按self_reported_done学习，这是继承设计；不代表7个函数每个步骤都错，也未证明它们都被后续使用。

适当修复方向是校验可观察的字段/操作后置条件并降低未经验证经验的采信，而不是把benchmark评估分数偷偷作为执行期oracle。

## 6. 恢复失败仍存在，但不能把相关性当因果

修复后18题记录recovery_unresolved，其中11题最终0分、7题成功。这个数字不能直接表述为11题均因恢复失败而失败。图观察和页面回滚不保证恢复任务数据，需逐题区分。

## 建议的处理顺序

1. 先修参数与UI字段绑定，并补A/B多步循环检测；当前代码只诊断，未修改。
2. 固定同一模型、同一任务序列、同一复杂度预算和环境，先逐题对照BASE/FULL45；再单独检验并行共享。
3. 保留现有全116主表；不要用删除长轨迹、挑成功交集或用截断旧轨迹代替新实验来修改成绩。
4. 对已有日志/分数重新计算不需要启动vLLM；本次未启动任务。

原始证据：analysis/diagnosis_20261029/audit.json、ui_text.jsonl、base_markor_ui.json、reproducer_results.json，以及data/archives中原tag日志。原始学长逐题记录和题序也保存在diagnosis目录，方便核对。
