# FULL45 50次调用全量结果诊断

2026-10-26读取服务器结果、逐题比对及轨迹，未修改算法或启动新实验。

| FULL45队列 | 完整成功 | 成功率 | 模型调用总数 |
| --- | ---: | ---: | ---: |
| 最初full232（含一题环境错误） | 49/116 | 42.24% | 1283 |
| 短信环境修复并替换7题后 | 53/116 | 45.69% | 1275 |
| 本轮空图、8容器、50次全量 | 57/116 | 49.14% | 2741 |

之前49指成功题数，当前49.14指百分比。对修复后的53题版本净增4题、3.45个百分点；调用均值10.99升至23.63。

逐题转移：47题保持完整成功；8题零分转完整成功；2题半分转完整成功；6题完整成功转零分；51题保持零分；2题零分转半分。此轮从头跑，而BASE仅重跑失败题保留旧成功，因此不能将BASE70/116与FULL45直接当作严格一致流程的对照。旧实验每组4容器，这轮8容器；在线共享图的可用内容随各容器进度变化，也并非单独改变步数的受控消融。

主要轨迹证据：

1. 50次上限确实生效，41题达到50调用，其中5题完整成功、36题仍零分。增加预算常被重复动作消耗。20题存在至少10轮连续完全相同的模型原生动作；例如CameraTakeVideo连续47次相同点击，SimpleCalendarAddOneEvent连续46次相同点击。重复动作不自动等价无效（部分删除任务最终成功），但上述两个退步任务明显陷入循环。
2. 全轮有35次joint_restore_fail、24次requery、684次stale_pixels计数、487次探索探针。stale_pixels是重拍检测计数，不能直接解释为684次错误执行，也不能仅凭这些数字证明所有失败由探索造成。SimpleCalendarAddOneEvent尾部模型持续要求设开始时间，steps记录的执行前界面却为桌面，存在截图/状态/历史失配的明确排查线索，根因尚未定位。NotesIsTodo本应查询，模型却点击Task list修改笔记再回答True；轨迹还含rollback_ok=false探针，说明模型行为和探索副作用均需要分别审计。
3. 图共享实际生效：graph_exec.jsonl有487条exec事件；8容器企稳时同一graph.jsonl inode，逐步shared_refresh读取增加的节点、观测、函数、成功episode。图执行ok表示操作层执行，不代表任务正确；不能据此认为图复用带来收益。部分失败任务大量重复图操作：RetroPlaylistDuration88次，TasksIncompleteTasksOnDate85次，TasksHighPriorityTasks42次。

退步6题：CameraTakeVideo（旧7调用成功，新50零分）、NotesIsTodo（6→7）、SimpleCalendarAddOneEvent（16→50）、SimpleCalendarAddOneEventTomorrow（14→50）、SimpleCalendarNextEvent（2→2）、SportsTrackerTotalDurationForCategoryThisWeek（6→50）。其中4题新预算耗尽，2题模型提前结束但验证失败。

结论：本轮确有提升，但新增成功被退步抵消；更多调用主要放大重复行为，探索/恢复及状态同步存在可疑失败链。应先定位恢复失败与截图/状态/历史失配，并将读任务的探索副作用单独审计；不能仅继续扩大预算。日志级环境/进程错误为0，不等于探索不会扰动任务。

原始数据目录：
/home/pxchen/workspace/android_world_mai8b_20261024/experiments/trace_reachability_validation/raw/mai_official/mai8b_full45_full116_calls50_emptyshared_20261025

## 恢复失败根因追踪（追加）

日历SimpleCalendarAddOneEvent第4步：先试探Event color，Back恢复成功；再试探无rid/text/desc/label的ImageButton，位于event_toolbar，坐标(73,211)。后者将EventActivity退出到MainActivity。通用回滚再按Back，使设备退出到NexusLauncher。图恢复无路径，relaunch_restore=False，无relaunch_replay日志。第4步明确requery=True，新capture0020_s4_pre_requery.jpg经人工查看确为桌面；随后模型继续输出“Set the start time to 19:00”，同坐标点击连续46次。故不能再把此例笼统解释为没有重新截图或没有重新推理。现有危险控件主要依赖文字、rid和类型；无标签工具栏导航未被有效拦截。通用Back也不是任意控件的逆操作。

MAI适配器的exploration上下文令record跳过探索及恢复动作。恢复成功时隐藏试探符合设计，但失败时decide_native仍使用原NativeHistory，未补充探索偏离/恢复失败的说明。模型会看到当前桌面及一段看似连续的日历编辑历史。代码有重新截图/重新推理，但没有失败后的明确恢复模式，亦无相同无效动作循环阻断。不能仅凭此日志断言全部重复行为由历史缺失造成。

NotesIsTodo第4步：RV探针改变内容，联合图恢复失败，relaunch_restore=True，同时learned_effectful记录数据不一致。full9.relaunch_restore用activity+layout判断返回成功，将content_sig不同仅写入_relaunch_content_same；FULL45据此标记危险控件，但仍将该次回滚设为成功并继续。之后模型进一步点Task list并回答True，最终零分。页面恢复不等于数据恢复。

本轮共有3个窗口记录relaunch_restore=True且learned_effectful：NotesIsTodo（零分）、SystemWifiTurnOff（成功）、TurnOffWifiAndTurnOnBluetooth（成功）。因此先前“最终恢复失败且零分”的9题只是明确rollback_failed标签关联统计，不包含所有恢复副作用候选；NotesIsTodo即属于形式恢复成功但内容已改变的额外失败候选。不能把9当作因果总数或完整上限。

建议修复顺序（本次仅诊断，未改算法）：拦截不具备明确逆操作的匿名工具栏导航；区分页面恢复与数据恢复，内容不一致进入失败分支；失败时立即停探索并向模型提供真实偏离状态，避免无说明沿用原历史；增加重复动作且状态不变的恢复/终止机制。改完对候选题按固定图条件做受控对照，才可测出恢复机制对成功率的因果影响。
