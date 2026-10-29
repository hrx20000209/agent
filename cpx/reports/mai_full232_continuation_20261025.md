# MAI-UI-8B 全量续跑记录

用户于2026-10-25要求跑完全量后，11:04（北京时间）启动续跑。

原始tag：`mai8b_full232_shared_20261025`。此前209条记录：BASE108，FULL45101；5个worker退出，3个正常结束。23条任务尚未执行，另外BASE/FULL45各有一条SimpleSmsReplyMostRecent初始化异常，尚未调用模型。

## 续跑方式

新增独立脚本`mai_resume_20261025.py`，不修改原driver、模型客户端或算法源码。启动时逐项校验原run_config.json记录的源码和配置哈希；沿用原seed30、4worker任务分配、20调用上限、模型服务和FULL45共享图。仅执行尚无result.json的任务，保留既有正常结果和错误。原始worker日志、状态文件、run_config均保留；续跑另记workers/*.continuation.log及*.continuation.status.json。

无障碍连接异常只在episode边界处理：保留原服务列表，重新绑定FastA11yService、回到HOME，然后重新通过原preflight。不会在任务运行中替模型执行操作。单个episode如果产生错误结果则保留错误并继续其余任务；没有结果的异常仍停止worker。

这是同一轮共享图的续跑，后续任务读取此前已积累的图，不能表述为重新从空图开始的独立实验。原轮起始时确为空图。

## 短信环境异常

SimpleSmsReplyMostRecent在android_world/task_evals/single/sms.py初始化读取收件箱第一条消息时IndexError。进一步检查发现SIM状态为空/NOT_READY，telephony处于无服务状态；`adb emu sms send`虽返回OK，收件箱仍没有消息。应将原短信相关零分列为环境风险，不直接归因于模型。

修复在已完成任务的空闲BASE worker1容器中验证，不改变正在运行的其他worker。任何修复后重试应独立存储并保留首次记录，明确标记续跑/重试，避免覆盖或选择性汇报。

补充：排查时曾向BASE worker2发送一条readiness-probe短信，并尝试重启modem，当时SimpleSmsSendReceivedAddress已启动。因此BASE该episode必须视为受人工环境探测影响，不能直接作为正式有效结果。FULL45对应episode未发送探测短信，但其电话环境同样没有通过验证。

## 全量覆盖完成与环境修复

截至北京时间11:26，原始全量已有232条唯一arm/task记录，BASE和FULL45各116条，无遗漏、无重复。其中各115条正常结束，各1条SimpleSmsReplyMostRecent初始化错误。全部原始和续跑worker均已退出。正常结束不代表任务成功；原始数据中BASE完整成功45条、部分得分0.5，FULL45完整成功49条、部分得分合计1.0。由于短信环境无效，不将这些计数作为干净的全量比较。

根因定位：Docker容器eth0的`net.ipv6.conf.eth0.disable_ipv6=1`，虽然lo有::1，`getaddrinfo('::1', ..., AI_ADDRCONFIG)`仍返回EAI_ADDRFAMILY。modem启动日志明确出现`Unable to connect character device modem: address resolution failed for ::1:...`。启用eth0 IPv6后，地址解析恢复；必要时在episode之间重启vendor.ril-daemon。8个容器SIM均变为LOADED，且均实际收到测试短信，随后删除诊断短信。

修复只作用于本次8个pxchen容器的网络命名空间，没有修改宿主机或学长容器。使用短生命周期的privileged辅助容器共享目标容器网络命名空间写入sysctl；辅助容器执行后即删除。直接docker exec修改/proc/sys因只读挂载失败，未修改宿主挂载设置。

后续新建容器可直接在Docker网络端点设置IPv6，已用临时容器验证eth0的disable_ipv6=0且AI_ADDRCONFIG解析::1成功：

```sh
--network 'name=bridge,driver-opt=com.docker.network.endpoint.sysctls=net.ipv6.conf.IFNAME.disable_ipv6=0'
```

本机Docker拒绝普通的`--sysctl net.ipv6.conf.eth0.disable_ipv6=0`，必须用上述端点driver-opt。现有容器的临时修复在网络命名空间重建后可能丢失，因此重启/重建后应先复查IPv6地址解析及实际短信接收，且只在任务未运行时发送诊断短信。

诊断期间仅在空闲BASE worker1试过35.2.10和33.1.24模拟器；最终已恢复37.1.11原二进制、原advancedFeatures配置，删除临时用户feature覆盖。8个补测容器均使用原37.1.11，未依赖降级版本。下载的旧版本诊断文件保留在pxchen/services/android_emulator_*，尚未清理。

## 独立短信补测

tag：`mai8b_sms_repair14_20261025`，BASE/FULL45各7个短信相关任务，共14条。沿用官方模型配置、seed30、原步数预算和4worker/组；FULL45补测从独立空图开始。包括MarkorCreateNoteAndSms及全部6个SimpleSms任务。此前的短信零分也有环境风险，因此补测所有短信任务，而非只挑选初始化报错的两条。

补测输出与原始232条分别保留；两轮共享图历史不同，不静默合并成同一轮完整正式结果。原始全量结果、completion_audit.json及SMS补测provenance.json均可审计。修复后的14条结果及最终状态另行补充。

## nohup后台与后续收尾

按用户要求，企稳后退出，后台继续执行。8个原补测worker使用docker exec -d启动，本就不依赖SSH；另用真正的nohup启动`mai_completion_watch.py`，每30秒写completion_audit.json，所有worker退出后核验14条唯一结果、错误记录及完成标记，写FINISHED或FAILED后退出。它不会把失败结果改成成功，不会操纵模型或自动用热图重试。

补测BASE/MarkorCreateNoteAndSms曾因ANSWER文本包含嵌套引号，在AndroidWorld的display_message overlay广播处失败。仅修复android_world/env/interface.py的display_message，使用shlex.join构造完整设备shell命令，模型答案及动作不变。已验证双引号、单引号、美元符号、反引号、命令替换文本逐字往返不被执行，实际overlay广播返回0。

该失败的结果、截图、日志和worker状态均保留，分别为retained_attempt_errors.jsonl、BASE/MarkorCreateNoteAndSms/p0_initial_overlay_error及.initial_overlay_error日志等；从补测当前runs.jsonl中移出该环境失败行后，BASE worker0通过nohup重新启动，依次重跑该任务并执行尚未运行的SimpleSmsSend。BASE不读共享图，因此不会用失败尝试积累的图来重试。补测provenance.json记录补丁SHA和重试原因。原始232条完全保留。

后台目录：`/home/pxchen/workspace/android_world_mai8b_20261024/experiments/trace_reachability_validation/raw/mai_official/mai8b_sms_repair14_20261025`。nohup_supervisor.log、nohup_supervisor.pid、completion_audit.json和workers/下状态文件可直接检查。

## 最终结果

短信补测于北京时间2026-10-25 11:41:37由nohup收尾进程确认为FINISHED：14/14唯一结果，缺项0，当前结果执行错误0，8个worker全部FINISHED。此前overlay失败尝试仍保留在独立错误档案中。

| 轮次 | BASE完整成功 | FULL45完整成功 | 说明 |
|---|---:|---:|---|
| 原始全量（各116题） | 45 | 49 | 每组115条正常结束、1条短信初始化错误；短信相关结果存在环境风险 |
| 独立短信补测（各7题） | 3 | 4 | 全部正常结束；单独空图，不混算成原轮成功率 |

原始全量含部分得分时，BASE累计45.5分，FULL45累计50.0分。这里仅记录原始计数，不能将这轮视为环境一致、没有中断的正式全量对照。

短信补测逐项完整成功（1成功，0任务失败；均无执行错误）：

| 任务 | BASE | FULL45 |
|---|---:|---:|
| MarkorCreateNoteAndSms | 0 | 0 |
| SimpleSmsReply | 1 | 0 |
| SimpleSmsReplyMostRecent | 0 | 1 |
| SimpleSmsResend | 0 | 0 |
| SimpleSmsSend | 1 | 1 |
| SimpleSmsSendClipboardContent | 1 | 1 |
| SimpleSmsSendReceivedAddress | 0 | 1 |

最终没有遗留运行中的worker。当前任务覆盖和补测已结束；如果需要正式全量成功率，应在修复后的环境下从空图完整重跑116题/组，统一共享图积累历史，而不是替换旧轮短信分数后直接合并。
