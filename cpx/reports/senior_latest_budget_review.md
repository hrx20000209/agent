# 学长新启动脚本：实际预算检查

读取仓库hrx20000209/android_world的main提交81a1531758da249c2acb74baa761d2da65e01b1c（2026-10-29）。agent仓库main仍为38fca149；本次新增启动脚本位于android_world。

正式Orin实验：scripts/launch/run_orin_e2e.sh -> scripts/e2e_driver.py -> scripts/full_run.py。不传--max_steps。继承rgs_run.Episode.max_steps，默认min(gelab_agent.MAX_AGENT_STEPS, suite_utils._allocate_step_budget(task.complexity))；前者为20，后者int(10*complexity)。所以默认min(20,int(10*complexity))。

LabServer seen/unseen评测同样不传--max_steps，使用默认复杂度预算。第一阶段warm seeds31/32同默认预算。warm2仅针对还无成功demo的seen模板，用seeds33/34，显式--max_steps40，直接覆盖默认预算。full_run手动调用decide/commit，不走GELABAgent.step中的20步退出检查。

FULL45主循环按m[calls]<budget且k<2*budget执行；图路径/函数可包含多次实际触控，所以2*budget不是触控总数上限。BASE按k<budget计循环步骤。FULL45的requery会再增加calls，但没有轮内剩余预算检查，最后一轮触发时可能比目标预算多1次；服务重试口径也不等于原始HTTP请求总数。

子进程另有timeout2400秒，是运行时间限制，不是步数。

最新实验组+SEENR：加载seenW1预热图和函数库，SAME_TPL/AGGR_COMMIT/TPL_DEMOS/EXPLORE_FIT/NAV_UNAMBIG；与之前MAI8B统一50调用、空图全量测试并非同一设置。+SEENR0不含EXPLORE_FIT/NAV_UNAMBIG。

纯函数验证（直接抽取本次源码max_steps方法，不执行Android/model）：默认complexity1/1.2/3分别10/12/20；显式40返回40，显式50返回50。启动脚本实际只在warm2传40，正式评测未传50。
