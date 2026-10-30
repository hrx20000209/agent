# Pure GPU paired full116 run, 2026-10-30

Started 17:24:17 Asia/Shanghai. New independent emulator containers: `pxchen-mai8b-base-puregpu-20261030` (BASE), `pxchen-mai8b-full45-puregpu-20261030` (FULL45). Both share only MAI-UI-8B vLLM at port 18197, GPU5, BF16, CPU offload 0. Host-resident nohup supervisor PID at launch: 1986632.

Upstream main 81a1531758da249c2acb74baa761d2da65e01b1c; senior task order, seed30, min(30,int(20*complexity)) call budget. FULL45 graph execution and exploration enabled, empty initial graph then online accumulation across its sequential tasks. BASE has neither. No SEENR warm start or CPX recovery_guard. Only upstream algorithm patch is recovery requery budget enforcement; MAI adapter and host infrastructure wrappers are included in package.tar.gz.

Previous CPU-offload run was manually interrupted and retained separately; no completed scores or interrupted tasks reused in this fresh cohort. Host supervisor handles device exits/timeouts, records failed task, restarts device before next task; three consecutive infrastructure failures stop that arm. This does not establish a fix for the original shared QEMU crash cause.

Fresh input benchmark: four requests per service, two concurrent, identical bytes per pair, changed goal prefix and image corner pixel to avoid full prefix/image cache reuse. Mean latency old 2GiB CPU offload 57.22s, pure GPU 5.75s, 9.94x. These are model request timings, not full AndroidWorld runtime predictions. See benchmark/fresh_summary.json.

run_config.json records exact launch options, source hashes and task list. package.tar.gz is the deployed source package. This folder archives launch settings, not final experiment results.
