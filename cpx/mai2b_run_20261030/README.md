# MAI-UI-2B paired full116 evaluation

BASE and FULL45 use separate new AndroidWorld containers, one sequential worker per arm; existing 8B experiment continues independently. Same upstream 81a1531 source, senior task order, seed30 and official MAI adapter as the 8B pure GPU run, except model endpoint 8100 and user-selected senior budget `min(20,int(10*complexity))` (8B remains `min(30,int(20*complexity))`). FULL45 empty graph with online accumulation, graph execution and exploration enabled; exploration window30s, max3 probes; BASE neither. No SEENR warm start, no CPX custom recovery_guard. Host nohup supervision and recovery-requery budget enforcement retained.

Service preflight confirms MAI-UI-2B BF16, context24576, images3, max_num_seqs2, gpu_memory_utilization0.38, GPU2. Server launch options are captured in run_config.json. These results will use a different budget from concurrent 8B, so success-rate differences alone cannot isolate model size.

package.tar.gz contains exact deployed source. This archive describes launch settings, not final results.

Startup: initial readiness preflight failed on stale config filename before any episode. Fixed readiness.py, recorded correction in run_config.json, restarted host supervisor PID2105962 at 17:37:20 Asia/Shanghai. Initial infra logs retained on server.
