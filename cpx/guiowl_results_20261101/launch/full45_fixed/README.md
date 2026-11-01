# FULL45 full116 rerun after open_app adapter repair

Only algorithm change: accept both `open` and `open_app`, with nonempty `text`, matching the official converter. No coordinate, answer, seed, prompt, budget or graph policy changes.

- Model: GUI-Owl-1.5-8B-Instruct, same checkpoint and running pure-GPU TP2 service on18197.
- Decoding: temperature0, top_p1, top_k-1, seed42, max_tokens2048.
- One dedicated fresh container; separate from the BASE seven-task repair.
- Exactly the original116-task senior order; direct per-task seed30.
- Budget: uncapped int(10*complexity), same FULL45 counted-action cap.
- Empty initial online graph; sequential sharing across all116 tasks in this single container.
- Same historical statistics assets; no final graph or results reused.
- Exploration enabled, window budget30seconds, max_probes3.
- Same upstream FULL45 code81a1531 plus existing recovery-requery boundary guard.
- Original paired experiment remains unchanged. BASE only retries its7 predeclared adapter exceptions.

Deployment verifies all old hashed sources, allowing only the parser alias, its tests, and host-runner tag/arm selection. It also compares the entire model config, emulator image/command, task order hash, and the exact permitted parser and host-runner diffs. Fourteen targeted tests passed locally; protocol tests and imports are repeated inside the actual container before nohup launch.

Remote root: /home/pxchen/workspace/android_world_mai8b_20261024/cpx_guiowl_full45_fix_20261101

Run tag: guiowl8b_full45_openapp_fix_20261101

Initial ETA: about5–6hours from formal episode start, based on the previous complete FULL45 run (~5hours). The existing2-request vLLM supports concurrent BASE/FULL45 containers; the additional BASE work consists of7tasks only.
