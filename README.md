# MobileExplorer

`real_phone_agent.py` is the standalone real-device entry point. It does not
depend on AndroidWorld: screenshots, accessibility trees, app launching, input,
and recovery all use one explicitly selected ADB serial.

```bash
python real_phone_agent.py \
  --task "Delete the recipes named Lentil Soup and Garlic Butter Shrimp from Broccoli" \
  --serial <adb-serial> \
  --api_url http://<host>:<port>/v1/chat/completions \
  --model GELAB-ZERO-4B \
  --max_steps 20 \
  --exploration on \
  --graph on \
  --skip off \
  --profile_memory on \
  --out_dir runs/broccoli
```

The three mechanism switches are independent. In particular, `--graph off`
does not disable exploration; it only discards probe results. `--skip` defaults
to `off` because the paper's simulator measurements found that unverified
skipping can hurt task success.

Each run writes `config.json`, `run_events.jsonl`, `memory_profile.jsonl`,
`belief_graph.json`, `screens/`, `prompts/`, `probe_trace.jsonl`, and
`filtered_elements.jsonl` under the selected output directory. Model decisions
read an immutable snapshot committed at the end of the previous step, so
current-step exploration cannot leak into the concurrent inference prompt.

The implementation reuses the deterministic modules in `Explorer/` and adds
the missing real-phone layer. Install the packages in `requirements.txt` (the
current repository's older prototype also imports Pillow, NumPy, and Requests)
before running on a device.
