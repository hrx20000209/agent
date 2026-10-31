from pathlib import Path
import json,gzip
b=Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_pair_puregpu_20261030/experiments/trace_reachability_validation/raw/mai_official/mai8b_pair_single_c30x20_puregpu_20261030');out={}
for task in ['MarkorCreateNote','MarkorCreateNoteFromClipboard','SimpleSmsReply','VlcCreatePlaylist','TasksHighPriorityTasks','SimpleCalendarDeleteEventsOnRelativeDay','TurnOffWifiAndTurnOnBluetooth']:
 for arm in ['BASE','FULL45']:
  p=b/arm/task/'p0';m=json.loads((p/'meta.json').read_text());ss=[json.loads(l) for l in (p/'steps.jsonl').read_text().splitlines()];a=list((p/'a11y').glob('*final*'));final=[gzip.open(q,'rt').read() for q in a]
  out[arm+'/'+task]={'goal':m['goal'],'params':m['params'],'steps':ss,'final_a11y':final}
print(json.dumps(out))
