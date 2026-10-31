from pathlib import Path
import json,collections,statistics
BASE=Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_pair_puregpu_20261030/experiments/trace_reachability_validation/raw/mai_official/mai8b_pair_single_c30x20_puregpu_20261030')
def read(p):
 return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
rows={a:read(BASE/(a+'.runs.jsonl')) for a in ['BASE','FULL45']};out={'episodes':{}}
for arm,rs in rows.items():
 for row in rs:
  p=BASE/arm/row['task']/'p0';res=json.loads((p/'result.json').read_text());steps=read(p/'steps.jsonl');ws=read(p/'windows.jsonl');ds=read(p/'mai_decisions.jsonl');ex=read(p/'mai_executed.jsonl');rq=read(p/'mai_requests.jsonl')
  rec=dict(row=row,result=res,step_kinds=dict(collections.Counter(x.get('kind','model') for x in steps)),windows=len(ws),probe_windows=sum(bool(w.get('probes')) for w in ws),probes=sum(len(w.get('probes',[])) for w in ws),probe_window_s=sum(w.get('window_s',0) for w in ws),rollback_failed=sum(not t.get('rollback_ok',False) for w in ws for t in w.get('probes',[])),joint_restore_fail=sum(w.get('joint_restore',{}).get('ok') is False for w in ws),relaunch_fail=sum(w.get('relaunch_restore') is False for w in ws),drift_unrestored=sum(bool(w.get('drift_unrestored')) for w in ws),model_latency_s=sum(d['latency_s'] for d in ds),request_graph_history_max=max([x.get('graph_history_actions',0) for x in rq] or [0]),last_responses=[d['response'] for d in ds[-2:]],steps_tail=steps[-3:])
  out['episodes'][arm+'/'+row['task']]=rec
print(json.dumps(out))
