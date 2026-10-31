"""Recompute full-success and call-count tables from unchanged per-task records."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
out={}
def passed(r):return r.get('success')==1 and not r.get('error') and not r.get('rc') and not r.get('teardown_error')
for model in ['8B','2B']:
 rows={a:[json.loads(l) for l in (ROOT/'summary'/model/(a+'.runs.jsonl')).read_text().splitlines()] for a in ['BASE','FULL45']}
 assert all(len(rs)==len({r['task'] for r in rs})==116 for rs in rows.values())
 assert {r['task'] for r in rows['BASE']}=={r['task'] for r in rows['FULL45']}
 b={r['task']:r for r in rows['BASE']};f={r['task']:r for r in rows['FULL45']};common=[k for k in b if passed(b[k]) and passed(f[k])];m={}
 for arm,rs in rows.items():
  calls=[r['model_calls'] for r in rs if r.get('model_calls') is not None];ss=[r for r in rs if passed(r)]
  m[arm]=dict(tasks=len(rs),successes=len(ss),success_rate_pct=100*len(ss)/len(rs),calls_observed_tasks=len(calls),mean_calls=sum(calls)/len(calls),success_mean_calls=sum(r['model_calls'] for r in ss)/len(ss),episode_wall_s=sum(r['e2e_s'] for r in rs),error_tasks=[r['task'] for r in rs if r.get('error') or r.get('rc') or r.get('teardown_error')],partial_scores=[dict(task=r['task'],score=r['success']) for r in rs if r.get('success') not in [None,0,1]])
 a=sum(f[k]['model_calls'] for k in common)/len(common);c=sum(b[k]['model_calls'] for k in common)/len(common)
 m['paired']=dict(common_successes=len(common),full_calls=a,base_calls=c,call_change_pct=100*(a/c-1),lost=[k for k in b if passed(b[k]) and not passed(f[k])],gained=[k for k in b if not passed(b[k]) and passed(f[k])],common_wall_change_pct=100*(sum(f[k]['e2e_s'] for k in common)/sum(b[k]['e2e_s'] for k in common)-1))
 m['all_episode_wall_change_pct']=100*(m['FULL45']['episode_wall_s']/m['BASE']['episode_wall_s']-1);out[model]=m
(ROOT/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
