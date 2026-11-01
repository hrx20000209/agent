"""Recompute GUI-Owl results from original per-task rows; no device access."""
from pathlib import Path
import json,statistics

def read(root,run,arm):
 rows=[json.loads(l) for l in (root/'records'/run/(arm+'.runs.jsonl')).read_text().splitlines() if l.strip()]
 assert len(rows)==len({r['task'] for r in rows})
 return rows
def success(r):return r.get('success')==1 and not r.get('error') and not r.get('rc')
def stats(rows):
 ok=[r for r in rows if success(r)];calls=[r['model_calls'] for r in rows if isinstance(r.get('model_calls'),(int,float))]
 return dict(tasks=len(rows),successes=len(ok),success_rate_pct=100*len(ok)/len(rows),mean_calls=statistics.mean(calls) if calls else None,known_calls_tasks=len(calls),successful_mean_calls=statistics.mean(r['model_calls'] for r in ok) if ok else None,errors=[{k:r.get(k) for k in ['task','error','rc','observed_decisions','termination']} for r in rows if r.get('error') or r.get('rc')])
def paired(base,full):
 a={r['task']:r for r in base};b={r['task']:r for r in full};assert set(a)==set(b)
 common=sorted(t for t in a if success(a[t]) and success(b[t]));bm=statistics.mean(a[t]['model_calls'] for t in common);fm=statistics.mean(b[t]['model_calls'] for t in common)
 return dict(common_success_tasks=common,n=len(common),base_mean=bm,full_mean=fm,change_pct=100*(fm/bm-1),gained=sorted(t for t in a if not success(a[t]) and success(b[t])),lost=sorted(t for t in a if success(a[t]) and not success(b[t])))
def compute(root):
 oldbase=read(root,'original','BASE');oldfull=read(root,'original','FULL45');retry=read(root,'base7','BASE');full=read(root,'full45_fixed','FULL45')
 assert len(oldbase)==len(oldfull)==len(full)==116 and len(retry)==7
 targets={r['task']:r for r in retry};assert set(targets)=={r['task'] for r in oldbase if r.get('error')}
 assert all('Unsupported GUI-Owl action: open_app' in r['error'] for r in oldbase if r['task'] in targets)
 merged=[targets.get(r['task'],r) for r in oldbase]
 assert sum(r==o for r,o in zip(merged,oldbase) if o['task'] not in targets)==109
 result=dict(original_base=stats(oldbase),original_full45=stats(oldfull),base7=stats(retry),corrected_base=stats(merged),corrected_full45=stats(full),original_pair=paired(oldbase,oldfull),corrected_pair=paired(merged,full),replaced_tasks=list(targets),base_mean_interpretation='original109 plus repaired7; not a fresh full116 run',step_definition='model calls, not total Android actions or exploration probes',mean_call_change_pct=100*(stats(full)['mean_calls']/stats(merged)['mean_calls']-1))
 return result,merged
if __name__=='__main__':
 root=Path(__file__).resolve().parent;result,merged=compute(root)
 out=root/'computed_summary.json'
 if out.exists():assert json.loads(out.read_text())==result,'Stored summary differs from raw records'
 else:out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 archived=root/'records/base7/merged_BASE.runs.jsonl'
 assert [json.loads(l) for l in archived.read_text().splitlines()]==merged
 print(json.dumps({k:v for k,v in result.items() if k not in ['original_base','original_full45']},ensure_ascii=False,indent=2))
