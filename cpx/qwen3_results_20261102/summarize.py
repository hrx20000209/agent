"""Recompute this single paired run; no model/device access and no retry merging."""
from pathlib import Path
import json,statistics,datetime
ROOT=Path(__file__).resolve().parent

def rows(arm):return [json.loads(l) for l in (ROOT/'records'/(arm+'.runs.jsonl')).read_text().splitlines() if l.strip()]
def normal(r):return not r.get('error') and not r.get('rc') and not r.get('teardown_error')
def successful(r):return r.get('success')==1 and normal(r)
def average(rs):return statistics.mean(r['model_calls'] for r in rs) if rs else None

def summarize():
 config=json.loads((ROOT/'records/run_config.json').read_text());out=dict(model=config['model'],tag=config['tag'],step_definition='LLM generation requests (model_calls), not exploration/graph/environment actions',errors_count_as_failure=True,missing_calls_imputed=False,earlier_runs_merged=False,arms={});data={}
 for arm in ['BASE','FULL45']:
  r=rows(arm);assert len(r)==116 and [x['task'] for x in r]==config['tasks'];assert len({x['task'] for x in r})==116
  assert all(type(x.get('model_calls')) is int and 0<=x['model_calls']<=x['allocated_budget'] for x in r)
  data[arm]={x['task']:x for x in r};ok=[x for x in r if normal(x)];succ=[x for x in r if successful(x)];status=json.loads((ROOT/'records'/(arm+'.status.json')).read_text())
  start=datetime.datetime.fromisoformat(status['started_at']);end=datetime.datetime.fromisoformat(status['ended_at'])
  out['arms'][arm]=dict(n=116,successes=len(succ),success_rate_pct=100*len(succ)/116,total_calls=sum(x['model_calls'] for x in r),all_mean_calls=average(r),normal_n=len(ok),normal_mean_calls=average(ok),success_mean_calls=average(succ),errors=[x for x in r if not normal(x)],ended_beijing=end.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),wall_seconds=(end-start).total_seconds())
 b=data['BASE'];f=data['FULL45'];common=sorted(t for t in b if successful(b[t]) and successful(f[t]));bm=average([b[t] for t in common]);fm=average([f[t] for t in common]);bs={t for t in b if successful(b[t])};fs={t for t in f if successful(f[t])}
 out['common_success']=dict(n=len(common),tasks=common,base_mean_calls=bm,full_mean_calls=fm,change_pct=100*(fm/bm-1),full_only=sorted(fs-bs),base_only=sorted(bs-fs));normal_tasks=sorted(t for t in b if normal(b[t]) and normal(f[t]));out['common_normal']=dict(n=len(normal_tasks),base_mean_calls=average([b[t] for t in normal_tasks]),full_mean_calls=average([f[t] for t in normal_tasks]))
 out['success_gain_pp']=out['arms']['FULL45']['success_rate_pct']-out['arms']['BASE']['success_rate_pct']
 out['unresolved_full_only_recovery_errors']=[]
 for task in sorted(t for t in f if not normal(f[t]) and normal(b[t])):
  result=json.loads((ROOT/'records/episodes/FULL45'/task/'result.json').read_text());trace=result.get('traceback','');assert 'relaunch_restore' in trace and 'screen_elements[action.index]' in trace
  out['unresolved_full_only_recovery_errors'].append(dict(task=task,root_cause='indexed scroll replay receives empty screen_elements',model_calls=f[task]['model_calls']))
 return out
if __name__=='__main__':
 result=summarize();(ROOT/'computed_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['arms','common_success']},ensure_ascii=False));print('BASE',result['arms']['BASE']['successes'],'FULL45',result['arms']['FULL45']['successes'],'common',result['common_success']['n'])
