"""Keep the original109 and replace only the seven predeclared adapter-error rows."""
import argparse,datetime,hashlib,json,statistics,time
from pathlib import Path

def summarize(rows):
 ok=[r for r in rows if r.get('success')==1 and not r.get('error') and not r.get('rc')]
 valid=[r['model_calls'] for r in rows if isinstance(r.get('model_calls'),(int,float))]
 return dict(tasks=len(rows),successes=len(ok),success_rate=100*len(ok)/len(rows),mean_model_calls=statistics.mean(valid) if valid else None,model_calls_known_tasks=len(valid),successful_mean_model_calls=statistics.mean(r['model_calls'] for r in ok) if ok else None,errors=[r['task'] for r in rows if r.get('error') or r.get('rc')])

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--wait',action='store_true');a=p.parse_args()
 m=json.loads((a.run/'run_config.json').read_text())
 while not (a.run/'final_audit.json').exists():
  if not a.wait:raise RuntimeError('Retry not complete')
  time.sleep(10)
 original=(a.run/'original_BASE.runs.jsonl').read_bytes()
 assert hashlib.sha256(original).hexdigest()==m['original_base_runs_sha256']
 old=[json.loads(l) for l in original.decode().splitlines()]
 retry=[json.loads(l) for l in (a.run/'BASE.runs.jsonl').read_text().splitlines()]
 assert len(retry)==len({r['task'] for r in retry})==7
 assert {r['task'] for r in retry}==set(m['tasks'])
 fixed={r['task']:r for r in retry}
 merged=[fixed.get(r['task'],r) for r in old]
 assert len(merged)==len({r['task'] for r in merged})==116
 assert all(r==o for r,o in zip(merged,old) if o['task'] not in fixed)
 report=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_run=m['original_run'],retry_run=str(a.run),interpretation='Mixed-version repair result: original109 plus seven predeclared adapter-error reruns; not a fresh full116 evaluation',original=summarize(old),retry=summarize(retry),merged=summarize(merged),replaced_tasks=m['tasks'])
 (a.run/'merged_BASE.runs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged))
 (a.run/'repair_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
