import os,json,time,statistics,datetime,subprocess,urllib.request
from pathlib import Path
ROOT=Path(os.environ['CPX_PAIR_ROOT']);RUN=ROOT/'experiments/trace_reachability_validation/raw/mai_official/mai8b_pair_single_c30x20_20261029';opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
while True:
 audit=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),arms={})
 finished=True
 for arm in ['BASE','FULL45']:
  p=RUN/(arm+'.status.json');s=json.loads(p.read_text()) if p.exists() else {'status':'NOT_STARTED','completed':[]}
  rows=s.get('completed',[]);good=[r for r in rows if not r.get('error') and not r.get('rc') and r.get('success') is not None];succ=[r for r in good if r['success']==1];calls=[r['model_calls'] for r in good if r.get('model_calls') is not None]
  audit['arms'][arm]=dict(status=s['status'],completed=len(rows),success=sum(r.get('success')==1 for r in good),errors=s.get('errors',[]),current_task=s.get('current_task'),mean_calls=statistics.mean(calls) if calls else None)
  if s['status'] not in ['FINISHED','FINISHED_WITH_ERRORS','FAILED']:finished=False
 try:
  text=opener.open('http://172.17.0.1:8097/metrics',timeout=10).read().decode();sel=[x for x in text.splitlines() if x.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))];audit['vllm_queue_metrics']=sel
 except Exception as e:audit['vllm_queue_error']=str(e)
 tmp=RUN/'completion_audit.tmp';tmp.write_text(json.dumps(audit,indent=2));os.replace(tmp,RUN/'completion_audit.json')
 if finished:
  (RUN/'final_audit.json').write_text(json.dumps(audit,indent=2));break
 time.sleep(30)
