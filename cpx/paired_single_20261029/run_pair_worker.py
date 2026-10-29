"""One AndroidWorld emulator per arm; senior order; explicit complexity budget."""
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['BASE','FULL45'],required=True);a=p.parse_args()
ROOT=Path(os.environ['CPX_PAIR_ROOT']);EXP=ROOT/'experiments/trace_reachability_validation';sys.path.insert(0,str(EXP/'scripts'));sys.path.insert(0,str(ROOT))
from mai_resume_20261025 import recover
from mai_warm_worker import sms_probe
TAG='mai8b_pair_single_c30x20_20261029';RUN=EXP/'raw/mai_official'/TAG
CONFIG=EXP/'configs/mai_ui_8b_pair_c30x20.json';c=json.loads(CONFIG.read_text());API='http://172.17.0.1:8097/v1/chat/completions';MODEL='MAI-UI-8B';manifest=json.loads((RUN/'run_config.json').read_text());tasks=manifest['tasks'];assert len(tasks)==len(set(tasks))==116
claim=open('/tmp/mai_device_experiment.claim.lock','a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=dict(status='STARTING',arm=a.arm,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=[],errors=[],pid=os.getpid(),budget='min(30,int(20*complexity))')
def save():
 dest=RUN/(a.arm+'.status.json');tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,dest)
def append(row):
 with (RUN/(a.arm+'.runs.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
save()
try:
 for name,h in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 recover(API,MODEL,c);sms_probe();state['status']='RUNNING';save()
 for task in tasks:
  out=RUN/a.arm/task/'p0';assert not out.exists(),'Previous episode retained: '+str(out)
  state['current_task']=task;save();recover(API,MODEL,c)
  env=dict(os.environ,FULL_ARM=a.arm,SEQ_ARM=a.arm,SEQ_APPDIR=str(RUN/('graph' if a.arm=='FULL45' else 'base_unused')),MAI_CONFIG=str(CONFIG),ONDEVICE_LOAD='0')
  for key in ['TPL_DEMOS','EXPLORE_FIT','WARM_EXPLORE','NO_EXEC','NO_SYS','PIPE_ONLY','GRAPH_PROMPT','RAND_EXPLORE','SAME_TPL','AGGR_COMMIT','NAV_UNAMBIG','MAI_SHARED_STATE','MAI_EPISODE_ROOTS_JSON','MAI_SMOKE_EMPTY_STATS']:env.pop(key,None)
  cmd=[sys.executable,str(EXP/'scripts/mai_upstream_probe.py'),'--task',task,'--seed','30','--variant','p0_pair','--mode','agent','--out',str(out),'--api_url',API,'--model',MODEL]
  started=time.time();timed_out=False
  with (RUN/(a.arm+'_'+task+'.log')).open('w') as log:
   try:rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=5400).returncode
   except subprocess.TimeoutExpired:rc=124;timed_out=True
  result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {'error':'no result'}
  meta=json.loads((out/'meta.json').read_text()) if (out/'meta.json').exists() else {};budget=min(30,int(20*meta['complexity'])) if 'complexity' in meta else None;calls=result.get('model_calls');attempts=out/'mai_attempts.jsonl'
  error=result.get('error') or result.get('eval_error') or ('episode timeout 5400s' if timed_out else None)
  if calls is not None and budget is not None and calls>budget:error='Budget exceeded: '+str(calls)+' > '+str(budget)
  row=dict(task=task,arm=a.arm,success=result.get('is_successful'),model_calls=calls,allocated_budget=budget,complexity=meta.get('complexity'),http_attempts=len(attempts.read_text().splitlines()) if attempts.exists() else 0,done=result.get('done'),rc=rc,error=error,teardown_error=result.get('teardown_error'),e2e_s=round(time.time()-started,2),termination='execution_error' if error or rc else 'budget' if not result.get('done') and calls is not None and budget is not None and calls>=budget else 'model_done')
  append(row);state['completed'].append(row)
  if error or rc or row['teardown_error']:state['errors'].append(task)
  save();print(json.dumps(row),flush=True)
 state['status']='FINISHED_WITH_ERRORS' if state['errors'] else 'FINISHED';state.pop('current_task',None);state['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
except BaseException as exc:
 state.update(status='FAILED',error=repr(exc));save();traceback.print_exc();raise
