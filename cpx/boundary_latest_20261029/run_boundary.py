"""Sequential boundary test of unchanged upstream FULL45 with MAI transport."""
import os,sys,json,time,subprocess,hashlib,datetime,fcntl,traceback
from pathlib import Path
ROOT=Path(os.environ.get('CPX_PROBE_ROOT','/androidworld'));EXP=ROOT/'experiments/trace_reachability_validation';SCRIPTS=EXP/'scripts';sys.path.insert(0,str(SCRIPTS));sys.path.insert(0,str(ROOT))
from mai_resume_20261025 import recover
from mai_warm_worker import sms_probe
TAG='mai8b_upstream81a1531_boundary_complexity_20261029';RUN=EXP/'raw/mai_official'/TAG
RUN.mkdir(parents=True,exist_ok=True)
API='http://172.17.0.1:8097/v1/chat/completions';MODEL='MAI-UI-8B'
CONFIG=EXP/'configs/mai_ui_8b_upstream_complexity.json'
c=json.loads((EXP/'configs/mai_ui_8b_full50_20261025.json').read_text());c['evaluation_note']='Single fresh emulator; upstream81a1531 FULL45 unchanged; official MAI adapter; complexity budget min(20,int(10*complexity)); no CPX recovery guard; six boundary tasks in senior order.';CONFIG.write_text(json.dumps(c,indent=2))
selected={'ExpenseDeleteDuplicates','ExpenseDeleteDuplicates2','MarkorCreateNote','NotesIsTodo','SimpleCalendarAddOneEvent','SimpleSmsReply'}
order=(EXP/'reports/e2e116n_task_order.txt').read_text().strip().split(',');tasks=[t for t in order if t in selected];assert len(tasks)==6
upstream=json.loads((ROOT/'upstream_sha256.json').read_text())
for n,sha in upstream.items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==sha,n
assert (ROOT/'experiments/runtime_graph_exploration/graph/observations.jsonl').stat().st_size==0
manifest=dict(tag=TAG,tasks=tasks,seed=30,arm='FULL45',upstream_commit='81a1531758da249c2acb74baa761d2da65e01b1c',upstream_verified_files=len(upstream),workers=1,budget='min(20,int(10*task.complexity))',explicit_max_steps=None,mai_config=c,algorithm_patches=[],graph_initialization='empty once, online accumulated only across selected six tasks',historical_observations=0,historical_statistics=True,transport='MAI official nonstreaming adapter',variant='p0_boundary',background_load=False,cold_subset_not_full_suite=True)
(RUN/'run_config.json').write_text(json.dumps(manifest,indent=2))
state=dict(status='WAITING_ENV',completed=[],errors=[],started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
def save():
 p=RUN/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,p)
def adb(*a):return subprocess.check_output(['adb',*a],text=True,timeout=30).strip()
claim=open('/tmp/mai_device_experiment.claim.lock','a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
save()
try:
 for i in range(90):
  try:
   if adb('shell','getprop','sys.boot_completed')=='1':break
  except Exception:pass
  time.sleep(3)
 else:raise RuntimeError('Emulator boot timeout')
 adb('root');adb('wait-for-device');adb('shell','input','keyevent','KEYCODE_HOME')
 recover(API,MODEL,c);sms_probe()
 state['status']='RUNNING';save()
 for task in tasks:
  out=RUN/'FULL45'/task/'p0';assert not out.exists(),'Existing episode retained'
  recover(API,MODEL,c)
  env=dict(os.environ,FULL_ARM='FULL45',SEQ_ARM='FULL45',SEQ_APPDIR=str(RUN/'graph'),MAI_CONFIG=str(CONFIG),ONDEVICE_LOAD='0')
  for name in ['TPL_DEMOS','EXPLORE_FIT','WARM_EXPLORE','NO_EXEC','NO_SYS','PIPE_ONLY','GRAPH_PROMPT','RAND_EXPLORE','SAME_TPL','AGGR_COMMIT','NAV_UNAMBIG','MAI_SHARED_STATE','MAI_EPISODE_ROOTS_JSON','MAI_SMOKE_EMPTY_STATS']:
   env.pop(name,None)
  cmd=[sys.executable,str(SCRIPTS/'mai_upstream_probe.py'),'--task',task,'--seed','30','--variant','p0_boundary','--mode','agent','--out',str(out),'--api_url',API,'--model',MODEL]
  state['current_task']=task;save();started=time.time()
  with (RUN/(task+'.log')).open('w') as log:
   try:rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
   except subprocess.TimeoutExpired:rc=124
  result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {'error':'no result'}
  meta=json.loads((out/'meta.json').read_text()) if (out/'meta.json').exists() else {}
  budget=min(20,int(10*meta['complexity'])) if 'complexity' in meta else None
  row=dict(task=task,rc=rc,success=result.get('is_successful'),model_calls=result.get('model_calls'),budget=budget,done=result.get('done'),error=result.get('error') or result.get('eval_error') or ('timeout' if rc==124 else None),e2e_s=round(time.time()-started,2),metrics={k:v for k,v in result.items() if k.startswith('m_')},explore_disabled=result.get('explore_disabled'))
  with (RUN/'runs.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
  state['completed'].append(row)
  if rc or row['error']:state['errors'].append(task)
  save();print(json.dumps(row),flush=True)
 state['status']='FINISHED_WITH_ERRORS' if state['errors'] else 'FINISHED';state.pop('current_task',None);state['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
except BaseException as exc:
 state['status']='FAILED';state['error']=repr(exc);save();traceback.print_exc();raise
