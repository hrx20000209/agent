"""Host-resident paired runner: emulator crashes cannot kill supervision.
Infrastructure-failed tasks are recorded once, never selected/retried for score.
"""
import argparse,concurrent.futures,datetime,fcntl,hashlib,json,os,signal,subprocess,threading,time,traceback,urllib.request
from pathlib import Path

TAG='mai8b_pair_single_c30x20_puregpu_20261030'
TERMINAL={'FINISHED','FINISHED_WITH_ERRORS','BLOCKED_ENV','FAILED'}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def read_json(p):
 try:return json.loads(p.read_text()),None
 except Exception as e:return {},repr(e)
def read_rows(p):
 if not p.exists():return []
 return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def harvest(out,task,arm,rc,reason,elapsed):
 result,e=read_json(out/'result.json');meta,me=read_json(out/'meta.json');budget=min(30,int(20*meta['complexity'])) if 'complexity' in meta else None
 calls=result.get('model_calls');error=reason or result.get('error') or result.get('eval_error') or (('invalid/missing result: '+str(e)) if e else None)
 if calls is not None and budget is not None and calls>budget:error='budget exceeded'
 attempts=out/'mai_attempts.jsonl';decisions=out/'mai_decisions.jsonl'
 return dict(task=task,arm=arm,success=result.get('is_successful'),model_calls=calls,allocated_budget=budget,complexity=meta.get('complexity'),http_attempts=len(attempts.read_text().splitlines()) if attempts.exists() else 0,observed_decisions=len(decisions.read_text().splitlines()) if decisions.exists() else 0,done=result.get('done'),rc=rc,error=error,teardown_error=result.get('teardown_error'),e2e_s=round(elapsed,2),termination='infrastructure_error' if reason else 'execution_error' if error or rc else 'budget' if not result.get('done') and calls is not None and budget is not None and calls>=budget else 'model_done',recorded_at=now())

class Runner:
 def __init__(self,root,arm,manifest):
  self.root=root;self.arm=arm;self.m=manifest;self.run=root/'experiments/trace_reachability_validation/raw/mai_official'/TAG;self.c=manifest['containers'][arm];self.inside='/androidworld';self.rows=read_rows(self.run/(arm+'.runs.jsonl'));assert len({x['task'] for x in self.rows})==len(self.rows)
  self.state=dict(status='STARTING',arm=arm,started_at=now(),host_pid=os.getpid(),completed=len(self.rows),errors=[r['task'] for r in self.rows if r.get('error') or r.get('rc') or r.get('teardown_error')],restarts=0)
  self.lock=threading.RLock();self.update()
 def update(self,**kw):
  with self.lock:self.state.update(kw);self.state['heartbeat_at']=now();atomic(self.run/(self.arm+'.status.json'),self.state)
 def command(self,args,timeout=30):return subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 def inspect(self):
  try:
   p=self.command(['docker','inspect','--format','{{json .State}}',self.c],15);return json.loads(p.stdout) if p.returncode==0 else {'Status':'missing','Running':False,'Error':p.stderr}
  except Exception as e:return {'Status':'inspect_error','Running':False,'Error':repr(e)}
 def event(self,kind,**kw):
  with (self.run/(self.arm+'.infra.jsonl')).open('a') as f:f.write(json.dumps(dict(t=now(),kind=kind,**kw))+'\n')
 def preserve_logs(self,task):
  p=self.command(['docker','logs','--tail','150','--timestamps',self.c],20);(self.run/(self.arm+'_'+task+'.container.log')).write_text(p.stdout+p.stderr)
 def ready(self):
  for attempt in range(3):
   self.update(status='WAITING_ENV',recovery_attempt=attempt+1)
   try:
    state=self.inspect()
    if not state.get('Running'):
     self.event('container_start',previous=state);p=self.command(['docker','start',self.c],30)
     if p.returncode:raise RuntimeError(p.stderr)
     self.command(['docker','exec',self.c,'rm','-f','/tmp/cpx_hostwatch_ready'],15)
     self.state['restarts']+=1
    self.update(status='WAITING_ENV',recovery_attempt=attempt+1)
    p=self.command(['docker','exec',self.c,'python','/androidworld/readiness.py'],timeout=330)
    if p.returncode:raise RuntimeError((p.stdout+p.stderr)[-1500:])
    self.event('readiness_passed');return True
   except Exception as e:
    self.event('readiness_failed',attempt=attempt+1,error=repr(e));self.update(status='RECOVERING',last_environment_error=repr(e))
    try:self.command(['docker','stop','-t','5',self.c],20)
    except Exception:pass
    time.sleep(10)
  self.update(status='BLOCKED_ENV');return False
 def commit(self,row):
  assert row['task'] not in {r['task'] for r in self.rows}
  with (self.run/(self.arm+'.runs.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  self.rows.append(row)
  if row.get('error') or row.get('rc') or row.get('teardown_error'):self.state['errors'].append(row['task'])
  self.update(completed=len(self.rows),last_result=row)
 def execute(self,task,out):
  cmd=['docker','exec','-e','FULL_ARM='+self.arm,'-e','SEQ_ARM='+self.arm,'-e','SEQ_APPDIR='+str(Path('/androidworld/experiments/trace_reachability_validation/raw/mai_official')/TAG/('graph' if self.arm=='FULL45' else 'base_unused')),'-e','MAI_CONFIG=/androidworld/experiments/trace_reachability_validation/configs/mai_ui_8b_pair_c30x20.json','-e','ONDEVICE_LOAD=0',self.c,'python','/androidworld/episode_entry.py','--task',task,'--seed','30','--variant','p0_pair','--mode','agent','--out','/androidworld/'+str(out.relative_to(self.root)),'--api_url','http://172.17.0.1:18197/v1/chat/completions','--model','MAI-UI-8B']
  if self.arm=='FULL45':cmd += ['--explore','--budget','30','--max_probes','3']
  start=time.time();reason=None;last_emit=0;rc=None
  with (self.run/(self.arm+'_'+task+'.log')).open('w') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while proc.poll() is None:
    stamp=time.time();state=self.inspect()
    if not state.get('Running'):reason='container exited: '+json.dumps(state);break
    activity=max([start]+[p.stat().st_mtime for p in out.rglob('*') if p.is_file()]) if out.exists() else start
    if stamp-start>2700:reason='episode hard timeout 2700s';break
    if stamp-activity>600:reason='no episode progress for600s';break
    if stamp-last_emit>15:
     self.update(status='RUNNING',current_task=task,episode_elapsed_s=round(stamp-start),last_activity_age_s=round(stamp-activity),docker_state=state.get('Status'));last_emit=stamp
    time.sleep(5)
   if reason:
    self.preserve_logs(task)
    # Kill the owned emulator as well as the local docker client: no orphan actions.
    try:self.command(['docker','stop','-t','5',self.c],20)
    except Exception:pass
    try:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
    except Exception:
     try:os.killpg(proc.pid,signal.SIGKILL)
     except ProcessLookupError:pass
   try:rc=proc.wait(timeout=10)
   except Exception:rc=124
  state=self.inspect()
  if not state.get('Running') and not reason:
   reason='container exited: '+json.dumps(state);self.preserve_logs(task)
  return harvest(out,task,self.arm,rc,reason,time.time()-start)
 def run_all(self):
  failures=0
  try:
   for task in self.m['tasks']:
    if task in {r['task'] for r in self.rows}:continue
    self.update(current_task=task)
    if not self.ready():return
    out=self.run/self.arm/task/'p0'
    if out.exists():
     # Host restart: harvest existing evidence, never silently repeat a task.
     result,e=read_json(out/'result.json')
     if not e and result.get('task')==task and result.get('is_successful') is not None:
      row=harvest(out,task,self.arm,0,None,result.get('wall_s',0));row['resumed_existing_result']=True
     else:row=harvest(out,task,self.arm,125,'interrupted before durable host completion; existing evidence retained',0)
    else:row=self.execute(task,out)
    self.commit(row)
    failures=failures+1 if row['termination']=='infrastructure_error' else 0
    if failures>=3:self.update(status='BLOCKED_ENV',last_environment_error='3 consecutive infrastructure failures; stop crash loop');return
   self.update(status='FINISHED_WITH_ERRORS' if self.state['errors'] else 'FINISHED',current_task=None,ended_at=now())
  except BaseException as e:self.update(status='FAILED',error=repr(e));traceback.print_exc();raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run=a.root/'experiments/trace_reachability_validation/raw/mai_official'/TAG;manifest=json.loads((run/'run_config.json').read_text());claim=open(run/'host_supervisor.lock','a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,h in manifest['source_sha256'].items():assert hashlib.sha256((a.root/name).read_bytes()).hexdigest()==h,name
 runners=[Runner(a.root,arm,manifest) for arm in ['BASE','FULL45']]
 with concurrent.futures.ThreadPoolExecutor(2) as pool:
  fs=[pool.submit(r.run_all) for r in runners]
  while True:
   audit=dict(updated_at=now(),host_pid=os.getpid(),arms={})
   for r in runners:
    with r.lock:s=dict(r.state);rows=list(r.rows)
    s['successes']=sum(x.get('success')==1 and not x.get('error') and not x.get('rc') for x in rows);audit['arms'][r.arm]=s
   try:
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}));s=op.open('http://127.0.0.1:18197/metrics',timeout=5).read().decode();audit['vllm_queue']=[l for l in s.splitlines() if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
   except Exception as e:audit['vllm_queue_error']=repr(e)
   atomic(run/'completion_audit.json',audit)
   if all(f.done() for f in fs):
    for f in fs:
     try:f.result()
     except Exception:pass
    atomic(run/'final_audit.json',audit);break
   time.sleep(10)
if __name__=='__main__':main()
