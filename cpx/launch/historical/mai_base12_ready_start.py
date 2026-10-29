import pathlib,json,subprocess,concurrent.futures,time,shlex,datetime
host=pathlib.Path('/home/pxchen/workspace/android_world_mai8b_20261024/experiments/trace_reachability_validation');root='/androidworld/experiments/trace_reachability_validation';tag='mai8b_base_full116_calls50_12w_20261026';run=host/'raw/mai_official'/tag
config=root+'/configs/mai_ui_8b_base_full50_20261026.json'
def run_cmd(c,args,timeout=30):
 return subprocess.run(['docker','exec',c,*args],check=True,capture_output=True,text=True,timeout=timeout)
def ready(i):
 c=f'pxchen-mai8b-base-w{i}-20261026';deadline=time.monotonic()+180
 while time.monotonic()<deadline:
  try:
   if run_cmd(c,['adb','shell','getprop','sys.boot_completed'],15).stdout.strip()=='1':break
  except Exception:pass
  time.sleep(5)
 else:raise RuntimeError(c+' boot timeout')
 run_cmd(c,['adb','root']);run_cmd(c,['adb','wait-for-device']);run_cmd(c,['adb','shell','input','keyevent','KEYCODE_HOME'])
 code="import sys,json;sys.path.insert(0,%r);from mai_resume_20261025 import recover;from mai_warm_worker import sms_probe;c=json.load(open(%r));recover('http://172.17.0.1:8097/v1/chat/completions',c['model'],c);sms_probe();print('READY: model/a11y/SMS')"%(root+'/scripts',config)
 result=run_cmd(c,['python','-c',code],90)
 print('PREFLIGHT',i,result.stdout.strip(),flush=True)
 argv=['nohup','python','-u',root+'/scripts/mai_base_full50_worker.py','--tag',tag,'--worker-id',str(i),'--workers','12','--max-calls','50','--config',config]
 log=root+'/raw/mai_official/'+tag+'/workers/BASE-'+str(i)+'.log'
 subprocess.run(['docker','exec','-d',c,'sh','-c','exec '+shlex.join(argv)+' > '+shlex.quote(log)+' 2>&1 < /dev/null'],check=True)
 print('LAUNCHED',i,c,flush=True)
 return i
with concurrent.futures.ThreadPoolExecutor(4) as pool:
 futures=[pool.submit(ready,i) for i in range(8,12)]
 for f in concurrent.futures.as_completed(futures):f.result()
launch=json.loads((run/'launch.json').read_text());launch.update(phase='all 12 workers launched',last_worker_launched_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(run/'launch.json').write_text(json.dumps(launch,indent=2))
print('ALL_12_STARTED',flush=True)
