import subprocess,json,pathlib,shlex,time,datetime
containers=['pxchen-mai8b-official-20261024','pxchen-mai8b-base-w1-20261025','pxchen-mai8b-base-w2-20261025','pxchen-mai8b-base-w3-20261025','pxchen-mai8b-shared-w1-20261024','pxchen-mai8b-full-w1-20261025','pxchen-mai8b-full-w2-20261025','pxchen-mai8b-full-w3-20261025']
tag='mai8b_full45_full116_calls50_recoveryfix_20261026'
host=pathlib.Path('/home/pxchen/workspace/android_world_mai8b_20261024/experiments/trace_reachability_validation')
root='/androidworld/experiments/trace_reachability_validation'
run=host/'raw/mai_official'/tag
assert not run.exists(), 'Tag already exists; do not launch twice'
# Check source compilation and exact partition before mutating device state.
subprocess.run(['docker','exec',containers[0],'python','-m','py_compile',root+'/scripts/mai_full45_full50_worker.py',root+'/scripts/mai_completion_watch.py'],check=True)
tasks=json.loads((host/'raw/mai_official/mai8b_full232_shared_20261025/run_config.json').read_text())['tasks']
parts=[tasks[i::8] for i in range(8)]
assert len(tasks)==len(set(tasks))==116 and sum(map(len,parts))==116
assert set.union(*(set(p) for p in parts))==set(tasks)
print('PARTITIONS',list(map(len,parts)),flush=True)
audit=json.loads((host/'raw/mai_official/mai8b_full45_full116_calls50_emptyshared_20261025/completion_audit.json').read_text())
assert audit['status']=='FINISHED'
expected=json.loads((host/'code_snapshots/recovery_fix_20261026_before/after_sha256.json').read_text())
import hashlib,urllib.request
for name,digest in expected.items():
 assert hashlib.sha256((host/name).read_bytes()).hexdigest()==digest,name
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
models=json.load(opener.open('http://127.0.0.1:8097/v1/models',timeout=10))['data']
assert any(m['id']=='MAI-UI-8B' and m['max_model_len']>=24576 for m in models)
run.mkdir(parents=True)
(run/'workers').mkdir()
# Preserve the exact fixed source so subsequent changes cannot obscure this run.
import shutil
for name in expected:
 target=run/'source_snapshot'/name
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(host/name,target)
(run/'source_snapshot/sha256.json').write_text(json.dumps(expected,indent=2))
launch=dict(tag=tag,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),containers=containers,partition_counts=list(map(len,parts)))
(run/'launch.json').write_text(json.dumps(launch,indent=2))
for i,c in enumerate(containers):
 argv=['nohup','python','-u',root+'/scripts/mai_full45_full50_worker.py','--tag',tag,'--worker-id',str(i),'--workers','8','--max-calls','50','--config',root+'/configs/mai_ui_8b_full50_20261025.json']
 log=root+'/raw/mai_official/'+tag+'/workers/FULL45-'+str(i)+'.log'
 command='exec '+shlex.join(argv)+' > '+shlex.quote(log)+' 2>&1 < /dev/null'
 subprocess.run(['docker','exec','-d',c,'sh','-c',command],check=True)
 print('LAUNCHED',i,c,flush=True)
for _ in range(20):
 if (run/'run_config.json').exists(): break
 time.sleep(.5)
assert (run/'run_config.json').exists(), 'Worker manifest missing'
argv=['nohup','python','-u',root+'/scripts/mai_completion_watch.py','--tag',tag]
log=root+'/raw/mai_official/'+tag+'/nohup_supervisor.log'
subprocess.run(['docker','exec','-d',containers[0],'sh','-c','exec '+shlex.join(argv)+' > '+shlex.quote(log)+' 2>&1 < /dev/null'],check=True)
print('SUPERVISOR_LAUNCHED',run,flush=True)
print('STARTED_UTC',launch['started_at'],flush=True)
