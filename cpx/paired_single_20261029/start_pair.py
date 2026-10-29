import subprocess,json,pathlib,shlex,datetime
root=pathlib.Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_pair_single_20261029');run=root/'experiments/trace_reachability_validation/raw/mai_official/mai8b_pair_single_c30x20_20261029';manifest=json.loads((run/'run_config.json').read_text());inside='/androidworld/cpx_pair_single_20261029'
assert not (run/'launch.json').exists(),'Already launched; inspect rather than duplicate'
boundary=pathlib.Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_upstream81a1531_boundary_20261029/experiments/trace_reachability_validation/raw/mai_official/mai8b_upstream81a1531_boundary_complexity_20261029/status.json');assert json.loads(boundary.read_text())['status']=='FINISHED'
for arm,c in manifest['containers'].items():
 assert not (run/(arm+'.status.json')).exists()
 code="import fcntl;f=open('/tmp/mai_device_experiment.claim.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);print('DEVICE_IDLE')"
 subprocess.run(['docker','exec',c,'python','-c',code],check=True)
 subprocess.run(['docker','exec',c,'python','-m','py_compile',inside+'/run_pair_worker.py',inside+'/watch_pair.py',inside+'/experiments/trace_reachability_validation/scripts/mai_upstream_probe.py'],check=True)
launch=dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),containers=manifest['containers'],tag=run.name,source_verified=True,background='docker exec -d + exec nohup + stdin /dev/null')
(run/'launch.json').write_text(json.dumps(launch,indent=2))
for arm,c in manifest['containers'].items():
 cmd='cd '+shlex.quote(inside)+' && exec nohup python -u run_pair_worker.py --arm '+arm+' > '+shlex.quote(str(pathlib.PurePosixPath(inside)/'experiments/trace_reachability_validation/raw/mai_official'/run.name/(arm+'.nohup.log'))) + ' 2>&1 < /dev/null'
 subprocess.run(['docker','exec','-d','-e','CPX_PAIR_ROOT='+inside,c,'sh','-c',cmd],check=True);print('LAUNCHED',arm,c,flush=True)
c=manifest['containers']['BASE'];log=inside+'/experiments/trace_reachability_validation/raw/mai_official/'+run.name+'/watch.nohup.log';cmd='cd '+inside+' && exec nohup python -u watch_pair.py > '+shlex.quote(log)+' 2>&1 < /dev/null';subprocess.run(['docker','exec','-d','-e','CPX_PAIR_ROOT='+inside,c,'sh','-c',cmd],check=True)
print('WATCHER_STARTED',launch['started_at'],flush=True)
